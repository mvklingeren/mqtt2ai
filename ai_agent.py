import subprocess
import os
import json
import logging
from datetime import datetime

from config import Config
from knowledge_base import KnowledgeBase


def timestamp() -> str:
    """Return current timestamp in [HH:MM:SS] format (helper for legacy format compatibility)."""
    return datetime.now().strftime("[%H:%M:%S]")


class AiAgent:
    """Handles interaction with AI CLI tools (Gemini or Claude)."""
    def __init__(self, config: Config):
        self.config = config

    def _get_cli_command(self) -> tuple[str, list[str]]:
        """Returns the CLI executable path and arguments based on the configured provider."""
        if self.config.ai_provider == "claude":
            cmd = self.config.claude_command
            args = [
                cmd,
                "--dangerously-skip-permissions",
                "--model", self.config.claude_model,
            ]
            # Add MCP config if specified
            if self.config.claude_mcp_config:
                args.extend(["--mcp-config", self.config.claude_mcp_config])
            return cmd, args
        else:  # gemini (default)
            cmd = self.config.gemini_command
            args = [
                cmd,
                "--yolo",
                "--model", self.config.gemini_model,
                "--allowed-mcp-server-names", "mqtt-tools",
            ]
            return cmd, args

    def run_analysis(self, messages_snapshot: str, kb: KnowledgeBase, trigger_reason: str):
        """Constructs the prompt and calls the configured AI CLI."""
        cyan, reset = "\033[96m", "\033[0m"
        provider = self.config.ai_provider.upper()
        logging.info(f"{cyan}--- AI Check Started [{provider}] (reason: {trigger_reason}) ---{reset}")
        
        demo_instruction = "**Demo mode is ENABLED.** " if self.config.demo_mode else ""
        
        # Helper to format sections
        def format_section(title, data, description):
            if not data or not data.get(list(data.keys())[0]): # Check if dict is empty or list inside is empty
                return ""
            return f"\n\n## {title}:\n{description}\n{json.dumps(data, indent=2)}"

        rules_section = format_section(
            "Learned Automation Rules", 
            kb.learned_rules, 
            "Execute these rules when their triggers match:"
        ) or "\n\n## Learned Automation Rules:\nNo learned rules yet.\n"

        patterns_section = format_section(
            "Pending Pattern Observations", 
            kb.pending_patterns, 
            "These patterns are being tracked but haven't reached 3 occurrences yet:"
        )

        rejected_section = format_section(
            "Rejected Patterns (DO NOT learn these)", 
            kb.rejected_patterns, 
            "These patterns have been explicitly rejected by the user. Do NOT record observations or create rules for them:"
        )

        # Safety Check
        safety_reminder = ""
        trigger_lower = trigger_reason.lower()
        if any(x in trigger_lower for x in ["temperature", "smoke", "water", "leak"]):
            safety_reminder = (
                "\n\n**SAFETY ALERT**: This analysis was triggered by a potential safety event. "
                "Check for temperature > 50C, smoke: true, or water_leak: true conditions and ACT IMMEDIATELY if found. "
                "Safety actions take PRIORITY over pattern learning.\n"
            )

        prompt = (
            f"You are a home automation AI with pattern learning capabilities. {demo_instruction}"
            f"{safety_reminder}"
            "IMPORTANT: Your PRIMARY task is to detect trigger→action patterns and call record_pattern_observation. "
            "Look for PIR/motion sensors (occupancy:true) followed by light/switch actions (/set topics with state:ON). "
            "When you find such a pattern, ALWAYS call record_pattern_observation with the trigger topic, field, action topic, and delay in seconds.\n\n"
            "## Rulebook:\n"
            f"{kb.rulebook_content}"
            f"{rules_section}"
            f"{patterns_section}"
            f"{rejected_section}\n\n"
            "## Latest MQTT Messages (analyze for trigger→action patterns):\n"
            f"{messages_snapshot}\n\n"
            "REMINDER: Look for patterns like 'zigbee2mqtt/xxx_pir {occupancy:true}' followed by 'zigbee2mqtt/xxx/set {state:ON}' and call record_pattern_observation!\n"
        )

        try:
            cli_cmd, cli_args = self._get_cli_command()
            
            if not os.access(cli_cmd, os.X_OK):
                logging.error(f"Error: {provider} CLI not found or not executable at '{cli_cmd}'")
                return

            # Call AI CLI
            result = subprocess.run(
                cli_args,
                input=prompt,
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            
            response_text = result.stdout.strip()
            light_blue, reset_color = "\033[94m", "\033[0m"
            # Using print here for distinct output visibility, or could use logging.info with special formatting
            print(f"{timestamp()} AI Response: {light_blue}{response_text}{reset_color}")

        except subprocess.TimeoutExpired:
            logging.error(f"{provider} CLI timed out after 120 seconds")
        except subprocess.CalledProcessError as e:
            logging.error(f"{provider} CLI failed with exit code {e.returncode}: {e.stderr}")
        except Exception as e:
            logging.error(f"Unexpected error during AI check: {e}")

