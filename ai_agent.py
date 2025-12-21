"""AI Agent module for the MQTT AI Daemon.

This module handles interaction with AI CLI tools (Gemini, Claude, or Codex)
for analyzing MQTT messages and making automation decisions.
"""
import subprocess
import shutil
import json
import logging
from datetime import datetime

from config import Config
from knowledge_base import KnowledgeBase


def timestamp() -> str:
    """Return current timestamp in [HH:MM:SS] format."""
    return datetime.now().strftime("[%H:%M:%S]")


class AiAgent:
    """Handles interaction with AI CLI tools (Gemini, Claude, or Codex)."""

    def __init__(self, config: Config):
        self.config = config

    def _get_cli_command(self) -> tuple[str, list[str]]:
        """Return the CLI executable path and arguments based on provider."""
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

        if self.config.ai_provider == "codex-openai":
            cmd = self.config.codex_command
            args = [
                cmd,
                "exec",
                "--model", self.config.codex_model,
                "--full-auto",  # Auto-approve mode
            ]
            return cmd, args

        # gemini (default)
        cmd = self.config.gemini_command
        args = [
            cmd,
            "--yolo",
            "--model", self.config.gemini_model,
            "--allowed-mcp-server-names", "mqtt-tools",
        ]
        return cmd, args

    def run_analysis(self, messages_snapshot: str, kb: KnowledgeBase,
                     trigger_reason: str):
        """Construct the prompt and call the configured AI CLI."""
        cyan, reset = "\033[96m", "\033[0m"
        provider = self.config.ai_provider.upper()
        logging.info(
            "%s--- AI Check Started [%s] (reason: %s) ---%s",
            cyan, provider, trigger_reason, reset
        )

        prompt = self._build_prompt(messages_snapshot, kb, trigger_reason)
        self._execute_ai_call(provider, prompt)

    def _build_prompt(self, messages_snapshot: str, kb: KnowledgeBase,
                      trigger_reason: str) -> str:
        """Build the prompt for the AI."""
        demo_instruction = (
            "**Demo mode is ENABLED.** " if self.config.demo_mode else ""
        )

        # Helper to format sections
        def format_section(title, data, description):
            if not data or not data.get(list(data.keys())[0]):
                return ""
            return (
                f"\n\n## {title}:\n{description}\n"
                f"{json.dumps(data, indent=2)}"
            )

        rules_section = format_section(
            "Learned Automation Rules",
            kb.learned_rules,
            "Execute these rules when their triggers match:"
        ) or "\n\n## Learned Automation Rules:\nNo learned rules yet.\n"

        patterns_section = format_section(
            "Pending Pattern Observations",
            kb.pending_patterns,
            "These patterns are being tracked but haven't reached 3 occurrences:"
        )

        rejected_section = format_section(
            "Rejected Patterns (DO NOT learn these)",
            kb.rejected_patterns,
            "These patterns have been explicitly rejected. Do NOT record or create:"
        )

        # Safety Check
        safety_reminder = ""
        trigger_lower = trigger_reason.lower()
        if any(x in trigger_lower for x in ["temperature", "smoke", "water", "leak"]):
            safety_reminder = (
                "\n\n**SAFETY ALERT**: This analysis was triggered by a potential "
                "safety event. Check for temperature > 50C, smoke: true, or "
                "water_leak: true conditions and ACT IMMEDIATELY if found. "
                "Safety actions take PRIORITY over pattern learning.\n"
            )

        prompt = (
            f"You are a home automation AI with pattern learning. {demo_instruction}"
            f"{safety_reminder}"
            "IMPORTANT: Your PRIMARY task is to detect trigger→action patterns "
            "and call record_pattern_observation. "
            "Look for PIR/motion sensors (occupancy:true) followed by light/switch "
            "actions (/set topics with state:ON). "
            "When you find such a pattern, ALWAYS call record_pattern_observation "
            "with the trigger topic, field, action topic, and delay in seconds.\n\n"
            "## Rulebook:\n"
            f"{kb.rulebook_content}"
            f"{rules_section}"
            f"{patterns_section}"
            f"{rejected_section}\n\n"
            "## Latest MQTT Messages (analyze for trigger→action patterns):\n"
            f"{messages_snapshot}\n\n"
            "REMINDER: Look for patterns like 'zigbee2mqtt/xxx_pir {occupancy:true}' "
            "followed by 'zigbee2mqtt/xxx/set {state:ON}' and call "
            "record_pattern_observation!\n"
        )
        return prompt

    def _execute_ai_call(self, provider: str, prompt: str):
        """Execute the AI CLI call."""
        try:
            cli_cmd, cli_args = self._get_cli_command()

            if not shutil.which(cli_cmd):
                logging.error(
                    "Error: %s CLI not found or not executable at '%s'",
                    provider, cli_cmd
                )
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
            print(f"{timestamp()} AI Response: {light_blue}{response_text}{reset_color}")

        except subprocess.TimeoutExpired:
            logging.error("%s CLI timed out after 120 seconds", provider)
        except subprocess.CalledProcessError as e:
            logging.error(
                "%s CLI failed with exit code %d: %s",
                provider, e.returncode, e.stderr
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Unexpected error during AI check: %s", e)

    def test_connection(self) -> tuple[bool, str]:
        """
        Test AI connection by asking it to write a joke and send it via MCP.

        Returns:
            Tuple of (success: bool, message: str with AI response or error)
        """
        provider = self.config.ai_provider.upper()
        cli_cmd, cli_args = self._get_cli_command()

        # Check if CLI executable exists
        if not shutil.which(cli_cmd):
            return False, f"{provider} CLI not found or not executable at '{cli_cmd}'"

        test_prompt = (
            "This is a connection test. Write a short, funny joke and send it using "
            "the send_mqtt_message tool to topic 'test/ai_joke'. "
            "After sending, confirm what you did."
        )

        try:
            result = subprocess.run(
                cli_args,
                input=test_prompt,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )

            response_text = result.stdout.strip()
            return True, response_text

        except subprocess.TimeoutExpired:
            return False, f"{provider} CLI timed out after 60 seconds"
        except subprocess.CalledProcessError as e:
            return False, f"{provider} CLI failed with exit code {e.returncode}: {e.stderr}"
        except Exception as e:  # pylint: disable=broad-exception-caught
            return False, f"Unexpected error: {e}"
