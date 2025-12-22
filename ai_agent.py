"""AI Agent module for the MQTT AI Daemon.

This module handles interaction with AI CLI tools (Gemini, Claude, or Codex)
or OpenAI-compatible APIs (Ollama, LM Studio, etc.) for analyzing MQTT 
messages and making automation decisions.
"""
import subprocess
import shutil
import json
import logging
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from config import Config
from knowledge_base import KnowledgeBase

# Import MCP tool implementations for OpenAI function calling
# pylint: disable=import-outside-toplevel
def _get_mcp_tools():
    """Lazy import of MCP tools to avoid circular imports."""
    import mcp_mqtt_server as mcp
    return mcp


# OpenAI function definitions for MCP tools
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send_mqtt_message",
            "description": "Send a message to an MQTT topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The MQTT topic to publish to (e.g., 'zigbee2mqtt/light/set')"
                    },
                    "payload": {
                        "type": "string",
                        "description": "The message payload, typically a JSON string (e.g., '{\"state\": \"ON\"}')"
                    }
                },
                "required": ["topic", "payload"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "record_pattern_observation",
            "description": "Record an observation of a potential trigger->action pattern. Call this when you detect a user manually performing an action after a trigger event. After 3 observations, use create_rule to formalize the pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trigger_topic": {
                        "type": "string",
                        "description": "The MQTT topic that triggered (e.g., 'zigbee2mqtt/hallway_pir')"
                    },
                    "trigger_field": {
                        "type": "string",
                        "description": "The field that changed (e.g., 'occupancy')"
                    },
                    "action_topic": {
                        "type": "string",
                        "description": "The action topic the user interacted with (e.g., 'zigbee2mqtt/hallway_light/set')"
                    },
                    "delay_seconds": {
                        "type": "number",
                        "description": "Time in seconds between trigger and user action"
                    }
                },
                "required": ["trigger_topic", "trigger_field", "action_topic", "delay_seconds"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_rule",
            "description": "Create or update an automation rule based on learned patterns. Use after 3+ pattern observations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id": {
                        "type": "string",
                        "description": "Unique identifier for the rule (e.g., 'hallway_pir_to_light')"
                    },
                    "trigger_topic": {
                        "type": "string",
                        "description": "MQTT topic that triggers the rule"
                    },
                    "trigger_field": {
                        "type": "string",
                        "description": "JSON field to monitor (e.g., 'occupancy', 'contact')"
                    },
                    "trigger_value": {
                        "type": "string",
                        "description": "Value that triggers the rule as JSON (e.g., 'true', '\"ON\"')"
                    },
                    "action_topic": {
                        "type": "string",
                        "description": "MQTT topic to publish to when triggered"
                    },
                    "action_payload": {
                        "type": "string",
                        "description": "Payload to send (e.g., '{\"state\": \"ON\"}')"
                    },
                    "avg_delay_seconds": {
                        "type": "number",
                        "description": "Average delay observed before user action"
                    },
                    "tolerance_seconds": {
                        "type": "number",
                        "description": "Tolerance window for timing"
                    }
                },
                "required": ["rule_id", "trigger_topic", "trigger_field", "trigger_value",
                            "action_topic", "action_payload", "avg_delay_seconds", "tolerance_seconds"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reject_pattern",
            "description": "Reject a pattern to prevent it from being learned. Use when a pattern is coincidental.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trigger_topic": {
                        "type": "string",
                        "description": "The MQTT trigger topic"
                    },
                    "trigger_field": {
                        "type": "string",
                        "description": "The field that triggers (e.g., 'occupancy')"
                    },
                    "action_topic": {
                        "type": "string",
                        "description": "The action topic"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional reason for rejection"
                    }
                },
                "required": ["trigger_topic", "trigger_field", "action_topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "report_undo",
            "description": "Report that a user undid an automated action. Call when user reversed your action within 30 seconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id": {
                        "type": "string",
                        "description": "The ID of the rule that was undone"
                    }
                },
                "required": ["rule_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_rule",
            "description": "Enable or disable a learned automation rule.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id": {
                        "type": "string",
                        "description": "The ID of the rule to toggle"
                    },
                    "enabled": {
                        "type": "boolean",
                        "description": "True to enable, False to disable"
                    }
                },
                "required": ["rule_id", "enabled"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_learned_rules",
            "description": "Get all learned automation rules.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pending_patterns",
            "description": "Get all pending patterns being tracked but not yet rules.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


def execute_tool_call(tool_name: str, arguments: dict) -> str:
    """Execute an MCP tool and return the result."""
    mcp = _get_mcp_tools()
    
    tool_map = {
        "send_mqtt_message": lambda args: mcp.send_mqtt_message(
            args["topic"], args["payload"]
        ),
        "record_pattern_observation": lambda args: mcp.record_pattern_observation(
            args["trigger_topic"], args["trigger_field"],
            args["action_topic"], args["delay_seconds"]
        ),
        "create_rule": lambda args: mcp.create_rule(
            args["rule_id"], args["trigger_topic"], args["trigger_field"],
            args["trigger_value"], args["action_topic"], args["action_payload"],
            args["avg_delay_seconds"], args["tolerance_seconds"]
        ),
        "reject_pattern": lambda args: mcp.reject_pattern(
            args["trigger_topic"], args["trigger_field"],
            args["action_topic"], args.get("reason", "")
        ),
        "report_undo": lambda args: mcp.report_undo(args["rule_id"]),
        "toggle_rule": lambda args: mcp.toggle_rule(
            args["rule_id"], args["enabled"]
        ),
        "get_learned_rules": lambda args: mcp.get_learned_rules(),
        "get_pending_patterns": lambda args: mcp.get_pending_patterns(),
    }
    
    if tool_name in tool_map:
        try:
            return tool_map[tool_name](arguments)
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"Error executing {tool_name}: {e}"
    return f"Unknown tool: {tool_name}"


def timestamp() -> str:
    """Return current timestamp in [HH:MM:SS] format."""
    return datetime.now().strftime("[%H:%M:%S]")


class AiAgent:
    """Handles interaction with AI CLI tools (Gemini, Claude, or Codex)."""

    # Fields to REMOVE from payloads (known noise, not useful for patterns)
    # Everything else is kept by default
    REMOVE_FIELDS = {
        # Zigbee device metadata (noise)
        "linkquality", "voltage", "energy",
        "update", "update_available",
        # Zigbee device settings (rarely change, not triggers)
        "child_lock", "countdown", "indicator_mode", "power_outage_memory",
        # Ring camera/device noise
        "timestamp", "type", "wirelessNetwork", "wirelessSignal",
        "firmwareStatus", "lastUpdate", "stream_Source", "still_Image_URL",
        # Version info
        "installed_version", "latest_version",
        # Tasmota device noise
        "Time", "Uptime", "UptimeSec", "Vcc", "Heap",
        "SleepMode", "Sleep", "LoadAvg", "MqttCount",
        "Hostname", "IPAddress",
    }

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

        # Compress the snapshot to reduce token usage
        compressed_snapshot = self._compress_snapshot(messages_snapshot)
        prompt = self._build_prompt(compressed_snapshot, kb, trigger_reason)
        self._execute_ai_call(provider, prompt)

    def _compress_snapshot(self, messages_snapshot: str) -> str:
        """Compress MQTT payloads by removing known noise fields.

        Uses a blacklist approach: removes known noise fields, keeps everything else.
        Also removes nested objects and null values.
        """
        compressed_lines = []

        for line in messages_snapshot.split('\n'):
            if not line.strip():
                continue

            # Try to find JSON payload in the line
            # Format is typically: [HH:MM:SS] topic/path {"key": "value", ...}
            try:
                # Find the start of JSON payload
                json_start = line.find('{')
                if json_start == -1:
                    # No JSON payload, keep line as-is
                    compressed_lines.append(line)
                    continue

                prefix = line[:json_start]
                json_str = line[json_start:]

                # Parse and filter the JSON
                payload = json.loads(json_str)
                if isinstance(payload, dict):
                    filtered = {
                        k: v for k, v in payload.items()
                        if k not in self.REMOVE_FIELDS  # Blacklist approach
                        and v is not None  # Remove null values
                        and not isinstance(v, dict)  # Remove nested objects
                    }
                    if filtered:
                        compressed_lines.append(
                            f"{prefix}{json.dumps(filtered, separators=(',', ':'))}"
                        )
                    # Skip lines with no remaining fields
                else:
                    # Not a dict, keep as-is
                    compressed_lines.append(line)

            except json.JSONDecodeError:
                # Not valid JSON, keep line as-is
                compressed_lines.append(line)

        return '\n'.join(compressed_lines)

    def _build_prompt(self, messages_snapshot: str, kb: KnowledgeBase,
                      trigger_reason: str) -> str:
        """Build the prompt for the AI."""
        demo_instruction = (
            "**Demo mode is ENABLED - you MUST send a unique joke to jokes/ topic (see Rule 4).** "
            if self.config.demo_mode else ""
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
            "You have MCP tools available. Use the send_mqtt_message tool to publish "
            "MQTT messages - do NOT use shell commands or file operations.\n\n"
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
        """Execute the AI call (CLI or API based on provider)."""
        if self.config.ai_provider == "openai-compatible":
            self._execute_openai_api_call(provider, prompt)
        else:
            self._execute_cli_call(provider, prompt)

    def _execute_openai_api_call(self, provider: str, prompt: str):
        """Execute AI call via OpenAI-compatible API with function calling."""
        if not OPENAI_AVAILABLE:
            logging.error(
                "OpenAI SDK not installed. Run: pip install openai"
            )
            return

        try:
            client = OpenAI(
                base_url=self.config.openai_api_base,
                api_key=self.config.openai_api_key,
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a home automation AI assistant with access to MQTT tools. "
                        "Analyze MQTT messages and take actions using the available functions. "
                        "When you detect trigger->action patterns, call record_pattern_observation. "
                        "After 3+ observations, call create_rule to formalize the automation. "
                        "Use send_mqtt_message to control devices directly when needed."
                    )
                },
                {"role": "user", "content": prompt}
            ]

            cyan, reset = "\033[96m", "\033[0m"
            purple = "\033[95m"  # Magenta/purple for tool calls
            max_iterations = 10  # Prevent infinite loops

            # Try with tool calling first, fall back to no tools if server doesn't support it
            use_tools = True

            for iteration in range(max_iterations):
                try:
                    if use_tools:
                        response = client.chat.completions.create(
                            model=self.config.openai_model,
                            messages=messages,
                            tools=OPENAI_TOOLS,
                            tool_choice="auto",
                            timeout=120,
                        )
                    else:
                        response = client.chat.completions.create(
                            model=self.config.openai_model,
                            messages=messages,
                            timeout=120,
                        )
                except Exception as api_err:
                    # Check if error is about tool calling not being supported
                    err_str = str(api_err)
                    if "tool" in err_str.lower() and use_tools:
                        logging.warning(
                            "Server doesn't support tool calling, falling back to text-only mode"
                        )
                        use_tools = False
                        response = client.chat.completions.create(
                            model=self.config.openai_model,
                            messages=messages,
                            timeout=120,
                        )
                    else:
                        raise

                message = response.choices[0].message

                # Check if the model wants to call functions
                if message.tool_calls:
                    # Add the assistant's response to messages
                    messages.append(message)

                    # Process each tool call
                    for tool_call in message.tool_calls:
                        func_name = tool_call.function.name
                        try:
                            func_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            func_args = {}

                        logging.info(
                            "%s[Tool Call] %s(%s)%s",
                            purple, func_name, json.dumps(func_args), reset
                        )

                        # Execute the tool
                        result = execute_tool_call(func_name, func_args)
                        logging.info(
                            "%s[Tool Result] %s%s", purple, result, reset
                        )

                        # Add the tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })
                else:
                    # No more tool calls, get the final response
                    if message.content:
                        logging.info(
                            "%sAI Response: %s%s",
                            cyan, message.content.strip(), reset
                        )
                    break
            else:
                logging.warning("Max tool call iterations reached")

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("OpenAI API error: %s", e)

    def _execute_cli_call(self, provider: str, prompt: str):
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
            cyan, reset = "\033[96m", "\033[0m"
            logging.info("%sAI Response: %s%s", cyan, response_text, reset)

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
        Test AI connection by asking it to write a joke.

        Returns:
            Tuple of (success: bool, message: str with AI response or error)
        """
        provider = self.config.ai_provider.upper()

        # Use OpenAI API for openai-compatible provider
        if self.config.ai_provider == "openai-compatible":
            return self._test_openai_connection(provider)

        return self._test_cli_connection(provider)

    def _test_openai_connection(self, provider: str) -> tuple[bool, str]:
        """Test connection to OpenAI-compatible API with function calling."""
        if not OPENAI_AVAILABLE:
            return False, "OpenAI SDK not installed. Run: pip install openai"

        try:
            client = OpenAI(
                base_url=self.config.openai_api_base,
                api_key=self.config.openai_api_key,
            )

            # First test: basic completion
            response = client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "user", "content": "Write a very short, funny one-liner joke."}
                ],
                timeout=60,
            )
            joke = response.choices[0].message.content.strip()

            # Second test: try function calling with send_mqtt_message
            tool_call_info = ""
            try:
                response = client.chat.completions.create(
                    model=self.config.openai_model,
                    messages=[
                        {
                            "role": "user",
                            "content": "Send a test message with your joke to MQTT topic 'test/ai_joke'. Use the send_mqtt_message function."
                        }
                    ],
                    tools=OPENAI_TOOLS,
                    tool_choice="auto",
                    timeout=60,
                )

                message = response.choices[0].message
                
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        func_name = tool_call.function.name
                        func_args = tool_call.function.arguments
                        tool_call_info = f"\n\nFunction calling works! Called: {func_name}({func_args})"
                        
                        # Actually execute the tool call for the test
                        try:
                            args = json.loads(func_args)
                            result = execute_tool_call(func_name, args)
                            tool_call_info += f"\nResult: {result}"
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            tool_call_info += f"\nExecution error: {e}"
                else:
                    tool_call_info = "\n\nNote: Model did not use function calling for the test request."

            except Exception as tool_err:  # pylint: disable=broad-exception-caught
                err_str = str(tool_err)
                if "tool" in err_str.lower():
                    tool_call_info = (
                        "\n\nNote: Server doesn't support tool calling. "
                        "To enable, restart vLLM with: --enable-auto-tool-choice --tool-call-parser hermes"
                    )
                else:
                    tool_call_info = f"\n\nFunction calling test failed: {tool_err}"

            return True, (
                f"Connected to {self.config.openai_api_base} using model {self.config.openai_model}\n\n"
                f"Joke: {joke}{tool_call_info}"
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            return False, f"OpenAI API error: {e}"

    def _test_cli_connection(self, provider: str) -> tuple[bool, str]:
        """Test connection to CLI-based AI provider."""
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
