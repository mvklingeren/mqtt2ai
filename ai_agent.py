"""AI Agent module for the MQTT AI Daemon.

This module handles interaction with AI CLI tools (Gemini, Claude, or Codex)
or OpenAI-compatible APIs (Ollama, LM Studio, etc.) for analyzing MQTT
messages and making automation decisions.
"""
import subprocess
import shutil
import json
import logging
import time
import os
import hashlib
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from config import Config
from knowledge_base import KnowledgeBase
from prompt_builder import PromptBuilder


def write_debug_output(debug_dir: str, url: str, body: dict, response: dict = None):
    """Write HTTP call details to a debug file with short hash filename.

    Args:
        debug_dir: Directory to write debug files to
        url: The API URL being called
        body: The request body (dict)
        response: Optional response data (dict)
    """
    # Create debug directory if it doesn't exist
    os.makedirs(debug_dir, exist_ok=True)

    # Generate short hash for filename
    timestamp = datetime.now().isoformat()
    hash_input = f"{timestamp}-{url}".encode()
    short_hash = hashlib.md5(hash_input).hexdigest()[:8]
    filename = os.path.join(debug_dir, f"{short_hash}.txt")

    # Calculate content length
    body_json = json.dumps(body, indent=2)
    content_length = len(body_json.encode('utf-8'))

    # Build debug output
    output_lines = [
        f"=== HTTP Request Debug ===",
        f"Timestamp: {timestamp}",
        f"URL: {url}",
        f"Content-Length: {content_length}",
        f"",
        f"=== Request Body ===",
        body_json,
    ]

    if response:
        response_json = json.dumps(response, indent=2)
        output_lines.extend([
            f"",
            f"=== Response ===",
            response_json,
        ])

    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    logging.debug("Debug output written to %s", filename)


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
            "description": "Record a NEW trigger->action pattern observation. IMPORTANT: First check SKIP PATTERNS section in the prompt - if the trigger[field]->action pair is listed there, DO NOT call this function (rule already exists). Only call for patterns NOT in SKIP PATTERNS.",
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
            "description": "Create a NEW automation rule after 3+ pattern observations. IMPORTANT: First check SKIP PATTERNS section - if the trigger[field]->action pair is listed there, DO NOT call this function (rule already exists).",
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

# Reduced tool set for rate-limited providers (Groq free tier, Ollama)
# Full 8 tools is ~4K tokens, this set is ~1K tokens
OPENAI_TOOLS_MINIMAL = [
    OPENAI_TOOLS[0],  # send_mqtt_message
    OPENAI_TOOLS[1],  # record_pattern_observation
    OPENAI_TOOLS[2],  # create_rule
]


def _announce_ai_action(topic: str, payload: str) -> None:
    """Publish a causation announcement for an AI-initiated MQTT action.
    
    This allows pattern learning to know that this action was automated
    by the AI, not performed manually by a user.
    """
    mcp = _get_mcp_tools()
    
    announcement = {
        "source": "ai_analysis",
        "rule_id": None,
        "trigger_topic": None,
        "trigger_field": None,
        "trigger_value": None,
        "action_topic": topic,
        "action_payload": payload,
        "timestamp": datetime.now().isoformat()
    }
    
    # Publish to the announce topic first
    announce_topic = "mqtt2ai/action/announce"
    try:
        mcp.send_mqtt_message(announce_topic, json.dumps(announcement))
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning("Failed to publish AI action announcement: %s", e)


def execute_tool_call(tool_name: str, arguments: dict) -> str:
    """Execute an MCP tool and return the result."""
    mcp = _get_mcp_tools()
    
    # For send_mqtt_message, publish announcement first
    if tool_name == "send_mqtt_message":
        topic = arguments.get("topic", "")
        payload = arguments.get("payload", "")
        
        # Skip announcement for announce topic itself to avoid recursion
        if not topic.startswith("mqtt2ai/"):
            _announce_ai_action(topic, payload)
        
        try:
            return mcp.send_mqtt_message(topic, payload)
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"Error executing {tool_name}: {e}"
    
    tool_map = {
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

    def __init__(self, config: Config):
        self.config = config
        self.prompt_builder = PromptBuilder(config)

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
                     trigger_reason: str, trigger_result=None):
        """Construct the prompt and call the configured AI CLI.
        
        Args:
            messages_snapshot: Raw MQTT messages as newline-separated string
            kb: KnowledgeBase with rules, patterns, and rulebook
            trigger_reason: Human-readable trigger reason string
            trigger_result: Optional TriggerResult with trigger context
        """
        cyan, reset = "\033[96m", "\033[0m"
        provider = self.config.ai_provider.upper()
        logging.info(
            "%s--- AI Check Started [%s] (reason: %s) ---%s",
            cyan, provider, trigger_reason, reset
        )

        # Use PromptBuilder for intelligent prompt construction
        # Use compact prompt for openai-compatible providers to stay within token limits
        # Groq free tier has 6-14K TPM limits, and tool call iterations accumulate tokens
        if self.config.ai_provider == "openai-compatible":
            prompt = self.prompt_builder.build_compact(
                messages_snapshot, kb, trigger_result, trigger_reason
            )
        else:
            prompt = self.prompt_builder.build(
                messages_snapshot, kb, trigger_result, trigger_reason
            )
        
        logging.debug("Prompt: %d chars (~%d tokens)", len(prompt), len(prompt)//4)

        # Get counts for stats display
        rules_count = len(kb.learned_rules.get('rules', []))
        patterns_count = len(kb.pending_patterns.get('patterns', []))

        self._execute_ai_call(provider, prompt, rules_count, patterns_count)

    def _execute_ai_call(self, provider: str, prompt: str,
                         rules_count: int = 0, patterns_count: int = 0):
        """Execute the AI call (CLI or API based on provider)."""
        if self.config.ai_provider == "openai-compatible":
            self._execute_openai_api_call(provider, prompt, rules_count, patterns_count)
        else:
            self._execute_cli_call(provider, prompt)

    def _call_with_retry(self, client, model: str, messages: list,
                          tools: list, extra_body: dict = None,
                          max_retries: int = 2, base_delay: float = 1.0):
        """Call the API with retry logic for transient errors.
        
        Args:
            client: OpenAI client
            model: Model name
            messages: Chat messages
            tools: Tool definitions
            extra_body: Extra parameters for the request
            max_retries: Maximum number of retries (default 2)
            base_delay: Base delay in seconds for exponential backoff
            
        Returns:
            API response
            
        Raises:
            Exception if all retries fail
        """
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    extra_body=extra_body,
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                last_error = e
                err_str = str(e)
                
                # Check if this is a retryable error (400/429/500/502/503/504)
                is_retryable = any(code in err_str for code in [
                    "400", "429", "500", "502", "503", "504",
                    "rate limit", "timeout", "temporarily"
                ])
                
                if attempt < max_retries and is_retryable:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logging.warning(
                        "API call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1, max_retries + 1, err_str[:100], delay
                    )
                    time.sleep(delay)
                else:
                    # Not retryable or out of retries
                    raise
        
        # Should not reach here, but just in case
        raise last_error  # type: ignore

    def _execute_openai_api_call(self, provider: str, prompt: str,
                                  rules_count: int = 0, patterns_count: int = 0):
        """Execute AI call via OpenAI-compatible API with function calling."""
        if not OPENAI_AVAILABLE:
            logging.error(
                "OpenAI SDK not installed. Run: pip install openai"
            )
            return

        try:
            # Use httpx timeout for proper timeout handling with Ollama
            import httpx
            client = OpenAI(
                base_url=self.config.openai_api_base,
                api_key=self.config.openai_api_key,
                timeout=httpx.Timeout(120.0, connect=10.0),  # Ollama with tools can take 30-60s
            )

            # Detect if using Ollama (for Qwen3 optimizations)
            is_ollama = "11434" in self.config.openai_api_base

            # System prompt with /no_think for Qwen3 models on Ollama
            system_content = (
                "/no_think\n" if is_ollama else ""
            ) + (
                "You are a home automation AI assistant with access to MQTT tools. "
                "Be brief and concise. "
                "Analyze MQTT messages and take actions using the available functions. "
                "When you detect trigger->action patterns, call record_pattern_observation. "
                "After 3+ observations, call create_rule to formalize the automation. "
                "Use send_mqtt_message to control devices directly when needed."
            )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]

            cyan, reset = "\033[96m", "\033[0m"
            purple = "\033[95m"  # Magenta/purple for tool calls
            orange_bold = "\033[1;38;5;208m"  # Bold orange for stats
            # Limit iterations to prevent infinite loops
            # With compact prompt (~500 tokens), Groq can handle the same as others
            max_iterations = 10

            # Try with tool calling first, fall back to no tools if server doesn't support it
            use_tools = True

            # Extra options for Ollama (num_predict limits thinking overhead)
            extra_body = {"options": {"num_predict": 200}} if is_ollama else None

            # Use minimal tools for rate-limited providers (Groq free tier, Ollama)
            # Full 8 tools is ~4K tokens, minimal 2 tools is ~500 tokens
            is_groq = "groq.com" in self.config.openai_api_base
            tools_to_use = OPENAI_TOOLS_MINIMAL if (is_ollama or is_groq) else OPENAI_TOOLS

            # Demo mode: only expose send_mqtt_message tool
            if self.config.demo_mode:
                tools_to_use = [t for t in tools_to_use if t["function"]["name"] == "send_mqtt_message"]

            # Get model for this request (round-robin across configured models)
            current_model = self.config.get_next_model()

            # Format prompt size for display (K for thousands)
            prompt_chars = len(prompt)
            prompt_display = f"{prompt_chars/1000:.1f}K" if prompt_chars >= 1000 else str(prompt_chars)
            est_tokens = prompt_chars // 4

            # Log AI request stats
            logging.info(
                "%s[AI Request] model=%s | prompt=%s chars (~%d tok) | rules=%d patterns=%d | tools=%d%s",
                orange_bold, current_model, prompt_display, est_tokens,
                rules_count, patterns_count, len(tools_to_use), reset
            )

            # Track total tokens across iterations
            total_prompt_tokens = 0
            total_completion_tokens = 0

            for iteration in range(max_iterations):
                try:
                    if use_tools:
                        response = self._call_with_retry(
                            client, current_model, messages, tools_to_use, extra_body
                        )
                    else:
                        response = client.chat.completions.create(
                            model=current_model,
                            messages=messages,
                            extra_body=extra_body,
                        )
                except Exception as api_err:
                    # Check if error is about tool calling not being supported
                    err_str = str(api_err)
                    logging.warning("API error: %s", err_str)
                    if "tool" in err_str.lower() and use_tools:
                        logging.warning(
                            "Server doesn't support tool calling, falling back to text-only mode"
                        )
                        use_tools = False
                        response = client.chat.completions.create(
                            model=current_model,
                            messages=messages,
                            extra_body=extra_body,
                        )
                    else:
                        raise

                message = response.choices[0].message

                # Write debug output if enabled
                if self.config.debug_output:
                    request_body = {
                        "model": current_model,
                        "messages": messages,
                    }
                    if use_tools:
                        request_body["tools"] = tools_to_use
                        request_body["tool_choice"] = "auto"
                    if extra_body:
                        request_body["extra_body"] = extra_body
                    response_data = {
                        "choices": [{"message": {
                            "role": message.role,
                            "content": message.content,
                            "tool_calls": [
                                {"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                for tc in (message.tool_calls or [])
                            ] if message.tool_calls else None
                        }}],
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                            "completion_tokens": response.usage.completion_tokens if response.usage else None,
                            "total_tokens": response.usage.total_tokens if response.usage else None,
                        } if response.usage else None
                    }
                    write_debug_output(
                        self.config.debug_output_dir,
                        f"{self.config.openai_api_base}/chat/completions",
                        request_body,
                        response_data
                    )

                # Track token usage if available
                if response.usage:
                    total_prompt_tokens += response.usage.prompt_tokens or 0
                    total_completion_tokens += response.usage.completion_tokens or 0
                    logging.info(
                        "%s[AI Tokens] iter %d/%d | %d prompt + %d completion = %d total%s",
                        orange_bold, iteration + 1, max_iterations,
                        response.usage.prompt_tokens or 0,
                        response.usage.completion_tokens or 0,
                        response.usage.total_tokens or 0, reset
                    )

                # Check if the model wants to call functions
                if message.tool_calls:
                    # Add the assistant's response to messages as a clean dict
                    # Using model_dump(exclude_none=True) removes null fields that may confuse
                    # non-OpenAI providers like Groq with Llama models
                    # Manually construct the message dictionary to ensure only expected fields are present
                    # and tool_calls are correctly formatted.
                    assistant_message_to_add = {"role": message.role}
                    if message.content:
                        assistant_message_to_add["content"] = message.content
                    if message.tool_calls:
                        assistant_message_to_add["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ]
                    messages.append(assistant_message_to_add)

                    # Track redundant tool calls for early termination
                    redundant_patterns = [
                        "already exists", "No need to re-learn", "No update needed"
                    ]
                    redundant_count = 0

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

                        # Track if this was a redundant call
                        if any(p in result for p in redundant_patterns):
                            redundant_count += 1

                        # Add the tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })

                    # Early termination: if all tool calls in this iteration were redundant
                    if redundant_count == len(message.tool_calls) and redundant_count > 0:
                        logging.info(
                            "%s[Early Stop] All %d tool calls redundant, stopping iteration%s",
                            orange_bold, redundant_count, reset
                        )
                        # Log total tokens
                        logging.info(
                            "%s[AI Total] %d iterations (early stop) | %d prompt + %d completion = %d tokens%s",
                            orange_bold, iteration + 1, total_prompt_tokens,
                            total_completion_tokens,
                            total_prompt_tokens + total_completion_tokens, reset
                        )
                        break

                    # Demo mode: stop after first send_mqtt_message (task complete)
                    if self.config.demo_mode:
                        for tool_call in message.tool_calls:
                            if tool_call.function.name == "send_mqtt_message":
                                logging.info(
                                    "%s[Demo Mode] Message sent, task complete%s",
                                    orange_bold, reset
                                )
                                # Log totals and exit
                                logging.info(
                                    "%s[AI Total] %d iterations (demo) | %d prompt + %d completion = %d tokens%s",
                                    orange_bold, iteration + 1, total_prompt_tokens,
                                    total_completion_tokens,
                                    total_prompt_tokens + total_completion_tokens, reset
                                )
                                # Force exit from while loop
                                iteration = max_iterations
                                break
                        if iteration >= max_iterations:
                            break

                    # After first iteration, compress the original prompt to save tokens
                    # The AI has already analyzed the MQTT data and made decisions
                    if iteration == 0 and len(messages) > 2:
                        messages[1] = {
                            "role": "user",
                            "content": "[MQTT analysis complete - context above. Continue with tool results.]"
                        }
                else:
                    # No more tool calls, get the final response
                    if message.content:
                        logging.info(
                            "%sAI Response: %s%s",
                            cyan, message.content.strip(), reset
                        )
                    # Log total tokens if we had multiple iterations
                    if iteration > 0:
                        logging.info(
                            "%s[AI Total] %d iterations | %d prompt + %d completion = %d tokens%s",
                            orange_bold, iteration + 1, total_prompt_tokens,
                            total_completion_tokens,
                            total_prompt_tokens + total_completion_tokens, reset
                        )
                    break
            else:
                logging.warning("Max tool call iterations reached")
                # Log total tokens even when max iterations reached
                logging.info(
                    "%s[AI Total] %d iterations (max) | %d prompt + %d completion = %d tokens%s",
                    orange_bold, max_iterations, total_prompt_tokens,
                    total_completion_tokens,
                    total_prompt_tokens + total_completion_tokens, reset
                )

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
            import httpx
            client = OpenAI(
                base_url=self.config.openai_api_base,
                api_key=self.config.openai_api_key,
                timeout=httpx.Timeout(120.0, connect=10.0),  # Ollama with tools can take 30-60s
            )

            # Detect if using Ollama (for Qwen3 optimizations)
            is_ollama = "11434" in self.config.openai_api_base
            extra_body = {"options": {"num_predict": 200}} if is_ollama else None
            system_msg = "/no_think\nBe brief." if is_ollama else None

            # Get model for this test (round-robin)
            current_model = self.config.get_next_model()

            # First test: basic completion
            messages = []
            if system_msg:
                messages.append({"role": "system", "content": system_msg})
            messages.append({"role": "user", "content": "Write a very short, funny one-liner joke."})

            response = client.chat.completions.create(
                model=current_model,
                messages=messages,
                extra_body=extra_body,
            )
            joke = response.choices[0].message.content.strip()

            # Second test: try function calling with send_mqtt_message
            tool_call_info = ""
            try:
                messages = []
                if system_msg:
                    messages.append({"role": "system", "content": system_msg})
                messages.append({
                    "role": "user",
                    "content": "Send a test message with your joke to MQTT topic 'test/ai_joke'. Use the send_mqtt_message function."
                })

                # Use minimal tools for Ollama and Groq (rate limited)
                is_groq = "groq.com" in self.config.openai_api_base
                tools_to_use = OPENAI_TOOLS_MINIMAL if (is_ollama or is_groq) else OPENAI_TOOLS

                response = client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    tools=tools_to_use,
                    tool_choice="auto",
                    extra_body=extra_body,
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
                f"Connected to {self.config.openai_api_base} using model {current_model}\n\n"
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
