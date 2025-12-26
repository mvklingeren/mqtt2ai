"""AI Agent module for the MQTT AI Daemon.

This module handles interaction with AI providers (OpenAI-compatible, Gemini, Claude)
using their native Python SDKs with function calling for analyzing MQTT
messages and making automation decisions.
"""
import json
import logging
import time
import os
import hashlib
from datetime import datetime

# Import tools module directly (no circular import anymore)
import tools

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

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


# OpenAI function definitions for tools
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
    },
    {
        "type": "function",
        "function": {
            "name": "raise_alert",
            "description": "Raise a security alert with severity 0.0-1.0. Use for suspicious events like motion+window at night. Low (0.0-0.3): log only. Medium (0.3-0.7): notification. High (0.7-1.0): AI takes action with device context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "number",
                        "description": "Alert severity from 0.0 to 1.0. Use 0.9 for break-in scenarios, 0.5 for suspicious activity, 0.2 for informational."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Human-readable description of why the alert is being raised (e.g., 'Motion detected on parking lot followed by window sensor at 04:30')"
                    },
                    "context": {
                        "type": "object",
                        "description": "Optional additional context like trigger topic, field values, timestamps"
                    }
                },
                "required": ["severity", "reason"]
            }
        }
    }
]

# Reduced tool set for rate-limited providers (Groq free tier, Ollama)
# Full 9 tools is ~4K tokens, this set is ~1.2K tokens
OPENAI_TOOLS_MINIMAL = [
    OPENAI_TOOLS[0],  # send_mqtt_message
    OPENAI_TOOLS[1],  # record_pattern_observation
    OPENAI_TOOLS[2],  # create_rule
    OPENAI_TOOLS[8],  # raise_alert - for security events
]


def _announce_ai_action(topic: str, payload: str) -> None:
    """Publish a causation announcement for an AI-initiated MQTT action.
    
    This allows pattern learning to know that this action was automated
    by the AI, not performed manually by a user.
    """
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
        tools.send_mqtt_message(announce_topic, json.dumps(announcement))
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning("Failed to publish AI action announcement: %s", e)


def execute_tool_call(tool_name: str, arguments: dict) -> str:
    """Execute a tool and return the result."""
    # Publish AI_TOOL_CALLED event for validation tracking
    from event_bus import event_bus, EventType
    event_bus.publish(EventType.AI_TOOL_CALLED, {
        "tool": tool_name,
        "arguments": arguments
    })
    
    # For send_mqtt_message, publish announcement first
    if tool_name == "send_mqtt_message":
        topic = arguments.get("topic", "")
        payload = arguments.get("payload", "")
        
        # Skip announcement for announce topic itself to avoid recursion
        if not topic.startswith("mqtt2ai/"):
            _announce_ai_action(topic, payload)
        
        try:
            return tools.send_mqtt_message(topic, payload)
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"Error executing {tool_name}: {e}"
    
    tool_map = {
        "record_pattern_observation": lambda args: tools.record_pattern_observation(
            args["trigger_topic"], args["trigger_field"],
            args["action_topic"], args["delay_seconds"]
        ),
        "create_rule": lambda args: tools.create_rule(
            args["rule_id"], args["trigger_topic"], args["trigger_field"],
            args["trigger_value"], args["action_topic"], args["action_payload"],
            args["avg_delay_seconds"], args["tolerance_seconds"]
        ),
        "reject_pattern": lambda args: tools.reject_pattern(
            args["trigger_topic"], args["trigger_field"],
            args["action_topic"], args.get("reason", "")
        ),
        "report_undo": lambda args: tools.report_undo(args["rule_id"]),
        "toggle_rule": lambda args: tools.toggle_rule(
            args["rule_id"], args["enabled"]
        ),
        "get_learned_rules": lambda args: tools.get_learned_rules(),
        "get_pending_patterns": lambda args: tools.get_pending_patterns(),
        "raise_alert": lambda args: raise_alert(
            args["severity"], args["reason"], args.get("context")
        ),
    }
    
    if tool_name in tool_map:
        try:
            return tool_map[tool_name](arguments)
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"Error executing {tool_name}: {e}"
    return f"Unknown tool: {tool_name}"


# Global reference to AiAgent for alert system (set by daemon)
_alert_agent: 'AiAgent' = None
_alert_config: Config = None


def set_alert_agent(agent: 'AiAgent', config: Config) -> None:
    """Set the global AI agent for alert system use."""
    global _alert_agent, _alert_config
    _alert_agent = agent
    _alert_config = config


def raise_alert(severity: float, reason: str, context: dict = None) -> str:
    """Raise an alert with severity 0.0-1.0.
    
    This function is called by the AI (via tool call) or directly by rules
    when a security-relevant event is detected.
    
    Severity levels:
    - 0.0-0.3: Low priority (log only)
    - 0.3-0.7: Medium priority (notification via MQTT)
    - 0.7-1.0: High priority (AI decides action with full device context)
    
    Args:
        severity: Alert severity from 0.0 to 1.0
        reason: Human-readable reason for the alert
        context: Optional additional context dict
        
    Returns:
        Status message about the alert handling
    """
    from daemon import get_device_tracker
    
    # Clamp severity to valid range
    severity = max(0.0, min(1.0, severity))
    
    # Color coding for logs
    if severity >= 0.7:
        color = "\033[91m"  # Red
        level_name = "HIGH"
    elif severity >= 0.3:
        color = "\033[93m"  # Yellow
        level_name = "MEDIUM"
    else:
        color = "\033[94m"  # Blue
        level_name = "LOW"
    reset = "\033[0m"
    
    logging.info(
        "%s[ALERT %s (%.1f)] %s%s",
        color, level_name, severity, reason, reset
    )
    
    if context:
        logging.debug("Alert context: %s", json.dumps(context))
    
    # Low priority: just log
    if severity < 0.3:
        return f"Alert logged (severity {severity:.1f}): {reason}"
    
    # Medium priority: send notification via MQTT
    if severity < 0.7:
        notification = {
            "severity": severity,
            "level": level_name,
            "reason": reason,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        try:
            tools.send_mqtt_message("mqtt2ai/alerts", json.dumps(notification))
            return f"Alert notification sent (severity {severity:.1f}): {reason}"
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Failed to send alert notification: %s", e)
            return f"Alert logged but notification failed: {e}"
    
    # High priority: AI decides action with full device context
    if _alert_agent is None or _alert_config is None:
        logging.error("Alert agent not initialized - cannot process high severity alert")
        return f"Alert logged but AI not available (severity {severity:.1f}): {reason}"
    
    # Get a point-in-time snapshot of device states for consistent AI decision-making
    # This prevents TOCTOU race conditions where states could change during AI processing
    device_tracker = get_device_tracker()
    if device_tracker:
        snapshot = device_tracker.get_snapshot()
        device_states = snapshot.states
        logging.debug(
            "Device snapshot taken: %d devices, age=%.3fs",
            snapshot.device_count, snapshot.age_seconds
        )
    else:
        device_states = {}
    
    # Build alert prompt for AI
    alert_prompt = _build_alert_prompt(severity, reason, context, device_states)
    
    logging.info(
        "%s[ALERT AI] Triggering AI response for high-severity alert...%s",
        color, reset
    )
    
    # Execute AI call synchronously for alerts (they need immediate response)
    try:
        _execute_alert_ai_call(_alert_agent, alert_prompt)
        return f"Alert processed by AI (severity {severity:.1f}): {reason}"
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Alert AI call failed: %s", e)
        return f"Alert AI call failed: {e}"


def _build_alert_prompt(
    severity: float, reason: str, context: dict, device_states: dict
) -> str:
    """Build the prompt for an alert AI call."""
    lines = [
        "# SECURITY ALERT",
        f"Severity: {severity:.1f} (HIGH PRIORITY)",
        f"Reason: {reason}",
        "",
    ]
    
    if context:
        lines.append("## Alert Context")
        lines.append(json.dumps(context, indent=2))
        lines.append("")
    
    lines.append("## Available Devices and Current States")
    lines.append("Use these to take appropriate action (e.g., activate sirens, turn on lights).")
    lines.append("")
    
    if device_states:
        for topic, state in sorted(device_states.items()):
            # Compact state representation
            state_str = json.dumps(state, separators=(',', ':'))
            if len(state_str) > 100:
                # Truncate long states
                state_str = state_str[:97] + "..."
            lines.append(f"- {topic}: {state_str}")
    else:
        lines.append("(No device states available)")
    
    lines.append("")
    lines.append("## Instructions")
    lines.append("Based on the alert severity and available devices:")
    lines.append("1. Identify appropriate response devices (sirens, lights, notifications)")
    lines.append("2. Use send_mqtt_message to activate them with correct payloads")
    lines.append("3. For sirens/alarms, set state to ON or true")
    lines.append("4. For lights, consider turning them ON at full brightness")
    lines.append("")
    lines.append("Take action NOW to respond to this security alert.")
    
    return "\n".join(lines)


def _execute_alert_ai_call(agent: 'AiAgent', prompt: str) -> None:
    """Execute an AI call for alert response."""
    if not OPENAI_AVAILABLE:
        logging.error("OpenAI SDK not available for alert AI call")
        return
    
    try:
        import httpx
        client = OpenAI(
            base_url=agent.config.openai_api_base,
            api_key=agent.config.openai_api_key,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        
        # Use minimal tools (just send_mqtt_message) for alerts
        alert_tools = [OPENAI_TOOLS[0]]  # send_mqtt_message only
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a security response AI. Your job is to respond to alerts "
                    "by activating appropriate devices. Be decisive and act immediately. "
                    "Use send_mqtt_message to control devices."
                )
            },
            {"role": "user", "content": prompt}
        ]
        
        current_model = agent.config.get_next_model()
        cyan, reset = "\033[96m", "\033[0m"
        purple = "\033[95m"
        
        # Single iteration for alerts (fast response)
        response = client.chat.completions.create(
            model=current_model,
            messages=messages,
            tools=alert_tools,
            tool_choice="auto",
        )
        
        message = response.choices[0].message
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}
                
                logging.info(
                    "%s[Alert Tool] %s(%s)%s",
                    purple, func_name, json.dumps(func_args), reset
                )
                
                result = execute_tool_call(func_name, func_args)
                logging.info("%s[Alert Result] %s%s", purple, result, reset)
        
        if message.content:
            logging.info("%sAlert AI: %s%s", cyan, message.content.strip(), reset)
            
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Alert AI call error: %s", e)
        raise


def timestamp() -> str:
    """Return current timestamp in [HH:MM:SS] format."""
    return datetime.now().strftime("[%H:%M:%S]")


class AiAgent:
    """Handles interaction with AI providers via their Python SDKs."""

    def __init__(self, config: Config):
        self.config = config
        self.prompt_builder = PromptBuilder(config)

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
        """Execute the AI call using the appropriate SDK based on provider."""
        if self.config.ai_provider == "openai-compatible":
            self._execute_openai_api_call(provider, prompt, rules_count, patterns_count)
        elif self.config.ai_provider == "gemini":
            self._execute_gemini_sdk_call(provider, prompt, rules_count, patterns_count)
        elif self.config.ai_provider == "claude":
            self._execute_claude_sdk_call(provider, prompt, rules_count, patterns_count)
        else:
            logging.error("Unknown AI provider: %s", self.config.ai_provider)

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

    def _execute_gemini_sdk_call(self, provider: str, prompt: str,
                                   rules_count: int = 0, patterns_count: int = 0):
        """Execute AI call via Google Gemini SDK with function calling."""
        if not GEMINI_AVAILABLE:
            logging.error(
                "Google GenAI SDK not installed. Run: pip install google-genai"
            )
            return

        try:
            # Initialize Gemini client
            client = genai.Client(api_key=self.config.gemini_api_key)

            cyan, reset = "\033[96m", "\033[0m"
            purple = "\033[95m"
            orange_bold = "\033[1;38;5;208m"

            # Define tools for Gemini
            gemini_tools = self._get_gemini_tool_declarations()

            # Format prompt size for display
            prompt_chars = len(prompt)
            prompt_display = f"{prompt_chars/1000:.1f}K" if prompt_chars >= 1000 else str(prompt_chars)

            logging.info(
                "%s[AI Request] model=%s | prompt=%s chars | rules=%d patterns=%d%s",
                orange_bold, self.config.gemini_model, prompt_display,
                rules_count, patterns_count, reset
            )

            # System instruction
            system_instruction = (
                "You are a home automation AI assistant with access to MQTT tools. "
                "Be brief and concise. "
                "Analyze MQTT messages and take actions using the available functions. "
                "When you detect trigger->action patterns, call record_pattern_observation. "
                "After 3+ observations, call create_rule to formalize the automation. "
                "Use send_mqtt_message to control devices directly when needed."
            )

            # Create config with tools
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=gemini_tools,
            )

            # Iterative tool calling loop
            max_iterations = 10
            contents = [prompt]

            for iteration in range(max_iterations):
                response = client.models.generate_content(
                    model=self.config.gemini_model,
                    contents=contents,
                    config=config,
                )

                # Check for function calls
                if response.candidates and response.candidates[0].content.parts:
                    has_function_call = False
                    for part in response.candidates[0].content.parts:
                        if part.function_call:
                            has_function_call = True
                            func_name = part.function_call.name
                            func_args = dict(part.function_call.args) if part.function_call.args else {}

                            logging.info(
                                "%s[Tool Call] %s(%s)%s",
                                purple, func_name, json.dumps(func_args), reset
                            )

                            # Execute the tool
                            result = execute_tool_call(func_name, func_args)
                            logging.info(
                                "%s[Tool Result] %s%s", purple, result, reset
                            )

                            # Add function response to continue conversation
                            contents.append(response.candidates[0].content)
                            contents.append(
                                types.Content(
                                    role="user",
                                    parts=[types.Part.from_function_response(
                                        name=func_name,
                                        response={"result": result}
                                    )]
                                )
                            )

                    if not has_function_call:
                        # No function calls, print final response
                        if response.text:
                            logging.info("%sAI Response: %s%s", cyan, response.text.strip(), reset)
                        break
                else:
                    # No content, done
                    if response.text:
                        logging.info("%sAI Response: %s%s", cyan, response.text.strip(), reset)
                    break
            else:
                logging.warning("Max tool call iterations reached")

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Gemini API error: %s", e)

    def _get_gemini_tool_declarations(self) -> list:
        """Get Gemini-format tool declarations."""
        # Convert OpenAI tool format to Gemini format
        gemini_tools = []
        for tool in OPENAI_TOOLS:
            func = tool["function"]
            gemini_tools.append(types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=func["name"],
                        description=func["description"],
                        parameters=func["parameters"]
                    )
                ]
            ))
        return gemini_tools

    def _execute_claude_sdk_call(self, provider: str, prompt: str,
                                  rules_count: int = 0, patterns_count: int = 0):
        """Execute AI call via Anthropic Claude SDK with function calling."""
        if not ANTHROPIC_AVAILABLE:
                logging.error(
                "Anthropic SDK not installed. Run: pip install anthropic"
                )
                return

        try:
            # Initialize Anthropic client
            client = Anthropic(api_key=self.config.claude_api_key)

            cyan, reset = "\033[96m", "\033[0m"
            purple = "\033[95m"
            orange_bold = "\033[1;38;5;208m"

            # Format prompt size for display
            prompt_chars = len(prompt)
            prompt_display = f"{prompt_chars/1000:.1f}K" if prompt_chars >= 1000 else str(prompt_chars)

            logging.info(
                "%s[AI Request] model=%s | prompt=%s chars | rules=%d patterns=%d%s",
                orange_bold, self.config.claude_model, prompt_display,
                rules_count, patterns_count, reset
            )

            # System prompt
            system_prompt = (
                "You are a home automation AI assistant with access to MQTT tools. "
                "Be brief and concise. "
                "Analyze MQTT messages and take actions using the available tools. "
                "When you detect trigger->action patterns, call record_pattern_observation. "
                "After 3+ observations, call create_rule to formalize the automation. "
                "Use send_mqtt_message to control devices directly when needed."
            )

            # Convert tools to Claude format
            claude_tools = self._get_claude_tool_definitions()

            # Build messages
            messages = [{"role": "user", "content": prompt}]

            max_iterations = 10
            for iteration in range(max_iterations):
                response = client.messages.create(
                    model=self.config.claude_model,
                    max_tokens=1024,
                    system=system_prompt,
                    tools=claude_tools,
                    messages=messages,
                )

                # Process the response
                has_tool_use = False
                tool_results = []

                for block in response.content:
                    if block.type == "tool_use":
                        has_tool_use = True
                        func_name = block.name
                        func_args = block.input if block.input else {}

                        logging.info(
                            "%s[Tool Call] %s(%s)%s",
                            purple, func_name, json.dumps(func_args), reset
                        )

                        # Execute the tool
                        result = execute_tool_call(func_name, func_args)
                        logging.info(
                            "%s[Tool Result] %s%s", purple, result, reset
                        )

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })

                    elif block.type == "text" and block.text:
                        logging.info("%sAI Response: %s%s", cyan, block.text.strip(), reset)

                if has_tool_use:
                    # Add assistant response and tool results to continue conversation
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})
                else:
                    # No more tool calls, done
                    break

                # Check stop reason
                if response.stop_reason == "end_turn":
                    break
            else:
                logging.warning("Max tool call iterations reached")

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Claude API error: %s", e)

    def _get_claude_tool_definitions(self) -> list:
        """Get Claude-format tool definitions."""
        # Convert OpenAI tool format to Claude format
        claude_tools = []
        for tool in OPENAI_TOOLS:
            func = tool["function"]
            claude_tools.append({
                "name": func["name"],
                "description": func["description"],
                "input_schema": func["parameters"]
            })
        return claude_tools

    def test_connection(self) -> tuple[bool, str]:
        """
        Test AI connection by asking it to write a joke.

        Returns:
            Tuple of (success: bool, message: str with AI response or error)
        """
        provider = self.config.ai_provider.upper()

        if self.config.ai_provider == "openai-compatible":
            return self._test_openai_connection(provider)
        elif self.config.ai_provider == "gemini":
            return self._test_gemini_connection(provider)
        elif self.config.ai_provider == "claude":
            return self._test_claude_connection(provider)
        else:
            return False, f"Unknown AI provider: {self.config.ai_provider}"

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

    def _test_gemini_connection(self, provider: str) -> tuple[bool, str]:
        """Test connection to Gemini API with function calling."""
        if not GEMINI_AVAILABLE:
            return False, "Google GenAI SDK not installed. Run: pip install google-genai"

        try:
            client = genai.Client(api_key=self.config.gemini_api_key)

            # Test basic completion
            response = client.models.generate_content(
                model=self.config.gemini_model,
                contents="Write a very short, funny one-liner joke.",
            )
            joke = response.text.strip() if response.text else "No response"

            # Test function calling
            tool_call_info = ""
            try:
                gemini_tools = self._get_gemini_tool_declarations()
                config = types.GenerateContentConfig(tools=gemini_tools)

                response = client.models.generate_content(
                    model=self.config.gemini_model,
                    contents="Send a test message with your joke to MQTT topic 'test/ai_joke'. Use the send_mqtt_message function.",
                    config=config,
                )

                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.function_call:
                            func_name = part.function_call.name
                            func_args = dict(part.function_call.args) if part.function_call.args else {}
                            tool_call_info = f"\n\nFunction calling works! Called: {func_name}({json.dumps(func_args)})"

                            # Execute the tool call
                            try:
                                result = execute_tool_call(func_name, func_args)
                                tool_call_info += f"\nResult: {result}"
                            except Exception as e:  # pylint: disable=broad-exception-caught
                                tool_call_info += f"\nExecution error: {e}"
                            break
                    else:
                        tool_call_info = "\n\nNote: Model did not use function calling for the test request."

            except Exception as tool_err:  # pylint: disable=broad-exception-caught
                tool_call_info = f"\n\nFunction calling test failed: {tool_err}"

            return True, (
                f"Connected to Gemini API using model {self.config.gemini_model}\n\n"
                f"Joke: {joke}{tool_call_info}"
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            return False, f"Gemini API error: {e}"

    def _test_claude_connection(self, provider: str) -> tuple[bool, str]:
        """Test connection to Claude API with function calling."""
        if not ANTHROPIC_AVAILABLE:
            return False, "Anthropic SDK not installed. Run: pip install anthropic"

        try:
            client = Anthropic(api_key=self.config.claude_api_key)

            # Test basic completion
            response = client.messages.create(
                model=self.config.claude_model,
                max_tokens=256,
                messages=[{"role": "user", "content": "Write a very short, funny one-liner joke."}],
            )
            joke = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    joke = block.text.strip()
                    break

            # Test function calling
            tool_call_info = ""
            try:
                claude_tools = self._get_claude_tool_definitions()

                response = client.messages.create(
                    model=self.config.claude_model,
                    max_tokens=256,
                    tools=claude_tools,
                    messages=[{
                        "role": "user",
                        "content": "Send a test message with your joke to MQTT topic 'test/ai_joke'. Use the send_mqtt_message function."
                    }],
                )

                for block in response.content:
                    if block.type == "tool_use":
                        func_name = block.name
                        func_args = block.input if block.input else {}
                        tool_call_info = f"\n\nFunction calling works! Called: {func_name}({json.dumps(func_args)})"

                        # Execute the tool call
                        try:
                            result = execute_tool_call(func_name, func_args)
                            tool_call_info += f"\nResult: {result}"
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            tool_call_info += f"\nExecution error: {e}"
                        break
                else:
                    tool_call_info = "\n\nNote: Model did not use function calling for the test request."

            except Exception as tool_err:  # pylint: disable=broad-exception-caught
                tool_call_info = f"\n\nFunction calling test failed: {tool_err}"

            return True, (
                f"Connected to Claude API using model {self.config.claude_model}\n\n"
                f"Joke: {joke}{tool_call_info}"
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            return False, f"Claude API error: {e}"
