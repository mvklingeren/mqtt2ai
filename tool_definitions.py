# tool_definitions.py

"""Defines the tool schemas for AI function calling."""

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
                        "description": (
                            "The MQTT topic to publish to "
                            "(e.g., 'zigbee2mqtt/light/set')"
                        )
                    },
                    "payload": {
                        "type": "string",
                        "description": (
                            "The message payload, typically a JSON string "
                            "(e.g., '{\"state\": \"ON\"}')"
                        )
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
            "description": (
                "Record a NEW trigger->action pattern observation. IMPORTANT: First check "
                "SKIP PATTERNS section in the prompt - if the trigger[field]->action pair "
                "is listed there, DO NOT call this function (rule already exists). "
                "Only call for patterns NOT in SKIP PATTERNS."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "trigger_topic": {
                        "type": "string",
                        "description": (
                            "The MQTT topic that triggered "
                            "(e.g., 'zigbee2mqtt/hallway_pir')"
                        )
                    },
                    "trigger_field": {
                        "type": "string",
                        "description": "The field that changed (e.g., 'occupancy')"
                    },
                    "action_topic": {
                        "type": "string",
                        "description": (
                            "The action topic the user interacted with "
                            "(e.g., 'zigbee2mqtt/hallway_light/set')"
                        )
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
            "description": (
                "Create a NEW automation rule after 3+ pattern observations. "
                "IMPORTANT: First check SKIP PATTERNS section - if the trigger[field]->action "
                "pair is listed there, DO NOT call this function (rule already exists)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id": {
                        "type": "string",
                        "description": (
                            "Unique identifier for the rule "
                            "(e.g., 'hallway_pir_to_light')"
                        )
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
                        "description": (
                            "Value that triggers the rule as JSON "
                            "(e.g., 'true', '\"ON\"')"
                        )
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
                "required": [
                    "rule_id", "trigger_topic", "trigger_field", "trigger_value",
                    "action_topic", "action_payload", "avg_delay_seconds", "tolerance_seconds"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reject_pattern",
            "description": (
                "Reject a pattern to prevent it from being learned. "
                "Use when a pattern is coincidental."
            ),
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
            "description": (
                "Report that a user undid an automated action. "
                "Call when user reversed your action within 30 seconds."
            ),
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
            "description": (
                "Raise a security alert with severity 0.0-1.0. "
                "Use for suspicious events like motion+window at night. "
                "Low (0.0-0.3): log only. Medium (0.3-0.7): notification. "
                "High (0.7-1.0): AI takes action with device context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "number",
                        "description": (
                            "Alert severity from 0.0 to 1.0. "
                            "Use 0.9 for break-in scenarios, 0.5 for suspicious activity, "
                            "0.2 for informational."
                        )
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Human-readable description of why the alert is being raised "
                            "(e.g., 'Motion detected on parking lot followed by window sensor')"
                        )
                    },
                    "context": {
                        "type": "object",
                        "description": (
                            "Optional additional context like trigger topic, "
                            "field values, timestamps"
                        )
                    }
                },
                "required": ["severity", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_telegram_message",
            "description": (
                "Send a message to all authorized Telegram users. "
                "Use for important notifications, confirmations, or status updates. "
                "Supports Markdown formatting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": (
                            "The message text to send. Supports Markdown formatting. "
                            "Example: '*Alert*: Motion detected in living room'"
                        )
                    }
                },
                "required": ["message"]
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
