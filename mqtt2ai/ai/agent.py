"""AI Agent module for the MQTT AI Daemon.

This module handles interaction with AI providers (OpenAI-compatible, Gemini, Claude)
using their native Python SDKs with function calling for analyzing MQTT
messages and making automation decisions.
"""
import json
import logging
from datetime import datetime
from typing import Optional, TYPE_CHECKING

# Import tools module directly (no circular import anymore)
from mqtt2ai.ai import tools
from mqtt2ai.ai.tool_definitions import OPENAI_TOOLS, OPENAI_TOOLS_MINIMAL
from mqtt2ai.ai.providers.openai import OpenAiProvider
from mqtt2ai.ai.providers.gemini import GeminiProvider
from mqtt2ai.ai.providers.claude import ClaudeProvider
from mqtt2ai.ai.alert_handler import AlertHandler
from mqtt2ai.core.event_bus import event_bus, EventType
from mqtt2ai.core.utils import write_debug_output


from mqtt2ai.core.context import RuntimeContext
from mqtt2ai.core.config import Config
from mqtt2ai.rules.knowledge_base import KnowledgeBase
from mqtt2ai.ai.prompt_builder import PromptBuilder

if TYPE_CHECKING:
    from mqtt2ai.rules.device_tracker import DeviceStateTracker
    from mqtt2ai.mqtt.client import MqttClient








def timestamp() -> str:
    """Return current timestamp in [HH:MM:SS] format."""
    return datetime.now().strftime("[%H:%M:%S]")


class AiAgent:
    """Handles interaction with AI providers via their Python SDKs."""

    # Fields to remove from MQTT payloads to save tokens
    REMOVE_FIELDS = [
        "linkquality", "update", "update_available", "voltage", "battery",
        "action_transaction", "action_group", "target_temperature_type"
    ]

    def __init__(
        self,
        config: Config,
        event_bus: 'EventBus',
        device_tracker: Optional['DeviceStateTracker'] = None,
        mqtt_client: Optional['MqttClient'] = None,
    ):
        self.config = config
        self.event_bus = event_bus
        self.device_tracker = device_tracker
        self.mqtt_client = mqtt_client
        self.prompt_builder = PromptBuilder(config)
        self.alert_handler = AlertHandler(config, self, device_tracker)
        self.ai_provider_instance = self._initialize_ai_provider()

    def set_telegram_bot(self, telegram_bot) -> None:
        """Set the Telegram bot reference for alert notifications.

        Args:
            telegram_bot: The TelegramBot instance
        """
        self.alert_handler.set_telegram_bot(telegram_bot)

    def _initialize_ai_provider(self):
        """Initialize the appropriate AI provider based on configuration."""
        if self.config.ai_provider == "openai-compatible":
            return OpenAiProvider(self.config, self)
        if self.config.ai_provider == "gemini":
            return GeminiProvider(self.config, self)
        if self.config.ai_provider == "claude":
            return ClaudeProvider(self.config, self)
        else:
            logging.error("Unknown AI provider configured: %s", self.config.ai_provider)
            return None

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
        if self.ai_provider_instance:
            self.ai_provider_instance.execute_call(prompt, rules_count, patterns_count)
        else:
            logging.error("AI provider not initialized, cannot execute call.")

    def _announce_ai_action(self, topic: str, payload: str) -> None:
        """Publish a causation announcement for an AI-initiated MQTT action."""
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

        announce_topic = "mqtt2ai/action/announce"
        try:
            ctx = RuntimeContext(mqtt_client=self.mqtt_client)
            tools.send_mqtt_message(announce_topic, json.dumps(announcement), context=ctx)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("Failed to publish AI action announcement: %s", e)

    def execute_tool_call(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return the result."""
        # Publish AI_TOOL_CALLED event

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
                self._announce_ai_action(topic, payload)

            try:
                ctx = RuntimeContext(mqtt_client=self.mqtt_client)
                return tools.send_mqtt_message(topic, payload, context=ctx)
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
            "raise_alert": lambda args: self.alert_handler.raise_alert(
                args["severity"], args["reason"], args.get("context")
            ),
            "send_telegram_message": lambda args: tools.send_telegram_message(
                args["message"]
            ),
        }

        if tool_name in tool_map:
            try:
                return tool_map[tool_name](arguments)
            except Exception as e:  # pylint: disable=broad-exception-caught
                return f"Error executing {tool_name}: {e}"
        return f"Unknown tool: {tool_name}"

    def process_telegram_query(self, prompt: str, user_message: str) -> str:
        """Process a Telegram user query through the AI.

        This is a synchronous call that processes the user's request and
        returns a text response suitable for sending back via Telegram.

        Args:
            prompt: The full prompt with device context
            user_message: The original user message (for logging)

        Returns:
            Response text for the user
        """
        cyan, reset = "\033[96m", "\033[0m"
        provider = self.config.ai_provider.upper()
        logging.info(
            "%s--- Telegram Query [%s] ---%s",
            cyan, provider, reset
        )

        if not self.ai_provider_instance:
            return "❌ AI provider not configured"

        try:
            response = self.ai_provider_instance.execute_telegram_query(prompt)
            return response
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Error in Telegram query: %s", e)
            return f"❌ Error processing request: {e}"






    def test_connection(self) -> tuple[bool, str]:
        """Test AI connection by asking it to write a joke."""
        provider = self.config.ai_provider.upper()

        if self.ai_provider_instance:
            return self.ai_provider_instance.test_connection()
        return False, f"Unknown AI provider: {self.config.ai_provider}"

