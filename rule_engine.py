"""Rule engine module for direct execution of learned automation rules.

This module contains the RuleEngine class that executes learned rules
directly without AI for matched triggers, providing fast, deterministic
execution of fixed automation rules.
"""
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from event_bus import event_bus, EventType
from trigger_analyzer import TriggerResult

# Avoid circular imports - use TYPE_CHECKING
if TYPE_CHECKING:
    from mqtt_client import MqttClient
    from knowledge_base import KnowledgeBase


class RuleEngine:
    """Executes learned rules directly without AI for matched triggers.

    This provides fast, deterministic execution of fixed automation rules
    while reserving AI for anomaly detection and pattern learning.
    """

    # Topic for publishing causation announcements
    ANNOUNCE_TOPIC = "mqtt2ai/action/announce"

    def __init__(self, mqtt_client: 'MqttClient', kb: 'KnowledgeBase'):
        self.mqtt = mqtt_client
        self.kb = kb

    def check_and_execute(
        self,
        topic: str,
        payload_str: str,
        trigger_result: TriggerResult
    ) -> bool:
        """Check if any enabled rule matches and execute directly.

        Args:
            topic: The MQTT topic that triggered
            payload_str: The raw payload string (JSON)
            trigger_result: The TriggerResult from TriggerAnalyzer

        Returns:
            True if a rule was executed (handled), False otherwise
        """
        if not trigger_result.should_trigger:
            return False

        # Parse payload
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            return False

        if not isinstance(payload, dict):
            return False

        # Reload rules to get latest state
        rules = self.kb.learned_rules.get("rules", [])

        for rule in rules:
            if not rule.get("enabled", True):
                continue

            if self._matches(rule, topic, payload, trigger_result):
                self._announce_and_execute(rule, topic, trigger_result)
                # Publish RULE_EXECUTED event for validation
                event_bus.publish(EventType.RULE_EXECUTED, {
                    "rule_id": rule.get("id"),
                    "trigger_topic": topic,
                    "trigger_field": trigger_result.field_name,
                    "action_topic": rule.get("action", {}).get("topic")
                })
                return True

        return False

    def _matches(
        self,
        rule: dict,
        topic: str,
        payload: dict,
        trigger_result: TriggerResult
    ) -> bool:
        """Check if a rule matches the current trigger event."""
        trigger = rule.get("trigger", {})

        # Check topic match
        if trigger.get("topic") != topic:
            return False

        # Check field match
        rule_field = trigger.get("field")
        if rule_field != trigger_result.field_name:
            return False

        # Check value match
        rule_value = trigger.get("value")
        actual_value = payload.get(rule_field)

        # Handle type conversion for comparison
        if rule_value == actual_value:
            return True

        # Try string comparison for booleans/numbers
        if str(rule_value).lower() == str(actual_value).lower():
            return True

        return False

    def _announce_and_execute(
        self,
        rule: dict,
        trigger_topic: str,
        trigger_result: TriggerResult
    ):
        """Publish causation announcement and then execute the rule action."""
        rule_id = rule.get("id", "unknown")
        trigger = rule.get("trigger", {})
        action = rule.get("action", {})

        action_topic = action.get("topic", "")
        action_payload = action.get("payload", "{}")

        # Build announcement message
        announcement = {
            "source": "direct_rule",
            "rule_id": rule_id,
            "trigger_topic": trigger_topic,
            "trigger_field": trigger_result.field_name,
            "trigger_value": trigger_result.new_value,
            "action_topic": action_topic,
            "action_payload": action_payload,
            "timestamp": datetime.now().isoformat()
        }

        # Publish announcement first
        green, reset = "\033[92m", "\033[0m"
        logging.info(
            "%s[DIRECT RULE] %s: %s[%s=%s] -> %s%s",
            green, rule_id, trigger_topic, trigger_result.field_name,
            trigger_result.new_value, action_topic, reset
        )

        self.mqtt.publish(self.ANNOUNCE_TOPIC, json.dumps(announcement))

        # Then execute the action
        self.mqtt.publish(action_topic, action_payload)

