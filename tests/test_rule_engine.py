"""Tests for the RuleEngine class."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from knowledge_base import KnowledgeBase
from trigger_analyzer import TriggerResult
from rule_engine import RuleEngine


@pytest.fixture
def mock_mqtt_client():
    """Create a mock MqttClient."""
    client = MagicMock()
    client.publish = MagicMock()
    return client


@pytest.fixture
def sample_rule():
    """A sample learned rule for testing."""
    return {
        "id": "test_pir_to_light",
        "trigger": {
            "topic": "zigbee2mqtt/test_pir",
            "field": "occupancy",
            "value": True
        },
        "action": {
            "topic": "zigbee2mqtt/test_light/set",
            "payload": '{"state": "ON"}'
        },
        "timing": {
            "avg_delay_seconds": 2.0,
            "tolerance_seconds": 1.0
        },
        "confidence": {
            "occurrences": 5,
            "last_triggered": "2024-01-01T12:00:00"
        },
        "enabled": True
    }


@pytest.fixture
def disabled_rule():
    """A disabled rule for testing."""
    return {
        "id": "disabled_rule",
        "trigger": {
            "topic": "zigbee2mqtt/test_pir",
            "field": "occupancy",
            "value": True
        },
        "action": {
            "topic": "zigbee2mqtt/other_light/set",
            "payload": '{"state": "ON"}'
        },
        "enabled": False
    }


@pytest.fixture
def kb_with_rules(config_with_temp_files, sample_rule):
    """Create a KnowledgeBase with a sample rule."""
    kb = KnowledgeBase(config_with_temp_files)
    kb.learned_rules = {"rules": [sample_rule]}
    kb.pending_patterns = {"patterns": []}
    kb.rejected_patterns = {"patterns": []}
    return kb


@pytest.fixture
def trigger_result_match():
    """A TriggerResult that matches the sample rule."""
    return TriggerResult(
        should_trigger=True,
        reason="State field 'occupancy' changed",
        field_name="occupancy",
        old_value=False,
        new_value=True
    )


@pytest.fixture
def trigger_result_no_match():
    """A TriggerResult that doesn't match (different field)."""
    return TriggerResult(
        should_trigger=True,
        reason="State field 'contact' changed",
        field_name="contact",
        old_value=True,
        new_value=False
    )


@pytest.fixture
def trigger_result_no_trigger():
    """A TriggerResult that indicates no trigger."""
    return TriggerResult(
        should_trigger=False,
        reason="No significant changes"
    )


class TestRuleEngineMatching:
    """Tests for rule matching logic."""

    def test_matches_exact_rule(self, mock_mqtt_client, kb_with_rules, trigger_result_match):
        """Test that a matching rule is detected."""
        engine = RuleEngine(mock_mqtt_client, kb_with_rules)

        topic = "zigbee2mqtt/test_pir"
        payload_str = '{"occupancy": true, "battery": 85}'

        result = engine.check_and_execute(topic, payload_str, trigger_result_match)

        assert result is True
        # Should have published 2 messages: announcement + action
        assert mock_mqtt_client.publish.call_count == 2

    def test_no_match_different_topic(self, mock_mqtt_client, kb_with_rules, trigger_result_match):
        """Test that wrong topic doesn't match."""
        engine = RuleEngine(mock_mqtt_client, kb_with_rules)

        topic = "zigbee2mqtt/other_pir"  # Different topic
        payload_str = '{"occupancy": true}'

        result = engine.check_and_execute(topic, payload_str, trigger_result_match)

        assert result is False
        mock_mqtt_client.publish.assert_not_called()

    def test_no_match_different_field(self, mock_mqtt_client, kb_with_rules, trigger_result_no_match):
        """Test that wrong field doesn't match."""
        engine = RuleEngine(mock_mqtt_client, kb_with_rules)

        topic = "zigbee2mqtt/test_pir"
        payload_str = '{"contact": false}'

        result = engine.check_and_execute(topic, payload_str, trigger_result_no_match)

        assert result is False
        mock_mqtt_client.publish.assert_not_called()

    def test_no_trigger_returns_false(self, mock_mqtt_client, kb_with_rules, trigger_result_no_trigger):
        """Test that non-triggering events return False immediately."""
        engine = RuleEngine(mock_mqtt_client, kb_with_rules)

        topic = "zigbee2mqtt/test_pir"
        payload_str = '{"occupancy": true}'

        result = engine.check_and_execute(topic, payload_str, trigger_result_no_trigger)

        assert result is False
        mock_mqtt_client.publish.assert_not_called()

    def test_disabled_rule_not_executed(self, mock_mqtt_client, config_with_temp_files,
                                         disabled_rule, trigger_result_match):
        """Test that disabled rules are not executed."""
        kb = KnowledgeBase(config_with_temp_files)
        kb.learned_rules = {"rules": [disabled_rule]}

        engine = RuleEngine(mock_mqtt_client, kb)

        topic = "zigbee2mqtt/test_pir"
        payload_str = '{"occupancy": true}'

        result = engine.check_and_execute(topic, payload_str, trigger_result_match)

        assert result is False
        mock_mqtt_client.publish.assert_not_called()

    def test_invalid_json_returns_false(self, mock_mqtt_client, kb_with_rules, trigger_result_match):
        """Test that invalid JSON payloads return False."""
        engine = RuleEngine(mock_mqtt_client, kb_with_rules)

        topic = "zigbee2mqtt/test_pir"
        payload_str = "not valid json"

        result = engine.check_and_execute(topic, payload_str, trigger_result_match)

        assert result is False
        mock_mqtt_client.publish.assert_not_called()

    def test_non_dict_payload_returns_false(self, mock_mqtt_client, kb_with_rules, trigger_result_match):
        """Test that non-dict JSON payloads return False."""
        engine = RuleEngine(mock_mqtt_client, kb_with_rules)

        topic = "zigbee2mqtt/test_pir"
        payload_str = '["array", "not", "dict"]'

        result = engine.check_and_execute(topic, payload_str, trigger_result_match)

        assert result is False
        mock_mqtt_client.publish.assert_not_called()


class TestRuleEngineExecution:
    """Tests for rule execution and announcement."""

    def test_announcement_published_before_action(self, mock_mqtt_client, kb_with_rules,
                                                   trigger_result_match):
        """Test that announcement is published before the action."""
        engine = RuleEngine(mock_mqtt_client, kb_with_rules)

        topic = "zigbee2mqtt/test_pir"
        payload_str = '{"occupancy": true}'

        engine.check_and_execute(topic, payload_str, trigger_result_match)

        # Check call order: announcement first, then action
        calls = mock_mqtt_client.publish.call_args_list
        assert len(calls) == 2

        # First call should be to announce topic
        announce_topic, announce_payload = calls[0][0]
        assert announce_topic == "mqtt2ai/action/announce"

        # Second call should be to action topic
        action_topic, action_payload = calls[1][0]
        assert action_topic == "zigbee2mqtt/test_light/set"
        assert action_payload == '{"state": "ON"}'

    def test_announcement_contains_required_fields(self, mock_mqtt_client, kb_with_rules,
                                                    trigger_result_match):
        """Test that announcement contains all required fields."""
        engine = RuleEngine(mock_mqtt_client, kb_with_rules)

        topic = "zigbee2mqtt/test_pir"
        payload_str = '{"occupancy": true}'

        engine.check_and_execute(topic, payload_str, trigger_result_match)

        # Parse the announcement payload
        announce_call = mock_mqtt_client.publish.call_args_list[0]
        announce_payload = json.loads(announce_call[0][1])

        # Check required fields
        assert announce_payload["source"] == "direct_rule"
        assert announce_payload["rule_id"] == "test_pir_to_light"
        assert announce_payload["trigger_topic"] == "zigbee2mqtt/test_pir"
        assert announce_payload["trigger_field"] == "occupancy"
        assert announce_payload["trigger_value"] is True
        assert announce_payload["action_topic"] == "zigbee2mqtt/test_light/set"
        assert announce_payload["action_payload"] == '{"state": "ON"}'
        assert "timestamp" in announce_payload

    def test_value_matching_with_string_boolean(self, mock_mqtt_client, config_with_temp_files,
                                                 trigger_result_match):
        """Test that string 'true' matches boolean True."""
        kb = KnowledgeBase(config_with_temp_files)
        kb.learned_rules = {"rules": [{
            "id": "string_bool_rule",
            "trigger": {
                "topic": "zigbee2mqtt/test_pir",
                "field": "occupancy",
                "value": "true"  # String instead of boolean
            },
            "action": {
                "topic": "zigbee2mqtt/test_light/set",
                "payload": '{"state": "ON"}'
            },
            "enabled": True
        }]}

        engine = RuleEngine(mock_mqtt_client, kb)

        topic = "zigbee2mqtt/test_pir"
        payload_str = '{"occupancy": true}'  # Boolean in payload

        result = engine.check_and_execute(topic, payload_str, trigger_result_match)

        assert result is True


class TestRuleEngineMultipleRules:
    """Tests for handling multiple rules."""

    def test_first_matching_rule_executed(self, mock_mqtt_client, config_with_temp_files,
                                           sample_rule, trigger_result_match):
        """Test that only the first matching rule is executed."""
        second_rule = {
            "id": "second_rule",
            "trigger": {
                "topic": "zigbee2mqtt/test_pir",
                "field": "occupancy",
                "value": True
            },
            "action": {
                "topic": "zigbee2mqtt/second_light/set",
                "payload": '{"state": "ON"}'
            },
            "enabled": True
        }

        kb = KnowledgeBase(config_with_temp_files)
        kb.learned_rules = {"rules": [sample_rule, second_rule]}

        engine = RuleEngine(mock_mqtt_client, kb)

        topic = "zigbee2mqtt/test_pir"
        payload_str = '{"occupancy": true}'

        result = engine.check_and_execute(topic, payload_str, trigger_result_match)

        assert result is True

        # Only first rule's action should be executed
        action_call = mock_mqtt_client.publish.call_args_list[1]
        assert action_call[0][0] == "zigbee2mqtt/test_light/set"

    def test_no_rules_returns_false(self, mock_mqtt_client, config_with_temp_files,
                                     trigger_result_match):
        """Test that no rules returns False."""
        kb = KnowledgeBase(config_with_temp_files)
        kb.learned_rules = {"rules": []}

        engine = RuleEngine(mock_mqtt_client, kb)

        topic = "zigbee2mqtt/test_pir"
        payload_str = '{"occupancy": true}'

        result = engine.check_and_execute(topic, payload_str, trigger_result_match)

        assert result is False
        mock_mqtt_client.publish.assert_not_called()

