"""Tests for the tools module."""
import json
import os
import sys
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

# Import the functions we want to test (after setting up mocks)
from mqtt2ai.ai.tools import (
    send_mqtt_message,
    send_telegram_message,
    create_rule,
    get_learned_rules,
    record_pattern_observation,
    get_pending_patterns,
    delete_rule,
    toggle_rule,
    clear_pending_patterns,
    reject_pattern,
    get_rejected_patterns,
    remove_rejected_pattern,
    report_undo,
    _clear_pending_pattern,
    _is_pattern_rejected,
    _add_rejected_pattern,
    ToolHandler,
    LEARNED_RULES_FILE,
    PENDING_PATTERNS_FILE,
    REJECTED_PATTERNS_FILE,
)


@pytest.fixture
def mock_files(temp_dir, monkeypatch):
    """Mock the file paths to use temp directory."""
    learned_rules = os.path.join(temp_dir, "learned_rules.json")
    pending_patterns = os.path.join(temp_dir, "pending_patterns.json")
    rejected_patterns = os.path.join(temp_dir, "rejected_patterns.json")

    # Patch the module-level constants
    monkeypatch.setattr("mqtt2ai.ai.tools.LEARNED_RULES_FILE", learned_rules)
    monkeypatch.setattr("mqtt2ai.ai.tools.PENDING_PATTERNS_FILE", pending_patterns)
    monkeypatch.setattr("mqtt2ai.ai.tools.REJECTED_PATTERNS_FILE", rejected_patterns)

    return {
        "learned_rules": learned_rules,
        "pending_patterns": pending_patterns,
        "rejected_patterns": rejected_patterns
    }


# Mock RuntimeContext for tests
class MockRuntimeContext:
    def __init__(self, mqtt_client_mock, telegram_bot=None, disable_new_rules=False):
        self.mqtt_client = mqtt_client_mock
        self.device_tracker = MagicMock()  # Device tracker not directly used by tools here
        self.telegram_bot = telegram_bot
        self.disable_new_rules = disable_new_rules

@pytest.fixture
def mock_context():
    """Provides a mock RuntimeContext for tool functions."""
    mock_client = MagicMock()
    mock_client.publish.return_value = True
    return MockRuntimeContext(mock_client)


class TestSendMqttMessage:
    """Tests for send_mqtt_message function."""

    def test_send_mqtt_message_success(self, mock_context):
        """Test successful MQTT message sending."""
        mock_context.mqtt_client.publish.return_value = True
        result = send_mqtt_message("test/topic", '{"state": "ON"}', context=mock_context)

        assert "Successfully" in result
        mock_context.mqtt_client.publish.assert_called_once()

    def test_send_mqtt_message_correct_topic(self, mock_context):
        """Test that correct topic is used."""
        mock_context.mqtt_client.publish.return_value = True
        send_mqtt_message("zigbee2mqtt/light/set", '{"state": "ON"}', context=mock_context)
        call_args = mock_context.mqtt_client.publish.call_args[0]
        assert call_args[0] == "zigbee2mqtt/light/set"

    def test_send_mqtt_message_correct_payload(self, mock_context):
        """Test that correct payload is used."""
        mock_context.mqtt_client.publish.return_value = True
        payload = '{"state": "ON", "brightness": 100}'
        send_mqtt_message("test/topic", payload, context=mock_context)

        call_args = mock_context.mqtt_client.publish.call_args[0]
        assert call_args[1] == payload

    def test_send_mqtt_message_failure(self, mock_context):
        """Test handling when publish fails."""
        mock_context.mqtt_client.publish.return_value = False
        result = send_mqtt_message("test/topic", '{"state": "ON"}', context=mock_context)

        assert "Error" in result or "Failed" in result


class TestCreateRule:
    """Tests for create_rule function."""

    def test_create_new_rule(self, mock_files):
        """Test creating a new rule."""
        result = create_rule(
            rule_id="test_rule",
            trigger_topic="zigbee2mqtt/pir",
            trigger_field="occupancy",
            trigger_value="true",
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}',
            avg_delay_seconds=2.0,
            tolerance_seconds=1.0
        )

        assert "created" in result.lower()

        # Verify file was created
        with open(mock_files["learned_rules"], "r") as f:
            data = json.load(f)
        assert len(data["rules"]) == 1
        assert data["rules"][0]["id"] == "test_rule"

    def test_create_rule_parses_boolean_trigger(self, mock_files):
        """Test that trigger value is parsed as JSON."""
        create_rule(
            rule_id="test_rule",
            trigger_topic="zigbee2mqtt/pir",
            trigger_field="occupancy",
            trigger_value="true",  # String "true"
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}',
            avg_delay_seconds=2.0,
            tolerance_seconds=1.0
        )

        with open(mock_files["learned_rules"], "r") as f:
            data = json.load(f)
        # Should be parsed to boolean True
        assert data["rules"][0]["trigger"]["value"] is True

    def test_update_existing_rule(self, mock_files):
        """Test updating an existing rule with different trigger value."""
        # Create initial rule with trigger_value="true"
        create_rule(
            rule_id="test_rule",
            trigger_topic="zigbee2mqtt/pir",
            trigger_field="occupancy",
            trigger_value="true",
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}',
            avg_delay_seconds=2.0,
            tolerance_seconds=1.0
        )

        # Update the same rule with a different trigger value
        # Note: same trigger value would return "already exists"
        result = create_rule(
            rule_id="test_rule",
            trigger_topic="zigbee2mqtt/pir",
            trigger_field="occupancy",
            trigger_value="false",  # Different value triggers update
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "OFF"}',
            avg_delay_seconds=3.0,
            tolerance_seconds=1.5
        )

        assert "updated" in result.lower()

        with open(mock_files["learned_rules"], "r") as f:
            data = json.load(f)
        # Should still be only one rule
        assert len(data["rules"]) == 1
        # Occurrences should be incremented
        assert data["rules"][0]["confidence"]["occurrences"] == 4

    def test_create_rule_rejected_pattern(self, mock_files):
        """Test that rejected patterns cannot be created as rules."""
        # Add to rejected patterns first
        _add_rejected_pattern(
            "zigbee2mqtt/pir",
            "occupancy",
            "zigbee2mqtt/light/set",
            "test rejection"
        )

        result = create_rule(
            rule_id="test_rule",
            trigger_topic="zigbee2mqtt/pir",
            trigger_field="occupancy",
            trigger_value="true",
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}',
            avg_delay_seconds=2.0,
            tolerance_seconds=1.0
        )

        assert "rejected" in result.lower()

    def test_create_rule_clears_pending(self, mock_files):
        """Test that creating a rule clears pending patterns."""
        # Create pending pattern first
        record_pattern_observation(
            "zigbee2mqtt/pir",
            "occupancy",
            "zigbee2mqtt/light/set",
            2.0
        )

        # Create rule
        create_rule(
            rule_id="test_rule",
            trigger_topic="zigbee2mqtt/pir",
            trigger_field="occupancy",
            trigger_value="true",
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}',
            avg_delay_seconds=2.0,
            tolerance_seconds=1.0
        )

        # Pending pattern should be cleared
        with open(mock_files["pending_patterns"], "r") as f:
            data = json.load(f)
        assert len(data["patterns"]) == 0


class TestGetLearnedRules:
    """Tests for get_learned_rules function."""

    def test_get_learned_rules_empty(self, mock_files):
        """Test getting rules when none exist."""
        result = get_learned_rules()
        data = json.loads(result)

        assert "rules" in data
        assert len(data["rules"]) == 0

    def test_get_learned_rules_with_data(self, mock_files):
        """Test getting rules when they exist."""
        # Create a rule first
        create_rule(
            rule_id="test_rule",
            trigger_topic="zigbee2mqtt/pir",
            trigger_field="occupancy",
            trigger_value="true",
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}',
            avg_delay_seconds=2.0,
            tolerance_seconds=1.0
        )

        result = get_learned_rules()
        data = json.loads(result)

        assert len(data["rules"]) == 1
        assert data["rules"][0]["id"] == "test_rule"


class TestRecordPatternObservation:
    """Tests for record_pattern_observation function."""

    def test_record_first_observation(self, mock_files):
        """Test recording first pattern observation."""
        result = record_pattern_observation(
            "zigbee2mqtt/pir",
            "occupancy",
            "zigbee2mqtt/light/set",
            2.0
        )

        assert "1/3" in result

        with open(mock_files["pending_patterns"], "r") as f:
            data = json.load(f)
        assert len(data["patterns"]) == 1
        assert len(data["patterns"][0]["observations"]) == 1

    def test_record_multiple_observations(self, mock_files):
        """Test recording multiple observations."""
        record_pattern_observation(
            "zigbee2mqtt/pir", "occupancy", "zigbee2mqtt/light/set", 2.0
        )
        record_pattern_observation(
            "zigbee2mqtt/pir", "occupancy", "zigbee2mqtt/light/set", 2.5
        )
        result = record_pattern_observation(
            "zigbee2mqtt/pir", "occupancy", "zigbee2mqtt/light/set", 3.0
        )

        assert "ready to create rule" in result.lower()

        with open(mock_files["pending_patterns"], "r") as f:
            data = json.load(f)
        assert len(data["patterns"][0]["observations"]) == 3

    def test_record_rejected_pattern(self, mock_files):
        """Test that rejected patterns are not recorded."""
        _add_rejected_pattern(
            "zigbee2mqtt/pir",
            "occupancy",
            "zigbee2mqtt/light/set",
            "test"
        )

        result = record_pattern_observation(
            "zigbee2mqtt/pir",
            "occupancy",
            "zigbee2mqtt/light/set",
            2.0
        )

        assert "rejected" in result.lower()

    def test_record_different_patterns(self, mock_files):
        """Test recording different patterns."""
        record_pattern_observation(
            "zigbee2mqtt/pir_a", "occupancy", "zigbee2mqtt/light_a/set", 2.0
        )
        record_pattern_observation(
            "zigbee2mqtt/pir_b", "occupancy", "zigbee2mqtt/light_b/set", 2.0
        )

        with open(mock_files["pending_patterns"], "r") as f:
            data = json.load(f)
        assert len(data["patterns"]) == 2


class TestGetPendingPatterns:
    """Tests for get_pending_patterns function."""

    def test_get_pending_patterns_empty(self, mock_files):
        """Test getting patterns when none exist."""
        result = get_pending_patterns()
        data = json.loads(result)

        assert "patterns" in data
        assert len(data["patterns"]) == 0

    def test_get_pending_patterns_with_data(self, mock_files):
        """Test getting patterns when they exist."""
        record_pattern_observation(
            "zigbee2mqtt/pir", "occupancy", "zigbee2mqtt/light/set", 2.0
        )

        result = get_pending_patterns()
        data = json.loads(result)

        assert len(data["patterns"]) == 1


class TestDeleteRule:
    """Tests for delete_rule function."""

    def test_delete_existing_rule(self, mock_files):
        """Test deleting an existing rule."""
        create_rule(
            rule_id="test_rule",
            trigger_topic="zigbee2mqtt/pir",
            trigger_field="occupancy",
            trigger_value="true",
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}',
            avg_delay_seconds=2.0,
            tolerance_seconds=1.0
        )

        result = delete_rule("test_rule")

        assert "deleted" in result.lower()

        with open(mock_files["learned_rules"], "r") as f:
            data = json.load(f)
        assert len(data["rules"]) == 0

    def test_delete_nonexistent_rule(self, mock_files):
        """Test deleting a rule that doesn't exist."""
        result = delete_rule("nonexistent_rule")

        assert "not found" in result.lower()


class TestToggleRule:
    """Tests for toggle_rule function."""

    def test_disable_rule(self, mock_files):
        """Test disabling a rule."""
        create_rule(
            rule_id="test_rule",
            trigger_topic="zigbee2mqtt/pir",
            trigger_field="occupancy",
            trigger_value="true",
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}',
            avg_delay_seconds=2.0,
            tolerance_seconds=1.0
        )

        result = toggle_rule("test_rule", False)

        assert "disabled" in result.lower()

        with open(mock_files["learned_rules"], "r") as f:
            data = json.load(f)
        assert data["rules"][0]["enabled"] is False

    def test_enable_rule(self, mock_files):
        """Test enabling a rule."""
        create_rule(
            rule_id="test_rule",
            trigger_topic="zigbee2mqtt/pir",
            trigger_field="occupancy",
            trigger_value="true",
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}',
            avg_delay_seconds=2.0,
            tolerance_seconds=1.0
        )
        toggle_rule("test_rule", False)

        result = toggle_rule("test_rule", True)

        assert "enabled" in result.lower()

    def test_toggle_nonexistent_rule(self, mock_files):
        """Test toggling a rule that doesn't exist."""
        result = toggle_rule("nonexistent_rule", True)

        assert "not found" in result.lower()


class TestClearPendingPatterns:
    """Tests for clear_pending_patterns function."""

    def test_clear_pending_patterns(self, mock_files):
        """Test clearing all pending patterns."""
        record_pattern_observation(
            "zigbee2mqtt/pir", "occupancy", "zigbee2mqtt/light/set", 2.0
        )
        record_pattern_observation(
            "zigbee2mqtt/pir_b", "occupancy", "zigbee2mqtt/light_b/set", 2.0
        )

        result = clear_pending_patterns()

        assert "cleared" in result.lower()

        with open(mock_files["pending_patterns"], "r") as f:
            data = json.load(f)
        assert len(data["patterns"]) == 0


class TestRejectPattern:
    """Tests for reject_pattern function."""

    def test_reject_pattern(self, mock_files):
        """Test rejecting a pattern."""
        result = reject_pattern(
            "zigbee2mqtt/pir",
            "occupancy",
            "zigbee2mqtt/light/set",
            "Coincidental pattern"
        )

        assert "rejected" in result.lower()

        with open(mock_files["rejected_patterns"], "r") as f:
            data = json.load(f)
        assert len(data["patterns"]) == 1

    def test_reject_clears_pending(self, mock_files):
        """Test that rejecting clears pending observations."""
        record_pattern_observation(
            "zigbee2mqtt/pir", "occupancy", "zigbee2mqtt/light/set", 2.0
        )

        reject_pattern(
            "zigbee2mqtt/pir",
            "occupancy",
            "zigbee2mqtt/light/set",
            "test"
        )

        with open(mock_files["pending_patterns"], "r") as f:
            data = json.load(f)
        assert len(data["patterns"]) == 0

    def test_reject_deletes_existing_rule(self, mock_files):
        """Test that rejecting deletes existing rule."""
        create_rule(
            rule_id="test_rule",
            trigger_topic="zigbee2mqtt/pir",
            trigger_field="occupancy",
            trigger_value="true",
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}',
            avg_delay_seconds=2.0,
            tolerance_seconds=1.0
        )

        result = reject_pattern(
            "zigbee2mqtt/pir",
            "occupancy",
            "zigbee2mqtt/light/set",
            "test"
        )

        assert "Deleted" in result

        with open(mock_files["learned_rules"], "r") as f:
            data = json.load(f)
        assert len(data["rules"]) == 0


class TestGetRejectedPatterns:
    """Tests for get_rejected_patterns function."""

    def test_get_rejected_patterns_empty(self, mock_files):
        """Test getting rejected patterns when none exist."""
        result = get_rejected_patterns()
        data = json.loads(result)

        assert "patterns" in data
        assert len(data["patterns"]) == 0

    def test_get_rejected_patterns_with_data(self, mock_files):
        """Test getting rejected patterns when they exist."""
        reject_pattern(
            "zigbee2mqtt/pir",
            "occupancy",
            "zigbee2mqtt/light/set",
            "test"
        )

        result = get_rejected_patterns()
        data = json.loads(result)

        assert len(data["patterns"]) == 1


class TestRemoveRejectedPattern:
    """Tests for remove_rejected_pattern function."""

    def test_remove_rejected_pattern(self, mock_files):
        """Test removing a rejected pattern."""
        reject_pattern(
            "zigbee2mqtt/pir",
            "occupancy",
            "zigbee2mqtt/light/set",
            "test"
        )

        result = remove_rejected_pattern(
            "zigbee2mqtt/pir",
            "occupancy",
            "zigbee2mqtt/light/set"
        )

        assert "Removed" in result

        with open(mock_files["rejected_patterns"], "r") as f:
            data = json.load(f)
        assert len(data["patterns"]) == 0

    def test_remove_nonexistent_rejected(self, mock_files):
        """Test removing a pattern that wasn't rejected."""
        result = remove_rejected_pattern(
            "zigbee2mqtt/nonexistent",
            "field",
            "zigbee2mqtt/action"
        )

        assert "not found" in result.lower()


class TestReportUndo:
    """Tests for report_undo function."""

    def test_report_undo_increments_count(self, mock_files):
        """Test that report_undo increments undo count."""
        create_rule(
            rule_id="test_rule",
            trigger_topic="zigbee2mqtt/pir",
            trigger_field="occupancy",
            trigger_value="true",
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}',
            avg_delay_seconds=2.0,
            tolerance_seconds=1.0
        )

        result = report_undo("test_rule")

        assert "1/3" in result

        with open(mock_files["learned_rules"], "r") as f:
            data = json.load(f)
        assert data["rules"][0]["undo_count"] == 1

    def test_report_undo_threshold_reached(self, mock_files):
        """Test that threshold message appears after 3 undos."""
        create_rule(
            rule_id="test_rule",
            trigger_topic="zigbee2mqtt/pir",
            trigger_field="occupancy",
            trigger_value="true",
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}',
            avg_delay_seconds=2.0,
            tolerance_seconds=1.0
        )

        report_undo("test_rule")
        report_undo("test_rule")
        result = report_undo("test_rule")

        assert "THRESHOLD" in result

    def test_report_undo_nonexistent_rule(self, mock_files):
        """Test reporting undo for nonexistent rule."""
        result = report_undo("nonexistent_rule")

        assert "not found" in result.lower()


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_pattern_rejected_true(self, mock_files):
        """Test _is_pattern_rejected returns True for rejected patterns."""
        _add_rejected_pattern(
            "zigbee2mqtt/pir",
            "occupancy",
            "zigbee2mqtt/light/set",
            "test"
        )

        result = _is_pattern_rejected(
            "zigbee2mqtt/pir",
            "occupancy",
            "zigbee2mqtt/light/set"
        )

        assert result is True

    def test_is_pattern_rejected_false(self, mock_files):
        """Test _is_pattern_rejected returns False for non-rejected patterns."""
        result = _is_pattern_rejected(
            "zigbee2mqtt/pir",
            "occupancy",
            "zigbee2mqtt/light/set"
        )

        assert result is False

    def test_add_rejected_pattern_no_duplicates(self, mock_files):
        """Test that duplicate rejections are not added."""
        _add_rejected_pattern(
            "zigbee2mqtt/pir",
            "occupancy",
            "zigbee2mqtt/light/set",
            "first"
        )
        _add_rejected_pattern(
            "zigbee2mqtt/pir",
            "occupancy",
            "zigbee2mqtt/light/set",
            "second"
        )

        with open(mock_files["rejected_patterns"], "r") as f:
            data = json.load(f)
        # Should only have one entry
        assert len(data["patterns"]) == 1

    def test_clear_pending_pattern(self, mock_files):
        """Test _clear_pending_pattern removes specific pattern."""
        record_pattern_observation(
            "zigbee2mqtt/pir_a", "occupancy", "zigbee2mqtt/light_a/set", 2.0
        )
        record_pattern_observation(
            "zigbee2mqtt/pir_b", "occupancy", "zigbee2mqtt/light_b/set", 2.0
        )

        _clear_pending_pattern(
            "zigbee2mqtt/pir_a",
            "occupancy",
            "zigbee2mqtt/light_a/set"
        )

        with open(mock_files["pending_patterns"], "r") as f:
            data = json.load(f)
        # Should only have one pattern left
        assert len(data["patterns"]) == 1
        assert data["patterns"][0]["trigger_topic"] == "zigbee2mqtt/pir_b"


class TestSendTelegramMessage:
    """Tests for send_telegram_message function."""

    def test_send_telegram_message_success(self):
        """Test successful Telegram message sending."""
        mock_bot = MagicMock()
        mock_bot.broadcast_message.return_value = 2
        context = MockRuntimeContext(MagicMock(), telegram_bot=mock_bot)

        result = send_telegram_message("Hello, world!", context=context)

        assert "Successfully" in result
        assert "2 user(s)" in result
        mock_bot.broadcast_message.assert_called_once_with("Hello, world!")

    def test_send_telegram_message_no_bot(self):
        """Test Telegram message when bot is not configured."""
        context = MockRuntimeContext(MagicMock(), telegram_bot=None)

        result = send_telegram_message("Hello", context=context)

        assert "Error" in result
        assert "not configured" in result

    def test_send_telegram_message_no_context(self):
        """Test Telegram message when no context is provided."""
        result = send_telegram_message("Hello")

        assert "Error" in result
        assert "not configured" in result

    def test_send_telegram_message_no_recipients(self):
        """Test Telegram message when no users receive it."""
        mock_bot = MagicMock()
        mock_bot.broadcast_message.return_value = 0
        context = MockRuntimeContext(MagicMock(), telegram_bot=mock_bot)

        result = send_telegram_message("Hello", context=context)

        assert "Error" in result
        assert "no authorized chats" in result.lower()


class TestToolHandler:
    """Tests for the ToolHandler class."""

    def test_tool_handler_send_mqtt_message(self):
        """Test ToolHandler.send_mqtt_message."""
        mock_client = MagicMock()
        mock_client.publish.return_value = True
        context = MockRuntimeContext(mock_client)
        handler = ToolHandler(context)

        result = handler.send_mqtt_message("test/topic", '{"state": "ON"}')

        assert "Successfully" in result
        mock_client.publish.assert_called_once()

    def test_tool_handler_send_telegram_message(self):
        """Test ToolHandler.send_telegram_message."""
        mock_bot = MagicMock()
        mock_bot.broadcast_message.return_value = 1
        context = MockRuntimeContext(MagicMock(), telegram_bot=mock_bot)
        handler = ToolHandler(context)

        result = handler.send_telegram_message("Test message")

        assert "Successfully" in result
        mock_bot.broadcast_message.assert_called_once_with("Test message")

    def test_tool_handler_create_rule_respects_disable_new_rules(self, mock_files):
        """Test that ToolHandler.create_rule respects disable_new_rules setting."""
        context = MockRuntimeContext(MagicMock(), disable_new_rules=True)
        handler = ToolHandler(context)

        handler.create_rule(
            rule_id="disabled_rule",
            trigger_topic="zigbee2mqtt/pir",
            trigger_field="occupancy",
            trigger_value="true",
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}',
            avg_delay_seconds=2.0,
            tolerance_seconds=1.0
        )

        with open(mock_files["learned_rules"], "r") as f:
            data = json.load(f)
        # Rule should be created disabled
        assert data["rules"][0]["enabled"] is False

    def test_tool_handler_delegates_to_standalone_functions(self, mock_files):
        """Test that ToolHandler methods delegate to standalone functions."""
        context = MockRuntimeContext(MagicMock())
        handler = ToolHandler(context)

        # Test record_pattern_observation delegation
        result = handler.record_pattern_observation(
            "zigbee2mqtt/pir", "occupancy", "zigbee2mqtt/light/set", 2.0
        )
        assert "1/3" in result

        # Test get_pending_patterns delegation
        result = handler.get_pending_patterns()
        data = json.loads(result)
        assert len(data["patterns"]) == 1

        # Test clear_pending_patterns delegation
        result = handler.clear_pending_patterns()
        assert "cleared" in result.lower()


class TestCreateRuleWithContext:
    """Tests for create_rule function with context parameter."""

    def test_create_rule_with_disable_new_rules_context(self, mock_files):
        """Test that create_rule respects disable_new_rules from context."""
        context = MockRuntimeContext(MagicMock(), disable_new_rules=True)

        create_rule(
            rule_id="context_test_rule",
            trigger_topic="zigbee2mqtt/pir",
            trigger_field="occupancy",
            trigger_value="true",
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}',
            avg_delay_seconds=2.0,
            tolerance_seconds=1.0,
            context=context
        )

        with open(mock_files["learned_rules"], "r") as f:
            data = json.load(f)
        # Rule should be created disabled due to context setting
        assert data["rules"][0]["enabled"] is False

    def test_create_rule_enabled_by_default_without_context(self, mock_files):
        """Test that create_rule enables rules by default without context."""
        create_rule(
            rule_id="default_test_rule",
            trigger_topic="zigbee2mqtt/pir",
            trigger_field="occupancy",
            trigger_value="true",
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}',
            avg_delay_seconds=2.0,
            tolerance_seconds=1.0
        )

        with open(mock_files["learned_rules"], "r") as f:
            data = json.load(f)
        # Rule should be enabled by default
        assert data["rules"][0]["enabled"] is True
