"""Isolated unit tests for ToolHandler with mocked dependencies.

These tests validate that the ToolHandler correctly interacts with
injected dependencies (mqtt_client, telegram_bot) via RuntimeContext,
without requiring a real MQTT broker or Telegram bot.
"""
import json
import os
from unittest.mock import MagicMock

import pytest

from mqtt2ai.ai.tools import ToolHandler
from mqtt2ai.core.context import RuntimeContext


class TestToolHandlerMqttIsolated:
    """Isolated tests for ToolHandler MQTT operations with mocked mqtt_client."""

    @pytest.fixture
    def mock_mqtt_client(self):
        """Create a mock MQTT client."""
        client = MagicMock()
        client.publish.return_value = True
        return client

    @pytest.fixture
    def tool_handler(self, mock_mqtt_client):
        """Create a ToolHandler with mocked RuntimeContext."""
        context = RuntimeContext(mqtt_client=mock_mqtt_client)
        return ToolHandler(context)

    def test_send_mqtt_message_calls_publish(self, tool_handler, mock_mqtt_client):
        """Test that send_mqtt_message calls mqtt_client.publish."""
        result = tool_handler.send_mqtt_message("test/topic", '{"state": "ON"}')

        assert "Successfully" in result
        mock_mqtt_client.publish.assert_called_once_with("test/topic", '{"state": "ON"}')

    def test_send_mqtt_message_with_correct_topic(self, tool_handler, mock_mqtt_client):
        """Test that the correct topic is passed to publish."""
        tool_handler.send_mqtt_message("zigbee2mqtt/living_room_light/set", '{"brightness": 100}')

        call_args = mock_mqtt_client.publish.call_args[0]
        assert call_args[0] == "zigbee2mqtt/living_room_light/set"

    def test_send_mqtt_message_with_correct_payload(self, tool_handler, mock_mqtt_client):
        """Test that the correct payload is passed to publish."""
        payload = '{"state": "ON", "brightness": 254}'
        tool_handler.send_mqtt_message("test/topic", payload)

        call_args = mock_mqtt_client.publish.call_args[0]
        assert call_args[1] == payload

    def test_send_mqtt_message_handles_publish_failure(self, tool_handler, mock_mqtt_client):
        """Test that send_mqtt_message handles publish failure gracefully."""
        mock_mqtt_client.publish.return_value = False

        result = tool_handler.send_mqtt_message("test/topic", '{"state": "ON"}')

        assert "Error" in result or "Failed" in result

    def test_send_mqtt_message_without_client(self):
        """Test that send_mqtt_message handles missing mqtt_client."""
        context = RuntimeContext(mqtt_client=None)
        handler = ToolHandler(context)

        result = handler.send_mqtt_message("test/topic", '{"state": "ON"}')

        assert "Error" in result or "Failed" in result


class TestToolHandlerTelegramIsolated:
    """Isolated tests for ToolHandler Telegram operations with mocked telegram_bot."""

    @pytest.fixture
    def mock_telegram_bot(self):
        """Create a mock Telegram bot."""
        bot = MagicMock()
        bot.broadcast_message.return_value = 2  # Simulate sending to 2 users
        return bot

    @pytest.fixture
    def tool_handler(self, mock_telegram_bot):
        """Create a ToolHandler with mocked RuntimeContext."""
        context = RuntimeContext(telegram_bot=mock_telegram_bot)
        return ToolHandler(context)

    def test_send_telegram_message_calls_broadcast(self, tool_handler, mock_telegram_bot):
        """Test that send_telegram_message calls telegram_bot.broadcast_message."""
        result = tool_handler.send_telegram_message("Hello, world!")

        assert "Successfully" in result
        mock_telegram_bot.broadcast_message.assert_called_once_with("Hello, world!")

    def test_send_telegram_message_reports_user_count(self, tool_handler, mock_telegram_bot):
        """Test that the result includes the number of users notified."""
        mock_telegram_bot.broadcast_message.return_value = 3

        result = tool_handler.send_telegram_message("Test message")

        assert "3 user(s)" in result

    def test_send_telegram_message_handles_no_recipients(self, tool_handler, mock_telegram_bot):
        """Test handling when no users receive the message."""
        mock_telegram_bot.broadcast_message.return_value = 0

        result = tool_handler.send_telegram_message("Test message")

        assert "Error" in result
        assert "no authorized chats" in result.lower()

    def test_send_telegram_message_handles_exception(self, tool_handler, mock_telegram_bot):
        """Test handling when broadcast_message raises an exception."""
        mock_telegram_bot.broadcast_message.side_effect = Exception("Connection failed")

        result = tool_handler.send_telegram_message("Test message")

        assert "Error" in result
        assert "Connection failed" in result

    def test_send_telegram_message_without_bot(self):
        """Test that send_telegram_message handles missing telegram_bot."""
        context = RuntimeContext(telegram_bot=None)
        handler = ToolHandler(context)

        result = handler.send_telegram_message("Test message")

        assert "Error" in result
        assert "not configured" in result


class TestToolHandlerCreateRuleIsolated:
    """Isolated tests for ToolHandler.create_rule with file mocking."""

    @pytest.fixture
    def mock_files(self, tmp_path, monkeypatch):
        """Mock the file paths to use temp directory."""
        learned_rules = str(tmp_path / "learned_rules.json")
        pending_patterns = str(tmp_path / "pending_patterns.json")
        rejected_patterns = str(tmp_path / "rejected_patterns.json")

        monkeypatch.setattr("mqtt2ai.ai.tools.LEARNED_RULES_FILE", learned_rules)
        monkeypatch.setattr("mqtt2ai.ai.tools.PENDING_PATTERNS_FILE", pending_patterns)
        monkeypatch.setattr("mqtt2ai.ai.tools.REJECTED_PATTERNS_FILE", rejected_patterns)

        return {
            "learned_rules": learned_rules,
            "pending_patterns": pending_patterns,
            "rejected_patterns": rejected_patterns,
        }

    @pytest.fixture
    def tool_handler(self):
        """Create a ToolHandler with minimal RuntimeContext."""
        context = RuntimeContext(disable_new_rules=False)
        return ToolHandler(context)

    def test_create_rule_writes_to_file(self, mock_files, tool_handler):
        """Test that create_rule writes the rule to the JSON file."""
        result = tool_handler.create_rule(
            rule_id="test_rule",
            trigger_topic="zigbee2mqtt/pir",
            trigger_field="occupancy",
            trigger_value="true",
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}',
            avg_delay_seconds=2.0,
            tolerance_seconds=1.0,
        )

        assert "created" in result.lower()

        # Verify file was written
        with open(mock_files["learned_rules"], "r") as f:
            data = json.load(f)

        assert len(data["rules"]) == 1
        assert data["rules"][0]["id"] == "test_rule"
        assert data["rules"][0]["trigger"]["topic"] == "zigbee2mqtt/pir"
        assert data["rules"][0]["action"]["topic"] == "zigbee2mqtt/light/set"

    def test_create_rule_parses_boolean_trigger_value(self, mock_files, tool_handler):
        """Test that trigger_value is parsed as JSON (boolean)."""
        tool_handler.create_rule(
            rule_id="bool_test",
            trigger_topic="zigbee2mqtt/sensor",
            trigger_field="occupancy",
            trigger_value="true",
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}',
            avg_delay_seconds=1.0,
            tolerance_seconds=0.5,
        )

        with open(mock_files["learned_rules"], "r") as f:
            data = json.load(f)

        # Should be parsed to boolean True, not string "true"
        assert data["rules"][0]["trigger"]["value"] is True

    def test_create_rule_respects_disable_new_rules(self, mock_files):
        """Test that create_rule respects disable_new_rules from context."""
        context = RuntimeContext(disable_new_rules=True)
        handler = ToolHandler(context)

        handler.create_rule(
            rule_id="disabled_rule",
            trigger_topic="zigbee2mqtt/pir",
            trigger_field="occupancy",
            trigger_value="true",
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}',
            avg_delay_seconds=2.0,
            tolerance_seconds=1.0,
        )

        with open(mock_files["learned_rules"], "r") as f:
            data = json.load(f)

        # Rule should be created disabled
        assert data["rules"][0]["enabled"] is False

    def test_create_rule_enabled_by_default(self, mock_files, tool_handler):
        """Test that rules are enabled by default when disable_new_rules=False."""
        tool_handler.create_rule(
            rule_id="enabled_rule",
            trigger_topic="zigbee2mqtt/pir",
            trigger_field="occupancy",
            trigger_value="true",
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}',
            avg_delay_seconds=2.0,
            tolerance_seconds=1.0,
        )

        with open(mock_files["learned_rules"], "r") as f:
            data = json.load(f)

        assert data["rules"][0]["enabled"] is True


class TestToolHandlerIntegrationIsolated:
    """Integration tests for ToolHandler with multiple mocked dependencies."""

    @pytest.fixture
    def full_mock_context(self, tmp_path, monkeypatch):
        """Create a fully mocked RuntimeContext with all dependencies."""
        # Mock file paths
        monkeypatch.setattr(
            "mqtt2ai.ai.tools.LEARNED_RULES_FILE",
            str(tmp_path / "learned_rules.json")
        )
        monkeypatch.setattr(
            "mqtt2ai.ai.tools.PENDING_PATTERNS_FILE",
            str(tmp_path / "pending_patterns.json")
        )
        monkeypatch.setattr(
            "mqtt2ai.ai.tools.REJECTED_PATTERNS_FILE",
            str(tmp_path / "rejected_patterns.json")
        )

        # Mock MQTT client
        mqtt_client = MagicMock()
        mqtt_client.publish.return_value = True

        # Mock Telegram bot
        telegram_bot = MagicMock()
        telegram_bot.broadcast_message.return_value = 1

        return RuntimeContext(
            mqtt_client=mqtt_client,
            telegram_bot=telegram_bot,
            disable_new_rules=False,
        )

    def test_tool_handler_with_all_dependencies(self, full_mock_context):
        """Test ToolHandler works correctly with all mocked dependencies."""
        handler = ToolHandler(full_mock_context)

        # Test MQTT
        mqtt_result = handler.send_mqtt_message("test/topic", '{"test": true}')
        assert "Successfully" in mqtt_result
        full_mock_context.mqtt_client.publish.assert_called_once()

        # Test Telegram
        telegram_result = handler.send_telegram_message("Test notification")
        assert "Successfully" in telegram_result
        full_mock_context.telegram_bot.broadcast_message.assert_called_once()

    def test_tool_handler_delegates_to_helper_functions(self, full_mock_context):
        """Test that ToolHandler methods delegate to helper functions correctly."""
        handler = ToolHandler(full_mock_context)

        # Test record_pattern_observation
        result = handler.record_pattern_observation(
            "zigbee2mqtt/pir",
            "occupancy",
            "zigbee2mqtt/light/set",
            2.5
        )
        assert "1/3" in result

        # Test get_pending_patterns
        patterns = handler.get_pending_patterns()
        data = json.loads(patterns)
        assert len(data["patterns"]) == 1

        # Test clear_pending_patterns
        clear_result = handler.clear_pending_patterns()
        assert "cleared" in clear_result.lower()

        # Verify patterns are cleared
        patterns_after = handler.get_pending_patterns()
        data_after = json.loads(patterns_after)
        assert len(data_after["patterns"]) == 0

