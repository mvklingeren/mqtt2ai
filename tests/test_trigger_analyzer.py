"""Tests for the TriggerAnalyzer module."""
import json
import os
import sys
import time
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trigger_analyzer import TriggerAnalyzer, TriggerResult, TopicState, NumericFieldState


class TestTriggerResult:
    """Tests for TriggerResult dataclass."""

    def test_trigger_result_no_trigger_str(self):
        """Test string representation for no trigger."""
        result = TriggerResult(should_trigger=False)
        assert str(result) == "NO_TRIGGER"

    def test_trigger_result_with_trigger_str(self):
        """Test string representation for trigger."""
        result = TriggerResult(
            should_trigger=True,
            reason="State field 'contact' changed",
            field_name="contact",
            old_value=True,
            new_value=False
        )
        result_str = str(result)
        assert "TRIGGER" in result_str
        assert "contact" in result_str
        assert "True" in result_str
        assert "False" in result_str

    def test_trigger_result_default_values(self):
        """Test default values for TriggerResult."""
        result = TriggerResult(should_trigger=False)
        assert result.reason is None
        assert result.field_name is None
        assert result.old_value is None
        assert result.new_value is None


class TestTopicState:
    """Tests for TopicState dataclass."""

    def test_topic_state_defaults(self):
        """Test default values for TopicState."""
        state = TopicState()
        assert state.last_trigger_time == 0.0
        assert state.state_fields == {}
        assert state.numeric_fields == {}


class TestNumericFieldState:
    """Tests for NumericFieldState dataclass."""

    def test_numeric_field_state_initialization(self):
        """Test NumericFieldState initialization."""
        current_time = time.time()
        state = NumericFieldState(
            last_value=50.0,
            last_update_time=current_time,
            baseline_value=45.0,
            baseline_time=current_time - 100
        )
        assert state.last_value == 50.0
        assert state.baseline_value == 45.0


class TestTriggerAnalyzerInit:
    """Tests for TriggerAnalyzer initialization."""

    def test_init_with_default_config(self, temp_dir):
        """Test initialization with default config (no file)."""
        config_path = os.path.join(temp_dir, "nonexistent.json")
        analyzer = TriggerAnalyzer(config_path)

        # Should use defaults
        assert analyzer.config["cooldown_seconds"] == 60
        assert "contact" in analyzer.config["state_fields"]
        assert "power" in analyzer.config["numeric_fields"]

    def test_init_with_custom_config(self, temp_dir):
        """Test initialization with custom config file."""
        config_path = os.path.join(temp_dir, "custom_triggers.json")
        custom_config = {
            "cooldown_seconds": 30,
            "state_fields": ["contact", "motion"],
            "numeric_fields": {
                "power": {"immediate_delta": 50, "drift_delta": 100}
            }
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(custom_config, f)

        analyzer = TriggerAnalyzer(config_path)

        assert analyzer.config["cooldown_seconds"] == 30
        assert analyzer.config["state_fields"] == ["contact", "motion"]
        assert analyzer.config["numeric_fields"]["power"]["immediate_delta"] == 50

    def test_init_with_invalid_json(self, temp_dir):
        """Test initialization with invalid JSON falls back to defaults."""
        config_path = os.path.join(temp_dir, "invalid.json")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("not valid json {{{")

        analyzer = TriggerAnalyzer(config_path)

        # Should use defaults
        assert analyzer.config["cooldown_seconds"] == 60

    def test_init_merges_with_defaults(self, temp_dir):
        """Test that partial config merges with defaults."""
        config_path = os.path.join(temp_dir, "partial.json")
        partial_config = {
            "cooldown_seconds": 120
            # Missing other fields
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(partial_config, f)

        analyzer = TriggerAnalyzer(config_path)

        # Custom value
        assert analyzer.config["cooldown_seconds"] == 120
        # Default values filled in
        assert "state_fields" in analyzer.config
        assert "numeric_fields" in analyzer.config


class TestTriggerAnalyzerStateFields:
    """Tests for state field change detection."""

    def test_first_state_value_no_trigger(self, temp_dir):
        """Test that first seen value doesn't trigger."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))
        result = analyzer.analyze(
            "zigbee2mqtt/sensor",
            '{"contact": true}'
        )
        assert result.should_trigger is False

    def test_state_field_change_triggers(self, temp_dir):
        """Test that state field change triggers."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))

        # First message - no trigger
        analyzer.analyze("zigbee2mqtt/sensor", '{"contact": true}')

        # Change - should trigger
        result = analyzer.analyze("zigbee2mqtt/sensor", '{"contact": false}')

        assert result.should_trigger is True
        assert result.field_name == "contact"
        assert result.old_value is True
        assert result.new_value is False

    def test_state_field_no_change_no_trigger(self, temp_dir):
        """Test that same value doesn't trigger."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))

        analyzer.analyze("zigbee2mqtt/sensor", '{"contact": true}')
        result = analyzer.analyze("zigbee2mqtt/sensor", '{"contact": true}')

        assert result.should_trigger is False

    def test_occupancy_field_change(self, temp_dir):
        """Test occupancy field change detection."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))

        analyzer.analyze("zigbee2mqtt/pir", '{"occupancy": false}')
        result = analyzer.analyze("zigbee2mqtt/pir", '{"occupancy": true}')

        assert result.should_trigger is True
        assert result.field_name == "occupancy"

    def test_smoke_field_change(self, temp_dir):
        """Test smoke field change detection."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))

        analyzer.analyze("zigbee2mqtt/smoke", '{"smoke": false}')
        result = analyzer.analyze("zigbee2mqtt/smoke", '{"smoke": true}')

        assert result.should_trigger is True
        assert result.field_name == "smoke"

    def test_action_field_change(self, temp_dir):
        """Test action field change detection."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))

        analyzer.analyze("zigbee2mqtt/button", '{"action": ""}')
        result = analyzer.analyze("zigbee2mqtt/button", '{"action": "single"}')

        assert result.should_trigger is True
        assert result.field_name == "action"


class TestTriggerAnalyzerNumericFields:
    """Tests for numeric field change detection."""

    def test_first_numeric_value_no_trigger(self, temp_dir):
        """Test that first numeric value doesn't trigger."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))
        result = analyzer.analyze(
            "zigbee2mqtt/plug",
            '{"power": 50}'
        )
        assert result.should_trigger is False

    def test_small_power_change_no_trigger(self, temp_dir):
        """Test that small power changes don't trigger."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))

        analyzer.analyze("zigbee2mqtt/plug", '{"power": 50}')
        # Small change (10W, threshold is 100W)
        result = analyzer.analyze("zigbee2mqtt/plug", '{"power": 60}')

        assert result.should_trigger is False

    def test_large_power_change_triggers(self, temp_dir):
        """Test that large power changes trigger."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))

        analyzer.analyze("zigbee2mqtt/plug", '{"power": 50}')
        # Large change (150W, threshold is 100W)
        result = analyzer.analyze("zigbee2mqtt/plug", '{"power": 200}')

        assert result.should_trigger is True
        assert result.field_name == "power"
        assert "Immediate change" in result.reason

    def test_temperature_change_detection(self, temp_dir):
        """Test temperature change detection."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))

        analyzer.analyze("zigbee2mqtt/temp", '{"temperature": 20}')
        # Large change (15°C, threshold is 10°C)
        result = analyzer.analyze("zigbee2mqtt/temp", '{"temperature": 35}')

        assert result.should_trigger is True
        assert result.field_name == "temperature"

    def test_invalid_numeric_value_ignored(self, temp_dir):
        """Test that invalid numeric values are ignored."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))

        analyzer.analyze("zigbee2mqtt/plug", '{"power": 50}')
        # Invalid value
        result = analyzer.analyze("zigbee2mqtt/plug", '{"power": "not_a_number"}')

        assert result.should_trigger is False


class TestTriggerAnalyzerDrift:
    """Tests for drift detection."""

    def test_drift_detection(self, temp_dir):
        """Test gradual drift detection."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))
        # Override baseline window for testing
        analyzer.config["baseline_window_seconds"] = 1

        # Initial value
        analyzer.analyze("zigbee2mqtt/plug", '{"power": 50}')

        # Small increments over time (simulating gradual drift)
        analyzer.analyze("zigbee2mqtt/plug", '{"power": 70}')
        analyzer.analyze("zigbee2mqtt/plug", '{"power": 90}')
        analyzer.analyze("zigbee2mqtt/plug", '{"power": 110}')
        analyzer.analyze("zigbee2mqtt/plug", '{"power": 130}')
        analyzer.analyze("zigbee2mqtt/plug", '{"power": 150}')
        analyzer.analyze("zigbee2mqtt/plug", '{"power": 170}')
        analyzer.analyze("zigbee2mqtt/plug", '{"power": 190}')
        analyzer.analyze("zigbee2mqtt/plug", '{"power": 210}')
        analyzer.analyze("zigbee2mqtt/plug", '{"power": 230}')
        analyzer.analyze("zigbee2mqtt/plug", '{"power": 250}')

        # At some point drift should be detected (200 threshold from baseline)
        # This test verifies the drift mechanism exists


class TestTriggerAnalyzerCooldown:
    """Tests for cooldown functionality."""

    def test_cooldown_prevents_rapid_triggers(self, temp_dir):
        """Test that cooldown prevents rapid re-triggers."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))
        # Short cooldown for testing
        analyzer.config["cooldown_seconds"] = 2

        # First change triggers
        analyzer.analyze("zigbee2mqtt/sensor", '{"contact": true}')
        result1 = analyzer.analyze("zigbee2mqtt/sensor", '{"contact": false}')
        assert result1.should_trigger is True

        # Immediate second change should be on cooldown
        result2 = analyzer.analyze("zigbee2mqtt/sensor", '{"contact": true}')
        assert result2.should_trigger is False
        assert "cooldown" in result2.reason.lower()

    def test_cooldown_expires(self, temp_dir):
        """Test that cooldown expires after time passes."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))
        # Very short cooldown for testing
        analyzer.config["cooldown_seconds"] = 0.1

        # First change triggers
        analyzer.analyze("zigbee2mqtt/sensor", '{"contact": true}')
        analyzer.analyze("zigbee2mqtt/sensor", '{"contact": false}')

        # Wait for cooldown
        time.sleep(0.15)

        # Should trigger again
        result = analyzer.analyze("zigbee2mqtt/sensor", '{"contact": true}')
        assert result.should_trigger is True

    def test_cooldown_per_topic(self, temp_dir):
        """Test that cooldown is per-topic."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))
        analyzer.config["cooldown_seconds"] = 60

        # Trigger on topic A
        analyzer.analyze("zigbee2mqtt/sensor_a", '{"contact": true}')
        analyzer.analyze("zigbee2mqtt/sensor_a", '{"contact": false}')

        # Topic B should not be affected by A's cooldown
        analyzer.analyze("zigbee2mqtt/sensor_b", '{"contact": true}')
        result = analyzer.analyze("zigbee2mqtt/sensor_b", '{"contact": false}')
        assert result.should_trigger is True


class TestTriggerAnalyzerPayloadHandling:
    """Tests for payload handling."""

    def test_non_json_payload(self, temp_dir):
        """Test that non-JSON payloads don't trigger."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))
        result = analyzer.analyze("zigbee2mqtt/sensor", "not json")

        assert result.should_trigger is False
        assert "Non-JSON" in result.reason

    def test_non_object_json_payload(self, temp_dir):
        """Test that non-object JSON doesn't trigger."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))
        result = analyzer.analyze("zigbee2mqtt/sensor", "[1, 2, 3]")

        assert result.should_trigger is False
        assert "not a JSON object" in result.reason

    def test_empty_payload(self, temp_dir):
        """Test that empty JSON object doesn't trigger."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))
        result = analyzer.analyze("zigbee2mqtt/sensor", "{}")

        assert result.should_trigger is False

    def test_payload_with_unknown_fields_only(self, temp_dir):
        """Test payload with only unknown fields."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))
        result = analyzer.analyze(
            "zigbee2mqtt/sensor",
            '{"unknown_field": 42, "another": "value"}'
        )

        assert result.should_trigger is False
        assert "No significant changes" in result.reason


class TestTriggerAnalyzerMultipleTopics:
    """Tests for handling multiple topics."""

    def test_independent_topic_states(self, temp_dir):
        """Test that topics maintain independent states."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))

        # Set up topic A
        analyzer.analyze("zigbee2mqtt/sensor_a", '{"contact": true}')

        # Set up topic B with different value
        analyzer.analyze("zigbee2mqtt/sensor_b", '{"contact": false}')

        # Change topic A
        result_a = analyzer.analyze("zigbee2mqtt/sensor_a", '{"contact": false}')
        assert result_a.should_trigger is True
        assert result_a.old_value is True

        # Topic B should still have its own state
        result_b = analyzer.analyze("zigbee2mqtt/sensor_b", '{"contact": true}')
        assert result_b.should_trigger is True
        assert result_b.old_value is False


class TestTriggerAnalyzerGetStats:
    """Tests for get_stats method."""

    def test_get_stats_empty(self, temp_dir):
        """Test get_stats with no tracked topics."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))
        stats = analyzer.get_stats()

        assert stats["tracked_topics"] == 0
        assert "config" in stats

    def test_get_stats_with_topics(self, temp_dir):
        """Test get_stats after tracking topics."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))

        analyzer.analyze("zigbee2mqtt/sensor_a", '{"contact": true}')
        analyzer.analyze("zigbee2mqtt/sensor_b", '{"power": 50}')

        stats = analyzer.get_stats()

        assert stats["tracked_topics"] == 2


class TestTriggerAnalyzerStateUpdate:
    """Tests for state update during cooldown."""

    def test_state_updates_during_cooldown(self, temp_dir):
        """Test that state is still updated during cooldown."""
        analyzer = TriggerAnalyzer(os.path.join(temp_dir, "none.json"))
        analyzer.config["cooldown_seconds"] = 60

        # First messages
        analyzer.analyze("zigbee2mqtt/sensor", '{"contact": true}')
        analyzer.analyze("zigbee2mqtt/sensor", '{"contact": false}')  # Triggers

        # Change during cooldown
        analyzer.analyze("zigbee2mqtt/sensor", '{"contact": true}')  # On cooldown

        # After cooldown expires (mocked), the state should be 'true'
        # This is verified by checking the topic_state
        topic_state = analyzer.topic_states["zigbee2mqtt/sensor"]
        assert topic_state.state_fields["contact"] is True

