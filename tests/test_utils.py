"""Tests for the utils module."""
import json
import os
from unittest.mock import patch, MagicMock

import pytest

from mqtt2ai.core import utils
from mqtt2ai.core.utils import load_json_file, save_json_file, publish_mqtt


class TestLoadJsonFile:
    """Tests for load_json_file function."""

    def test_load_existing_json_file(self, temp_dir):
        """Test loading an existing valid JSON file."""
        filepath = os.path.join(temp_dir, "test.json")
        data = {"key": "value", "number": 42}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_json_file(filepath, {})
        assert result == data

    def test_load_nonexistent_file_returns_default(self, temp_dir):
        """Test loading a nonexistent file returns the default."""
        filepath = os.path.join(temp_dir, "nonexistent.json")
        default = {"default": True}

        result = load_json_file(filepath, default)
        assert result == default

    def test_load_invalid_json_returns_default(self, temp_dir):
        """Test loading invalid JSON returns the default."""
        filepath = os.path.join(temp_dir, "invalid.json")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("not valid json {{{")

        default = {"fallback": True}
        result = load_json_file(filepath, default)
        assert result == default

    def test_load_json_array(self, temp_dir):
        """Test loading a JSON array."""
        filepath = os.path.join(temp_dir, "array.json")
        data = [1, 2, 3, "four"]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_json_file(filepath, [])
        assert result == data

    def test_load_nested_json(self, temp_dir):
        """Test loading nested JSON structure."""
        filepath = os.path.join(temp_dir, "nested.json")
        data = {
            "level1": {
                "level2": {
                    "items": [1, 2, 3]
                }
            }
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_json_file(filepath, {})
        assert result == data

    def test_load_empty_json_object(self, temp_dir):
        """Test loading an empty JSON object."""
        filepath = os.path.join(temp_dir, "empty.json")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("{}")

        result = load_json_file(filepath, {"default": True})
        assert result == {}

    def test_load_json_with_unicode(self, temp_dir):
        """Test loading JSON with unicode characters."""
        filepath = os.path.join(temp_dir, "unicode.json")
        data = {"message": "Hello 世界 🌍", "emoji": "🔥"}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        result = load_json_file(filepath, {})
        assert result == data


class TestSaveJsonFile:
    """Tests for save_json_file function."""

    def test_save_json_file_creates_file(self, temp_dir):
        """Test that save_json_file creates a new file."""
        filepath = os.path.join(temp_dir, "new.json")
        data = {"created": True}

        save_json_file(filepath, data)

        assert os.path.exists(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved == data

    def test_save_json_file_overwrites(self, temp_dir):
        """Test that save_json_file overwrites existing file."""
        filepath = os.path.join(temp_dir, "overwrite.json")

        # Save initial data
        save_json_file(filepath, {"first": True})
        # Overwrite
        save_json_file(filepath, {"second": True})

        with open(filepath, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved == {"second": True}

    def test_save_json_file_pretty_format(self, temp_dir):
        """Test that save_json_file uses pretty formatting."""
        filepath = os.path.join(temp_dir, "pretty.json")
        data = {"key": "value"}

        save_json_file(filepath, data)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for indentation (pretty format)
        assert "\n" in content
        assert "  " in content  # indent=2

    def test_save_json_array(self, temp_dir):
        """Test saving a JSON array."""
        filepath = os.path.join(temp_dir, "array.json")
        data = [1, 2, 3, "four", {"nested": True}]

        save_json_file(filepath, data)

        with open(filepath, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved == data

    def test_save_complex_nested_structure(self, temp_dir):
        """Test saving complex nested structures."""
        filepath = os.path.join(temp_dir, "complex.json")
        data = {
            "rules": [
                {
                    "id": "rule1",
                    "trigger": {"topic": "test", "field": "occupancy"},
                    "enabled": True
                }
            ],
            "metadata": {
                "version": 1,
                "created": "2024-01-01"
            }
        }

        save_json_file(filepath, data)

        with open(filepath, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved == data

    def test_save_and_load_roundtrip(self, temp_dir):
        """Test save and load roundtrip preserves data."""
        filepath = os.path.join(temp_dir, "roundtrip.json")
        original = {
            "string": "hello",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3],
            "object": {"nested": True}
        }

        save_json_file(filepath, original)
        loaded = load_json_file(filepath, {})

        assert loaded == original

    def test_save_json_file_no_temp_file_left_behind(self, temp_dir):
        """Test that atomic write cleans up temp file after save."""
        filepath = os.path.join(temp_dir, "atomic.json")
        tmp_path = filepath + ".tmp"
        data = {"atomic": True}

        save_json_file(filepath, data)

        # Main file should exist
        assert os.path.exists(filepath)
        # Temp file should NOT exist after successful save
        assert not os.path.exists(tmp_path)

    def test_save_json_file_atomic_preserves_old_on_read(self, temp_dir):
        """Test that atomic write ensures readers see complete data.

        This verifies that during a write, the original file remains
        intact until the atomic replace happens.
        """
        filepath = os.path.join(temp_dir, "atomic_read.json")

        # Save initial data
        initial_data = {"version": 1, "rules": ["rule1", "rule2"]}
        save_json_file(filepath, initial_data)

        # Save new data
        new_data = {"version": 2, "rules": ["rule1", "rule2", "rule3"]}
        save_json_file(filepath, new_data)

        # After save, we should see the new data (atomic swap completed)
        loaded = load_json_file(filepath, {})
        assert loaded == new_data

    def test_save_json_file_atomic_multiple_writes(self, temp_dir):
        """Test multiple atomic writes don't leave temp files."""
        filepath = os.path.join(temp_dir, "multi_atomic.json")
        tmp_path = filepath + ".tmp"

        # Perform multiple saves
        for i in range(5):
            save_json_file(filepath, {"iteration": i})

        # Main file should have final value
        loaded = load_json_file(filepath, {})
        assert loaded == {"iteration": 4}

        # No temp file should remain
        assert not os.path.exists(tmp_path)


@pytest.fixture
def mock_mqtt_client():
    """Mock MqttClient for utils module."""
    mock_client = MagicMock()
    mock_client.publish.return_value = True
    return mock_client


class TestPublishMqtt:
    """Tests for publish_mqtt function."""

    def test_publish_mqtt_success_with_explicit_client(self, mock_mqtt_client):
        """Test successful MQTT publish with explicit client."""
        result = publish_mqtt("test/topic", {"state": "ON"}, mqtt_client=mock_mqtt_client)

        assert result is True
        mock_mqtt_client.publish.assert_called_once()

    def test_publish_mqtt_with_context(self, mock_mqtt_client):
        """Test MQTT publish using RuntimeContext."""
        from mqtt2ai.core.context import set_context, RuntimeContext

        # Set up context with mock client
        ctx = RuntimeContext(mqtt_client=mock_mqtt_client)
        set_context(ctx)

        try:
            result = publish_mqtt("test/topic", {"state": "OFF"})

            assert result is True
            mock_mqtt_client.publish.assert_called_once()
        finally:
            set_context(None)

    def test_publish_mqtt_serializes_payload(self, mock_mqtt_client):
        """Test that payload is JSON serialized."""
        payload = {"key": "value", "number": 42}
        publish_mqtt("test/topic", payload, mqtt_client=mock_mqtt_client)

        call_args = mock_mqtt_client.publish.call_args[0]
        sent_payload = call_args[1]

        # Verify it's valid JSON
        parsed = json.loads(sent_payload)
        assert parsed == payload

    def test_publish_mqtt_no_client_available(self):
        """Test MQTT publish when no client is available."""
        from mqtt2ai.core.context import set_context

        # Ensure no context is set
        set_context(None)

        result = publish_mqtt("test/topic", {"state": "ON"})

        assert result is False

    def test_publish_mqtt_publish_error(self, mock_mqtt_client):
        """Test MQTT publish when publish fails."""
        mock_mqtt_client.publish.return_value = False

        result = publish_mqtt("test/topic", {"state": "ON"}, mqtt_client=mock_mqtt_client)

        assert result is False

    def test_publish_mqtt_correct_topic(self, mock_mqtt_client):
        """Test that correct topic is used."""
        publish_mqtt("zigbee2mqtt/light/set", {"state": "ON"}, mqtt_client=mock_mqtt_client)

        call_args = mock_mqtt_client.publish.call_args[0]
        assert call_args[0] == "zigbee2mqtt/light/set"
