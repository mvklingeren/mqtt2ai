"""Tests for the utils module."""
import json
import os
import sys

from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_json_file, save_json_file, publish_mqtt  # pylint: disable=wrong-import-position


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


@pytest.fixture
def mock_paho_client():
    """Mock paho.mqtt.client.Client for utils module."""
    with patch("utils.mqtt.Client") as mock_client_class:
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance

        # Mock publish result
        mock_result = MagicMock()
        mock_result.rc = 0  # MQTT_ERR_SUCCESS
        mock_result.wait_for_publish = MagicMock()
        mock_instance.publish.return_value = mock_result

        yield mock_instance


@pytest.fixture(autouse=True)
def reset_utils_client():
    """Reset the module-level MQTT client before each test."""
    import utils
    utils._MQTT_CLIENT = None
    yield
    utils._MQTT_CLIENT = None


class TestPublishMqtt:
    """Tests for publish_mqtt function."""

    def test_publish_mqtt_success(self, mock_paho_client):
        """Test successful MQTT publish."""
        result = publish_mqtt("test/topic", {"state": "ON"})

        assert result is True
        mock_paho_client.publish.assert_called_once()

    def test_publish_mqtt_with_custom_host_port(self, mock_paho_client):
        """Test MQTT publish with custom host and port."""
        result = publish_mqtt(
            "test/topic",
            {"state": "OFF"},
            host="10.0.0.1",
            port="8883"
        )

        assert result is True
        # Verify connection was made to custom host/port
        mock_paho_client.connect.assert_called_with("10.0.0.1", 8883, keepalive=60)

    def test_publish_mqtt_serializes_payload(self, mock_paho_client):
        """Test that payload is JSON serialized."""
        payload = {"key": "value", "number": 42}
        publish_mqtt("test/topic", payload)

        call_args = mock_paho_client.publish.call_args[0]
        sent_payload = call_args[1]

        # Verify it's valid JSON
        parsed = json.loads(sent_payload)
        assert parsed == payload

    def test_publish_mqtt_connection_error(self):
        """Test MQTT publish when connection fails."""
        with patch("utils.mqtt.Client") as mock_client_class:
            mock_client_class.return_value.connect.side_effect = Exception("Connection refused")
            result = publish_mqtt("test/topic", {"state": "ON"})

        assert result is False

    def test_publish_mqtt_publish_error(self, mock_paho_client):
        """Test MQTT publish when publish fails."""
        mock_result = mock_paho_client.publish.return_value
        mock_result.rc = 1  # Not SUCCESS

        result = publish_mqtt("test/topic", {"state": "ON"})

        assert result is False

    def test_publish_mqtt_uses_qos_1(self, mock_paho_client):
        """Test that publish uses QoS 1."""
        publish_mqtt("test/topic", {"state": "ON"})

        call_kwargs = mock_paho_client.publish.call_args[1]
        assert call_kwargs.get("qos") == 1

    def test_publish_mqtt_correct_topic(self, mock_paho_client):
        """Test that correct topic is used."""
        publish_mqtt("zigbee2mqtt/light/set", {"state": "ON"})

        call_args = mock_paho_client.publish.call_args[0]
        assert call_args[0] == "zigbee2mqtt/light/set"
