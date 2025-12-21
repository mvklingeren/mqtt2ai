"""Tests for the utils module."""
import json
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_json_file, save_json_file, publish_mqtt


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


class TestPublishMqtt:
    """Tests for publish_mqtt function."""

    def test_publish_mqtt_success(self, mock_subprocess_run):
        """Test successful MQTT publish."""
        result = publish_mqtt("test/topic", {"state": "ON"})

        assert result is True
        mock_subprocess_run.assert_called_once()

        # Verify the command arguments
        call_args = mock_subprocess_run.call_args[0][0]
        assert "mosquitto_pub" in call_args
        assert "-t" in call_args
        assert "test/topic" in call_args
        assert "-m" in call_args

    def test_publish_mqtt_with_custom_host_port(self, mock_subprocess_run):
        """Test MQTT publish with custom host and port."""
        result = publish_mqtt(
            "test/topic",
            {"state": "OFF"},
            host="10.0.0.1",
            port="8883"
        )

        assert result is True
        call_args = mock_subprocess_run.call_args[0][0]
        assert "-h" in call_args
        assert "10.0.0.1" in call_args
        assert "-p" in call_args
        assert "8883" in call_args

    def test_publish_mqtt_serializes_payload(self, mock_subprocess_run):
        """Test that payload is JSON serialized."""
        payload = {"key": "value", "number": 42}
        publish_mqtt("test/topic", payload)

        call_args = mock_subprocess_run.call_args[0][0]
        # Find the message argument
        msg_idx = call_args.index("-m") + 1
        sent_payload = call_args[msg_idx]

        # Verify it's valid JSON
        parsed = json.loads(sent_payload)
        assert parsed == payload

    def test_publish_mqtt_file_not_found(self):
        """Test MQTT publish when mosquitto_pub is not found."""
        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            result = publish_mqtt("test/topic", {"state": "ON"})

        assert result is False

    def test_publish_mqtt_subprocess_error(self):
        """Test MQTT publish when subprocess fails."""
        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "mosquitto_pub")
            mock_run.side_effect.stderr = "Connection refused"
            result = publish_mqtt("test/topic", {"state": "ON"})

        assert result is False

    def test_publish_mqtt_uses_default_host_port(self, mock_subprocess_run):
        """Test MQTT publish uses default host and port."""
        publish_mqtt("test/topic", {"state": "ON"})

        call_args = mock_subprocess_run.call_args[0][0]
        assert "192.168.1.245" in call_args
        assert "1883" in call_args

