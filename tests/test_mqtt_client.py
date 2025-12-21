"""Tests for the MqttClient module."""
import os
import subprocess
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mqtt_client import MqttClient
from config import Config


class TestMqttClientInit:
    """Tests for MqttClient initialization."""

    def test_init_with_config(self, config):
        """Test initialization with a config."""
        client = MqttClient(config)
        assert client.config == config

    def test_init_uses_config_values(self):
        """Test that client uses config values."""
        config = Config()
        config.mqtt_host = "10.0.0.1"
        config.mqtt_port = "8883"

        client = MqttClient(config)

        assert client.config.mqtt_host == "10.0.0.1"
        assert client.config.mqtt_port == "8883"


class TestMqttClientPublish:
    """Tests for the publish method."""

    def test_publish_calls_mosquitto_pub(self, config, mock_subprocess_run):
        """Test that publish calls mosquitto_pub."""
        client = MqttClient(config)
        client.publish("test/topic", '{"state": "ON"}')

        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args[0][0]
        assert "mosquitto_pub" in call_args

    def test_publish_with_correct_host(self, mock_subprocess_run):
        """Test publish uses correct host."""
        config = Config()
        config.mqtt_host = "mqtt.example.com"
        client = MqttClient(config)

        client.publish("test/topic", '{"state": "ON"}')

        call_args = mock_subprocess_run.call_args[0][0]
        assert "-h" in call_args
        host_idx = call_args.index("-h") + 1
        assert call_args[host_idx] == "mqtt.example.com"

    def test_publish_with_correct_port(self, mock_subprocess_run):
        """Test publish uses correct port."""
        config = Config()
        config.mqtt_port = "8883"
        client = MqttClient(config)

        client.publish("test/topic", '{"state": "ON"}')

        call_args = mock_subprocess_run.call_args[0][0]
        assert "-p" in call_args
        port_idx = call_args.index("-p") + 1
        assert call_args[port_idx] == "8883"

    def test_publish_with_correct_topic(self, config, mock_subprocess_run):
        """Test publish uses correct topic."""
        client = MqttClient(config)
        client.publish("zigbee2mqtt/light/set", '{"state": "ON"}')

        call_args = mock_subprocess_run.call_args[0][0]
        assert "-t" in call_args
        topic_idx = call_args.index("-t") + 1
        assert call_args[topic_idx] == "zigbee2mqtt/light/set"

    def test_publish_with_correct_payload(self, config, mock_subprocess_run):
        """Test publish uses correct payload."""
        client = MqttClient(config)
        payload = '{"state": "ON", "brightness": 100}'
        client.publish("test/topic", payload)

        call_args = mock_subprocess_run.call_args[0][0]
        assert "-m" in call_args
        msg_idx = call_args.index("-m") + 1
        assert call_args[msg_idx] == payload

    def test_publish_handles_file_not_found(self, config):
        """Test publish handles mosquitto_pub not found."""
        client = MqttClient(config)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            # Should not raise, just log error
            client.publish("test/topic", '{"state": "ON"}')

    def test_publish_handles_subprocess_error(self, config):
        """Test publish handles subprocess errors."""
        client = MqttClient(config)

        with patch("subprocess.run") as mock_run:
            error = subprocess.CalledProcessError(1, "mosquitto_pub")
            error.stderr = "Connection refused"
            mock_run.side_effect = error
            # Should not raise, just log error
            client.publish("test/topic", '{"state": "ON"}')

    def test_publish_uses_check_true(self, config, mock_subprocess_run):
        """Test publish calls subprocess with check=True."""
        client = MqttClient(config)
        client.publish("test/topic", '{"state": "ON"}')

        call_kwargs = mock_subprocess_run.call_args[1]
        assert call_kwargs.get("check") is True

    def test_publish_captures_output(self, config, mock_subprocess_run):
        """Test publish captures subprocess output."""
        client = MqttClient(config)
        client.publish("test/topic", '{"state": "ON"}')

        call_kwargs = mock_subprocess_run.call_args[1]
        assert call_kwargs.get("capture_output") is True
        assert call_kwargs.get("text") is True


class TestMqttClientStartListenerProcess:
    """Tests for the start_listener_process method."""

    def test_start_listener_calls_popen(self, config, mock_subprocess_popen):
        """Test that start_listener_process calls Popen."""
        client = MqttClient(config)
        process = client.start_listener_process()

        mock_subprocess_popen.assert_called_once()
        assert process is not None

    def test_start_listener_with_correct_command(self, config, mock_subprocess_popen):
        """Test start_listener_process uses correct command."""
        client = MqttClient(config)
        client.start_listener_process()

        call_args = mock_subprocess_popen.call_args[0][0]
        assert "mosquitto_sub" in call_args

    def test_start_listener_with_correct_host(self, mock_subprocess_popen):
        """Test start_listener_process uses correct host."""
        config = Config()
        config.mqtt_host = "mqtt.example.com"
        client = MqttClient(config)

        client.start_listener_process()

        call_args = mock_subprocess_popen.call_args[0][0]
        assert "-h" in call_args
        host_idx = call_args.index("-h") + 1
        assert call_args[host_idx] == "mqtt.example.com"

    def test_start_listener_with_correct_port(self, mock_subprocess_popen):
        """Test start_listener_process uses correct port."""
        config = Config()
        config.mqtt_port = "8883"
        client = MqttClient(config)

        client.start_listener_process()

        call_args = mock_subprocess_popen.call_args[0][0]
        assert "-p" in call_args
        port_idx = call_args.index("-p") + 1
        assert call_args[port_idx] == "8883"

    def test_start_listener_with_correct_topics(self, mock_subprocess_popen):
        """Test start_listener_process uses correct topics."""
        config = Config()
        config.mqtt_topics = ["zigbee2mqtt/#", "jokes/#"]
        client = MqttClient(config)

        client.start_listener_process()

        call_args = mock_subprocess_popen.call_args[0][0]
        # Check that both topics are present with -t flags
        assert call_args.count("-t") == 2
        first_topic_idx = call_args.index("-t") + 1
        assert call_args[first_topic_idx] == "zigbee2mqtt/#"
        second_topic_idx = call_args.index("-t", first_topic_idx) + 1
        assert call_args[second_topic_idx] == "jokes/#"

    def test_start_listener_with_verbose_flag(self, config, mock_subprocess_popen):
        """Test start_listener_process uses -v flag."""
        client = MqttClient(config)
        client.start_listener_process()

        call_args = mock_subprocess_popen.call_args[0][0]
        assert "-v" in call_args

    def test_start_listener_pipes_stdout(self, config, mock_subprocess_popen):
        """Test start_listener_process pipes stdout."""
        client = MqttClient(config)
        client.start_listener_process()

        call_kwargs = mock_subprocess_popen.call_args[1]
        assert call_kwargs.get("stdout") == subprocess.PIPE

    def test_start_listener_pipes_stderr(self, config, mock_subprocess_popen):
        """Test start_listener_process pipes stderr."""
        client = MqttClient(config)
        client.start_listener_process()

        call_kwargs = mock_subprocess_popen.call_args[1]
        assert call_kwargs.get("stderr") == subprocess.PIPE


class TestMqttClientIntegration:
    """Integration-style tests for MqttClient."""

    def test_publish_and_listener_use_same_config(self, mock_subprocess_run, mock_subprocess_popen):
        """Test that publish and listener use the same config values."""
        config = Config()
        config.mqtt_host = "shared.host.com"
        config.mqtt_port = "9999"
        config.mqtt_topics = ["test/#"]

        client = MqttClient(config)

        # Start listener
        client.start_listener_process()
        listener_args = mock_subprocess_popen.call_args[0][0]

        # Publish
        client.publish("test/topic", '{"test": true}')
        publish_args = mock_subprocess_run.call_args[0][0]

        # Both should use same host
        listener_host = listener_args[listener_args.index("-h") + 1]
        publish_host = publish_args[publish_args.index("-h") + 1]
        assert listener_host == publish_host == "shared.host.com"

        # Both should use same port
        listener_port = listener_args[listener_args.index("-p") + 1]
        publish_port = publish_args[publish_args.index("-p") + 1]
        assert listener_port == publish_port == "9999"

