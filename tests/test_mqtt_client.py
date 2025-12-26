"""Tests for the MqttClient module."""
import os
import subprocess
import sys
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mqtt_client import MqttClient
from config import Config


@pytest.fixture
def mock_paho_client():
    """Mock paho.mqtt.client.Client."""
    with patch("mqtt_client.mqtt.Client") as mock_client_class:
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance
        
        # Mock publish result
        mock_result = MagicMock()
        mock_result.rc = 0  # MQTT_ERR_SUCCESS
        mock_result.wait_for_publish = MagicMock()
        mock_instance.publish.return_value = mock_result
        
        yield mock_instance


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

    def test_init_client_not_connected(self, config):
        """Test that client is not connected initially."""
        client = MqttClient(config)
        assert not client._connected.is_set()
        assert client._client is None


class TestMqttClientConnect:
    """Tests for the connect method."""

    def test_connect_creates_paho_client(self, config, mock_paho_client):
        """Test that connect creates a paho client."""
        client = MqttClient(config)
        
        # Simulate successful connection by triggering the callback
        # reason_code == 0 means success in paho-mqtt
        def trigger_connect(*args, **kwargs):
            # Trigger the on_connect callback with reason_code=0 (success)
            client._on_connect(mock_paho_client, None, MagicMock(), 0)
        
        mock_paho_client.connect.side_effect = trigger_connect
        
        result = client.connect()
        
        assert result is True
        mock_paho_client.connect.assert_called_once()

    def test_connect_uses_correct_host_and_port(self, mock_paho_client):
        """Test that connect uses config host and port."""
        config = Config()
        config.mqtt_host = "mqtt.example.com"
        config.mqtt_port = "8883"
        client = MqttClient(config)
        
        def trigger_connect(*args, **kwargs):
            client._on_connect(mock_paho_client, None, MagicMock(), 0)
        
        mock_paho_client.connect.side_effect = trigger_connect
        client.connect()
        
        mock_paho_client.connect.assert_called_with("mqtt.example.com", 8883, keepalive=60)

    def test_connect_starts_loop(self, config, mock_paho_client):
        """Test that connect starts the network loop."""
        client = MqttClient(config)
        
        def trigger_connect(*args, **kwargs):
            client._on_connect(mock_paho_client, None, MagicMock(), 0)
        
        mock_paho_client.connect.side_effect = trigger_connect
        client.connect()
        
        mock_paho_client.loop_start.assert_called_once()

    def test_connect_returns_false_on_timeout(self, config, mock_paho_client):
        """Test that connect returns False if connection times out."""
        client = MqttClient(config)
        
        # Don't trigger the callback - simulate timeout
        with patch.object(client._connected, 'wait', return_value=False):
            result = client.connect()
        
        assert result is False

    def test_connect_returns_true_if_already_connected(self, config, mock_paho_client):
        """Test that connect returns True immediately if already connected."""
        client = MqttClient(config)
        client._connected.set()
        client._client = mock_paho_client
        
        result = client.connect()
        
        assert result is True
        # Should not call connect again
        mock_paho_client.connect.assert_not_called()


class TestMqttClientDisconnect:
    """Tests for the disconnect method."""

    def test_disconnect_stops_loop(self, config, mock_paho_client):
        """Test that disconnect stops the network loop."""
        client = MqttClient(config)
        client._client = mock_paho_client
        client._connected.set()
        
        client.disconnect()
        
        mock_paho_client.loop_stop.assert_called_once()

    def test_disconnect_calls_disconnect(self, config, mock_paho_client):
        """Test that disconnect calls client.disconnect."""
        client = MqttClient(config)
        client._client = mock_paho_client
        client._connected.set()
        
        client.disconnect()
        
        mock_paho_client.disconnect.assert_called_once()

    def test_disconnect_clears_connected_flag(self, config, mock_paho_client):
        """Test that disconnect clears the connected flag."""
        client = MqttClient(config)
        client._client = mock_paho_client
        client._connected.set()
        
        client.disconnect()
        
        assert not client._connected.is_set()

    def test_disconnect_handles_no_client(self, config):
        """Test that disconnect handles case where client is None."""
        client = MqttClient(config)
        # Should not raise
        client.disconnect()


class TestMqttClientPublish:
    """Tests for the publish method."""

    def test_publish_connects_if_not_connected(self, config, mock_paho_client):
        """Test that publish auto-connects if not connected."""
        client = MqttClient(config)
        
        # Simulate connection happening during publish
        # reason_code == 0 means success in paho-mqtt
        def trigger_connect(*args, **kwargs):
            client._client = mock_paho_client
            client._on_connect(mock_paho_client, None, MagicMock(), 0)
        
        mock_paho_client.connect.side_effect = trigger_connect
        
        result = client.publish("test/topic", '{"state": "ON"}')
        
        assert result is True
        mock_paho_client.publish.assert_called_once()

    def test_publish_with_correct_topic(self, config, mock_paho_client):
        """Test publish uses correct topic."""
        client = MqttClient(config)
        client._client = mock_paho_client
        client._connected.set()
        
        client.publish("zigbee2mqtt/light/set", '{"state": "ON"}')
        
        call_args = mock_paho_client.publish.call_args
        assert call_args[0][0] == "zigbee2mqtt/light/set"

    def test_publish_with_correct_payload(self, config, mock_paho_client):
        """Test publish uses correct payload."""
        client = MqttClient(config)
        client._client = mock_paho_client
        client._connected.set()
        
        payload = '{"state": "ON", "brightness": 100}'
        client.publish("test/topic", payload)
        
        call_args = mock_paho_client.publish.call_args
        assert call_args[0][1] == payload

    def test_publish_uses_qos_1(self, config, mock_paho_client):
        """Test publish uses QoS 1."""
        client = MqttClient(config)
        client._client = mock_paho_client
        client._connected.set()
        
        client.publish("test/topic", '{"state": "ON"}')
        
        call_kwargs = mock_paho_client.publish.call_args[1]
        assert call_kwargs.get("qos") == 1

    def test_publish_waits_for_publish(self, config, mock_paho_client):
        """Test publish waits for message to be sent."""
        client = MqttClient(config)
        client._client = mock_paho_client
        client._connected.set()
        
        mock_result = mock_paho_client.publish.return_value
        
        client.publish("test/topic", '{"state": "ON"}')
        
        mock_result.wait_for_publish.assert_called_once_with(timeout=5.0)

    def test_publish_returns_true_on_success(self, config, mock_paho_client):
        """Test publish returns True on success."""
        client = MqttClient(config)
        client._client = mock_paho_client
        client._connected.set()
        
        result = client.publish("test/topic", '{"state": "ON"}')
        
        assert result is True

    def test_publish_returns_false_on_failure(self, config, mock_paho_client):
        """Test publish returns False on failure."""
        client = MqttClient(config)
        client._client = mock_paho_client
        client._connected.set()
        
        mock_result = mock_paho_client.publish.return_value
        mock_result.rc = 1  # Not SUCCESS
        
        result = client.publish("test/topic", '{"state": "ON"}')
        
        assert result is False

    def test_publish_returns_false_if_not_connected(self, config, mock_paho_client):
        """Test publish returns False if connection fails."""
        client = MqttClient(config)
        
        # Simulate connection failure
        mock_paho_client.connect.side_effect = Exception("Connection refused")
        
        result = client.publish("test/topic", '{"state": "ON"}')
        
        assert result is False


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


class TestMqttClientAnnounce:
    """Tests for the announce method."""

    def test_announce_publishes_to_announce_topic(self, config, mock_paho_client):
        """Test that announce publishes to the announce topic."""
        client = MqttClient(config)
        client._client = mock_paho_client
        client._connected.set()
        
        client.announce(
            source="direct_rule",
            action_topic="zigbee2mqtt/light/set",
            action_payload='{"state": "ON"}'
        )
        
        call_args = mock_paho_client.publish.call_args[0]
        assert call_args[0] == "mqtt2ai/action/announce"

    def test_announce_includes_source(self, config, mock_paho_client):
        """Test that announce includes source in payload."""
        import json
        
        client = MqttClient(config)
        client._client = mock_paho_client
        client._connected.set()
        
        client.announce(
            source="ai_analysis",
            action_topic="test/topic",
            action_payload='{}'
        )
        
        call_args = mock_paho_client.publish.call_args[0]
        payload = json.loads(call_args[1])
        assert payload["source"] == "ai_analysis"

    def test_announce_removes_none_values(self, config, mock_paho_client):
        """Test that announce removes None values from payload."""
        import json
        
        client = MqttClient(config)
        client._client = mock_paho_client
        client._connected.set()
        
        client.announce(
            source="direct_rule",
            action_topic="test/topic",
            action_payload='{}',
            rule_id=None,  # Should be removed
            reason="test reason"
        )
        
        call_args = mock_paho_client.publish.call_args[0]
        payload = json.loads(call_args[1])
        assert "rule_id" not in payload
        assert payload["reason"] == "test reason"
