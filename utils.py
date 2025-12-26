"""Shared utility functions for the MQTT AI Daemon.

This module contains common utilities used across multiple modules,
eliminating code duplication for JSON file operations and MQTT publishing.
"""
import json
import sys
from typing import Any, Optional, TypeVar

import paho.mqtt.client as mqtt

T = TypeVar('T')

# Default MQTT configuration - can be overridden by callers
DEFAULT_MQTT_HOST = "192.168.1.245"
DEFAULT_MQTT_PORT = "1883"

# Module-level client for connection reuse (lazy initialized)
_mqtt_client: Optional[mqtt.Client] = None


def load_json_file(filepath: str, default: T) -> T:
    """Load a JSON file, returning default if not found or invalid.

    Args:
        filepath: Path to the JSON file
        default: Default value to return if file doesn't exist or is invalid

    Returns:
        The loaded JSON data, or the default value
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json_file(filepath: str, data: Any) -> None:
    """Save data to a JSON file with pretty formatting.

    Args:
        filepath: Path to the JSON file
        data: Data to save (must be JSON serializable)
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _get_mqtt_client(host: str, port: int) -> Optional[mqtt.Client]:
    """Get or create the module-level MQTT client.
    
    Uses a simple synchronous connection for one-off publishes.
    For high-frequency publishing, use MqttClient class instead.
    """
    global _mqtt_client
    
    if _mqtt_client is not None:
        return _mqtt_client
    
    try:
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            protocol=mqtt.MQTTv311
        )
        client.connect(host, port, keepalive=60)
        client.loop_start()
        _mqtt_client = client
        return client
    except Exception as e:
        print(f"Error connecting to MQTT broker: {e}", file=sys.stderr)
        return None


def publish_mqtt(
    topic: str,
    payload: dict,
    host: str = DEFAULT_MQTT_HOST,
    port: str = DEFAULT_MQTT_PORT
) -> bool:
    """Publish a message to an MQTT topic using paho-mqtt.

    Args:
        topic: The MQTT topic to publish to
        payload: The payload as a dict (will be JSON serialized)
        host: MQTT broker host
        port: MQTT broker port (as string for backwards compatibility)

    Returns:
        True if successful, False otherwise
    """
    payload_str = json.dumps(payload)
    
    try:
        client = _get_mqtt_client(host, int(port))
        if client is None:
            return False
        
        result = client.publish(topic, payload_str, qos=1)
        result.wait_for_publish(timeout=5.0)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            return True
        else:
            print(f"Error publishing MQTT: rc={result.rc}", file=sys.stderr)
            return False
            
    except Exception as e:
        print(f"Error sending MQTT: {e}", file=sys.stderr)
        return False
