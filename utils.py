"""Shared utility functions for the MQTT AI Daemon.

This module contains common utilities used across multiple modules,
eliminating code duplication for JSON file operations and MQTT publishing.
"""
import json
import subprocess
import sys
from typing import Any, TypeVar

T = TypeVar('T')

# Default MQTT configuration - can be overridden by callers
DEFAULT_MQTT_HOST = "192.168.1.245"
DEFAULT_MQTT_PORT = "1883"


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


def publish_mqtt(
    topic: str,
    payload: dict,
    host: str = DEFAULT_MQTT_HOST,
    port: str = DEFAULT_MQTT_PORT
) -> bool:
    """Publish a message to an MQTT topic using mosquitto_pub.

    Args:
        topic: The MQTT topic to publish to
        payload: The payload as a dict (will be JSON serialized)
        host: MQTT broker host
        port: MQTT broker port

    Returns:
        True if successful, False otherwise
    """
    payload_str = json.dumps(payload)
    try:
        subprocess.run(
            [
                "mosquitto_pub", "-h", host, "-p", port,
                "-t", topic, "-m", payload_str
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except FileNotFoundError:
        print(
            "Error: 'mosquitto_pub' not found. Install mosquitto-clients.",
            file=sys.stderr
        )
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error sending MQTT: {e.stderr}", file=sys.stderr)
        return False
