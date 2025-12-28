"""Shared utility functions for the MQTT AI Daemon.

This module contains common utilities used across multiple modules,
eliminating code duplication for JSON file operations and MQTT publishing.
"""
import json
import sys
from typing import Any, Optional, TypeVar
import os
import hashlib
from datetime import datetime
import logging

import paho.mqtt.client as mqtt

T = TypeVar('T')

# Module-level client for connection reuse (lazy initialized)
_MQTT_CLIENT: Optional[mqtt.Client] = None


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
    """Save data to a JSON file atomically with pretty formatting.

    Uses a temp file + rename pattern to ensure readers never see
    partial/empty files during writes. This prevents race conditions
    where a reader might see a truncated file while a write is in progress.

    Args:
        filepath: Path to the JSON file
        data: Data to save (must be JSON serializable)
    """
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())  # Ensure data is on disk before rename

    os.replace(tmp_path, filepath)  # Atomic swap


def _get_mqtt_client(host: str, port: int) -> Optional[mqtt.Client]:
    """Get or create the module-level MQTT client.

    Uses a simple synchronous connection for one-off publishes.
    For high-frequency publishing, use MqttClient class instead.
    """
    global _MQTT_CLIENT

    if _MQTT_CLIENT is not None:
        return _MQTT_CLIENT

    try:
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            protocol=mqtt.MQTTv311
        )
        client.connect(host, port, keepalive=60)
        client.loop_start()
        _MQTT_CLIENT = client
        return client
    except Exception as e:
        print(f"Error connecting to MQTT broker: {e}", file=sys.stderr)
        return None


def publish_mqtt(
    topic: str,
    payload: dict,
    host: str,
    port: str
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
        print(f"Error publishing MQTT: rc={result.rc}", file=sys.stderr)
        return False

    except Exception as e:
        print(f"Error sending MQTT: {e}", file=sys.stderr)
        return False

def write_debug_output(debug_dir: str, url: str, body: dict, response: dict = None):
    """Write HTTP call details to a debug file with short hash filename."""
    # Create debug directory if it doesn't exist
    os.makedirs(debug_dir, exist_ok=True)

    # Generate short hash for filename
    timestamp = datetime.now().isoformat()
    hash_input = f"{timestamp}-{url}".encode()
    short_hash = hashlib.md5(hash_input).hexdigest()[:8]
    filename = os.path.join(debug_dir, f"{short_hash}.txt")

    # Calculate content length
    body_json = json.dumps(body, indent=2)
    content_length = len(body_json.encode('utf-8'))

    # Build debug output
    output_lines = [
        "=== HTTP Request Debug ===",
        f"Timestamp: {timestamp}",
        f"URL: {url}",
        f"Content-Length: {content_length}",
        "",
        "=== Request Body ===",
        body_json,
    ]

    if response:
        response_json = json.dumps(response, indent=2)
        output_lines.extend([
            "",
            "=== Response ===",
            response_json,
        ])

    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    logging.debug("Debug output written to %s", filename)

