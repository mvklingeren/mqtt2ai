"""Shared utility functions for the MQTT AI Daemon.

This module contains common utilities used across multiple modules,
eliminating code duplication for JSON file operations and MQTT publishing.
"""
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, TypeVar

if TYPE_CHECKING:
    from mqtt2ai.mqtt.client import MqttClient

T = TypeVar('T')


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


def publish_mqtt(
    topic: str,
    payload: dict,
    mqtt_client: Optional['MqttClient'] = None
) -> bool:
    """Publish a message to an MQTT topic.

    Uses the provided MqttClient or falls back to the RuntimeContext.
    This function exists for backward compatibility; prefer using
    MqttClient.publish() directly for new code.

    Args:
        topic: The MQTT topic to publish to
        payload: The payload as a dict (will be JSON serialized)
        mqtt_client: Optional MqttClient instance. If not provided,
                     uses the client from RuntimeContext.

    Returns:
        True if successful, False otherwise
    """
    # Import here to avoid circular imports
    from mqtt2ai.core.context import get_context

    client = mqtt_client
    if client is None:
        ctx = get_context()
        if ctx is None or ctx.mqtt_client is None:
            logging.error("Cannot publish MQTT: no client available")
            return False
        client = ctx.mqtt_client

    payload_str = json.dumps(payload)
    return client.publish(topic, payload_str)

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

