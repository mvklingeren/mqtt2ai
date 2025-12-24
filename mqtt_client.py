"""MQTT Client module for the MQTT AI Daemon.

This module handles low-level MQTT subprocess operations including
publishing messages and starting the subscription listener process.
"""
import json
import subprocess
import logging
from datetime import datetime
from typing import Any, Optional

from config import Config


# Topic for publishing causation announcements
ANNOUNCE_TOPIC = "mqtt2ai/action/announce"


class MqttClient:
    """Handles low-level MQTT subprocess operations."""

    def __init__(self, config: Config):
        self.config = config

    def announce(
        self,
        source: str,
        action_topic: str,
        action_payload: str,
        rule_id: Optional[str] = None,
        trigger_topic: Optional[str] = None,
        trigger_field: Optional[str] = None,
        trigger_value: Any = None,
        reason: Optional[str] = None
    ):
        """Publish a causation announcement before an automated action.
        
        This allows pattern learning to skip actions that were automated
        (not user-initiated).
        
        Args:
            source: Source of the action ("direct_rule" or "ai_analysis")
            action_topic: The MQTT topic being acted upon
            action_payload: The payload being sent
            rule_id: Optional rule ID (for direct_rule source)
            trigger_topic: The topic that triggered this action
            trigger_field: The field that triggered this action
            trigger_value: The value that triggered this action
            reason: Optional AI reasoning (for ai_analysis source)
        """
        announcement = {
            "source": source,
            "rule_id": rule_id,
            "trigger_topic": trigger_topic,
            "trigger_field": trigger_field,
            "trigger_value": trigger_value,
            "action_topic": action_topic,
            "action_payload": action_payload,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
        # Remove None values for cleaner JSON
        announcement = {k: v for k, v in announcement.items() if v is not None}
        
        self.publish(ANNOUNCE_TOPIC, json.dumps(announcement))

    def publish(self, topic: str, payload: str):
        """Publish a message using mosquitto_pub."""
        logging.info("-> Sending MQTT: Topic='%s', Payload='%s'", topic, payload)
        try:
            subprocess.run(
                [
                    "mosquitto_pub",
                    "-h", self.config.mqtt_host,
                    "-p", self.config.mqtt_port,
                    "-t", topic,
                    "-m", payload
                ],
                check=True,
                capture_output=True,
                text=True
            )
        except FileNotFoundError:
            logging.error("Error: 'mosquitto_pub' command not found.")
        except subprocess.CalledProcessError as e:
            logging.error("Error sending MQTT message: %s", e.stderr)

    def start_listener_process(self) -> subprocess.Popen:
        """Start the mosquitto_sub process."""
        cmd = [
            "mosquitto_sub",
            "-h", self.config.mqtt_host,
            "-p", self.config.mqtt_port,
        ]
        # Add each topic with its own -t flag
        for topic in self.config.mqtt_topics:
            cmd.extend(["-t", topic])
        cmd.append("-v")
        logging.info("Starting MQTT listener: %s", ' '.join(cmd))
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
