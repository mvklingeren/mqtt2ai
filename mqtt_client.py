"""MQTT Client module for the MQTT AI Daemon.

This module handles low-level MQTT subprocess operations including
publishing messages and starting the subscription listener process.
"""
import subprocess
import logging

from config import Config


class MqttClient:
    """Handles low-level MQTT subprocess operations."""

    def __init__(self, config: Config):
        self.config = config

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
