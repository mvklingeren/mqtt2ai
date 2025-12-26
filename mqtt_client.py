"""MQTT Client module for the MQTT AI Daemon.

This module handles MQTT operations using paho-mqtt for persistent connections
and subprocess for the subscription listener process.
"""
import json
import subprocess
import logging
import threading
from datetime import datetime
from typing import Any, Optional

import paho.mqtt.client as mqtt

from config import Config


# Topic for publishing causation announcements
ANNOUNCE_TOPIC = "mqtt2ai/action/announce"


class MqttClient:
    """Handles MQTT operations with persistent paho-mqtt connection."""

    def __init__(self, config: Config):
        self.config = config
        self._client: Optional[mqtt.Client] = None
        self._connected = threading.Event()
        self._lock = threading.Lock()

    def connect(self) -> bool:
        """Establish a persistent connection to the MQTT broker.
        
        Returns:
            True if connection was successful, False otherwise.
        """
        if self._client is not None and self._connected.is_set():
            return True
        
        with self._lock:
            if self._client is not None and self._connected.is_set():
                return True
            
            try:
                # Create client with protocol v5 for modern MQTT
                self._client = mqtt.Client(
                    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                    protocol=mqtt.MQTTv311
                )
                
                # Set up callbacks
                self._client.on_connect = self._on_connect
                self._client.on_disconnect = self._on_disconnect
                
                # Connect to broker
                port = int(self.config.mqtt_port)
                logging.info(
                    "Connecting to MQTT broker at %s:%d",
                    self.config.mqtt_host, port
                )
                self._client.connect(self.config.mqtt_host, port, keepalive=60)
                
                # Start the network loop in a background thread
                self._client.loop_start()
                
                # Wait for connection with timeout
                if self._connected.wait(timeout=10.0):
                    return True
                else:
                    logging.error("MQTT connection timeout")
                    return False
                    
            except Exception as e:
                logging.error("Failed to connect to MQTT broker: %s", e)
                self._client = None
                return False

    def disconnect(self):
        """Disconnect from the MQTT broker and clean up."""
        with self._lock:
            if self._client is not None:
                logging.info("Disconnecting from MQTT broker")
                self._client.loop_stop()
                self._client.disconnect()
                self._client = None
                self._connected.clear()

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: mqtt.ConnectFlags,
        reason_code: mqtt.ReasonCode,
        properties: Optional[mqtt.Properties] = None
    ):
        """Callback when connected to MQTT broker."""
        if reason_code == 0:
            logging.info("Connected to MQTT broker successfully")
            self._connected.set()
        else:
            logging.error("MQTT connection failed with code: %s", reason_code)

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        disconnect_flags: mqtt.DisconnectFlags,
        reason_code: mqtt.ReasonCode,
        properties: Optional[mqtt.Properties] = None
    ):
        """Callback when disconnected from MQTT broker."""
        self._connected.clear()
        if reason_code != 0:
            logging.warning(
                "Unexpected MQTT disconnection (code: %s). Will auto-reconnect.",
                reason_code
            )
        else:
            logging.info("Disconnected from MQTT broker")

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

    def publish(self, topic: str, payload: str) -> bool:
        """Publish a message to the MQTT broker.
        
        Uses the persistent paho-mqtt connection. Will attempt to connect
        if not already connected.
        
        Args:
            topic: The MQTT topic to publish to.
            payload: The message payload as a string.
            
        Returns:
            True if the message was published successfully, False otherwise.
        """
        logging.info("-> Sending MQTT: Topic='%s', Payload='%s'", topic, payload)
        
        # Ensure we're connected
        if not self._connected.is_set():
            if not self.connect():
                logging.error("Cannot publish: not connected to MQTT broker")
                return False
        
        try:
            result = self._client.publish(topic, payload, qos=1)
            # Wait for publish to complete with timeout
            result.wait_for_publish(timeout=5.0)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                return True
            else:
                logging.error("MQTT publish failed with code: %s", result.rc)
                return False
                
        except Exception as e:
            logging.error("Error publishing MQTT message: %s", e)
            return False

    def start_listener_process(self) -> subprocess.Popen:
        """Start the mosquitto_sub process for subscription.
        
        Note: Subscriptions still use subprocess for now as it requires
        a larger architectural change to handle incoming messages.
        """
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
