"""MQTT Client module for the MQTT AI Daemon.

This module handles MQTT operations using paho-mqtt for persistent connections
including both publishing and subscribing to MQTT messages.
"""
import json
import logging
import queue
import threading
from datetime import datetime
from typing import Any, Optional

import paho.mqtt.client as mqtt

from mqtt2ai.core.config import Config
from mqtt2ai.core.constants import MqttTopics


class MqttClient:
    """Handles MQTT operations with persistent paho-mqtt connection."""

    def __init__(self, config: Config):
        self.config = config
        self._client: Optional[mqtt.Client] = None
        self._connected = threading.Event()
        self._lock = threading.Lock()
        self._message_queue: Optional[queue.Queue] = None
        self._subscribed_topics: list[str] = []

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
            # Resubscribe to topics on reconnection
            for topic in self._subscribed_topics:
                client.subscribe(topic)
                logging.debug("Resubscribed to topic: %s", topic)
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

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: Any,
        msg: mqtt.MQTTMessage
    ):
        """Callback when a message is received on a subscribed topic.

        Puts the message (topic, payload, is_retained) tuple into the message queue
        for processing by the daemon. Retained messages are flagged so they can
        be excluded from AI analysis (they represent historical state, not new events).
        """
        if self._message_queue is None:
            logging.warning("Received message but no queue configured")
            return

        try:
            payload = msg.payload.decode("utf-8", errors="replace")
            # Include retain flag so daemon can filter retained messages from AI
            self._message_queue.put((msg.topic, payload, msg.retain))
        except Exception as e:
            logging.error("Error processing received message: %s", e)

    def subscribe(self, topics: list[str], message_queue: queue.Queue) -> bool:
        """Subscribe to MQTT topics and put received messages in the queue.

        Uses the persistent paho-mqtt connection. Will connect if not already
        connected. Subscriptions are automatically restored on reconnection.

        Args:
            topics: List of MQTT topic patterns to subscribe to.
            message_queue: Queue where received (topic, payload) tuples are put.

        Returns:
            True if subscriptions were successful, False otherwise.
        """
        self._message_queue = message_queue

        # Ensure we're connected
        if not self._connected.is_set():
            if not self.connect():
                logging.error("Cannot subscribe: not connected to MQTT broker")
                return False

        # Set the message callback
        self._client.on_message = self._on_message

        # Subscribe to each topic
        try:
            for topic in topics:
                result, _ = self._client.subscribe(topic)
                if result == mqtt.MQTT_ERR_SUCCESS:
                    self._subscribed_topics.append(topic)
                    logging.info("Subscribed to topic: %s", topic)
                else:
                    logging.error("Failed to subscribe to %s: %s", topic, result)
                    return False
            return True
        except Exception as e:
            logging.error("Error subscribing to topics: %s", e)
            return False

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

        self.publish(MqttTopics.ACTION_ANNOUNCE, json.dumps(announcement))

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
            logging.error("MQTT publish failed with code: %s", result.rc)
            return False

        except Exception as e:
            logging.error("Error publishing MQTT message: %s", e)
            return False

