import subprocess
import logging

from config import Config


class MqttClient:
    """Handles low-level MQTT subprocess operations."""
    def __init__(self, config: Config):
        self.config = config

    def publish(self, topic: str, payload: str):
        """Publishes a message using mosquitto_pub."""
        logging.info(f"-> Sending MQTT: Topic='{topic}', Payload='{payload}'")
        try:
            subprocess.run(
                ["mosquitto_pub", "-h", self.config.mqtt_host, "-p", self.config.mqtt_port, "-t", topic, "-m", payload],
                check=True, capture_output=True, text=True
            )
        except FileNotFoundError:
            logging.error("Error: 'mosquitto_pub' command not found.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error sending MQTT message: {e.stderr}")

    def start_listener_process(self) -> subprocess.Popen:
        """Starts the mosquitto_sub process."""
        cmd = ["mosquitto_sub", "-h", self.config.mqtt_host, "-p", self.config.mqtt_port, "-t", self.config.mqtt_topic, "-v"]
        logging.info(f"Starting MQTT listener: {' '.join(cmd)}")
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

