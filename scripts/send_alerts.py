#!/usr/bin/env python3
"""Send test MQTT alerts for testing purposes."""
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from mqtt2ai.core.config import Config
from mqtt2ai.mqtt.client import MqttClient
from mqtt2ai.core.context import RuntimeContext
from mqtt2ai.ai.tools import ToolHandler


def main():
    """Main entry point for sending test alerts."""
    # Initialize tools with MQTT client
    config = Config()
    mqtt_client = MqttClient(config)
    mqtt_client.connect()

    # Create runtime context with the MQTT client
    context = RuntimeContext(mqtt_client=mqtt_client)
    tool_handler = ToolHandler(context)

    tool_handler.send_mqtt_message(
        topic='alert/power',
        payload=(
            '{"device": "0xa4c138725761cac1", "type": "spike", '
            '"from_watts": 0, "to_watts": 1585, '
            '"message": "Unusual power consumption detected"}'
        )
    )

    tool_handler.send_mqtt_message(
        topic='jokes/',
        payload=(
            '{"joke": "Why did the scarecrow win an award? '
            'Because he was outstanding in his field!"}'
        )
    )

    # Disconnect when done
    mqtt_client.disconnect()


if __name__ == "__main__":
    main()

