"""Send test MQTT alerts for testing purposes."""
from config import Config
from mqtt_client import MqttClient
from context import RuntimeContext
import tools

# Initialize tools with MQTT client
config = Config()
mqtt_client = MqttClient(config)
mqtt_client.connect()

# Create runtime context with the MQTT client
context = RuntimeContext(mqtt_client=mqtt_client)

tools.send_mqtt_message(
    topic='alert/power',
    payload=(
        '{"device": "0xa4c138725761cac1", "type": "spike", '
        '"from_watts": 0, "to_watts": 1585, '
        '"message": "Unusual power consumption detected"}'
    ),
    context=context
)

tools.send_mqtt_message(
    topic='jokes/',
    payload=(
        '{"joke": "Why did the scarecrow win an award? '
        'Because he was outstanding in his field!"}'
    ),
    context=context
)

# Disconnect when done
mqtt_client.disconnect()
