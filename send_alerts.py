from mcp_mqtt_server import send_mqtt_message

send_mqtt_message(
    topic='alert/power',
    payload='{"device": "0xa4c138725761cac1", "type": "spike", "from_watts": 0, "to_watts": 1585, "message": "Unusual power consumption detected"}'
)

send_mqtt_message(
    topic='jokes/',
    payload='{"joke": "Why did the scarecrow win an award? Because he was outstanding in his field!"}'
)
