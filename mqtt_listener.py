"""MQTT Listener module - Legacy standalone listener.

This module provides a standalone MQTT listener that monitors messages
and periodically triggers AI analysis using Gemini.
"""
import subprocess
import sys
import collections
import os
import threading
import time
import json
from typing import Optional

import paho.mqtt.client as mqtt

# --- Configuration ---
MAX_MESSAGES = 100
MQTT_HOST = "192.168.1.245"
MQTT_PORT = 1883
MQTT_TOPIC = "zigbee2mqtt/#"
RULEBOOK_FILE = "rulebook.md"
GEMINI_CLI_COMMAND = "/opt/homebrew/bin/gemini"
CHECK_INTERVAL_SECONDS = 10

# Module-level MQTT client for publishing
_MQTT_CLIENT: Optional[mqtt.Client] = None


def _get_mqtt_client() -> Optional[mqtt.Client]:
    """Get or create the module-level MQTT client for publishing."""
    global _MQTT_CLIENT

    if _MQTT_CLIENT is not None:
        return _MQTT_CLIENT

    try:
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            protocol=mqtt.MQTTv311
        )
        client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
        client.loop_start()
        _MQTT_CLIENT = client
        return client
    except Exception as e:
        print(f"Error connecting to MQTT broker: {e}", file=sys.stderr)
        return None


def send_mqtt_message(topic, payload):
    """Publish a generic message to a specified MQTT topic using paho-mqtt."""
    print(f"  -> Sending MQTT: Topic='{topic}', Payload='{payload}'")

    try:
        client = _get_mqtt_client()
        if client is None:
            print("Error: Could not connect to MQTT broker.", file=sys.stderr)
            return

        result = client.publish(topic, payload, qos=1)
        result.wait_for_publish(timeout=5.0)

        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"Error publishing MQTT: rc={result.rc}", file=sys.stderr)

    except Exception as e:
        print(f"Error sending MQTT message: {e}", file=sys.stderr)


def check_for_alarms(messages, rulebook_content):
    """
    Use Gemini to analyze recent messages and return a JSON object.

    This function runs in a separate thread.
    """
    print("\n--- Verify with Ai on decision-making ---")

    prompt = (
        "You are a home automation AI. Analyze the following MQTT messages "
        "based on the provided rulebook. "
        "Your response MUST be a single, valid JSON object and nothing else. "
        "Follow the JSON structure and rules defined in the rulebook.\n\n"
        "## Rulebook:\n"
        f"{rulebook_content}\n\n"
        "## Latest MQTT Messages:\n"
        f"{messages}\n"
    )

    try:
        if not os.access(GEMINI_CLI_COMMAND, os.X_OK):
            print(
                f"Error: Gemini CLI not found or not executable at "
                f"'{GEMINI_CLI_COMMAND}'",
                file=sys.stderr
            )
            return

        result = subprocess.run(
            [GEMINI_CLI_COMMAND],
            input=prompt,
            capture_output=True,
            text=True,
            check=True,
        )
        response_text = result.stdout.strip()

        # --- JSON Parsing Logic ---
        light_blue = "\033[94m"
        reset_color = "\033[0m"

        try:
            data = json.loads(response_text)
            summary = data.get("summary", "No summary provided.")
            actions = data.get("actions", [])

            print(f"AI Summary: {light_blue}{summary}{reset_color}")

            if actions:
                print("AI Actions:")
                for action in actions:
                    topic = action.get("topic")
                    payload = action.get("payload")
                    if topic and payload is not None:
                        payload_str = (
                            json.dumps(payload) if isinstance(payload, dict)
                            else str(payload)
                        )
                        send_mqtt_message(topic, payload_str)
                    else:
                        print(f"  - Invalid action found: {action}")

        except json.JSONDecodeError:
            print(
                f"Error: AI did not return valid JSON. Response was:\n"
                f"{response_text}",
                file=sys.stderr
            )

    except FileNotFoundError:
        print(
            f"Error: Gemini CLI not found at '{GEMINI_CLI_COMMAND}'",
            file=sys.stderr
        )
    except subprocess.CalledProcessError as e:
        print(f"Error calling Gemini CLI: {e.stderr}", file=sys.stderr)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"An unexpected error occurred during AI check: {e}", file=sys.stderr)


def listen_to_mqtt():
    """
    Listen to MQTT messages and periodically check for alarms.

    Stores messages and periodically checks for alarms in a background thread.
    """
    try:
        with open(RULEBOOK_FILE, "r", encoding="utf-8") as f:
            rulebook_content = f.read()
    except FileNotFoundError:
        print(f"Error: '{RULEBOOK_FILE}' not found.", file=sys.stderr)
        sys.exit(1)

    messages = collections.deque(maxlen=MAX_MESSAGES)
    last_check_time = time.time()

    mqtt_command = [
        "mosquitto_sub", "-h", MQTT_HOST, "-p", str(MQTT_PORT),
        "-t", MQTT_TOPIC, "-v",
    ]

    try:
        with subprocess.Popen(
            mqtt_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        ) as process:

            print(
                f"Listening for messages on topic '{MQTT_TOPIC}' "
                f"at {MQTT_HOST}:{MQTT_PORT}..."
            )
            print(f"AI checks will run every {CHECK_INTERVAL_SECONDS} seconds.")

            # Main loop to read messages
            while True:
                # Check if it's time to run the AI analysis
                if time.time() - last_check_time >= CHECK_INTERVAL_SECONDS:
                    if messages:
                        # Create a snapshot and start the background check
                        message_snapshot = "\n".join(list(messages))
                        thread = threading.Thread(
                            target=check_for_alarms,
                            args=(message_snapshot, rulebook_content)
                        )
                        thread.daemon = True
                        thread.start()
                    last_check_time = time.time()

                # Read the next line without blocking indefinitely
                line = process.stdout.readline()
                if line:
                    line = line.strip()
                    if line:
                        print(line)
                        messages.append(line)

                # Check if the subprocess has ended
                if process.poll() is not None:
                    break

                # Small sleep to prevent this loop from pegging a CPU core
                time.sleep(0.01)

            # Handle process exit
            stderr_output = process.communicate()[1]
            if process.returncode != 0:
                print(
                    f"Error: mosquitto_sub exited with code {process.returncode}",
                    file=sys.stderr
                )
                if stderr_output:
                    print(f"stderr:\n{stderr_output}", file=sys.stderr)

    except FileNotFoundError:
        print("Error: 'mosquitto_sub' command not found.", file=sys.stderr)
        print(
            "Please ensure that mosquitto-clients is installed and in your PATH.",
            file=sys.stderr
        )
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    listen_to_mqtt()
