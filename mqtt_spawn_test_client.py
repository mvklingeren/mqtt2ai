#!/usr/bin/env python3
"""
MQTT Test Client - Spawns test events to verify rule learning.

This script simulates real-world scenarios to test the pattern learning system:
- PIR sensor triggering followed by manual light switch actions
- Repeatable patterns to trigger rule creation after 3 occurrences

Usage:
    python mqtt_spawn_test_client.py                    # Run default PIR->Light scenario
    python mqtt_spawn_test_client.py --scenario door    # Run door->light scenario
    python mqtt_spawn_test_client.py --list             # List available scenarios
    python mqtt_spawn_test_client.py --custom           # Interactive custom event mode
"""

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime

from utils import publish_mqtt

# MQTT Configuration (match daemon settings)
MQTT_HOST = "192.168.1.245"
MQTT_PORT = "1883"


@dataclass
class TestEvent:
    """Represents a single MQTT test event."""
    topic: str
    payload: dict
    description: str
    delay_before: float = 0.0  # Seconds to wait before sending


@dataclass
class TestScenario:
    """A sequence of events that simulates a pattern."""
    name: str
    description: str
    events: list[TestEvent]
    repeat_count: int = 3  # How many times to repeat to trigger rule creation
    repeat_delay: float = 5.0  # Seconds between repetitions


def send_mqtt(topic: str, payload: dict) -> bool:
    """Send an MQTT message."""
    return publish_mqtt(topic, payload, MQTT_HOST, MQTT_PORT)


def timestamp():
    """Return current timestamp."""
    return datetime.now().strftime("[%H:%M:%S]")


# ============================================================================
# PREDEFINED TEST SCENARIOS
# ============================================================================

SCENARIOS = {
    "pir_light": TestScenario(
        name="PIR → Light",
        description="PIR sensor detects motion, then user turns on kitchen light",
        events=[
            TestEvent(
                topic="zigbee2mqtt/pir_hallway",
                payload={"occupancy": True, "battery": 95, "linkquality": 120},
                description="PIR detects motion (occupancy=true)",
            ),
            TestEvent(
                topic="zigbee2mqtt/light_kitchen/set",
                payload={"state": "ON", "brightness": 254},
                description="User turns on kitchen light",
                delay_before=0,  # Will be randomized
            ),
            TestEvent(
                topic="zigbee2mqtt/pir_hallway",
                payload={"occupancy": False, "battery": 95, "linkquality": 120},
                description="PIR clears (occupancy=false) - resets for next trigger",
                delay_before=2.0,
            ),
        ],
        repeat_count=3,
        repeat_delay=5.0,
    ),

    "door_light": TestScenario(
        name="Door → Hallway Light",
        description="Front door opens, user turns on hallway light",
        events=[
            TestEvent(
                topic="zigbee2mqtt/front_door",
                payload={"contact": False, "battery": 88, "linkquality": 95},
                description="Front door opens",
            ),
            TestEvent(
                topic="zigbee2mqtt/light_hallway/set",
                payload={"state": "ON"},
                description="User turns on hallway light",
                delay_before=0,  # Will be randomized
            ),
            TestEvent(
                topic="zigbee2mqtt/front_door",
                payload={"contact": True, "battery": 88, "linkquality": 95},
                description="Front door closes",
                delay_before=5.0,
            ),
        ],
        repeat_count=3,
        repeat_delay=10.0,
    ),

    "motion_fan": TestScenario(
        name="Motion → Fan",
        description="Motion in bathroom triggers user to turn on exhaust fan",
        events=[
            TestEvent(
                topic="zigbee2mqtt/motion_bathroom",
                payload={"occupancy": True, "illuminance": 50},
                description="Motion detected in bathroom",
            ),
            TestEvent(
                topic="zigbee2mqtt/fan_bathroom/set",
                payload={"state": "ON", "speed": "high"},
                description="User turns on exhaust fan",
                delay_before=0,  # Will be randomized
            ),
        ],
        repeat_count=3,
        repeat_delay=15.0,
    ),

    "quick_test": TestScenario(
        name="Quick Test",
        description="Fast scenario for quick testing (short delays)",
        events=[
            TestEvent(
                topic="zigbee2mqtt/test_pir",
                payload={"occupancy": True},
                description="Test PIR triggers (occupancy=true)",
            ),
            TestEvent(
                topic="zigbee2mqtt/test_light/set",
                payload={"state": "ON"},
                description="Test light on",
                delay_before=0,  # Will be randomized
            ),
            TestEvent(
                topic="zigbee2mqtt/test_pir",
                payload={"occupancy": False},
                description="Test PIR clears (occupancy=false)",
                delay_before=1.0,
            ),
        ],
        repeat_count=3,
        repeat_delay=3.0,
    ),
}


def run_scenario(
    scenario: TestScenario,
    user_delay_range: tuple[float, float] = (3.0, 8.0)
):
    """
    Run a test scenario multiple times to trigger rule learning.

    Args:
        scenario: The scenario to run
        user_delay_range: (min, max) seconds for randomized user action delay
    """
    cyan, yellow, green, magenta, reset = (
        "\033[96m", "\033[93m", "\033[92m", "\033[95m", "\033[0m"
    )
    bold = "\033[1m"

    print(f"\n{bold}{cyan}{'='*60}{reset}")
    print(f"{bold}{cyan}Running Scenario: {scenario.name}{reset}")
    print(f"{cyan}{scenario.description}{reset}")
    print(f"{cyan}Repeating {scenario.repeat_count}x to trigger rule creation{reset}")
    print(f"{cyan}{'='*60}{reset}\n")

    # INITIALIZATION PHASE: Prime the daemon with initial states
    print(f"{magenta}--- Initialization Phase: Priming daemon with baseline ---{reset}")
    for event in scenario.events:
        # Send the "opposite" state to prime the analyzer
        primed_payload = event.payload.copy()
        if "occupancy" in primed_payload:
            primed_payload["occupancy"] = not primed_payload["occupancy"]
        elif "contact" in primed_payload:
            primed_payload["contact"] = not primed_payload["contact"]
        elif "state" in primed_payload:
            primed_payload["state"] = (
                "OFF" if primed_payload["state"] == "ON" else "ON"
            )
        else:
            continue  # No state field to prime

        print(f"{timestamp()} {magenta}Priming: {event.topic} with opposite state{reset}")
        send_mqtt(event.topic, primed_payload)
        time.sleep(0.5)

    print(f"{timestamp()} Waiting 2s for daemon to process initial states...\n")
    time.sleep(2.0)

    for rep in range(scenario.repeat_count):
        # Randomize the user action delay for realism
        user_delay = random.uniform(*user_delay_range)

        print(
            f"{yellow}--- Repetition {rep + 1}/{scenario.repeat_count} "
            f"(user delay: {user_delay:.1f}s) ---{reset}"
        )

        for i, event in enumerate(scenario.events):
            # Determine delay
            if i == 1 and event.delay_before == 0:
                # This is the "user action" - use randomized delay
                delay = user_delay
            else:
                delay = event.delay_before

            if delay > 0:
                print(f"{timestamp()} Waiting {delay:.1f}s...")
                time.sleep(delay)

            # Send the event
            print(f"{timestamp()} {green}→ {event.description}{reset}")
            print(f"           Topic: {event.topic}")
            print(f"           Payload: {json.dumps(event.payload)}")

            if not send_mqtt(event.topic, event.payload):
                print("Failed to send event. Aborting scenario.", file=sys.stderr)
                return False

        # Wait between repetitions
        if rep < scenario.repeat_count - 1:
            print(
                f"\n{timestamp()} Waiting {scenario.repeat_delay}s "
                "before next repetition...\n"
            )
            time.sleep(scenario.repeat_delay)

    print(f"\n{bold}{green}✓ Scenario complete! Check if a rule was created.{reset}")
    print(f"{green}  Run: cat learned_rules.json{reset}\n")
    return True


def interactive_mode():
    """Interactive mode for sending custom events."""
    cyan, green, reset = "\033[96m", "\033[92m", "\033[0m"

    print(f"\n{cyan}=== Interactive MQTT Test Mode ==={reset}")
    print("Send custom MQTT messages. Type 'quit' to exit.\n")

    while True:
        try:
            topic = input("Topic (or 'quit'): ").strip()
            if topic.lower() == 'quit':
                break

            payload_str = input("Payload (JSON): ").strip()
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                print("Invalid JSON. Try again.")
                continue

            print(f"{green}Sending...{reset}")
            if send_mqtt(topic, payload):
                print(f"{green}✓ Sent successfully{reset}\n")

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except EOFError:
            break


def list_scenarios():
    """List all available test scenarios."""
    cyan, reset = "\033[96m", "\033[0m"
    bold = "\033[1m"

    print(f"\n{bold}{cyan}Available Test Scenarios:{reset}\n")
    for key, scenario in SCENARIOS.items():
        print(f"  {bold}{key}{reset}")
        print(f"    {scenario.description}")
        print(f"    Events: {len(scenario.events)}, Repeats: {scenario.repeat_count}x")
        print()


def main():
    """Main entry point for the test client."""
    parser = argparse.ArgumentParser(
        description="MQTT Test Client - Simulate events for rule learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mqtt_spawn_test_client.py                     # Run default PIR->Light
  python mqtt_spawn_test_client.py --scenario door_light
  python mqtt_spawn_test_client.py --list              # List all scenarios
  python mqtt_spawn_test_client.py --custom            # Interactive mode
  python mqtt_spawn_test_client.py --min-delay 2 --max-delay 6
        """
    )
    parser.add_argument(
        "--scenario", "-s",
        choices=list(SCENARIOS.keys()),
        default="pir_light",
        help="Scenario to run (default: pir_light)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available scenarios"
    )
    parser.add_argument(
        "--custom", "-c",
        action="store_true",
        help="Interactive mode for custom events"
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=3.0,
        help="Minimum user action delay in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=8.0,
        help="Maximum user action delay in seconds (default: 8.0)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Override repeat count (default: use scenario's count)"
    )

    args = parser.parse_args()

    if args.list:
        list_scenarios()
        return

    if args.custom:
        interactive_mode()
        return

    # Run the selected scenario
    scenario = SCENARIOS[args.scenario]

    # Allow overriding repeat count
    if args.repeat:
        scenario.repeat_count = args.repeat

    run_scenario(scenario, user_delay_range=(args.min_delay, args.max_delay))


if __name__ == "__main__":
    main()
