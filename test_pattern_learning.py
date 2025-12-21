#!/usr/bin/env python3
"""
Pattern Learning Test Script.

Simulates trigger→action patterns for the AI daemon to learn.
This script creates clear patterns: PIR motion → light turns on.

The key to successful pattern learning:
1. Send a PIR trigger (occupancy: false → true)
2. Wait a realistic delay (2-5 seconds, like a human would)
3. Send a light action (state: ON via /set topic)
4. Repeat 3+ times for the AI to recognize the pattern
"""

import argparse
import json
import time
from datetime import datetime

from utils import publish_mqtt

# MQTT Configuration
MQTT_HOST = "192.168.1.245"
MQTT_PORT = "1883"

# Test device names
PIR_TOPIC = "zigbee2mqtt/test_pir"
LIGHT_TOPIC = "zigbee2mqtt/test_light/set"


def get_timestamp():
    """Return current timestamp string."""
    return datetime.now().strftime("[%H:%M:%S]")


def send_mqtt(topic: str, payload: dict) -> bool:
    """Send an MQTT message and return success status."""
    return publish_mqtt(topic, payload, MQTT_HOST, MQTT_PORT)


def print_header(text):
    """Print a formatted header."""
    cyan = "\033[96m"
    bold = "\033[1m"
    reset = "\033[0m"
    print(f"\n{bold}{cyan}{'='*60}{reset}")
    print(f"{bold}{cyan}{text}{reset}")
    print(f"{bold}{cyan}{'='*60}{reset}\n")


def print_event(description, topic, payload):
    """Print a formatted event."""
    green = "\033[92m"
    reset = "\033[0m"
    print(f"{get_timestamp()} {green}→ {description}{reset}")
    print(f"           Topic: {topic}")
    print(f"           Payload: {json.dumps(payload)}")


def run_pattern_test(num_repetitions=4, action_delay=3.0, between_delay=8.0):
    """
    Run the pattern learning test.

    Args:
        num_repetitions: How many times to repeat (3+ needed for rule creation)
        action_delay: Seconds between PIR trigger and light action
        between_delay: Seconds between pattern repetitions
    """
    yellow = "\033[93m"
    magenta = "\033[95m"
    green = "\033[92m"
    reset = "\033[0m"

    print_header("Pattern Learning Test")
    print("This test simulates: PIR motion → Light turns ON")
    print(f"Repetitions: {num_repetitions}x (need 3+ for rule creation)")
    print(f"Action delay: {action_delay}s (time between motion and light)")
    print(f"Between delay: {between_delay}s (time between repetitions)")

    # =========================================================================
    # STEP 1: Initialize state (set PIR to false first)
    # =========================================================================
    print(f"\n{magenta}--- Phase 1: Initialize PIR state ---{reset}")
    print(f"{get_timestamp()} Setting PIR to occupancy=false (baseline state)")

    send_mqtt(PIR_TOPIC, {"occupancy": False})

    print(f"{get_timestamp()} Waiting 6s for cooldown to expire...")
    time.sleep(6)

    # =========================================================================
    # STEP 2: Run the pattern multiple times
    # =========================================================================
    print(f"\n{magenta}--- Phase 2: Run trigger→action patterns ---{reset}")

    for rep in range(1, num_repetitions + 1):
        print(f"\n{yellow}━━━ Pattern #{rep}/{num_repetitions} ━━━{reset}")

        # 2a. PIR detects motion (trigger)
        print_event(
            "PIR sensor detects motion (TRIGGER)",
            PIR_TOPIC,
            {"occupancy": True}
        )
        send_mqtt(PIR_TOPIC, {"occupancy": True})

        # 2b. Wait (simulates human reaction time)
        print(f"{get_timestamp()} Waiting {action_delay}s (simulating human reaction)...")
        time.sleep(action_delay)

        # 2c. Light turns on (action)
        print_event(
            "Light turns ON (ACTION)",
            LIGHT_TOPIC,
            {"state": "ON"}
        )
        send_mqtt(LIGHT_TOPIC, {"state": "ON"})

        # 2d. Clear the PIR (so next trigger is a change)
        time.sleep(1)
        print_event(
            "PIR clears (resets for next trigger)",
            PIR_TOPIC,
            {"occupancy": False}
        )
        send_mqtt(PIR_TOPIC, {"occupancy": False})

        # Wait between repetitions (but not after the last one)
        if rep < num_repetitions:
            print(f"\n{get_timestamp()} Waiting {between_delay}s before next pattern...")
            time.sleep(between_delay)

    # =========================================================================
    # STEP 3: Done
    # =========================================================================
    print_header("Test Complete!")
    print(f"{green}✓ Sent {num_repetitions} trigger→action patterns{reset}")
    print(f"\n{yellow}Check if a rule was learned:{reset}")
    print("  cat learned_rules.json")
    print("  cat pending_patterns.json")
    print()


def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(
        description="Test pattern learning for MQTT AI daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_pattern_learning.py                    # Default: 4 patterns, 3s delay
  python test_pattern_learning.py -n 5 -d 2         # 5 patterns, 2s delay
  python test_pattern_learning.py --quick           # Quick test: 3 patterns, 2s delay
        """
    )
    parser.add_argument(
        "-n", "--repetitions",
        type=int,
        default=4,
        help="Number of pattern repetitions (default: 4)"
    )
    parser.add_argument(
        "-d", "--delay",
        type=float,
        default=3.0,
        help="Seconds between trigger and action (default: 3.0)"
    )
    parser.add_argument(
        "-b", "--between",
        type=float,
        default=8.0,
        help="Seconds between repetitions (default: 8.0)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode: 3 repetitions, 2s delays"
    )

    args = parser.parse_args()

    if args.quick:
        run_pattern_test(num_repetitions=3, action_delay=2.0, between_delay=6.0)
    else:
        run_pattern_test(
            num_repetitions=args.repetitions,
            action_delay=args.delay,
            between_delay=args.between
        )


if __name__ == "__main__":
    main()
