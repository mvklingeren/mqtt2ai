"""MQTT Simulator module for the MQTT AI Daemon.

This module provides a simulated MQTT message source that reads
predefined scenarios from JSON files and replays them with timing,
allowing testing of pattern detection and rule learning without
a real MQTT broker.
"""
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Generator, Optional, Tuple


@dataclass
class SimulatedMessage:
    """A simulated MQTT message with timing information."""
    topic: str
    payload: str  # JSON string
    delay: float  # Delay in seconds before this message


class MqttSimulator:
    """Simulates MQTT messages from a predefined scenario file.

    Reads a JSON scenario file and yields messages with appropriate
    timing delays, allowing testing of the full daemon pipeline
    without a real MQTT broker.
    """

    def __init__(self, scenario_path: str, speed_multiplier: Optional[float] = None):
        """Initialize the simulator with a scenario file.

        Args:
            scenario_path: Path to the JSON scenario file
            speed_multiplier: Override the scenario's speed_multiplier (optional)
        """
        self.scenario_path = scenario_path
        self.scenario = self._load_scenario(scenario_path)

        # Use provided multiplier or fall back to scenario's value
        if speed_multiplier is not None:
            self.speed_multiplier = speed_multiplier
        else:
            self.speed_multiplier = self.scenario.get("speed_multiplier", 1.0)

        self.variables = self.scenario.get("variables", {})
        self.messages = self.scenario.get("messages", [])

        logging.info(
            "Loaded simulation scenario: %s (%d messages, speed=%.1fx)",
            self.scenario.get("name", "Unnamed"),
            len(self.messages),
            self.speed_multiplier
        )

    def _load_scenario(self, path: str) -> dict:
        """Load scenario from JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error("Scenario file not found: %s", path)
            raise
        except json.JSONDecodeError as e:
            logging.error("Invalid JSON in scenario file %s: %s", path, e)
            raise

    def _substitute_variables(self, value: Any) -> Any:
        """Replace {{variable}} placeholders with values from variables section."""
        if isinstance(value, str):
            # Match {{variable_name}} pattern
            pattern = r"\{\{(\w+)\}\}"
            matches = re.findall(pattern, value)

            for match in matches:
                if match in self.variables:
                    # If the entire string is just the variable, return the raw value
                    if value == f"{{{{{match}}}}}":
                        return self.variables[match]
                    # Otherwise do string substitution
                    value = value.replace(f"{{{{{match}}}}}", str(self.variables[match]))

            return value
        return value

    def _parse_delay(self, delay_value: Any) -> float:
        """Parse a delay value, substituting variables if needed."""
        delay = self._substitute_variables(delay_value)
        try:
            return float(delay)
        except (TypeError, ValueError):
            logging.warning("Invalid delay value '%s', using 0", delay_value)
            return 0.0

    def _format_payload(self, payload: Any) -> str:
        """Convert payload to JSON string."""
        if isinstance(payload, str):
            return payload
        return json.dumps(payload)

    def generate_messages(self) -> Generator[SimulatedMessage, None, None]:
        """Generate simulated messages with timing.

        Yields SimulatedMessage objects. The caller is responsible for
        sleeping between messages based on the delay.
        """
        for msg in self.messages:
            # Skip comment-only entries (entries with just _comment key)
            if "_comment" in msg and "topic" not in msg:
                continue

            topic = msg.get("topic", "")
            if not topic:
                continue  # Skip entries without a topic

            payload = self._format_payload(msg.get("payload", ""))
            raw_delay = msg.get("delay", 0)
            delay = self._parse_delay(raw_delay) / self.speed_multiplier

            yield SimulatedMessage(
                topic=topic,
                payload=payload,
                delay=delay
            )

    def run_simulation(self) -> Generator[Tuple[str, str], None, None]:
        """Run the simulation, yielding (topic, payload) tuples.

        This handles the timing delays internally and yields
        messages as they become "available".
        """
        cyan, reset = "\033[96m", "\033[0m"
        dim = "\033[2m"

        print(f"\n{cyan}=== MQTT Simulation Started ==={reset}")
        print(f"{dim}Scenario: {self.scenario.get('name', 'Unnamed')}{reset}")
        print(f"{dim}Description: {self.scenario.get('description', 'No description')}{reset}")
        print(f"{dim}Speed: {self.speed_multiplier}x (delays divided by this){reset}")
        print()

        start_time = time.time()

        for msg in self.generate_messages():
            if msg.delay > 0:
                logging.debug("Waiting %.2fs before next message...", msg.delay)
                time.sleep(msg.delay)

            elapsed = time.time() - start_time
            timestamp = datetime.now().strftime("[%H:%M:%S]")

            # Print the simulated message
            print(f"{timestamp} {dim}[SIM +{elapsed:.1f}s]{reset} {msg.topic} {msg.payload}")

            yield (msg.topic, msg.payload)

        print(f"\n{cyan}=== MQTT Simulation Complete ==={reset}")
        print(f"{dim}Total time: {time.time() - start_time:.1f}s{reset}\n")

    def get_info(self) -> dict:
        """Get information about the loaded scenario."""
        return {
            "name": self.scenario.get("name", "Unnamed"),
            "description": self.scenario.get("description", ""),
            "message_count": len(self.messages),
            "speed_multiplier": self.speed_multiplier,
            "variables": self.variables
        }


# CLI test mode
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="[%H:%M:%S]"
    )

    if len(sys.argv) < 2:
        print("Usage: python mqtt_simulator.py <scenario.json> [speed_multiplier]")
        print("\nExample: python mqtt_simulator.py scenarios/pattern_learning.json 10")
        sys.exit(1)

    scenario_file = sys.argv[1]
    speed = float(sys.argv[2]) if len(sys.argv) > 2 else None

    simulator = MqttSimulator(scenario_file, speed)

    print("\nScenario Info:")
    info = simulator.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    print("Running simulation...\n")
    for topic, payload in simulator.run_simulation():
        # In real use, these would be passed to the daemon's message handling
        pass

