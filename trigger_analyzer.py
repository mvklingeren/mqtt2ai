"""
Trigger Analyzer Module.

Provides intelligent filtering for MQTT messages to detect meaningful changes
while ignoring routine noise (like periodic sensor updates with minor fluctuations).

Key features:
- Tracks state fields (contact, smoke, occupancy, etc.) and triggers on value changes
- Tracks numeric fields (power, temperature, etc.) and triggers on significant deltas
- Detects gradual drift by comparing current values to baseline from X minutes ago
- Implements per-topic cooldown to prevent rapid re-triggers
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Dict


@dataclass
class TriggerResult:
    """Result of analyzing an MQTT message for triggering."""

    should_trigger: bool
    reason: Optional[str] = None
    field_name: Optional[str] = None
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    topic: Optional[str] = None  # The MQTT topic that triggered

    def __str__(self):
        if self.should_trigger:
            return (
                f"TRIGGER: {self.reason} "
                f"(field={self.field_name}, {self.old_value} -> {self.new_value})"
            )
        return "NO_TRIGGER"


@dataclass
class NumericFieldState:
    """Tracks state for a numeric field including baseline for drift detection."""

    last_value: float
    last_update_time: float
    baseline_value: float
    baseline_time: float


@dataclass
class TopicState:
    """Tracks the state of a single MQTT topic."""

    last_trigger_time: float = 0.0
    state_fields: Dict[str, Any] = field(default_factory=dict)
    numeric_fields: Dict[str, NumericFieldState] = field(default_factory=dict)


class TriggerAnalyzer:
    """
    Analyzes MQTT messages to determine if they should trigger an AI check.

    Maintains per-topic state to detect:
    - Boolean/state field changes (e.g., contact: true -> false)
    - Significant numeric changes (e.g., power jumping by 100W)
    - Gradual drift in numeric values over time (e.g., temperature slowly rising)
    """

    def __init__(
        self, config_path: str = "filtered_triggers.json",
        simulation_mode: bool = False
    ):
        """Initialize the analyzer with configuration from a JSON file.
        
        Args:
            config_path: Path to the trigger configuration JSON file
            simulation_mode: If True, disables cooldown to allow rapid triggering
        """
        self.config = self._load_config(config_path)
        self.topic_states: Dict[str, TopicState] = {}
        self.simulation_mode = simulation_mode

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file with sensible defaults."""
        default_config = {
            "cooldown_seconds": 60,
            "baseline_window_seconds": 600,
            "state_fields": [
                "contact", "smoke", "occupancy", "water_leak",
                "tamper", "motion", "state", "action"
            ],
            "numeric_fields": {
                "power": {"immediate_delta": 100, "drift_delta": 200},
                "temperature": {"immediate_delta": 10, "drift_delta": 15},
                "current": {"immediate_delta": 2, "drift_delta": 5},
                "humidity": {"immediate_delta": 20, "drift_delta": 30},
                "illuminance": {"immediate_delta": 500, "drift_delta": 1000}
            }
        }

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded_config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in loaded_config:
                        loaded_config[key] = value
                return loaded_config
        except FileNotFoundError:
            print(f"Info: '{config_path}' not found. Using default configuration.")
            return default_config
        except json.JSONDecodeError as e:
            print(f"Error loading '{config_path}': {e}. Using default configuration.")
            return default_config

    def _get_or_create_topic_state(self, topic: str) -> TopicState:
        """Get existing topic state or create a new one."""
        if topic not in self.topic_states:
            self.topic_states[topic] = TopicState()
        return self.topic_states[topic]

    def _is_on_cooldown(self, topic_state: TopicState) -> bool:
        """Check if the topic is still in cooldown period.
        
        In simulation mode, cooldown is disabled to allow rapid triggering
        for testing pattern learning.
        """
        # Disable cooldown in simulation mode
        if self.simulation_mode:
            return False
        
        if topic_state.last_trigger_time == 0:
            return False
        elapsed = time.time() - topic_state.last_trigger_time
        return elapsed < self.config["cooldown_seconds"]

    def _check_state_fields(
        self, topic_state: TopicState, payload: dict
    ) -> Optional[TriggerResult]:
        """Check if any state field has changed value."""
        for field_name in self.config["state_fields"]:
            if field_name not in payload:
                continue

            new_value = payload[field_name]
            old_value = topic_state.state_fields.get(field_name)

            # Update stored value
            topic_state.state_fields[field_name] = new_value

            # If this is the first time we see this field, don't trigger
            if old_value is None:
                continue

            # Check if value changed
            if new_value != old_value:
                return TriggerResult(
                    should_trigger=True,
                    reason=f"State field '{field_name}' changed",
                    field_name=field_name,
                    old_value=old_value,
                    new_value=new_value
                )

        return None

    def _check_numeric_fields(  # pylint: disable=too-many-locals
        self, topic_state: TopicState, payload: dict
    ) -> Optional[TriggerResult]:
        """Check if any numeric field has changed significantly or drifted."""
        current_time = time.time()
        baseline_window = self.config["baseline_window_seconds"]

        for field_name, thresholds in self.config["numeric_fields"].items():
            if field_name not in payload:
                continue

            try:
                new_value = float(payload[field_name])
            except (TypeError, ValueError):
                continue

            immediate_delta = thresholds.get("immediate_delta", float("inf"))
            drift_delta = thresholds.get("drift_delta", float("inf"))

            # Get or initialize numeric field state
            if field_name not in topic_state.numeric_fields:
                # First time seeing this field - initialize baseline
                topic_state.numeric_fields[field_name] = NumericFieldState(
                    last_value=new_value,
                    last_update_time=current_time,
                    baseline_value=new_value,
                    baseline_time=current_time
                )
                continue

            field_state = topic_state.numeric_fields[field_name]
            old_value = field_state.last_value
            baseline_value = field_state.baseline_value

            # Check immediate delta (compared to last value)
            immediate_change = abs(new_value - old_value)
            if immediate_change >= immediate_delta:
                # Update state before returning
                field_state.last_value = new_value
                field_state.last_update_time = current_time
                # Reset baseline on significant change
                field_state.baseline_value = new_value
                field_state.baseline_time = current_time

                return TriggerResult(
                    should_trigger=True,
                    reason=(
                        f"Immediate change in '{field_name}' "
                        f"(delta={immediate_change:.1f}, threshold={immediate_delta})"
                    ),
                    field_name=field_name,
                    old_value=old_value,
                    new_value=new_value
                )

            # Check drift from baseline
            drift_change = abs(new_value - baseline_value)
            if drift_change >= drift_delta:
                # Update state before returning
                field_state.last_value = new_value
                field_state.last_update_time = current_time
                # Reset baseline on drift detection
                field_state.baseline_value = new_value
                field_state.baseline_time = current_time

                return TriggerResult(
                    should_trigger=True,
                    reason=(
                        f"Drift detected in '{field_name}' from baseline "
                        f"(drift={drift_change:.1f}, threshold={drift_delta})"
                    ),
                    field_name=field_name,
                    old_value=baseline_value,
                    new_value=new_value
                )

            # Update last value
            field_state.last_value = new_value
            field_state.last_update_time = current_time

            # Update baseline if window has elapsed
            baseline_age = current_time - field_state.baseline_time
            if baseline_age >= baseline_window:
                field_state.baseline_value = new_value
                field_state.baseline_time = current_time

        return None

    def analyze(self, topic: str, raw_payload: str) -> TriggerResult:
        """
        Analyze an MQTT message to determine if it should trigger an AI check.

        Args:
            topic: The MQTT topic the message was received on
            raw_payload: The raw payload string (expected to be JSON)

        Returns:
            TriggerResult indicating whether to trigger and why
        """
        # Try to parse payload as JSON
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            # Non-JSON payloads can't be analyzed for field changes
            return TriggerResult(should_trigger=False, reason="Non-JSON payload")

        if not isinstance(payload, dict):
            return TriggerResult(
                should_trigger=False, reason="Payload is not a JSON object"
            )

        topic_state = self._get_or_create_topic_state(topic)

        # Check cooldown first
        if self._is_on_cooldown(topic_state):
            # Still update state even during cooldown
            self._check_state_fields(topic_state, payload)
            self._check_numeric_fields(topic_state, payload)
            return TriggerResult(should_trigger=False, reason="Topic is on cooldown")

        # Check state fields for changes
        result = self._check_state_fields(topic_state, payload)
        if result and result.should_trigger:
            topic_state.last_trigger_time = time.time()
            result.topic = topic  # Include the topic for deduplication
            return result

        # Check numeric fields for significant changes
        result = self._check_numeric_fields(topic_state, payload)
        if result and result.should_trigger:
            topic_state.last_trigger_time = time.time()
            result.topic = topic  # Include the topic for deduplication
            return result

        return TriggerResult(
            should_trigger=False, reason="No significant changes detected", topic=topic
        )

    def get_stats(self) -> dict:
        """Get statistics about tracked topics and their states."""
        return {
            "tracked_topics": len(self.topic_states),
            "config": self.config
        }


# Simple test/demo when run directly
if __name__ == "__main__":
    analyzer = TriggerAnalyzer()

    print("=== Trigger Analyzer Test ===\n")
    print(f"Config: {json.dumps(analyzer.config, indent=2)}\n")

    # Simulate a sequence of messages
    test_messages = [
        ("zigbee2mqtt/power_plug_1", '{"power": 50, "state": "ON"}'),
        ("zigbee2mqtt/power_plug_1", '{"power": 55, "state": "ON"}'),  # Small change
        ("zigbee2mqtt/power_plug_1", '{"power": 200, "state": "ON"}'),  # Big jump
        ("zigbee2mqtt/power_plug_1", '{"power": 210, "state": "ON"}'),  # On cooldown
        ("zigbee2mqtt/door_sensor", '{"contact": true}'),
        ("zigbee2mqtt/door_sensor", '{"contact": true}'),  # No change
        ("zigbee2mqtt/door_sensor", '{"contact": false}'),  # Changed! Should trigger
        ("zigbee2mqtt/smoke_detector", '{"smoke": false, "temperature": 22}'),
        ("zigbee2mqtt/smoke_detector", '{"smoke": true, "temperature": 25}'),  # Smoke!
    ]

    print("Simulating message sequence:\n")
    for test_topic, test_payload in test_messages:
        test_result = analyzer.analyze(test_topic, test_payload)
        icon = "🔴 TRIGGER" if test_result.should_trigger else "⚪ no trigger"
        print(f"{icon} | {test_topic}")
        print(f"         Payload: {test_payload}")
        print(f"         Result: {test_result}")
        print()
