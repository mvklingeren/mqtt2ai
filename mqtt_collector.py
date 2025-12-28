"""MQTT Collector for message collection and trigger detection.

This module handles collecting MQTT messages (real or simulated),
detecting triggers, and routing them to either the rule engine or AI.
"""
import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from ai_agent import AiAgent
from config import Config
from device_state_tracker import DeviceStateTracker
from event_bus import event_bus, EventType
from knowledge_base import KnowledgeBase
from mqtt_client import MqttClient
from mqtt_simulator import MqttSimulator
from rule_engine import RuleEngine
from trigger_analyzer import TriggerAnalyzer, TriggerResult


def timestamp() -> str:
    """Return current timestamp in [HH:MM:SS] format."""
    return datetime.now().strftime("[%H:%M:%S]")


@dataclass
class CollectorCallbacks:
    """Callbacks for collector events."""
    on_trigger: Callable[[TriggerResult, float], None]
    on_shutdown: Callable[[str], None]
    wait_for_ai: Callable[[], None]


class MqttCollector:
    """Collects MQTT messages and detects triggers.
    
    This class handles:
    - Real MQTT message collection via paho-mqtt subscription
    - Simulated message replay from scenario files
    - Trigger detection and routing to rule engine or AI
    - Message filtering and display
    - Device state tracking
    """

    def __init__(
        self,
        config: Config,
        mqtt_client: MqttClient,
        trigger_analyzer: TriggerAnalyzer,
        device_tracker: DeviceStateTracker,
        rule_engine: RuleEngine,
        knowledge_base: KnowledgeBase,
        messages_deque: deque,
        lock: threading.Lock,
        callbacks: CollectorCallbacks
    ):
        """Initialize the MQTT collector.
        
        Args:
            config: Application configuration
            mqtt_client: MQTT client for subscriptions
            trigger_analyzer: Analyzer for detecting triggers
            device_tracker: Tracker for device states
            rule_engine: Engine for direct rule execution
            knowledge_base: Knowledge base for rules and patterns
            messages_deque: Shared deque for storing messages
            lock: Thread lock for deque access
            callbacks: Callbacks for collector events
        """
        self.config = config
        self.mqtt = mqtt_client
        self.trigger_analyzer = trigger_analyzer
        self.device_tracker = device_tracker
        self.rule_engine = rule_engine
        self.kb = knowledge_base
        self.messages_deque = messages_deque
        self.lock = lock
        self.callbacks = callbacks

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._new_message_count = 0
        self._message_count_lock = threading.Lock()

        # Queue for receiving MQTT messages from paho-mqtt subscription
        # Tuple format: (topic, payload, is_retained)
        self._mqtt_message_queue: queue.Queue[tuple[str, str, bool]] = queue.Queue()

    def start(self) -> None:
        """Start the collector thread (real or simulation)."""
        if self._running:
            logging.warning("Collector already running")
            return

        self._running = True

        if self.config.simulation_file:
            self._thread = threading.Thread(
                target=self._simulation_collector_loop,
                daemon=True,
                name="Simulation-Collector"
            )
        else:
            self._thread = threading.Thread(
                target=self._collector_loop,
                daemon=True,
                name="MQTT-Collector"
            )

        self._thread.start()
        logging.debug("Collector thread started")

    def stop(self) -> None:
        """Stop the collector thread."""
        self._running = False
        # Thread will stop on next iteration due to _running check

    def is_alive(self) -> bool:
        """Check if collector thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def get_new_message_count(self) -> int:
        """Get current new message count."""
        with self._message_count_lock:
            return self._new_message_count

    def reset_message_count(self) -> None:
        """Reset the new message count to zero."""
        with self._message_count_lock:
            self._new_message_count = 0

    def _increment_message_count(self) -> None:
        """Increment the new message count."""
        with self._message_count_lock:
            self._new_message_count += 1

    def _collector_loop(self):  # pylint: disable=too-many-branches,too-many-statements
        """Background thread that collects MQTT messages via paho-mqtt subscription."""
        logging.info("Starting MQTT collector...")

        try:
            # Subscribe to topics using paho-mqtt
            if not self.mqtt.subscribe(
                self.config.mqtt_topics,
                self._mqtt_message_queue
            ):
                logging.critical("Failed to subscribe to MQTT topics")
                self.callbacks.on_shutdown("Failed to subscribe to MQTT topics")
                return

            while self._running:
                # Get message from queue with timeout to allow checking self._running
                try:
                    current_topic, current_payload, is_retained = self._mqtt_message_queue.get(
                        timeout=1.0
                    )
                except queue.Empty:
                    continue

                self._process_message(current_topic, current_payload, is_retained)

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Collector thread error: %s", e)

    def _simulation_collector_loop(self):  # pylint: disable=too-many-branches,too-many-statements
        """Background thread that replays simulated MQTT messages from a scenario file."""
        logging.info(
            "Starting SIMULATION collector from '%s'...",
            self.config.simulation_file
        )

        try:
            simulator = MqttSimulator(
                self.config.simulation_file,
                self.config.simulation_speed
            )

            for topic, payload in simulator.run_simulation():
                if not self._running:
                    break

                self._process_message(topic, payload, is_retained=False, simulation_mode=True)

            # Simulation complete - wait for any pending AI analysis
            logging.info("Simulation complete. Waiting for AI to finish processing...")

            # Give AI time to process final messages
            time.sleep(2.0)

            # Wait for AI queue to be empty
            self.callbacks.wait_for_ai()

            # Publish SIMULATION_COMPLETE event for validation
            event_bus.publish(EventType.SIMULATION_COMPLETE, {
                "total_messages": len(self.messages_deque)
            })

            logging.info("All AI processing complete. Shutting down...")
            self._running = False

        except FileNotFoundError as e:
            logging.critical("CRITICAL: Scenario file not found: %s", e)
            self.callbacks.on_shutdown(f"Scenario file not found: {e}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Simulation collector error: %s", e)
            self._running = False

    def _process_message(
        self,
        topic: str,
        payload: str,
        is_retained: bool = False,
        simulation_mode: bool = False
    ):
        """Process a single MQTT message.
        
        Args:
            topic: MQTT topic
            payload: Message payload
            is_retained: Whether message is a retained message
            simulation_mode: Whether running in simulation mode
        """
        # 1. Topic Filtering (Prefixes)
        if any(
            topic.startswith(p)
            for p in self.config.ignore_printing_prefixes
        ):
            return

        # 2. Payload validation
        if not payload or not payload.isprintable():
            return

        # Build display line (topic + payload format for consistency)
        line = f"{topic} {payload}"

        # 3. Display Logic
        should_print = topic not in self.config.ignore_printing_topics

        # 4. Analysis
        trigger_result = self.trigger_analyzer.analyze(topic, payload)

        is_trigger_line = False
        if trigger_result.should_trigger:
            self._print_trigger(trigger_result, line)

            # Publish TRIGGER_FIRED event for validation (simulation mode)
            if simulation_mode:
                event_bus.publish(EventType.TRIGGER_FIRED, {
                    "topic": topic,
                    "field": trigger_result.field_name,
                    "old_value": trigger_result.old_value,
                    "new_value": trigger_result.new_value
                })

            # Try direct rule execution first (fast path, no AI)
            # Reload knowledge base to get latest rules
            self.kb.load_all()
            rule_handled = self.rule_engine.check_and_execute(
                topic, payload, trigger_result
            )

            if not rule_handled:
                # No rule matched - queue for AI (pattern learning or anomaly)
                if simulation_mode:
                    # Publish RULE_NOT_MATCHED event for validation
                    event_bus.publish(EventType.RULE_NOT_MATCHED, {
                        "trigger_topic": topic,
                        "trigger_field": trigger_result.field_name
                    })

                # Notify via callback
                self.callbacks.on_trigger(trigger_result, time.time())

                # In simulation mode, wait for AI to finish before continuing
                # This ensures deterministic pattern learning
                if simulation_mode:
                    self.callbacks.wait_for_ai()

            is_trigger_line = True

        # Verbose printing for non-trigger lines
        if should_print and not is_trigger_line and self.config.verbose:
            if self.config.compress_output:
                compressed = self._compress_line(line)
                if compressed:  # Skip if fully compressed to nothing
                    print(f"{timestamp()} {compressed}")
            else:
                print(f"{timestamp()} {line}")

        # 5. Update device tracker
        self.device_tracker.update(topic, payload)

        # 6. Store with [RETAINED] prefix for retained messages
        # Retained messages represent historical state, not new events,
        # and are excluded from AI analysis snapshots
        if is_retained:
            msg_prefix = "[RETAINED] "
        else:
            msg_prefix = ""

        with self.lock:
            self.messages_deque.append(f"{msg_prefix}{timestamp()} {line}")

        self._increment_message_count()

    def _compress_line(self, line: str) -> Optional[str]:
        """Compress a single MQTT message line by removing noise fields.
        
        Returns None if the line has no relevant fields after compression.
        """
        try:
            json_start = line.find('{')
            if json_start == -1:
                return line  # No JSON, keep as-is

            prefix = line[:json_start]
            json_str = line[json_start:]

            payload = json.loads(json_str)
            if isinstance(payload, dict):
                filtered = {
                    k: v for k, v in payload.items()
                    if k not in AiAgent.REMOVE_FIELDS
                    and v is not None
                    and not isinstance(v, dict)
                }
                if filtered:
                    return f"{prefix}{json.dumps(filtered, separators=(',', ':'))}"
                return None  # All fields were noise
            return line
        except json.JSONDecodeError:
            return line

    def _print_trigger(self, trigger_result: TriggerResult, line: str):
        """Print a formatted trigger notification."""
        yellow, bold, reset = "\033[93m", "\033[1m", "\033[0m"
        display_line = self._compress_line(line) if self.config.compress_output else line
        if display_line is None:
            display_line = line  # Fallback to original if fully compressed
        print(f"{timestamp()} {display_line}")
        print(
            f"{timestamp()} {bold}{yellow}"
            f">>> SMART TRIGGER: {trigger_result.reason} <<<{reset}"
        )
        print(
            f"           {yellow}Field: {trigger_result.field_name}, "
            f"Change: {trigger_result.old_value} -> {trigger_result.new_value}{reset}"
        )

