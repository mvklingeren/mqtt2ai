"""Daemon module for the MQTT AI Daemon.

This module contains the main MqttAiDaemon class that orchestrates
the MQTT message collection, trigger analysis, and AI integration.
"""
import collections
import json
import os
import queue
import select
import sys
import threading
import time
import logging
import signal
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque as Deque
from typing import Optional

from config import Config
from knowledge_base import KnowledgeBase
from mqtt_client import MqttClient
from mqtt_simulator import MqttSimulator
from ai_agent import AiAgent, set_alert_agent
from trigger_analyzer import TriggerAnalyzer, TriggerResult
from event_bus import event_bus, EventType
import tools

VERSION = "0.2"

BANNER = r"""
  __  __  ___ _____ _____ ____    _    ___ 
 |  \/  |/ _ \_   _|_   _|___ \  / \  |_ _|
 | |\/| | | | || |   | |   __) |/ _ \  | | 
 | |  | | |_| || |   | |  / __// ___ \ | | 
 |_|  |_|\__\_\|_|   |_| |_____/_/   \_\___|
"""


@dataclass
class AiRequest:
    """Represents a request for AI analysis."""
    snapshot: str
    reason: str
    trigger_result: Optional[TriggerResult] = None
    created_at: float = field(default_factory=time.time)

    def is_expired(self, max_age_seconds: float = 30.0) -> bool:
        """Check if this request is too old to be relevant."""
        return (time.time() - self.created_at) > max_age_seconds


import fnmatch
import re


@dataclass
class DeviceStateSnapshot:
    """An immutable point-in-time snapshot of device states.
    
    This ensures the AI makes decisions based on a consistent view
    of device states that won't change during processing.
    """
    states: dict[str, dict]
    timestamp: float
    
    def get_state(self, topic: str) -> Optional[dict]:
        """Get the state for a specific topic from this snapshot."""
        return self.states.get(topic)
    
    @property
    def device_count(self) -> int:
        """Get the number of devices in this snapshot."""
        return len(self.states)
    
    @property
    def age_seconds(self) -> float:
        """Get the age of this snapshot in seconds."""
        return time.time() - self.timestamp


class DeviceStateTracker:
    """Tracks the last known state of devices matching a topic pattern.
    
    This provides an in-memory cache of device states that can be used
    by the alert system to give the AI full context about available devices.
    
    Thread Safety:
        This class is thread-safe. All read operations return deep copies
        or immutable snapshots to prevent TOCTOU race conditions between
        the Collector Thread (writer) and AI Worker Thread (reader).
    """
    
    def __init__(self, pattern: str = "zigbee2mqtt/*"):
        """Initialize the tracker with a topic pattern.
        
        Args:
            pattern: Glob pattern for topics to track (e.g., "zigbee2mqtt/*")
        """
        self.pattern = pattern
        # Convert glob to regex for matching
        # zigbee2mqtt/* -> ^zigbee2mqtt/[^/]+$
        regex_pattern = pattern.replace("*", "[^/]+")
        self._pattern_re = re.compile(f"^{regex_pattern}$")
        self._states: dict[str, dict] = {}
        self._lock = threading.Lock()
    
    def should_track(self, topic: str) -> bool:
        """Check if a topic should be tracked.
        
        Excludes:
        - /set and /get suffixes (commands, not state)
        - bridge/ topics (not device state)
        """
        if topic.endswith("/set") or topic.endswith("/get"):
            return False
        if "/bridge/" in topic:
            return False
        return bool(self._pattern_re.match(topic))
    
    def update(self, topic: str, payload: str) -> None:
        """Update the state for a device topic.
        
        Args:
            topic: The MQTT topic
            payload: The raw payload string (JSON expected)
        """
        if not self.should_track(topic):
            return
        
        try:
            state = json.loads(payload)
            if isinstance(state, dict):
                with self._lock:
                    # Store a copy to prevent external mutation
                    self._states[topic] = dict(state)
        except json.JSONDecodeError:
            pass  # Ignore non-JSON payloads
    
    def get_snapshot(self) -> DeviceStateSnapshot:
        """Get an immutable point-in-time snapshot of all device states.
        
        This is the preferred method for AI/alert processing as it provides
        a consistent view that won't change during decision-making.
        
        Returns:
            DeviceStateSnapshot with deep-copied states and timestamp
        """
        with self._lock:
            # Deep copy all states to ensure immutability
            states_copy = {
                topic: dict(state) for topic, state in self._states.items()
            }
            return DeviceStateSnapshot(
                states=states_copy,
                timestamp=time.time()
            )
    
    def get_all_states(self) -> dict[str, dict]:
        """Get a deep copy of all tracked device states.
        
        Note: Prefer get_snapshot() for AI processing to get timestamp info.
        
        Returns:
            Dict mapping topic -> last known state (deep copied)
        """
        with self._lock:
            # Deep copy to prevent mutation of returned dicts
            return {topic: dict(state) for topic, state in self._states.items()}
    
    def get_state(self, topic: str) -> Optional[dict]:
        """Get a copy of the last known state for a specific topic.
        
        Args:
            topic: The MQTT topic
            
        Returns:
            A copy of the last known state dict, or None if not tracked
        """
        with self._lock:
            state = self._states.get(topic)
            return dict(state) if state else None
    
    def get_device_count(self) -> int:
        """Get the number of tracked devices."""
        with self._lock:
            return len(self._states)


# Global device tracker instance (initialized by daemon)
_device_tracker: Optional['DeviceStateTracker'] = None


def get_device_tracker() -> Optional['DeviceStateTracker']:
    """Get the global device state tracker instance."""
    return _device_tracker


def set_device_tracker(tracker: 'DeviceStateTracker') -> None:
    """Set the global device state tracker instance."""
    global _device_tracker
    _device_tracker = tracker


class RuleEngine:
    """Executes learned rules directly without AI for matched triggers.
    
    This provides fast, deterministic execution of fixed automation rules
    while reserving AI for anomaly detection and pattern learning.
    """

    # Topic for publishing causation announcements
    ANNOUNCE_TOPIC = "mqtt2ai/action/announce"

    def __init__(self, mqtt_client: 'MqttClient', kb: 'KnowledgeBase'):
        self.mqtt = mqtt_client
        self.kb = kb

    def check_and_execute(
        self,
        topic: str,
        payload_str: str,
        trigger_result: TriggerResult
    ) -> bool:
        """Check if any enabled rule matches and execute directly.
        
        Args:
            topic: The MQTT topic that triggered
            payload_str: The raw payload string (JSON)
            trigger_result: The TriggerResult from TriggerAnalyzer
            
        Returns:
            True if a rule was executed (handled), False otherwise
        """
        if not trigger_result.should_trigger:
            return False

        # Parse payload
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            return False

        if not isinstance(payload, dict):
            return False

        # Reload rules to get latest state
        rules = self.kb.learned_rules.get("rules", [])

        for rule in rules:
            if not rule.get("enabled", True):
                continue

            if self._matches(rule, topic, payload, trigger_result):
                self._announce_and_execute(rule, topic, trigger_result)
                # Publish RULE_EXECUTED event for validation
                event_bus.publish(EventType.RULE_EXECUTED, {
                    "rule_id": rule.get("id"),
                    "trigger_topic": topic,
                    "trigger_field": trigger_result.field_name,
                    "action_topic": rule.get("action", {}).get("topic")
                })
                return True

        return False

    def _matches(
        self,
        rule: dict,
        topic: str,
        payload: dict,
        trigger_result: TriggerResult
    ) -> bool:
        """Check if a rule matches the current trigger event."""
        trigger = rule.get("trigger", {})
        
        # Check topic match
        if trigger.get("topic") != topic:
            return False

        # Check field match
        rule_field = trigger.get("field")
        if rule_field != trigger_result.field_name:
            return False

        # Check value match
        rule_value = trigger.get("value")
        actual_value = payload.get(rule_field)

        # Handle type conversion for comparison
        if rule_value == actual_value:
            return True

        # Try string comparison for booleans/numbers
        if str(rule_value).lower() == str(actual_value).lower():
            return True

        return False

    def _announce_and_execute(
        self,
        rule: dict,
        trigger_topic: str,
        trigger_result: TriggerResult
    ):
        """Publish causation announcement and then execute the rule action."""
        rule_id = rule.get("id", "unknown")
        trigger = rule.get("trigger", {})
        action = rule.get("action", {})

        action_topic = action.get("topic", "")
        action_payload = action.get("payload", "{}")

        # Build announcement message
        announcement = {
            "source": "direct_rule",
            "rule_id": rule_id,
            "trigger_topic": trigger_topic,
            "trigger_field": trigger_result.field_name,
            "trigger_value": trigger_result.new_value,
            "action_topic": action_topic,
            "action_payload": action_payload,
            "timestamp": datetime.now().isoformat()
        }

        # Publish announcement first
        green, reset = "\033[92m", "\033[0m"
        logging.info(
            "%s[DIRECT RULE] %s: %s[%s=%s] -> %s%s",
            green, rule_id, trigger_topic, trigger_result.field_name,
            trigger_result.new_value, action_topic, reset
        )

        self.mqtt.publish(self.ANNOUNCE_TOPIC, json.dumps(announcement))

        # Then execute the action
        self.mqtt.publish(action_topic, action_payload)


def setup_logging(config: Config):
    """Configure the logging module."""
    level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="[%H:%M:%S]"
    )


def print_banner(config: Config):
    """Print the MQTT2AI ASCII art banner with version and AI info."""
    cyan, yellow, reset = "\033[96m", "\033[93m", "\033[0m"
    dim, bold = "\033[2m", "\033[1m"

    # Get the appropriate model based on provider
    provider = config.ai_provider
    if provider == "gemini":
        model = config.gemini_model
    elif provider == "claude":
        model = config.claude_model
    elif provider == "codex-openai":
        model = config.codex_model
    elif provider == "openai-compatible":
        models = config.openai_models
        model = f"{len(models)} models (round-robin)" if len(models) > 1 else models[0] if models else "none"
    else:
        model = "unknown"

    print(f"{cyan}{BANNER}{reset}")
    print(f"  {dim}v{VERSION}{reset}  {bold}AI:{reset} {yellow}{provider}{reset} / {model}")
    
    # Show simulation mode if active
    if config.simulation_file:
        magenta = "\033[95m"
        print(f"  {bold}{magenta}[SIMULATION MODE]{reset} {dim}{config.simulation_file}{reset}")
    
    print()


def timestamp() -> str:
    """Return current timestamp in [HH:MM:SS] format."""
    return datetime.now().strftime("[%H:%M:%S]")


class MqttAiDaemon:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """Main daemon class orchestrating the components."""

    def __init__(self, config: Config):
        self.config = config
        self.kb = KnowledgeBase(config)
        self.mqtt = MqttClient(config)
        self.ai = AiAgent(config)
        # In simulation mode, disable cooldown to allow rapid triggering
        self.trigger_analyzer = TriggerAnalyzer(
            config.filtered_triggers_file,
            simulation_mode=bool(config.simulation_file)
        )
        
        # Inject MQTT client and config into tools module
        tools.set_mqtt_client(self.mqtt)
        tools.set_disable_new_rules(config.disable_new_rules)
        
        # Rule engine for direct execution of learned rules (no AI needed)
        self.rule_engine = RuleEngine(self.mqtt, self.kb)
        
        # Device state tracker for alert system context
        self.device_tracker = DeviceStateTracker(config.device_track_pattern)
        set_device_tracker(self.device_tracker)  # Make globally accessible
        
        # Set up alert agent for raise_alert() function
        set_alert_agent(self.ai, config)

        self.messages_deque: Deque[str] = collections.deque(maxlen=config.max_messages)
        self.new_message_count = 0
        self.lock = threading.Lock()

        self.ai_event = threading.Event()
        self.running = True
        self.collector_thread: Optional[threading.Thread] = None

        # AI worker thread and queue for async AI calls
        # maxsize=3 limits queue depth to prevent backlog of stale requests
        self.ai_queue: queue.Queue[Optional[AiRequest]] = queue.Queue(maxsize=3)
        self.ai_thread: Optional[threading.Thread] = None
        self.ai_busy = threading.Event()  # Set when AI is processing a request

        # Keyboard input thread for manual triggers
        self.keyboard_thread: Optional[threading.Thread] = None
        self.manual_trigger = False  # Flag to track if trigger was manual
        self.last_trigger_result: Optional[TriggerResult] = None  # Last trigger result

        # Deduplication: track last queued trigger to avoid redundant requests
        self._last_queued_trigger: Optional[tuple[str, str]] = None  # (topic, field)

        # Queue for receiving MQTT messages from paho-mqtt subscription
        self.mqtt_message_queue: queue.Queue[tuple[str, str]] = queue.Queue()

    def start(self):
        """Start the daemon."""
        setup_logging(self.config)
        print_banner(self.config)

        # Load initial state
        self.kb.load_all()

        # Print trigger analyzer stats
        stats = self.trigger_analyzer.get_stats()
        logging.info("Smart trigger configuration loaded:")
        logging.info("  - State fields: %s", stats['config']['state_fields'])
        logging.info(
            "  - Numeric fields: %s",
            list(stats['config']['numeric_fields'].keys())
        )
        
        # Log rule engine status
        rules_count = len(self.kb.learned_rules.get('rules', []))
        enabled_count = sum(
            1 for r in self.kb.learned_rules.get('rules', [])
            if r.get('enabled', True)
        )
        logging.info(
            "Rule engine initialized: %d rules (%d enabled for direct execution)",
            rules_count, enabled_count
        )
        
        # Log device tracker status
        logging.info(
            "Device tracker initialized: pattern='%s'",
            self.config.device_track_pattern
        )

        # Connect MQTT client for publishing (persistent connection)
        if not self.mqtt.connect():
            logging.warning(
                "Failed to connect MQTT client for publishing. "
                "Will retry on first publish."
            )

        # Start AI worker thread (processes AI requests asynchronously)
        if not self.config.no_ai:
            self.ai_thread = threading.Thread(
                target=self._ai_worker_loop, daemon=True, name="AI-Worker"
            )
            self.ai_thread.start()

        # Start keyboard input thread for manual triggers
        self.keyboard_thread = threading.Thread(
            target=self._keyboard_input_loop, daemon=True, name="Keyboard-Input"
        )
        self.keyboard_thread.start()

        # Start collector (simulation or real MQTT)
        if self.config.simulation_file:
            self.collector_thread = threading.Thread(
                target=self._simulation_collector_loop, daemon=True, name="Simulation-Collector"
            )
        else:
            self.collector_thread = threading.Thread(
                target=self._collector_loop, daemon=True, name="MQTT-Collector"
            )
        self.collector_thread.start()

        if self.config.no_ai:
            logging.info("Daemon started in NO-AI MODE (logging only, no AI calls)")
        else:
            triggers = []
            if not self.config.disable_interval_trigger:
                triggers.append(f"every {self.config.ai_check_interval}s")
            if not self.config.disable_threshold_trigger:
                triggers.append(f"{self.config.ai_check_threshold} msgs")
            triggers.append("smart trigger")
            logging.info(
                "Daemon started. AI checks: %s",
                ", ".join(triggers)
            )

        self._main_loop()

    def _main_loop(self):
        """Main event loop for the daemon."""
        last_check_time = time.time()

        try:
            while self.running and self.collector_thread.is_alive():
                instant_trigger = self.ai_event.wait(timeout=1.0)

                with self.lock:
                    should_check_count = (
                        not self.config.disable_threshold_trigger
                        and self.new_message_count >= self.config.ai_check_threshold
                    )

                should_check_time = (
                    not self.config.disable_interval_trigger
                    and (time.time() - last_check_time) >= self.config.ai_check_interval
                )

                if instant_trigger or should_check_count or should_check_time:
                    reason = self._determine_trigger_reason(
                        instant_trigger, should_check_count
                    )
                    if instant_trigger:
                        self.ai_event.clear()

                    # Capture snapshot and trigger result
                    with self.lock:
                        snapshot = "\n".join(list(self.messages_deque))
                        self.new_message_count = 0
                        last_check_time = time.time()
                        trigger_result = self.last_trigger_result
                        self.last_trigger_result = None  # Clear after use

                    if snapshot:
                        self._handle_ai_check(snapshot, reason, trigger_result)

        except KeyboardInterrupt:
            logging.info("Stopping daemon (KeyboardInterrupt)...")
        finally:
            self._shutdown()

    def _determine_trigger_reason(
        self, instant_trigger: bool, should_check_count: bool
    ) -> str:
        """Determine the reason for triggering an AI check."""
        if self.manual_trigger:
            self.manual_trigger = False
            return "manual (Enter pressed)"
        if instant_trigger:
            return "smart_trigger"
        if should_check_count:
            return f"message_count ({self.config.ai_check_threshold})"
        return f"interval ({self.config.ai_check_interval}s)"

    def _keyboard_input_loop(self):
        """Background thread that listens for Enter key to trigger manual AI check."""
        # Don't start keyboard input in simulation mode (stdin may not be available)
        if self.config.simulation_file:
            logging.debug("Keyboard input disabled in simulation mode")
            return
        
        # Check if stdin is a TTY (interactive terminal)
        if not sys.stdin.isatty():
            logging.debug("Keyboard input disabled (stdin is not a TTY)")
            return
        
        logging.info("Press [Enter] to trigger an immediate AI check")

        while self.running:
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 1.0)
                if ready:
                    line = sys.stdin.readline()
                    if line == '':
                        # EOF reached, stop listening
                        break
                    if line == '\n':
                        self.manual_trigger = True
                        self.ai_event.set()
                        logging.info("Manual AI check triggered")
            except Exception:  # pylint: disable=broad-exception-caught
                # Stdin might not be available (e.g., when running as service)
                time.sleep(1.0)

    def _handle_ai_check(
        self, snapshot: str, reason: str,
        trigger_result: Optional[TriggerResult] = None
    ):
        """Handle the AI check based on mode."""
        if self.config.no_ai:
            logging.info(
                "[NO-AI MODE] Would have triggered AI check (reason: %s), %d messages",
                reason,
                len(snapshot.splitlines())
            )
        else:
            # Queue the AI request for async processing
            # Skip if AI is already busy to avoid queue buildup
            if self.ai_busy.is_set():
                logging.debug(
                    "AI is busy, skipping request (reason: %s)", reason
                )
                return

            # Deduplicate: skip if same topic/field as last queued trigger
            if trigger_result and trigger_result.topic and trigger_result.field_name:
                trigger_key = (trigger_result.topic, trigger_result.field_name)
                if trigger_key == self._last_queued_trigger:
                    logging.debug(
                        "Deduplicating trigger for %s[%s]",
                        trigger_result.topic, trigger_result.field_name
                    )
                    return
                self._last_queued_trigger = trigger_key

            request = AiRequest(
                snapshot=snapshot, reason=reason, trigger_result=trigger_result
            )
            try:
                self.ai_queue.put_nowait(request)
                logging.debug("Queued AI request (reason: %s)", reason)
            except queue.Full:
                logging.debug(
                    "AI queue full, dropping request (reason: %s)", reason
                )

    def _ai_worker_loop(self):
        """Background thread that processes AI requests from the queue."""
        logging.info("AI worker thread started")

        while self.running:
            try:
                # Wait for a request with timeout to allow checking self.running
                request = self.ai_queue.get(timeout=1.0)

                # None is the shutdown signal
                if request is None:
                    logging.debug("AI worker received shutdown signal")
                    break

                # Skip expired requests to avoid processing stale data
                if request.is_expired():
                    age = time.time() - request.created_at
                    logging.info(
                        "Skipping expired AI request (age: %.1fs, reason: %s)",
                        age, request.reason
                    )
                    self.ai_queue.task_done()
                    continue

                # Mark as busy to prevent queue buildup
                self.ai_busy.set()

                try:
                    # Reload knowledge base to get any updates
                    self.kb.load_all()
                    self.ai.run_analysis(
                        request.snapshot, self.kb, request.reason,
                        request.trigger_result
                    )
                finally:
                    self.ai_busy.clear()
                    self.ai_queue.task_done()

            except queue.Empty:
                # Timeout, just continue to check self.running
                continue
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.error("AI worker error: %s", e)
                self.ai_busy.clear()

        logging.info("AI worker thread stopped")

    def _wait_for_ai_completion(self, timeout: float = 60.0):
        """Wait for the AI worker to finish processing the current request.
        
        Used in simulation mode to ensure deterministic pattern learning
        by waiting for each trigger to be fully processed before continuing.
        """
        # First, wait a short time for the request to be picked up from the queue
        time.sleep(0.1)
        
        # Then wait for AI to finish (ai_busy will be set while processing)
        start_wait = time.time()
        while self.ai_busy.is_set() and self.running:
            if time.time() - start_wait > timeout:
                logging.warning("AI completion wait timed out after %.1fs", timeout)
                break
            time.sleep(0.1)

    def _shutdown(self):
        """Gracefully shutdown the daemon and its threads."""
        self.running = False

        # Signal the AI worker thread to stop
        if self.ai_thread and self.ai_thread.is_alive():
            logging.debug("Signaling AI worker to stop...")
            self.ai_queue.put(None)  # Shutdown signal
            self.ai_thread.join(timeout=5.0)
            if self.ai_thread.is_alive():
                logging.warning("AI worker thread did not stop in time")

        # Disconnect MQTT client
        self.mqtt.disconnect()

        logging.info("Daemon shutdown complete")

    def _collector_loop(self):  # pylint: disable=too-many-branches,too-many-statements
        """Background thread that collects MQTT messages via paho-mqtt subscription."""
        logging.info(
            "Starting MQTT collector (quiet for %ds)...",
            self.config.skip_printing_seconds
        )

        try:
            # Subscribe to topics using paho-mqtt
            if not self.mqtt.subscribe(
                self.config.mqtt_topics,
                self.mqtt_message_queue
            ):
                logging.critical("Failed to subscribe to MQTT topics")
                os.kill(os.getpid(), signal.SIGTERM)
                return

            start_time = time.time()
            quiet_period_over = False

            while self.running:
                # Get message from queue with timeout to allow checking self.running
                try:
                    current_topic, current_payload = self.mqtt_message_queue.get(
                        timeout=1.0
                    )
                except queue.Empty:
                    continue

                # 1. Topic Filtering (Prefixes)
                if any(
                    current_topic.startswith(p)
                    for p in self.config.ignore_printing_prefixes
                ):
                    continue

                # 2. Payload validation
                if not current_payload or not current_payload.isprintable():
                    continue

                # Build display line (topic + payload format for consistency)
                line = f"{current_topic} {current_payload}"

                # 3. Display Logic
                elapsed = time.time() - start_time
                should_print = False

                if elapsed > self.config.skip_printing_seconds:
                    if not quiet_period_over:
                        logging.info("--- Initial quiet period over. Monitoring... ---")
                        quiet_period_over = True

                    if current_topic not in self.config.ignore_printing_topics:
                        should_print = True

                # 4. Analysis
                trigger_result = self.trigger_analyzer.analyze(
                    current_topic, current_payload
                )

                is_trigger_line = False
                if trigger_result.should_trigger:
                    self._print_trigger(trigger_result, line)
                    
                    # Try direct rule execution first (fast path, no AI)
                    # Reload knowledge base to get latest rules
                    self.kb.load_all()
                    rule_handled = self.rule_engine.check_and_execute(
                        current_topic, current_payload, trigger_result
                    )
                    
                    if not rule_handled:
                        # No rule matched - queue for AI (pattern learning or anomaly)
                        with self.lock:
                            self.last_trigger_result = trigger_result
                        self.ai_event.set()
                    # If rule_handled, skip AI - the rule was executed directly
                    
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
                self.device_tracker.update(current_topic, current_payload)

                # 6. Store
                with self.lock:
                    self.messages_deque.append(f"{timestamp()} {line}")
                    self.new_message_count += 1

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

            start_time = time.time()
            quiet_period_over = False

            for topic, payload in simulator.run_simulation():
                if not self.running:
                    break

                # Skip ignored prefixes
                if any(
                    topic.startswith(p)
                    for p in self.config.ignore_printing_prefixes
                ):
                    continue

                # Build the line in mosquitto_sub format: "topic payload"
                line = f"{topic} {payload}"

                current_topic = topic
                current_payload = payload

                if not current_payload or not current_payload.isprintable():
                    continue

                # Display Logic
                elapsed = time.time() - start_time
                should_print = False

                if elapsed > self.config.skip_printing_seconds:
                    if not quiet_period_over:
                        logging.info("--- Initial quiet period over. Monitoring... ---")
                        quiet_period_over = True

                    if current_topic not in self.config.ignore_printing_topics:
                        should_print = True

                # Analysis
                trigger_result = self.trigger_analyzer.analyze(
                    current_topic, current_payload
                )

                is_trigger_line = False
                if trigger_result.should_trigger:
                    self._print_trigger(trigger_result, line)
                    
                    # Publish TRIGGER_FIRED event for validation
                    event_bus.publish(EventType.TRIGGER_FIRED, {
                        "topic": current_topic,
                        "field": trigger_result.field_name,
                        "old_value": trigger_result.old_value,
                        "new_value": trigger_result.new_value
                    })
                    
                    # Try direct rule execution first (fast path, no AI)
                    # Reload knowledge base to get latest rules
                    self.kb.load_all()
                    rule_handled = self.rule_engine.check_and_execute(
                        current_topic, current_payload, trigger_result
                    )
                    
                    if not rule_handled:
                        # No rule matched - queue for AI (pattern learning or anomaly)
                        # Publish RULE_NOT_MATCHED event for validation
                        event_bus.publish(EventType.RULE_NOT_MATCHED, {
                            "trigger_topic": current_topic,
                            "trigger_field": trigger_result.field_name
                        })
                        
                        with self.lock:
                            self.last_trigger_result = trigger_result
                        self.ai_event.set()
                        
                        # In simulation mode, wait for AI to finish before continuing
                        # This ensures deterministic pattern learning
                        if not self.config.no_ai:
                            self._wait_for_ai_completion()
                    # If rule_handled, skip AI - the rule was executed directly
                    
                    is_trigger_line = True

                # Verbose printing for non-trigger lines
                if should_print and not is_trigger_line and self.config.verbose:
                    if self.config.compress_output:
                        compressed = self._compress_line(line)
                        if compressed:  # Skip if fully compressed to nothing
                            print(f"{timestamp()} {compressed}")
                    else:
                        print(f"{timestamp()} {line}")

                # Update device tracker
                self.device_tracker.update(current_topic, current_payload)

                # Store
                with self.lock:
                    self.messages_deque.append(f"{timestamp()} {line}")
                    self.new_message_count += 1

            # Simulation complete - wait for any pending AI analysis
            logging.info("Simulation complete. Waiting for AI to finish processing...")
            
            # Give AI time to process final messages
            time.sleep(2.0)
            
            # Wait for AI queue to be empty
            if not self.config.no_ai:
                self.ai_queue.join()
            
            # Publish SIMULATION_COMPLETE event for validation
            event_bus.publish(EventType.SIMULATION_COMPLETE, {
                "total_messages": len(self.messages_deque)
            })
            
            logging.info("All AI processing complete. Shutting down...")
            self.running = False

        except FileNotFoundError as e:
            logging.critical("CRITICAL: Scenario file not found: %s", e)
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Simulation collector error: %s", e)
            self.running = False

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

    def _print_trigger(self, trigger_result, line: str):
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
