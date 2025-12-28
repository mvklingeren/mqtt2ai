"""Daemon module for the MQTT AI Daemon.

This module contains the main MqttAiDaemon class that orchestrates
the MQTT message collection, trigger analysis, and AI integration.
"""
import collections
import json
import queue
import select
import sys
import threading
import time
import logging
from dataclasses import dataclass
from datetime import datetime
from collections import deque as Deque
from typing import Dict, Optional

from mqtt2ai.core.config import Config
from mqtt2ai.core.event_bus import event_bus
from mqtt2ai.core.scenario_validator import (
    ScenarioValidator,
    print_test_report,
    write_json_report
)
from mqtt2ai.rules.knowledge_base import KnowledgeBase
from mqtt2ai.mqtt.client import MqttClient
from mqtt2ai.ai.agent import AiAgent
from mqtt2ai.ai.orchestrator import AiOrchestrator
from mqtt2ai.mqtt.collector import MqttCollector, CollectorCallbacks
from mqtt2ai.rules.trigger_analyzer import TriggerAnalyzer, TriggerResult
from mqtt2ai.telegram.bot import TelegramBot
from mqtt2ai.telegram.handler import TelegramHandler
from mqtt2ai.rules.engine import RuleEngine
from mqtt2ai.rules.device_tracker import DeviceStateTracker
from mqtt2ai.ai import tools

VERSION = "0.2"

BANNER = r"""
__  __  ___ _____ _____ ____    _    ___
| |  \/  |/ _ \_   _|_   _|___ \  / \  |_ _|
| | |\/| | | | || |   | |   __) |/ _ \  | |
| | |  | | |_| || |   | |  / __// ___ \ | |
| |_|  |_|\__\_\|_|   |_| |_____/_/   \_\___|
"""


# Context for dependency injection into tools
@dataclass
class RuntimeContext:
    """Holds runtime dependencies for injection."""
    mqtt_client: 'MqttClient'
    device_tracker: 'DeviceStateTracker'


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
        if len(models) > 1:
            model = f"{len(models)} models (round-robin)"
        elif models:
            model = models[0]
        else:
            model = "none"
    else:
        model = "unknown"

    print(f"{cyan}{BANNER}{reset}")
    print(f"  {dim}v{VERSION}{reset}  {bold}AI:{reset} {yellow}{provider}{reset} / {model}")

    # Show simulation mode if active
    if config.simulation_file:
        magenta = "\033[95m"
        print(
            f"  {bold}{magenta}[SIMULATION MODE]{reset} "
            f"{dim}{config.simulation_file}{reset}"
        )

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

        # Device state tracker for alert system context
        self.device_tracker = DeviceStateTracker(config.device_track_pattern)

        # Initialize AI agent with event_bus and mqtt_client
        self.ai = AiAgent(
            config,
            event_bus=event_bus,
            device_tracker=self.device_tracker,
            mqtt_client=self.mqtt
        )

        # In simulation mode, disable cooldown to allow rapid triggering
        self.trigger_analyzer = TriggerAnalyzer(
            config.filtered_triggers_file,
            simulation_mode=bool(config.simulation_file)
        )

        # Inject disable_new_rules setting into tools
        tools.set_disable_new_rules(config.disable_new_rules)

        # Rule engine for direct execution of learned rules (no AI needed)
        self.rule_engine = RuleEngine(self.mqtt, self.kb)

        self.messages_deque: Deque[str] = collections.deque(maxlen=config.max_messages)
        self.new_message_count = 0
        self.lock = threading.Lock()

        self.ai_event = threading.Event()
        self.running = True

        # Graceful shutdown signaling from collector threads
        self.shutdown_event = threading.Event()
        self.shutdown_reason: Optional[str] = None

        # AI orchestrator for async AI request processing
        self.ai_orchestrator = AiOrchestrator(
            ai_agent=self.ai,
            knowledge_base=self.kb,
            no_ai=config.no_ai
        )

        # Trigger queue to decouple detection from main loop processing
        # Stores (trigger_result, timestamp) tuples
        self.trigger_queue: queue.Queue[tuple[TriggerResult, float]] = queue.Queue()

        # MQTT collector for message collection and trigger detection
        self.collector = MqttCollector(
            config=config,
            mqtt_client=self.mqtt,
            trigger_analyzer=self.trigger_analyzer,
            device_tracker=self.device_tracker,
            rule_engine=self.rule_engine,
            knowledge_base=self.kb,
            messages_deque=self.messages_deque,
            lock=self.lock,
            callbacks=CollectorCallbacks(
                on_trigger=self._on_trigger,
                on_shutdown=self._request_shutdown,
                wait_for_ai=self.ai_orchestrator.wait_for_completion
            )
        )

        # Keyboard input thread for manual triggers
        self.keyboard_thread: Optional[threading.Thread] = None
        self.manual_trigger = False  # Flag to track if trigger was manual

        # Telegram bot for bidirectional communication
        self.telegram_bot: Optional[TelegramBot] = None
        if config.telegram_enabled:
            self.telegram_handler = TelegramHandler(
                config=config,
                ai_agent=self.ai,
                device_tracker=self.device_tracker
            )
            self.telegram_bot = TelegramBot(
                config,
                device_tracker=self.device_tracker,
                on_user_message=self.telegram_handler.handle_message
            )
            # Connect Telegram bot to AI agent for alert notifications
            self.ai.set_telegram_bot(self.telegram_bot)
            # Set global reference for tools module
            tools.set_telegram_bot(self.telegram_bot)

        # Test mode: load assertions from scenario file
        self._scenario_assertions: Dict[str, dict] = {}
        self._scenario_name: str = ""
        self._test_passed: Optional[bool] = None  # None = not in test mode

    def _load_scenario_assertions(self):
        """Load assertions from the scenario file for test mode validation."""
        try:
            with open(self.config.simulation_file, "r", encoding="utf-8") as f:
                scenario = json.load(f)
            self._scenario_assertions = scenario.get("assertions", {})
            self._scenario_name = scenario.get("name", "")
            if self._scenario_assertions:
                logging.info(
                    "Test mode: loaded %d assertions from scenario",
                    len(self._scenario_assertions)
                )
            else:
                logging.warning(
                    "Test mode: no assertions found in scenario file"
                )
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error("Failed to load scenario assertions: %s", e)
            self._scenario_assertions = {}

    def _request_shutdown(self, reason: str):
        """Signal the main loop to perform a graceful shutdown.

        This method is thread-safe and can be called from collector threads
        to request a clean shutdown instead of using os.kill().
        """
        self.shutdown_reason = reason
        self.running = False
        self.shutdown_event.set()
        self.ai_event.set()  # Wake up main loop

    def _on_trigger(self, trigger_result: TriggerResult, timestamp: float):
        """Handle a trigger detected by the collector.
        
        This callback is invoked by the MqttCollector when a trigger is detected
        and no rule matched. It queues the trigger for AI processing.
        
        Args:
            trigger_result: The trigger detection result
            timestamp: When the trigger was detected
        """
        self.trigger_queue.put((trigger_result, timestamp))
        self.ai_event.set()

    def start(self):
        """Start the daemon."""
        setup_logging(self.config)
        print_banner(self.config)

        # Test mode: load assertions from scenario file (after logging is configured)
        if self.config.test_mode and self.config.simulation_file:
            self._load_scenario_assertions()

        # Load initial state (force reload to ensure files are read on startup)
        self.kb.force_reload()

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

        # Start AI orchestrator (processes AI requests asynchronously)
        self.ai_orchestrator.start()

        # Start keyboard input thread for manual triggers
        self.keyboard_thread = threading.Thread(
            target=self._keyboard_input_loop, daemon=True, name="Keyboard-Input"
        )
        self.keyboard_thread.start()

        # Start Telegram bot if configured
        if self.telegram_bot:
            if self.telegram_bot.start():
                logging.info("Telegram bot started successfully")
            else:
                logging.warning("Telegram bot failed to start")

        # Start MQTT collector (handles real MQTT or simulation)
        self.collector.start()

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
            while self.running and self.collector.is_alive():
                # Wait for trigger or timeout (1s)
                self.ai_event.wait(timeout=1.0)

                # Clear event immediately so we can set it again
                if self.ai_event.is_set():
                    self.ai_event.clear()

                # Check for shutdown signal from collector threads
                if self.shutdown_event.is_set():
                    if self.shutdown_reason:
                        logging.critical("Shutdown requested: %s", self.shutdown_reason)
                    break

                with self.lock:
                    should_check_count = (
                        not self.config.disable_threshold_trigger
                        and self.new_message_count >= self.config.ai_check_threshold
                    )

                should_check_time = (
                    not self.config.disable_interval_trigger
                    and (time.time() - last_check_time) >= self.config.ai_check_interval
                )

                # Process all queued triggers
                triggers_processed = 0
                while not self.trigger_queue.empty():
                    try:
                        trigger_result, _ = self.trigger_queue.get_nowait()
                        reason = "smart_trigger"

                        # Get snapshot for this trigger, excluding init messages
                        with self.lock:
                            snapshot = self._create_filtered_snapshot()

                        if snapshot:
                            self.ai_orchestrator.queue_request(snapshot, reason, trigger_result)
                            triggers_processed += 1
                    except queue.Empty:
                        break

                # If no specific triggers but threshold/interval met
                should_check = (
                    triggers_processed == 0
                    and (should_check_count or should_check_time or self.manual_trigger)
                )
                if should_check:
                    reason = self._determine_trigger_reason(
                        False, should_check_count
                    )

                    with self.lock:
                        snapshot = self._create_filtered_snapshot()
                        self.new_message_count = 0
                        last_check_time = time.time()

                    if snapshot:
                        self.ai_orchestrator.queue_request(snapshot, reason, None)

        except KeyboardInterrupt:
            logging.info("Stopping daemon (KeyboardInterrupt)...")
        finally:
            self._shutdown()

    def _determine_trigger_reason(
        self,
        instant_trigger: bool,
        should_check_count: bool
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

    def _create_filtered_snapshot(self) -> str:
        """Create a snapshot excluding retained messages.

        Retained MQTT messages (stored by the broker and delivered upon
        subscription) are marked with [RETAINED] prefix and excluded from
        AI analysis since they represent historical state, not new events.

        Must be called while holding self.lock.
        """
        filtered_lines = [
            line for line in self.messages_deque
            if not line.startswith("[RETAINED]")
        ]
        return "\n".join(filtered_lines)

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

    def _shutdown(self):
        """Gracefully shutdown the daemon and its threads."""
        self.running = False

        # Stop collector
        self.collector.stop()

        # Stop AI orchestrator
        self.ai_orchestrator.stop()

        # Stop Telegram bot
        if self.telegram_bot:
            logging.debug("Stopping Telegram bot...")
            self.telegram_bot.stop()

        # Disconnect MQTT client
        self.mqtt.disconnect()

        logging.info("Daemon shutdown complete")

        # Test mode: run validation and print results
        if self.config.test_mode and self._scenario_assertions:
            self._run_test_validation()

    def _run_test_validation(self):
        """Run scenario validation and print test results."""
        validator = ScenarioValidator(self._scenario_assertions)
        results = validator.validate()

        # Print results to terminal
        self._test_passed = print_test_report(results, self._scenario_name)

        # Write JSON report if requested
        if self.config.test_report_file:
            write_json_report(
                results,
                self.config.test_report_file,
                self._scenario_name
            )

    def get_test_result(self) -> Optional[bool]:
        """Get the test result for exit code determination.

        Returns:
            True if all tests passed, False if any failed, None if not in test mode
        """
        return self._test_passed
