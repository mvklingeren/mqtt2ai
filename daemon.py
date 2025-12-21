"""Daemon module for the MQTT AI Daemon.

This module contains the main MqttAiDaemon class that orchestrates
the MQTT message collection, trigger analysis, and AI integration.
"""
import collections
import os
import threading
import time
import logging
import signal
from datetime import datetime
from collections import deque as Deque

from config import Config
from knowledge_base import KnowledgeBase
from mqtt_client import MqttClient
from ai_agent import AiAgent
from trigger_analyzer import TriggerAnalyzer


def setup_logging(config: Config):
    """Configure the logging module."""
    level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="[%H:%M:%S]"
    )


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
        self.trigger_analyzer = TriggerAnalyzer(config.filtered_triggers_file)

        self.messages_deque: Deque[str] = collections.deque(maxlen=config.max_messages)
        self.new_message_count = 0
        self.lock = threading.Lock()

        self.ai_event = threading.Event()
        self.running = True
        self.collector_thread = None

    def start(self):
        """Start the daemon."""
        setup_logging(self.config)

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

        # Start collector
        self.collector_thread = threading.Thread(
            target=self._collector_loop, daemon=True
        )
        self.collector_thread.start()

        if self.config.no_ai:
            logging.info("Daemon started in NO-AI MODE (logging only, no AI calls)")
        else:
            logging.info(
                "Daemon started. AI checks every %ds, %d msgs, or on smart trigger.",
                self.config.ai_check_interval,
                self.config.ai_check_threshold
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
                        self.new_message_count >= self.config.ai_check_threshold
                    )

                should_check_time = (
                    (time.time() - last_check_time) >= self.config.ai_check_interval
                )

                if instant_trigger or should_check_count or should_check_time:
                    reason = self._determine_trigger_reason(
                        instant_trigger, should_check_count
                    )
                    if instant_trigger:
                        self.ai_event.clear()

                    # Capture snapshot
                    with self.lock:
                        snapshot = "\n".join(list(self.messages_deque))
                        self.new_message_count = 0
                        last_check_time = time.time()

                    if snapshot:
                        self._handle_ai_check(snapshot, reason)

        except KeyboardInterrupt:
            logging.info("Stopping daemon (KeyboardInterrupt)...")
        finally:
            self.running = False

    def _determine_trigger_reason(
        self, instant_trigger: bool, should_check_count: bool
    ) -> str:
        """Determine the reason for triggering an AI check."""
        if instant_trigger:
            return "smart_trigger"
        if should_check_count:
            return f"message_count ({self.config.ai_check_threshold})"
        return f"interval ({self.config.ai_check_interval}s)"

    def _handle_ai_check(self, snapshot: str, reason: str):
        """Handle the AI check based on mode."""
        if self.config.no_ai:
            logging.info(
                "[NO-AI MODE] Would have triggered AI check (reason: %s), %d messages",
                reason,
                len(snapshot.splitlines())
            )
        else:
            # Reload knowledge base to get any updates from tools/external edits
            self.kb.load_all()
            self.ai.run_analysis(snapshot, self.kb, reason)

    def _collector_loop(self):  # pylint: disable=too-many-branches,too-many-statements
        """Background thread that collects MQTT messages."""
        logging.info(
            "Starting MQTT collector (quiet for %ds)...",
            self.config.skip_printing_seconds
        )

        try:
            process = self.mqtt.start_listener_process()
            start_time = time.time()
            quiet_period_over = False

            for raw_line in iter(process.stdout.readline, b""):
                if not self.running:
                    break

                # 1. Safe Topic Extraction
                try:
                    space_idx = raw_line.index(b" ")
                    raw_topic = raw_line[:space_idx].decode("ascii")
                except (ValueError, UnicodeDecodeError):
                    continue  # Skip invalid lines

                # 2. Binary Topic Filtering (Prefixes)
                if any(
                    raw_topic.startswith(p)
                    for p in self.config.ignore_printing_prefixes
                ):
                    continue

                # 3. Payload Decoding
                try:
                    line = raw_line.decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue

                if not line:
                    continue

                current_topic = raw_topic
                current_payload = (
                    line[len(current_topic) + 1:].strip()
                    if len(line) > len(current_topic) else ""
                )

                if not current_payload or not current_payload.isprintable():
                    continue

                # 4. Display Logic
                elapsed = time.time() - start_time
                should_print = False

                if elapsed > self.config.skip_printing_seconds:
                    if not quiet_period_over:
                        logging.info("--- Initial quiet period over. Monitoring... ---")
                        quiet_period_over = True

                    if current_topic not in self.config.ignore_printing_topics:
                        should_print = True

                # 5. Analysis
                trigger_result = self.trigger_analyzer.analyze(
                    current_topic, current_payload
                )

                is_trigger_line = False
                if trigger_result.should_trigger:
                    self._print_trigger(trigger_result, line)
                    self.ai_event.set()
                    is_trigger_line = True

                # Verbose printing for non-trigger lines
                if should_print and not is_trigger_line and self.config.verbose:
                    print(f"{timestamp()} {line}")

                # 6. Store
                with self.lock:
                    self.messages_deque.append(f"{timestamp()} {line}")
                    self.new_message_count += 1

            # Cleanup if process ends
            process.communicate()
            if process.returncode != 0:
                logging.error(
                    "mosquitto_sub exited with code %d", process.returncode
                )

        except FileNotFoundError:
            logging.critical(
                "CRITICAL: 'mosquitto_sub' not found. Cannot collect messages."
            )
            os.kill(os.getpid(), signal.SIGTERM)  # Kill daemon
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Collector thread error: %s", e)

    def _print_trigger(self, trigger_result, line: str):
        """Print a formatted trigger notification."""
        yellow, bold, reset = "\033[93m", "\033[1m", "\033[0m"
        print(f"{timestamp()} {line}")
        print(
            f"{timestamp()} {bold}{yellow}"
            f">>> SMART TRIGGER: {trigger_result.reason} <<<{reset}"
        )
        print(
            f"           {yellow}Field: {trigger_result.field_name}, "
            f"Change: {trigger_result.old_value} -> {trigger_result.new_value}{reset}"
        )
