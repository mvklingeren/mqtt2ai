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
from dataclasses import dataclass
from datetime import datetime
from collections import deque as Deque
from typing import Optional

from config import Config
from knowledge_base import KnowledgeBase
from mqtt_client import MqttClient
from ai_agent import AiAgent
from trigger_analyzer import TriggerAnalyzer

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
    else:
        model = "unknown"

    print(f"{cyan}{BANNER}{reset}")
    print(f"  {dim}v{VERSION}{reset}  {bold}AI:{reset} {yellow}{provider}{reset} / {model}")
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
        self.trigger_analyzer = TriggerAnalyzer(config.filtered_triggers_file)

        self.messages_deque: Deque[str] = collections.deque(maxlen=config.max_messages)
        self.new_message_count = 0
        self.lock = threading.Lock()

        self.ai_event = threading.Event()
        self.running = True
        self.collector_thread: Optional[threading.Thread] = None

        # AI worker thread and queue for async AI calls
        self.ai_queue: queue.Queue[Optional[AiRequest]] = queue.Queue()
        self.ai_thread: Optional[threading.Thread] = None
        self.ai_busy = threading.Event()  # Set when AI is processing a request

        # Keyboard input thread for manual triggers
        self.keyboard_thread: Optional[threading.Thread] = None
        self.manual_trigger = False  # Flag to track if trigger was manual

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

        # Start collector
        self.collector_thread = threading.Thread(
            target=self._collector_loop, daemon=True, name="MQTT-Collector"
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
        logging.info("Press [Enter] to trigger an immediate AI check")

        while self.running:
            try:
                # Use select for non-blocking check with timeout on Unix
                if sys.platform != 'win32':
                    ready, _, _ = select.select([sys.stdin], [], [], 1.0)
                    if ready:
                        line = sys.stdin.readline()
                        if line == '\n' or line == '':
                            self.manual_trigger = True
                            self.ai_event.set()
                            logging.info("Manual AI check triggered")
                else:
                    # Windows fallback - blocking read
                    import msvcrt  # pylint: disable=import-outside-toplevel
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b'\r' or key == b'\n':
                            self.manual_trigger = True
                            self.ai_event.set()
                            logging.info("Manual AI check triggered")
                    else:
                        time.sleep(0.1)
            except Exception:  # pylint: disable=broad-exception-caught
                # Stdin might not be available (e.g., when running as service)
                time.sleep(1.0)

    def _handle_ai_check(self, snapshot: str, reason: str):
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

            request = AiRequest(snapshot=snapshot, reason=reason)
            self.ai_queue.put(request)
            logging.debug("Queued AI request (reason: %s)", reason)

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

                # Mark as busy to prevent queue buildup
                self.ai_busy.set()

                try:
                    # Reload knowledge base to get any updates
                    self.kb.load_all()
                    self.ai.run_analysis(request.snapshot, self.kb, request.reason)
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

        logging.info("Daemon shutdown complete")

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
                    if self.config.compress_output:
                        compressed = self._compress_line(line)
                        if compressed:  # Skip if fully compressed to nothing
                            print(f"{timestamp()} {compressed}")
                    else:
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
