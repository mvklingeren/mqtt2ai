#!/usr/bin/env python3
import argparse
import subprocess
import sys
import collections
import os
import threading
import time
import json
import logging
import signal
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

try:
    from trigger_analyzer import TriggerAnalyzer
except ImportError:
    print("Error: trigger_analyzer.py not found in current directory.", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---

@dataclass
class Config:
    mqtt_host: str = "192.168.1.245"
    mqtt_port: str = "1883"
    mqtt_topic: str = "#"
    max_messages: int = 1000
    ai_check_interval: int = 300  # 5 minutes
    ai_check_threshold: int = 500  # messages
    
    # Files
    rulebook_file: str = "rulebook.md"
    filtered_triggers_file: str = "filtered_triggers.json"
    learned_rules_file: str = "learned_rules.json"
    pending_patterns_file: str = "pending_patterns.json"
    rejected_patterns_file: str = "rejected_patterns.json"
    
    # Gemini
    gemini_command: str = "/opt/homebrew/bin/gemini"
    gemini_model: str = "gemini-2.5-flash"
    
    # Filtering & Display
    verbose: bool = False
    demo_mode: bool = False
    skip_printing_seconds: int = 3
    ignore_printing_topics: List[str] = field(default_factory=lambda: ["zigbee2mqtt/bridge/logging", "zigbee2mqtt/bridge/health"])
    ignore_printing_prefixes: List[str] = field(default_factory=list)

    @classmethod
    def from_args(cls) -> 'Config':
        parser = argparse.ArgumentParser(description="MQTT AI Daemon - Smart home automation with AI")
        
        parser.add_argument("--mqtt-host", default=os.environ.get("MQTT_HOST", "192.168.1.245"), help="MQTT Broker Host")
        parser.add_argument("--mqtt-port", default=os.environ.get("MQTT_PORT", "1883"), help="MQTT Broker Port")
        parser.add_argument("--gemini-command", default=os.environ.get("GEMINI_CLI_COMMAND", "/opt/homebrew/bin/gemini"), help="Path to Gemini CLI")
        parser.add_argument("--model", dest="gemini_model", default="gemini-2.5-flash", help="Gemini Model ID")
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
        parser.add_argument("--demo", action="store_true", help="Enable demo mode")
        
        args = parser.parse_args()
        
        c = cls()
        c.mqtt_host = args.mqtt_host
        c.mqtt_port = args.mqtt_port
        c.gemini_command = args.gemini_command
        c.gemini_model = args.gemini_model
        c.verbose = args.verbose
        c.demo_mode = args.demo
        return c

# --- Logging Setup ---

def setup_logging(config: Config):
    """Configures the logging module."""
    level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="[%H:%M:%S]"
    )

def timestamp() -> str:
    """Return current timestamp in [HH:MM:SS] format (helper for legacy format compatibility)."""
    return datetime.now().strftime("[%H:%M:%S]")

# --- Core Classes ---

class KnowledgeBase:
    """Manages loading and reloading of rules and patterns."""
    def __init__(self, config: Config):
        self.config = config
        self.learned_rules: Dict = {"rules": []}
        self.pending_patterns: Dict = {"patterns": []}
        self.rejected_patterns: Dict = {"patterns": []}
        self.rulebook_content: str = ""

    def load_all(self):
        """Reloads all configuration files from disk."""
        self.learned_rules = self._load_json(self.config.learned_rules_file, {"rules": []})
        self.pending_patterns = self._load_json(self.config.pending_patterns_file, {"patterns": []})
        self.rejected_patterns = self._load_json(self.config.rejected_patterns_file, {"patterns": []})
        
        try:
            with open(self.config.rulebook_file, "r") as f:
                self.rulebook_content = f.read()
        except FileNotFoundError:
            logging.error(f"Rulebook file '{self.config.rulebook_file}' not found.")
            # Don't exit, just continue with empty rulebook
            self.rulebook_content = ""
            
        logging.debug(f"Loaded {len(self.learned_rules.get('rules', []))} rules, {len(self.pending_patterns.get('patterns', []))} patterns.")

    def _load_json(self, filepath: str, default: Any) -> Any:
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default

class MqttClient:
    """Handles low-level MQTT subprocess operations."""
    def __init__(self, config: Config):
        self.config = config

    def publish(self, topic: str, payload: str):
        """Publishes a message using mosquitto_pub."""
        logging.info(f"-> Sending MQTT: Topic='{topic}', Payload='{payload}'")
        try:
            subprocess.run(
                ["mosquitto_pub", "-h", self.config.mqtt_host, "-p", self.config.mqtt_port, "-t", topic, "-m", payload],
                check=True, capture_output=True, text=True
            )
        except FileNotFoundError:
            logging.error("Error: 'mosquitto_pub' command not found.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error sending MQTT message: {e.stderr}")

    def start_listener_process(self) -> subprocess.Popen:
        """Starts the mosquitto_sub process."""
        cmd = ["mosquitto_sub", "-h", self.config.mqtt_host, "-p", self.config.mqtt_port, "-t", self.config.mqtt_topic, "-v"]
        logging.info(f"Starting MQTT listener: {' '.join(cmd)}")
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

class AiAgent:
    """Handles interaction with the Gemini CLI."""
    def __init__(self, config: Config):
        self.config = config

    def run_analysis(self, messages_snapshot: str, kb: KnowledgeBase, trigger_reason: str):
        """Constructs the prompt and calls the Gemini CLI."""
        cyan, reset = "\033[96m", "\033[0m"
        logging.info(f"{cyan}--- AI Check Started (reason: {trigger_reason}) ---{reset}")
        
        demo_instruction = "**Demo mode is ENABLED.** " if self.config.demo_mode else ""
        
        # Helper to format sections
        def format_section(title, data, description):
            if not data or not data.get(list(data.keys())[0]): # Check if dict is empty or list inside is empty
                return ""
            return f"\n\n## {title}:\n{description}\n{json.dumps(data, indent=2)}"

        rules_section = format_section(
            "Learned Automation Rules", 
            kb.learned_rules, 
            "Execute these rules when their triggers match:"
        ) or "\n\n## Learned Automation Rules:\nNo learned rules yet.\n"

        patterns_section = format_section(
            "Pending Pattern Observations", 
            kb.pending_patterns, 
            "These patterns are being tracked but haven't reached 3 occurrences yet:"
        )

        rejected_section = format_section(
            "Rejected Patterns (DO NOT learn these)", 
            kb.rejected_patterns, 
            "These patterns have been explicitly rejected by the user. Do NOT record observations or create rules for them:"
        )

        # Safety Check
        safety_reminder = ""
        trigger_lower = trigger_reason.lower()
        if any(x in trigger_lower for x in ["temperature", "smoke", "water", "leak"]):
            safety_reminder = (
                "\n\n**SAFETY ALERT**: This analysis was triggered by a potential safety event. "
                "Check for temperature > 50C, smoke: true, or water_leak: true conditions and ACT IMMEDIATELY if found. "
                "Safety actions take PRIORITY over pattern learning.\n"
            )

        prompt = (
            f"You are a home automation AI with pattern learning capabilities. {demo_instruction}"
            f"{safety_reminder}"
            "IMPORTANT: Your PRIMARY task is to detect trigger→action patterns and call record_pattern_observation. "
            "Look for PIR/motion sensors (occupancy:true) followed by light/switch actions (/set topics with state:ON). "
            "When you find such a pattern, ALWAYS call record_pattern_observation with the trigger topic, field, action topic, and delay in seconds.\n\n"
            "## Rulebook:\n"
            f"{kb.rulebook_content}"
            f"{rules_section}"
            f"{patterns_section}"
            f"{rejected_section}\n\n"
            "## Latest MQTT Messages (analyze for trigger→action patterns):\n"
            f"{messages_snapshot}\n\n"
            "REMINDER: Look for patterns like 'zigbee2mqtt/xxx_pir {occupancy:true}' followed by 'zigbee2mqtt/xxx/set {state:ON}' and call record_pattern_observation!\n"
        )

        try:
            if not os.access(self.config.gemini_command, os.X_OK):
                logging.error(f"Error: Gemini CLI not found or not executable at '{self.config.gemini_command}'")
                return

            # Call Gemini
            result = subprocess.run(
                [
                    self.config.gemini_command,
                    "--yolo",
                    "--model", self.config.gemini_model,
                    "--allowed-mcp-server-names", "mqtt-tools",
                ],
                input=prompt,
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            
            response_text = result.stdout.strip()
            light_blue, reset_color = "\033[94m", "\033[0m"
            # Using print here for distinct output visibility, or could use logging.info with special formatting
            print(f"{timestamp()} AI Response: {light_blue}{response_text}{reset_color}")

        except subprocess.TimeoutExpired:
            logging.error("Gemini CLI timed out after 120 seconds")
        except subprocess.CalledProcessError as e:
            logging.error(f"Gemini CLI failed with exit code {e.returncode}: {e.stderr}")
        except Exception as e:
            logging.error(f"Unexpected error during AI check: {e}")


class MqttAiDaemon:
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
        """Starts the daemon."""
        setup_logging(self.config)
        
        # Load initial state
        self.kb.load_all()
        
        # Print trigger analyzer stats
        stats = self.trigger_analyzer.get_stats()
        logging.info("Smart trigger configuration loaded:")
        logging.info(f"  - State fields: {stats['config']['state_fields']}")
        logging.info(f"  - Numeric fields: {list(stats['config']['numeric_fields'].keys())}")
        
        # Start collector
        self.collector_thread = threading.Thread(target=self._collector_loop, daemon=True)
        self.collector_thread.start()
        
        logging.info(f"Daemon started. AI checks every {self.config.ai_check_interval}s, {self.config.ai_check_threshold} msgs, or on smart trigger.")
        
        self._main_loop()

    def _main_loop(self):
        last_check_time = time.time()
        
        try:
            while self.running and self.collector_thread.is_alive():
                instant_trigger = self.ai_event.wait(timeout=1.0)
                
                with self.lock:
                    should_check_count = self.new_message_count >= self.config.ai_check_threshold
                
                should_check_time = (time.time() - last_check_time) >= self.config.ai_check_interval
                
                if instant_trigger or should_check_count or should_check_time:
                    if instant_trigger:
                        reason = "smart_trigger"
                        self.ai_event.clear()
                    elif should_check_count:
                        reason = f"message_count ({self.config.ai_check_threshold})"
                    else:
                        reason = f"interval ({self.config.ai_check_interval}s)"
                    
                    # Capture snapshot
                    with self.lock:
                        snapshot = "\n".join(list(self.messages_deque))
                        self.new_message_count = 0
                        last_check_time = time.time()
                    
                    if snapshot:
                        # Reload knowledge base to get any updates from tools/external edits
                        self.kb.load_all()
                        self.ai.run_analysis(snapshot, self.kb, reason)
                        
        except KeyboardInterrupt:
            logging.info("Stopping daemon (KeyboardInterrupt)...")
        finally:
            self.running = False

    def _collector_loop(self):
        logging.info(f"Starting MQTT collector (quiet for {self.config.skip_printing_seconds}s)...")
        
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
                    continue # Skip invalid lines
                
                # 2. Binary Topic Filtering (Prefixes)
                if any(raw_topic.startswith(p) for p in self.config.ignore_printing_prefixes):
                    continue

                # 3. Payload Decoding
                try:
                    line = raw_line.decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue
                
                if not line:
                    continue

                current_topic = raw_topic
                current_payload = line[len(current_topic) + 1:].strip() if len(line) > len(current_topic) else ""
                
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
                trigger_result = self.trigger_analyzer.analyze(current_topic, current_payload)
                
                is_trigger_line = False
                if trigger_result.should_trigger:
                    yellow, bold, reset = "\033[93m", "\033[1m", "\033[0m"
                    # We always print significant triggers
                    print(f"{timestamp()} {line}")
                    print(f"{timestamp()} {bold}{yellow}>>> SMART TRIGGER: {trigger_result.reason} <<<{reset}")
                    print(f"           {yellow}Field: {trigger_result.field_name}, Change: {trigger_result.old_value} -> {trigger_result.new_value}{reset}")
                    
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
                logging.error(f"mosquitto_sub exited with code {process.returncode}")

        except FileNotFoundError:
            logging.critical("CRITICAL: 'mosquitto_sub' not found. Cannot collect messages.")
            os.kill(os.getpid(), signal.SIGTERM) # Kill daemon
        except Exception as e:
            logging.error(f"Collector thread error: {e}")

# --- Entry Point ---

def main():
    config = Config.from_args()
    daemon = MqttAiDaemon(config)
    
    # Signal Handling
    def handle_signal(signum, frame):
        logging.info("Signal received, shutting down...")
        daemon.running = False
        # Allow instant exit if blocked on I/O
        if signum == signal.SIGTERM:
            sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    daemon.start()

if __name__ == "__main__":
    main()