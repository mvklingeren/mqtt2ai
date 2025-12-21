import argparse
import subprocess
import sys
import collections
import os
import threading
import time
import json
from datetime import datetime

from trigger_analyzer import TriggerAnalyzer

# --- Runtime Flags (set via command line) ---
VERBOSE = False


def timestamp():
    """Return current timestamp in [HH:MM:SS] format."""
    return datetime.now().strftime("[%H:%M:%S]")


# --- Configuration Constants ---
MAX_MESSAGES = 1000
# Trigger AI check when either of these conditions are met
AI_CHECK_INTERVAL_SECONDS = 300  # 5 minutes (periodic check)
AI_CHECK_MESSAGE_THRESHOLD = 50   # 50 messages accumulated
# Don't print messages for the first N seconds to avoid initial state dump
# This is now minimal as topic filtering handles persistent verbose messages.
SKIP_PRINTING_FOR_SECONDS = 3

# --- MQTT and AI Configuration ---
MQTT_HOST = "192.168.1.245"
MQTT_PORT = "1883"
MQTT_TOPIC = "#" # All topics, use: zigbee2mqtt/# for specific topics
RULEBOOK_FILE = "rulebook.md"
FILTERED_TRIGGERS_FILE = "filtered_triggers.json"
LEARNED_RULES_FILE = "learned_rules.json"
PENDING_PATTERNS_FILE = "pending_patterns.json"
REJECTED_PATTERNS_FILE = "rejected_patterns.json"
GEMINI_CLI_COMMAND = "/opt/homebrew/bin/gemini"
GEMINI_MODEL = "gemini-2.5-flash"  # Better tool calling than lite version
# Topics to ignore printing to console, but still process for AI
# Note: Ring camera snapshot topics contain binary image data that causes garbled output
IGNORE_PRINTING_TOPICS = ["zigbee2mqtt/bridge/logging", "zigbee2mqtt/bridge/health"]
# Topic prefixes to ignore (for binary data like camera snapshots)
IGNORE_PRINTING_PREFIXES = [] #["ring/"]
# Demo mode: when enabled, AI will always call a mcp send mqtt for a joke action
DEMO_MODE = False  # Set to True for testing

# --- Shared State ---
messages_deque = collections.deque(maxlen=MAX_MESSAGES)
new_message_count = 0
shared_state_lock = threading.Lock()

def load_json_file(filepath, default=None):
    """Load a JSON file, returning default if not found or invalid."""
    if default is None:
        default = {}
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def send_mqtt_message(topic, payload):
    """Publishes a generic message to a specified MQTT topic."""
    print(f"  -> Sending MQTT: Topic='{topic}', Payload='{payload}'")
    try:
        subprocess.run(
            ["mosquitto_pub", "-h", MQTT_HOST, "-p", MQTT_PORT, "-t", topic, "-m", payload],
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError:
        print("Error: 'mosquitto_pub' command not found.", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error sending MQTT message: {e.stderr}", file=sys.stderr)

def run_ai_orchestration(messages_snapshot, rulebook_content, learned_rules, pending_patterns, rejected_patterns, trigger_reason="scheduled"):
    """
    Calls Gemini AI with MCP tool calling enabled.
    The AI can directly invoke the send_mqtt_message tool via MCP.
    Also passes learned rules and pending patterns for intelligent automation.
    """
    cyan, reset = "\033[96m", "\033[0m"
    print(f"\n{timestamp()} {cyan}--- AI Check Started (reason: {trigger_reason}) ---{reset}")
    demo_mode_instruction = "**Demo mode is ENABLED.** " if DEMO_MODE else ""
    
    # Format learned rules for the prompt
    rules_section = ""
    if learned_rules.get("rules"):
        rules_section = "\n\n## Learned Automation Rules:\n"
        rules_section += "Execute these rules when their triggers match:\n"
        rules_section += json.dumps(learned_rules, indent=2)
    else:
        rules_section = "\n\n## Learned Automation Rules:\nNo learned rules yet. Watch for repeatable patterns to create new rules.\n"
    
    # Format pending patterns for the prompt
    patterns_section = ""
    if pending_patterns.get("patterns"):
        patterns_section = "\n\n## Pending Pattern Observations:\n"
        patterns_section += "These patterns are being tracked but haven't reached 3 occurrences yet:\n"
        patterns_section += json.dumps(pending_patterns, indent=2)
    
    # Format rejected patterns for the prompt
    rejected_section = ""
    if rejected_patterns.get("patterns"):
        rejected_section = "\n\n## Rejected Patterns (DO NOT learn these):\n"
        rejected_section += "These patterns have been explicitly rejected by the user. Do NOT record observations or create rules for them:\n"
        rejected_section += json.dumps(rejected_patterns, indent=2)
    
    # Add safety reminder for safety-related triggers
    safety_reminder = ""
    trigger_lower = trigger_reason.lower()
    if "temperature" in trigger_lower or "smoke" in trigger_lower or "water" in trigger_lower or "leak" in trigger_lower:
        safety_reminder = (
            "\n\n**SAFETY ALERT**: This analysis was triggered by a potential safety event. "
            "Check for temperature > 50C, smoke: true, or water_leak: true conditions and ACT IMMEDIATELY if found. "
            "Safety actions take PRIORITY over pattern learning.\n"
        )
    
    prompt = (
        f"You are a home automation AI with pattern learning capabilities. {demo_mode_instruction}"
        f"{safety_reminder}"
        "IMPORTANT: Your PRIMARY task is to detect trigger→action patterns and call record_pattern_observation. "
        "Look for PIR/motion sensors (occupancy:true) followed by light/switch actions (/set topics with state:ON). "
        "When you find such a pattern, ALWAYS call record_pattern_observation with the trigger topic, field, action topic, and delay in seconds.\n\n"
        "## Rulebook:\n"
        f"{rulebook_content}"
        f"{rules_section}"
        f"{patterns_section}"
        f"{rejected_section}\n\n"
        "## Latest MQTT Messages (analyze for trigger→action patterns):\n"
        f"{messages_snapshot}\n\n"
        "REMINDER: Look for patterns like 'zigbee2mqtt/xxx_pir {occupancy:true}' followed by 'zigbee2mqtt/xxx/set {state:ON}' and call record_pattern_observation!\n"
    )

    try:
        if not os.access(GEMINI_CLI_COMMAND, os.X_OK):
             print(f"Error: Gemini CLI not found or not executable at '{GEMINI_CLI_COMMAND}'", file=sys.stderr)
             return

        # Run Gemini with MCP tool calling enabled
        # --yolo: auto-approve all tool calls
        # --model: use Flash model for lower quota usage
        # --allowed-mcp-server-names: only allow the mqtt-tools server
        result = subprocess.run(
            [
                GEMINI_CLI_COMMAND,
                "--yolo",
                "--model", GEMINI_MODEL,
                "--allowed-mcp-server-names", "mqtt-tools",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,  # 2 minute timeout
        )
        response_text = result.stdout.strip()
        
        light_blue, reset_color = "\033[94m", "\033[0m"
        print(f"{timestamp()} AI Response: {light_blue}{response_text}{reset_color}")
            
    except subprocess.TimeoutExpired:
        print(f"Error: Gemini CLI timed out after 120 seconds", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error: Gemini CLI failed with exit code {e.returncode}", file=sys.stderr)
        if e.stderr:
            print(f"stderr: {e.stderr}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during AI check: {e}", file=sys.stderr)

def mqtt_collector_thread(trigger_analyzer, trigger_event):
    """
    A dedicated thread that runs mosquitto_sub, collects messages, and uses
    the TriggerAnalyzer to detect significant changes that should trigger AI.
    """
    global new_message_count
    mqtt_command = ["mosquitto_sub", "-h", MQTT_HOST, "-p", MQTT_PORT, "-t", MQTT_TOPIC, "-v"]
    
    print(f"Starting MQTT collector thread (initial quiet period for {SKIP_PRINTING_FOR_SECONDS} seconds)...")
    try:
        process = subprocess.Popen(
            mqtt_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        start_time = time.time()
        quiet_period_over = False

        for raw_line in iter(process.stdout.readline, b""):
            # Extract topic from raw bytes FIRST (before full decode) to check if we should skip
            # Topics are ASCII, so this is safe
            try:
                space_idx = raw_line.index(b" ")
                raw_topic = raw_line[:space_idx].decode("ascii")
            except (ValueError, UnicodeDecodeError):
                # No space found or topic isn't ASCII - skip this line entirely
                continue
            
            # Check if topic should be completely ignored (binary data like camera snapshots)
            is_binary_topic = any(raw_topic.startswith(prefix) for prefix in IGNORE_PRINTING_PREFIXES)
            if is_binary_topic:
                # Don't even try to decode binary payloads - skip entirely
                continue
            
            # Now safe to decode the full line
            try:
                line = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError:
                # Skip messages with binary payloads that can't be decoded
                continue
            if not line:
                continue
            
            # Extract topic and payload for filtering (now we know topic is valid)
            current_topic = raw_topic
            current_payload = line[len(current_topic) + 1:].strip() if len(line) > len(current_topic) else ""
            
            # Skip messages with empty or non-printable payloads (likely binary garbage)
            if not current_payload or not current_payload.isprintable():
                continue

            elapsed_time = time.time() - start_time
            should_print = False

            # Decide whether to print the line based on quiet period and topic filters
            if elapsed_time > SKIP_PRINTING_FOR_SECONDS:
                if not quiet_period_over:
                    print(f"{timestamp()} --- Initial quiet period over. Now printing filtered new messages. ---")
                    quiet_period_over = True
                
                # Check exact topic match for filtering
                is_ignored = current_topic in IGNORE_PRINTING_TOPICS
                if not is_ignored:
                    should_print = True
            
            # Use TriggerAnalyzer to check for significant changes
            # This works regardless of quiet period to build up state
            trigger_result = trigger_analyzer.analyze(current_topic, current_payload)
            
            instant_trigger_matched_for_line = False
            if trigger_result.should_trigger:
                yellow, bold, reset = "\033[93m", "\033[1m", "\033[0m"
                print(f"{timestamp()} {line}")  # Print the message that caused the trigger
                print(f"{timestamp()} {bold}{yellow}>>> SMART TRIGGER: {trigger_result.reason} <<<{reset}")
                print(f"           {yellow}Field: {trigger_result.field_name}, Change: {trigger_result.old_value} -> {trigger_result.new_value}{reset}")
                trigger_event.set()
                instant_trigger_matched_for_line = True
            
            if should_print and not instant_trigger_matched_for_line and VERBOSE:
                print(f"{timestamp()} {line}")
            
            with shared_state_lock:
                # Include timestamp so AI can calculate delays between events
                timestamped_line = f"{timestamp()} {line}"
                messages_deque.append(timestamped_line)
                new_message_count += 1
        
        stderr_output = process.communicate()[1]
        if process.returncode != 0:
            print(f"Error: mosquitto_sub exited with code {process.returncode}", file=sys.stderr)
            if stderr_output:
                print(f"stderr:\n{stderr_output.decode('utf-8', errors='replace')}", file=sys.stderr)

    except FileNotFoundError:
        print("Critical Error: 'mosquitto_sub' not found. The collector thread cannot start.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred in the collector thread: {e}", file=sys.stderr)

def main():
    """
    Main function to orchestrate the collector thread and the AI check loop.
    """
    global new_message_count
    
    try:
        with open(RULEBOOK_FILE, "r") as f:
            rulebook_content = f.read()
    except FileNotFoundError:
        print(f"Error: '{RULEBOOK_FILE}' not found. Cannot start daemon.", file=sys.stderr)
        sys.exit(1)

    # Load learned rules, pending patterns, and rejected patterns
    learned_rules = load_json_file(LEARNED_RULES_FILE, {"rules": []})
    pending_patterns = load_json_file(PENDING_PATTERNS_FILE, {"patterns": []})
    rejected_patterns = load_json_file(REJECTED_PATTERNS_FILE, {"patterns": []})
    print(f"Loaded {len(learned_rules.get('rules', []))} learned rules, {len(pending_patterns.get('patterns', []))} pending patterns, {len(rejected_patterns.get('patterns', []))} rejected patterns")

    # Initialize the smart trigger analyzer
    trigger_analyzer = TriggerAnalyzer(FILTERED_TRIGGERS_FILE)
    stats = trigger_analyzer.get_stats()
    print(f"Loaded smart trigger configuration:")
    print(f"  - State fields: {stats['config']['state_fields']}")
    print(f"  - Numeric fields: {list(stats['config']['numeric_fields'].keys())}")
    print(f"  - Cooldown: {stats['config']['cooldown_seconds']}s, Baseline window: {stats['config']['baseline_window_seconds']}s")

    ai_check_trigger_event = threading.Event()
    
    collector = threading.Thread(target=mqtt_collector_thread, args=(trigger_analyzer, ai_check_trigger_event))
    collector.daemon = True
    collector.start()

    last_check_time = time.time()
    print(f"Daemon started. AI checks will run every {AI_CHECK_INTERVAL_SECONDS}s, after {AI_CHECK_MESSAGE_THRESHOLD} messages, or on a smart trigger.")

    while True:
        instant_trigger_fired = ai_check_trigger_event.wait(timeout=1.0)
        
        with shared_state_lock:
            should_check_by_count = new_message_count >= AI_CHECK_MESSAGE_THRESHOLD
        
        should_check_by_time = (time.time() - last_check_time) >= AI_CHECK_INTERVAL_SECONDS

        if collector.is_alive() and (instant_trigger_fired or should_check_by_count or should_check_by_time):
            
            # Determine the reason for this AI check
            if instant_trigger_fired:
                trigger_reason = "smart_trigger"
            elif should_check_by_count:
                trigger_reason = f"message_count ({AI_CHECK_MESSAGE_THRESHOLD})"
            else:
                trigger_reason = f"interval ({AI_CHECK_INTERVAL_SECONDS}s)"
            
            with shared_state_lock:
                messages_snapshot = "\n".join(list(messages_deque))
                new_message_count = 0
                last_check_time = time.time()
            
            if instant_trigger_fired:
                ai_check_trigger_event.clear()

            if messages_snapshot:
                # Reload rules files to get latest state (they may have been updated by MCP tools)
                learned_rules = load_json_file(LEARNED_RULES_FILE, {"rules": []})
                pending_patterns = load_json_file(PENDING_PATTERNS_FILE, {"patterns": []})
                rejected_patterns = load_json_file(REJECTED_PATTERNS_FILE, {"patterns": []})
                run_ai_orchestration(messages_snapshot, rulebook_content, learned_rules, pending_patterns, rejected_patterns, trigger_reason)
        
        elif not collector.is_alive():
            print("Collector thread has terminated. Exiting daemon.", file=sys.stderr)
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MQTT AI Daemon - Smart home automation with AI")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print all MQTT messages to console")
    args = parser.parse_args()
    VERBOSE = args.verbose
    main()
