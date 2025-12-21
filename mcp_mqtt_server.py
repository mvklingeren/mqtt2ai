#!/usr/bin/env python3
"""
MCP Server exposing MQTT tools for AI CLI agents (Gemini, Claude, Codex).
This server provides the send_mqtt_message tool that allows the AI
to publish messages directly to MQTT topics, as well as tools for
managing learned automation rules.
"""

import json
import os
import subprocess
from datetime import datetime
from mcp.server.fastmcp import FastMCP

# MQTT Configuration
MQTT_HOST = "192.168.1.245"
MQTT_PORT = "1883"

# Rules files
LEARNED_RULES_FILE = "learned_rules.json"
PENDING_PATTERNS_FILE = "pending_patterns.json"
REJECTED_PATTERNS_FILE = "rejected_patterns.json"

# Initialize MCP server
mcp = FastMCP("mqtt-tools")


def _load_json_file(filepath: str, default: dict) -> dict:
    """Load a JSON file, returning default if not found or invalid."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json_file(filepath: str, data: dict) -> None:
    """Save data to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


@mcp.tool()
def send_mqtt_message(topic: str, payload: str) -> str:
    """
    Send a message to an MQTT topic.
    
    Args:
        topic: The MQTT topic to publish to (e.g., 'alert/power', 'zigbee2mqtt/device/set')
        payload: The message payload, typically a JSON string
    
    Returns:
        A confirmation message indicating success or failure
    """
    try:
        subprocess.run(
            ["mosquitto_pub", "-h", MQTT_HOST, "-p", MQTT_PORT, "-t", topic, "-m", payload],
            check=True,
            capture_output=True,
            text=True,
        )
        return f"Successfully sent message to topic '{topic}'"
    except FileNotFoundError:
        return "Error: 'mosquitto_pub' command not found. Please install mosquitto-clients."
    except subprocess.CalledProcessError as e:
        return f"Error sending MQTT message: {e.stderr}"


@mcp.tool()
def create_rule(
    rule_id: str,
    trigger_topic: str,
    trigger_field: str,
    trigger_value: str,
    action_topic: str,
    action_payload: str,
    avg_delay_seconds: float,
    tolerance_seconds: float
) -> str:
    """
    Create or update an automation rule based on learned patterns.
    
    This tool is used by the AI to formalize patterns it has detected after
    observing the same trigger->action sequence multiple times.
    
    Args:
        rule_id: Unique identifier for the rule (e.g., 'pir_hallway_to_light_kitchen')
        trigger_topic: MQTT topic that triggers the rule (e.g., 'zigbee2mqtt/pir_hallway')
        trigger_field: JSON field to monitor (e.g., 'occupancy', 'contact')
        trigger_value: Value that triggers the rule (e.g., 'true', 'false') - will be parsed as JSON
        action_topic: MQTT topic to publish to when triggered (e.g., 'zigbee2mqtt/light_kitchen/set')
        action_payload: Payload to send (e.g., '{"state": "ON"}')
        avg_delay_seconds: Average delay observed before user action
        tolerance_seconds: Tolerance window for timing (AI determines based on variance)
    
    Returns:
        Confirmation message indicating the rule was created or updated
    """
    # Load existing rules
    rules_data = _load_json_file(LEARNED_RULES_FILE, {"rules": []})
    
    # Parse the trigger value as JSON to handle booleans, numbers, etc.
    try:
        parsed_trigger_value = json.loads(trigger_value)
    except json.JSONDecodeError:
        parsed_trigger_value = trigger_value
    
    # Check if rule already exists by ID
    existing_rule_by_id = None
    for i, rule in enumerate(rules_data["rules"]):
        if rule["id"] == rule_id:
            existing_rule_by_id = i
            break
    
    # Also check if a rule with same trigger+action already exists (prevent duplicates)
    existing_rule_by_logic = None
    for i, rule in enumerate(rules_data["rules"]):
        if (rule["trigger"]["topic"] == trigger_topic and
            rule["trigger"]["field"] == trigger_field and
            rule["action"]["topic"] == action_topic):
            existing_rule_by_logic = i
            break
    
    new_rule = {
        "id": rule_id,
        "trigger": {
            "topic": trigger_topic,
            "field": trigger_field,
            "value": parsed_trigger_value
        },
        "action": {
            "topic": action_topic,
            "payload": action_payload
        },
        "timing": {
            "avg_delay_seconds": avg_delay_seconds,
            "tolerance_seconds": tolerance_seconds
        },
        "confidence": {
            "occurrences": 3,
            "last_triggered": datetime.now().isoformat()
        },
        "enabled": True
    }
    
    # Check if this pattern has been rejected
    if _is_pattern_rejected(trigger_topic, trigger_field, action_topic):
        return f"Pattern '{rule_id}' is in the rejected patterns list and will not be created. Use remove_rejected_pattern to allow it again."
    
    # Determine which index to update (prefer ID match, then logic match)
    existing_index = existing_rule_by_id if existing_rule_by_id is not None else existing_rule_by_logic
    
    if existing_index is not None:
        # Update existing rule (increment occurrences if updating)
        # PRESERVE the enabled state from the existing rule
        old_enabled = rules_data["rules"][existing_index].get("enabled", True)
        old_occurrences = rules_data["rules"][existing_index].get("confidence", {}).get("occurrences", 0)
        new_rule["confidence"]["occurrences"] = old_occurrences + 1
        new_rule["enabled"] = old_enabled  # Keep the user's choice
        rules_data["rules"][existing_index] = new_rule
        action = "updated"
    else:
        # Add new rule
        rules_data["rules"].append(new_rule)
        action = "created"
    
    # Save updated rules
    _save_json_file(LEARNED_RULES_FILE, rules_data)
    
    # Clear pending pattern observations for this trigger+action (they're now a rule)
    _clear_pending_pattern(trigger_topic, trigger_field, action_topic)
    
    return f"Successfully {action} rule '{rule_id}': {trigger_topic}[{trigger_field}={trigger_value}] -> {action_topic}"


def _clear_pending_pattern(trigger_topic: str, trigger_field: str, action_topic: str) -> None:
    """Remove a pending pattern after it becomes a rule."""
    patterns_data = _load_json_file(PENDING_PATTERNS_FILE, {"patterns": []})
    
    patterns_data["patterns"] = [
        p for p in patterns_data["patterns"]
        if not (p["trigger_topic"] == trigger_topic and
                p["trigger_field"] == trigger_field and
                p["action_topic"] == action_topic)
    ]
    
    _save_json_file(PENDING_PATTERNS_FILE, patterns_data)


def _is_pattern_rejected(trigger_topic: str, trigger_field: str, action_topic: str) -> bool:
    """Check if a pattern is in the rejected list."""
    rejected_data = _load_json_file(REJECTED_PATTERNS_FILE, {"patterns": []})
    
    for pattern in rejected_data["patterns"]:
        if (pattern["trigger_topic"] == trigger_topic and
            pattern["trigger_field"] == trigger_field and
            pattern["action_topic"] == action_topic):
            return True
    return False


def _add_rejected_pattern(trigger_topic: str, trigger_field: str, action_topic: str, reason: str = "") -> None:
    """Add a pattern to the rejected list."""
    rejected_data = _load_json_file(REJECTED_PATTERNS_FILE, {"patterns": []})
    
    # Check if already rejected
    if _is_pattern_rejected(trigger_topic, trigger_field, action_topic):
        return
    
    rejected_data["patterns"].append({
        "trigger_topic": trigger_topic,
        "trigger_field": trigger_field,
        "action_topic": action_topic,
        "reason": reason,
        "rejected_at": datetime.now().isoformat()
    })
    
    _save_json_file(REJECTED_PATTERNS_FILE, rejected_data)


@mcp.tool()
def get_learned_rules() -> str:
    """
    Get all learned automation rules.
    
    Returns:
        JSON string containing all learned rules
    """
    rules_data = _load_json_file(LEARNED_RULES_FILE, {"rules": []})
    return json.dumps(rules_data, indent=2)


@mcp.tool()
def record_pattern_observation(
    trigger_topic: str,
    trigger_field: str,
    action_topic: str,
    delay_seconds: float
) -> str:
    """
    Record an observation of a potential trigger->action pattern.
    
    The AI should call this when it detects a user manually performing an action
    after a trigger event. After 3 observations, the AI should use create_rule
    to formalize the pattern.
    
    Args:
        trigger_topic: The MQTT topic that triggered (e.g., 'zigbee2mqtt/pir_hallway')
        trigger_field: The field that changed (e.g., 'occupancy')
        action_topic: The action topic the user interacted with (e.g., 'zigbee2mqtt/light_kitchen/set')
        delay_seconds: Time in seconds between trigger and user action
    
    Returns:
        Status message indicating observation count
    """
    # Check if this pattern has been rejected - don't track it
    if _is_pattern_rejected(trigger_topic, trigger_field, action_topic):
        return f"Pattern {trigger_topic}[{trigger_field}] -> {action_topic} is rejected and will not be tracked."
    
    patterns_data = _load_json_file(PENDING_PATTERNS_FILE, {"patterns": []})
    
    # Find existing pattern or create new one
    existing_pattern = None
    for pattern in patterns_data["patterns"]:
        if (pattern["trigger_topic"] == trigger_topic and 
            pattern["trigger_field"] == trigger_field and
            pattern["action_topic"] == action_topic):
            existing_pattern = pattern
            break
    
    observation = {
        "delay_seconds": delay_seconds,
        "timestamp": datetime.now().isoformat()
    }
    
    if existing_pattern:
        existing_pattern["observations"].append(observation)
        count = len(existing_pattern["observations"])
    else:
        new_pattern = {
            "trigger_topic": trigger_topic,
            "trigger_field": trigger_field,
            "action_topic": action_topic,
            "observations": [observation]
        }
        patterns_data["patterns"].append(new_pattern)
        count = 1
    
    # Save updated patterns
    _save_json_file(PENDING_PATTERNS_FILE, patterns_data)
    
    if count >= 3:
        return f"Pattern has {count} observations - ready to create rule! Call create_rule to formalize this automation."
    else:
        return f"Pattern observation recorded ({count}/3 needed to create rule)"


@mcp.tool()
def get_pending_patterns() -> str:
    """
    Get all pending patterns that are being tracked but haven't become rules yet.
    
    Returns:
        JSON string containing all pending patterns and their observations
    """
    patterns_data = _load_json_file(PENDING_PATTERNS_FILE, {"patterns": []})
    return json.dumps(patterns_data, indent=2)


@mcp.tool()
def delete_rule(rule_id: str) -> str:
    """
    Delete a learned automation rule.
    
    Args:
        rule_id: The ID of the rule to delete
    
    Returns:
        Confirmation message
    """
    rules_data = _load_json_file(LEARNED_RULES_FILE, {"rules": []})
    
    original_count = len(rules_data["rules"])
    rules_data["rules"] = [r for r in rules_data["rules"] if r["id"] != rule_id]
    
    if len(rules_data["rules"]) < original_count:
        _save_json_file(LEARNED_RULES_FILE, rules_data)
        return f"Successfully deleted rule '{rule_id}'"
    else:
        return f"Rule '{rule_id}' not found"


@mcp.tool()
def toggle_rule(rule_id: str, enabled: bool) -> str:
    """
    Enable or disable a learned automation rule.
    
    Args:
        rule_id: The ID of the rule to toggle
        enabled: True to enable, False to disable
    
    Returns:
        Confirmation message
    """
    rules_data = _load_json_file(LEARNED_RULES_FILE, {"rules": []})
    
    for rule in rules_data["rules"]:
        if rule["id"] == rule_id:
            rule["enabled"] = enabled
            _save_json_file(LEARNED_RULES_FILE, rules_data)
            state = "enabled" if enabled else "disabled"
            return f"Rule '{rule_id}' is now {state}"
    
    return f"Rule '{rule_id}' not found"


@mcp.tool()
def clear_pending_patterns() -> str:
    """
    Clear all pending pattern observations.
    
    Use this to reset pattern learning when observations are stale or incorrect.
    
    Returns:
        Confirmation message
    """
    _save_json_file(PENDING_PATTERNS_FILE, {"patterns": []})
    return "All pending patterns cleared"


@mcp.tool()
def reject_pattern(
    trigger_topic: str,
    trigger_field: str,
    action_topic: str,
    reason: str = ""
) -> str:
    """
    Reject a pattern to prevent it from being learned or re-enabled.
    
    Use this when a pattern is coincidental (e.g., walking past a PIR downstairs 
    before going upstairs and turning on a light there - not a real automation).
    
    Rejected patterns will:
    - Not be tracked in pending patterns
    - Not be created as new rules
    - Have their pending observations cleared
    - Have any existing rule deleted
    
    Args:
        trigger_topic: The MQTT trigger topic (e.g., 'zigbee2mqtt/pir_hallway')
        trigger_field: The field that triggers (e.g., 'occupancy')
        action_topic: The action topic (e.g., 'zigbee2mqtt/light_kitchen/set')
        reason: Optional reason for rejection (e.g., 'coincidental - different floors')
    
    Returns:
        Confirmation message
    """
    # Add to rejected patterns list
    _add_rejected_pattern(trigger_topic, trigger_field, action_topic, reason)
    
    # Clear any pending observations for this pattern
    _clear_pending_pattern(trigger_topic, trigger_field, action_topic)
    
    # Delete any existing rule that matches
    rules_data = _load_json_file(LEARNED_RULES_FILE, {"rules": []})
    original_count = len(rules_data["rules"])
    rules_data["rules"] = [
        r for r in rules_data["rules"]
        if not (r["trigger"]["topic"] == trigger_topic and
                r["trigger"]["field"] == trigger_field and
                r["action"]["topic"] == action_topic)
    ]
    rules_deleted = original_count - len(rules_data["rules"])
    if rules_deleted > 0:
        _save_json_file(LEARNED_RULES_FILE, rules_data)
    
    result = f"Pattern rejected: {trigger_topic}[{trigger_field}] -> {action_topic}"
    if reason:
        result += f" (reason: {reason})"
    if rules_deleted > 0:
        result += f". Deleted {rules_deleted} existing rule(s)."
    return result


@mcp.tool()
def get_rejected_patterns() -> str:
    """
    Get all rejected patterns that will never be learned.
    
    Returns:
        JSON string containing all rejected patterns
    """
    rejected_data = _load_json_file(REJECTED_PATTERNS_FILE, {"patterns": []})
    return json.dumps(rejected_data, indent=2)


@mcp.tool()
def remove_rejected_pattern(
    trigger_topic: str,
    trigger_field: str,
    action_topic: str
) -> str:
    """
    Remove a pattern from the rejected list, allowing it to be learned again.
    
    Args:
        trigger_topic: The MQTT trigger topic
        trigger_field: The field that triggers
        action_topic: The action topic
    
    Returns:
        Confirmation message
    """
    rejected_data = _load_json_file(REJECTED_PATTERNS_FILE, {"patterns": []})
    
    original_count = len(rejected_data["patterns"])
    rejected_data["patterns"] = [
        p for p in rejected_data["patterns"]
        if not (p["trigger_topic"] == trigger_topic and
                p["trigger_field"] == trigger_field and
                p["action_topic"] == action_topic)
    ]
    
    if len(rejected_data["patterns"]) < original_count:
        _save_json_file(REJECTED_PATTERNS_FILE, rejected_data)
        return f"Removed rejection for: {trigger_topic}[{trigger_field}] -> {action_topic}. Pattern can now be learned."
    else:
        return f"Pattern not found in rejected list."


@mcp.tool()
def report_undo(rule_id: str) -> str:
    """
    Report that a user undid an automated action.
    
    Call this when you detect that a user reversed your automated action
    (e.g., you turned a light ON, user turned it OFF within 30 seconds).
    
    After 3 undos, the threshold is reached and you should call reject_pattern()
    to permanently disable this unwanted automation.
    
    Args:
        rule_id: The ID of the rule that was undone
    
    Returns:
        Current undo count and whether threshold is reached
    """
    rules_data = _load_json_file(LEARNED_RULES_FILE, {"rules": []})
    
    for rule in rules_data["rules"]:
        if rule["id"] == rule_id:
            rule["undo_count"] = rule.get("undo_count", 0) + 1
            _save_json_file(LEARNED_RULES_FILE, rules_data)
            count = rule["undo_count"]
            if count >= 3:
                return f"Undo count for '{rule_id}' is now {count}. THRESHOLD REACHED - call reject_pattern() to disable this unwanted rule."
            return f"Undo count for '{rule_id}' is now {count}/3"
    
    return f"Rule '{rule_id}' not found"


if __name__ == "__main__":
    mcp.run()
