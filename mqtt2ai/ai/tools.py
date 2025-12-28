"""Tool implementations for AI function calling.

This module provides a ToolHandler class that encapsulates tool functions
called by AI providers (OpenAI, Gemini, Claude) through their function
calling / tool use APIs. Dependencies are injected via RuntimeContext.
"""

import json
from datetime import datetime
from typing import TYPE_CHECKING

from mqtt2ai.core.utils import load_json_file, save_json_file

if TYPE_CHECKING:
    from mqtt2ai.core.context import RuntimeContext

# Rules files
LEARNED_RULES_FILE = "configs/learned_rules.json"
PENDING_PATTERNS_FILE = "configs/pending_patterns.json"
REJECTED_PATTERNS_FILE = "configs/rejected_patterns.json"


class ToolHandler:
    """Handles tool execution with injected dependencies.

    This class encapsulates all tool functions and receives dependencies
    through the RuntimeContext, avoiding global state.

    Attributes:
        context: The RuntimeContext containing all dependencies
    """

    def __init__(self, context: 'RuntimeContext'):
        """Initialize the ToolHandler with a RuntimeContext.

        Args:
            context: RuntimeContext containing mqtt_client, telegram_bot, etc.
        """
        self.context = context

    def send_mqtt_message(self, topic: str, payload: str) -> str:
        """Send a message to an MQTT topic.

        Args:
            topic: The MQTT topic to publish to (e.g., 'alert/power')
            payload: The message payload, typically a JSON string

        Returns:
            A confirmation message indicating success or failure
        """
        client = self.context.mqtt_client if self.context else None
        if client and client.publish(topic, payload):
            return f"Successfully sent message to topic '{topic}'"
        return "Error: Failed to send MQTT message. Check connection to broker."

    def send_telegram_message(self, message: str) -> str:
        """Send a message to all authorized Telegram users.

        Use this to notify users about important events, confirmations,
        or status updates. The message will be sent to all configured
        Telegram chat IDs.

        Args:
            message: The message text to send (Markdown supported)

        Returns:
            A confirmation message indicating success or failure
        """
        telegram_bot = self.context.telegram_bot if self.context else None
        if not telegram_bot:
            return "Error: Telegram bot not configured or not running"

        try:
            count = telegram_bot.broadcast_message(message)
            if count > 0:
                return f"Successfully sent Telegram message to {count} user(s)"
            return "Error: No Telegram messages sent (no authorized chats)"
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"Error sending Telegram message: {e}"

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def create_rule(
        self,
        rule_id: str,
        trigger_topic: str,
        trigger_field: str,
        trigger_value: str,
        action_topic: str,
        action_payload: str,
        avg_delay_seconds: float,
        tolerance_seconds: float
    ) -> str:
        """Create or update an automation rule based on learned patterns.

        This tool is used by the AI to formalize patterns it has detected after
        observing the same trigger->action sequence multiple times.

        Args:
            rule_id: Unique identifier for the rule
            trigger_topic: MQTT topic that triggers the rule
            trigger_field: JSON field to monitor (e.g., 'occupancy', 'contact')
            trigger_value: Value that triggers the rule - will be parsed as JSON
            action_topic: MQTT topic to publish to when triggered
            action_payload: Payload to send (e.g., '{"state": "ON"}')
            avg_delay_seconds: Average delay observed before user action
            tolerance_seconds: Tolerance window for timing

        Returns:
            Confirmation message indicating the rule was created or updated
        """
        # Load existing rules
        rules_data = load_json_file(LEARNED_RULES_FILE, {"rules": []})

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

        # Also check if a rule with same trigger+action already exists
        existing_rule_by_logic = None
        for i, rule in enumerate(rules_data["rules"]):
            if (rule["trigger"]["topic"] == trigger_topic and
                    rule["trigger"]["field"] == trigger_field and
                    rule["action"]["topic"] == action_topic):
                existing_rule_by_logic = i
                break

        # Get disable_new_rules from context
        disable_new_rules = (
            self.context.disable_new_rules if self.context else False
        )

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
            "enabled": not disable_new_rules
        }

        # Check if this pattern has been rejected
        if _is_pattern_rejected(trigger_topic, trigger_field, action_topic):
            return (
                f"Pattern '{rule_id}' is in the rejected patterns list and will not "
                "be created. Use remove_rejected_pattern to allow it again."
            )

        # Determine which index to update (prefer ID match, then logic match)
        existing_index = (
            existing_rule_by_id if existing_rule_by_id is not None
            else existing_rule_by_logic
        )

        # If rule already exists with same trigger+action, don't update unnecessarily
        if existing_index is not None:
            existing_rule = rules_data["rules"][existing_index]
            # Check if the rule is essentially the same (same trigger value)
            if existing_rule.get("trigger", {}).get("value") == parsed_trigger_value:
                return (
                    f"Rule '{existing_rule.get('id', rule_id)}' already exists for "
                    f"{trigger_topic}[{trigger_field}={trigger_value}] -> {action_topic}. "
                    "No update needed."
                )

        if existing_index is not None:
            # Update existing rule (increment occurrences if updating)
            # PRESERVE the enabled state from the existing rule
            old_enabled = rules_data["rules"][existing_index].get("enabled", True)
            old_occurrences = (
                rules_data["rules"][existing_index]
                .get("confidence", {})
                .get("occurrences", 0)
            )
            new_rule["confidence"]["occurrences"] = old_occurrences + 1
            new_rule["enabled"] = old_enabled  # Keep the user's choice
            rules_data["rules"][existing_index] = new_rule
            action = "updated"
        else:
            # Add new rule
            rules_data["rules"].append(new_rule)
            action = "created"

        # Save updated rules
        save_json_file(LEARNED_RULES_FILE, rules_data)

        # Clear pending pattern observations for this trigger+action
        _clear_pending_pattern(trigger_topic, trigger_field, action_topic)

        return (
            f"Successfully {action} rule '{rule_id}': "
            f"{trigger_topic}[{trigger_field}={trigger_value}] -> {action_topic}"
        )

    def record_pattern_observation(
        self,
        trigger_topic: str,
        trigger_field: str,
        action_topic: str,
        delay_seconds: float
    ) -> str:
        """Record an observation of a potential trigger->action pattern.

        The AI should call this when it detects a user manually performing an action
        after a trigger event. After 3 observations, the AI should use create_rule
        to formalize the pattern.

        Args:
            trigger_topic: The MQTT topic that triggered
            trigger_field: The field that changed (e.g., 'occupancy')
            action_topic: The action topic the user interacted with
            delay_seconds: Time in seconds between trigger and user action

        Returns:
            Status message indicating observation count
        """
        return record_pattern_observation(
            trigger_topic, trigger_field, action_topic, delay_seconds
        )

    def get_learned_rules(self) -> str:
        """Get all learned automation rules.

        Returns:
            JSON string containing all learned rules
        """
        return get_learned_rules()

    def get_pending_patterns(self) -> str:
        """Get all pending patterns that are being tracked.

        Returns:
            JSON string containing all pending patterns and their observations
        """
        return get_pending_patterns()

    def delete_rule(self, rule_id: str) -> str:
        """Delete a learned automation rule.

        Args:
            rule_id: The ID of the rule to delete

        Returns:
            Confirmation message
        """
        return delete_rule(rule_id)

    def toggle_rule(self, rule_id: str, enabled: bool) -> str:
        """Enable or disable a learned automation rule.

        Args:
            rule_id: The ID of the rule to toggle
            enabled: True to enable, False to disable

        Returns:
            Confirmation message
        """
        return toggle_rule(rule_id, enabled)

    def clear_pending_patterns(self) -> str:
        """Clear all pending pattern observations.

        Returns:
            Confirmation message
        """
        return clear_pending_patterns()

    def reject_pattern(
        self,
        trigger_topic: str,
        trigger_field: str,
        action_topic: str,
        reason: str = ""
    ) -> str:
        """Reject a pattern to prevent it from being learned.

        Args:
            trigger_topic: The MQTT trigger topic
            trigger_field: The field that triggers
            action_topic: The action topic
            reason: Optional reason for rejection

        Returns:
            Confirmation message
        """
        return reject_pattern(trigger_topic, trigger_field, action_topic, reason)

    def get_rejected_patterns(self) -> str:
        """Get all rejected patterns.

        Returns:
            JSON string containing all rejected patterns
        """
        return get_rejected_patterns()

    def remove_rejected_pattern(
        self,
        trigger_topic: str,
        trigger_field: str,
        action_topic: str
    ) -> str:
        """Remove a pattern from the rejected list.

        Args:
            trigger_topic: The MQTT trigger topic
            trigger_field: The field that triggers
            action_topic: The action topic

        Returns:
            Confirmation message
        """
        return remove_rejected_pattern(trigger_topic, trigger_field, action_topic)

    def report_undo(self, rule_id: str) -> str:
        """Report that a user undid an automated action.

        Args:
            rule_id: The ID of the rule that was undone

        Returns:
            Current undo count and whether threshold is reached
        """
        return report_undo(rule_id)


# =============================================================================
# Helper functions (stateless operations on files)
# =============================================================================

def _clear_pending_pattern(
    trigger_topic: str, trigger_field: str, action_topic: str
) -> None:
    """Remove a pending pattern after it becomes a rule."""
    patterns_data = load_json_file(PENDING_PATTERNS_FILE, {"patterns": []})

    patterns_data["patterns"] = [
        p for p in patterns_data["patterns"]
        if not (p["trigger_topic"] == trigger_topic and
                p["trigger_field"] == trigger_field and
                p["action_topic"] == action_topic)
    ]

    save_json_file(PENDING_PATTERNS_FILE, patterns_data)


def _is_pattern_rejected(
    trigger_topic: str, trigger_field: str, action_topic: str
) -> bool:
    """Check if a pattern is in the rejected list."""
    rejected_data = load_json_file(REJECTED_PATTERNS_FILE, {"patterns": []})

    for pattern in rejected_data["patterns"]:
        if (pattern["trigger_topic"] == trigger_topic and
                pattern["trigger_field"] == trigger_field and
                pattern["action_topic"] == action_topic):
            return True
    return False


def _add_rejected_pattern(
    trigger_topic: str, trigger_field: str, action_topic: str, reason: str = ""
) -> None:
    """Add a pattern to the rejected list."""
    rejected_data = load_json_file(REJECTED_PATTERNS_FILE, {"patterns": []})

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

    save_json_file(REJECTED_PATTERNS_FILE, rejected_data)


def get_learned_rules() -> str:
    """Get all learned automation rules.

    Returns:
        JSON string containing all learned rules
    """
    rules_data = load_json_file(LEARNED_RULES_FILE, {"rules": []})
    return json.dumps(rules_data, indent=2)


def _rule_exists_for_pattern(
    trigger_topic: str, trigger_field: str, action_topic: str
) -> bool:
    """Check if a rule already exists for this trigger->action pattern."""
    rules_data = load_json_file(LEARNED_RULES_FILE, {"rules": []})

    for rule in rules_data["rules"]:
        if (rule.get("trigger", {}).get("topic") == trigger_topic and
                rule.get("trigger", {}).get("field") == trigger_field and
                rule.get("action", {}).get("topic") == action_topic):
            return True
    return False


def record_pattern_observation(
    trigger_topic: str,
    trigger_field: str,
    action_topic: str,
    delay_seconds: float
) -> str:
    """Record an observation of a potential trigger->action pattern.

    The AI should call this when it detects a user manually performing an action
    after a trigger event. After 3 observations, the AI should use create_rule
    to formalize the pattern.

    Args:
        trigger_topic: The MQTT topic that triggered
        trigger_field: The field that changed (e.g., 'occupancy')
        action_topic: The action topic the user interacted with
        delay_seconds: Time in seconds between trigger and user action

    Returns:
        Status message indicating observation count
    """
    # Check if a rule already exists for this pattern - don't re-learn
    if _rule_exists_for_pattern(trigger_topic, trigger_field, action_topic):
        return (
            f"Rule already exists for {trigger_topic}[{trigger_field}] -> "
            f"{action_topic}. No need to re-learn this pattern."
        )

    # Check if this pattern has been rejected - don't track it
    if _is_pattern_rejected(trigger_topic, trigger_field, action_topic):
        return (
            f"Pattern {trigger_topic}[{trigger_field}] -> {action_topic} "
            "is rejected and will not be tracked."
        )

    patterns_data = load_json_file(PENDING_PATTERNS_FILE, {"patterns": []})

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
    save_json_file(PENDING_PATTERNS_FILE, patterns_data)

    if count >= 3:
        return (
            f"Pattern has {count} observations - ready to create rule! "
            "Call create_rule to formalize this automation."
        )
    return f"Pattern observation recorded ({count}/3 needed to create rule)"


def get_pending_patterns() -> str:
    """Get all pending patterns that are being tracked but haven't become rules yet.

    Returns:
        JSON string containing all pending patterns and their observations
    """
    patterns_data = load_json_file(PENDING_PATTERNS_FILE, {"patterns": []})
    return json.dumps(patterns_data, indent=2)


def delete_rule(rule_id: str) -> str:
    """Delete a learned automation rule.

    Args:
        rule_id: The ID of the rule to delete

    Returns:
        Confirmation message
    """
    rules_data = load_json_file(LEARNED_RULES_FILE, {"rules": []})

    original_count = len(rules_data["rules"])
    rules_data["rules"] = [r for r in rules_data["rules"] if r["id"] != rule_id]

    if len(rules_data["rules"]) < original_count:
        save_json_file(LEARNED_RULES_FILE, rules_data)
        return f"Successfully deleted rule '{rule_id}'"
    return f"Rule '{rule_id}' not found"


def toggle_rule(rule_id: str, enabled: bool) -> str:
    """Enable or disable a learned automation rule.

    Args:
        rule_id: The ID of the rule to toggle
        enabled: True to enable, False to disable

    Returns:
        Confirmation message
    """
    rules_data = load_json_file(LEARNED_RULES_FILE, {"rules": []})

    for rule in rules_data["rules"]:
        if rule["id"] == rule_id:
            rule["enabled"] = enabled
            save_json_file(LEARNED_RULES_FILE, rules_data)
            state = "enabled" if enabled else "disabled"
            return f"Rule '{rule_id}' is now {state}"

    return f"Rule '{rule_id}' not found"


def clear_pending_patterns() -> str:
    """Clear all pending pattern observations.

    Use this to reset pattern learning when observations are stale or incorrect.

    Returns:
        Confirmation message
    """
    save_json_file(PENDING_PATTERNS_FILE, {"patterns": []})
    return "All pending patterns cleared"


def reject_pattern(
    trigger_topic: str,
    trigger_field: str,
    action_topic: str,
    reason: str = ""
) -> str:
    """Reject a pattern to prevent it from being learned or re-enabled.

    Use this when a pattern is coincidental (e.g., walking past a PIR downstairs
    before going upstairs and turning on a light there - not a real automation).

    Rejected patterns will:
    - Not be tracked in pending patterns
    - Not be created as new rules
    - Have their pending observations cleared
    - Have any existing rule deleted

    Args:
        trigger_topic: The MQTT trigger topic
        trigger_field: The field that triggers (e.g., 'occupancy')
        action_topic: The action topic
        reason: Optional reason for rejection

    Returns:
        Confirmation message
    """
    # Add to rejected patterns list
    _add_rejected_pattern(trigger_topic, trigger_field, action_topic, reason)

    # Clear any pending observations for this pattern
    _clear_pending_pattern(trigger_topic, trigger_field, action_topic)

    # Delete any existing rule that matches
    rules_data = load_json_file(LEARNED_RULES_FILE, {"rules": []})
    original_count = len(rules_data["rules"])
    rules_data["rules"] = [
        r for r in rules_data["rules"]
        if not (r["trigger"]["topic"] == trigger_topic and
                r["trigger"]["field"] == trigger_field and
                r["action"]["topic"] == action_topic)
    ]
    rules_deleted = original_count - len(rules_data["rules"])
    if rules_deleted > 0:
        save_json_file(LEARNED_RULES_FILE, rules_data)

    result = (
        f"Pattern rejected: {trigger_topic}[{trigger_field}] -> {action_topic}"
    )
    if reason:
        result += f" (reason: {reason})"
    if rules_deleted > 0:
        result += f". Deleted {rules_deleted} existing rule(s)."
    return result


def get_rejected_patterns() -> str:
    """Get all rejected patterns that will never be learned.

    Returns:
        JSON string containing all rejected patterns
    """
    rejected_data = load_json_file(REJECTED_PATTERNS_FILE, {"patterns": []})
    return json.dumps(rejected_data, indent=2)


def remove_rejected_pattern(
    trigger_topic: str,
    trigger_field: str,
    action_topic: str
) -> str:
    """Remove a pattern from the rejected list, allowing it to be learned again.

    Args:
        trigger_topic: The MQTT trigger topic
        trigger_field: The field that triggers
        action_topic: The action topic

    Returns:
        Confirmation message
    """
    rejected_data = load_json_file(REJECTED_PATTERNS_FILE, {"patterns": []})

    original_count = len(rejected_data["patterns"])
    rejected_data["patterns"] = [
        p for p in rejected_data["patterns"]
        if not (p["trigger_topic"] == trigger_topic and
                p["trigger_field"] == trigger_field and
                p["action_topic"] == action_topic)
    ]

    if len(rejected_data["patterns"]) < original_count:
        save_json_file(REJECTED_PATTERNS_FILE, rejected_data)
        return (
            f"Removed rejection for: {trigger_topic}[{trigger_field}] -> "
            f"{action_topic}. Pattern can now be learned."
        )
    return "Pattern not found in rejected list."


def report_undo(rule_id: str) -> str:
    """Report that a user undid an automated action.

    Call this when you detect that a user reversed your automated action
    (e.g., you turned a light ON, user turned it OFF within 30 seconds).

    After 3 undos, the threshold is reached and you should call reject_pattern()
    to permanently disable this unwanted automation.

    Args:
        rule_id: The ID of the rule that was undone

    Returns:
        Current undo count and whether threshold is reached
    """
    rules_data = load_json_file(LEARNED_RULES_FILE, {"rules": []})

    for rule in rules_data["rules"]:
        if rule["id"] == rule_id:
            rule["undo_count"] = rule.get("undo_count", 0) + 1
            save_json_file(LEARNED_RULES_FILE, rules_data)
            count = rule["undo_count"]
            if count >= 3:
                return (
                    f"Undo count for '{rule_id}' is now {count}. "
                    "THRESHOLD REACHED - call reject_pattern() to disable "
                    "this unwanted rule."
                )
            return f"Undo count for '{rule_id}' is now {count}/3"

    return f"Rule '{rule_id}' not found"
