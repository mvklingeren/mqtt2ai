"""Templates for system prompts and rulebooks.

This module contains the constant strings used for system prompts, rulebooks,
and other static text content used by the AI agent.
"""

# Compact rulebook - core rules only, no verbose documentation
COMPACT_RULEBOOK = """## Core Rules

### Response Format
- If NO action is needed: respond with only "No action required" or similar brief statement
- Do NOT explain why each function wasn't called
- Keep non-action responses under 20 words

### CHECK SKIP PATTERNS FIRST
If a trigger->action pair is in SKIP PATTERNS, do NOT call record_pattern_observation or create_rule.
This saves tokens and prevents redundant operations.

### Safety (CRITICAL) - Use raise_alert()
- smoke:true → raise_alert(1.0, "Smoke detected", {topic, field})
- water_leak:true → raise_alert(1.0, "Water leak detected", {topic})
- temperature > 50°C → raise_alert(0.9, "High temperature", {topic, value})

### Security Awareness - Use raise_alert()
Detect suspicious patterns and raise alerts instead of direct device control.
The alert system has full device context and will decide appropriate actions.

SUSPICIOUS PATTERNS (use raise_alert with high severity 0.7-1.0):
- Motion sensor + window/door open within 5 min, especially at night (00:00-06:00)
  → raise_alert(0.9, "Motion followed by entry point breach at {time}")
- Multiple motion sensors in sequence (perimeter → interior) within minutes
  → raise_alert(0.85, "Sequential motion detected: perimeter to interior")
- Door/window opened while system armed
  → raise_alert(0.9, "Entry point breached while armed")

INFORMATIONAL (use raise_alert with low severity 0.1-0.3):
- Unusual device activity times
  → raise_alert(0.2, "Device activity at unusual hour")

DO NOT directly send_mqtt_message to sirens/alarms for security events.
Use raise_alert() and let the alert AI decide based on full device context.

### Pattern Learning
1. [SKIP-LEARNED] messages have rules - DO NOT call record_pattern_observation
2. [STATUS] messages are device feedback - NEVER use as triggers
3. [AUTO] messages are automated actions - NEVER use for learning
4. Only SENSORS (PIR, contact, button) are triggers
5. Valid: sensor event → /set command (0.5-30s delay)
6. INVALID patterns (IGNORE):
   - [STATUS] → anything
   - [AUTO] → anything
   - topic/set → same topic (circular)
   - topic → topic/set (same device loop)
7. Record valid sensor→action patterns ONLY (no [AUTO]/[STATUS])
8. Create rule after 3+ observations
9. NO MQTT messages while learning

### Rule Execution
- Fixed rules execute DIRECTLY (no AI)
- AI only for: anomalies, learning, safety
- [AUTO] means action already taken - do not duplicate

### Undo
- User reverses action <30s → report_undo
- 3 undos → reject_pattern
"""

