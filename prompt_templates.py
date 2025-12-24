# Compact rulebook - core rules only, no verbose documentation
COMPACT_RULEBOOK = """## Core Rules

### CHECK SKIP PATTERNS FIRST
If a trigger->action pair is in SKIP PATTERNS, do NOT call record_pattern_observation or create_rule.
This saves tokens and prevents redundant operations.

### Safety (CRITICAL)
- smoke:true → Activate siren, notify, turn on ALL known lights
- water_leak:true → Activate siren, notify
- temperature > 50°C → Activate siren, notify

### Security (when armed)
- Door/window opened (contact:false) or Motion (occupancy:true) → Activate siren + notify

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
