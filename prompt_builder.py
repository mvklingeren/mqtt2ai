"""Prompt Builder module for the MQTT AI Daemon.

This module handles intelligent prompt construction with:
- Smart MQTT message compression (deduplication with counts, aggregation)
- Context-aware rule/pattern filtering based on trigger
- Compact rulebook formatting
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from config import Config
from knowledge_base import KnowledgeBase
from trigger_analyzer import TriggerResult
from prompt_templates import COMPACT_RULEBOOK

@dataclass
class MessageStats:
    """Statistics for a deduplicated message."""
    topic: str
    payload: Dict[str, Any]
    timestamp: str
    count: int = 1
    first_seen: Optional[str] = None
    # For numeric fields, track min/max
    numeric_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)


class PromptBuilder:
    """Builds optimized prompts for AI analysis with intelligent compression."""

    # Fields to remove from payloads (noise)
    REMOVE_FIELDS = {
        "linkquality", "voltage", "energy", "update", "update_available",
        "child_lock", "countdown", "indicator_mode", "power_outage_memory",
        "timestamp", "type", "wirelessNetwork", "wirelessSignal",
        "firmwareStatus", "lastUpdate", "stream_Source", "still_Image_URL",
        "installed_version", "latest_version",
        "Time", "Uptime", "UptimeSec", "Vcc", "Heap",
        "SleepMode", "Sleep", "LoadAvg", "MqttCount", "Hostname", "IPAddress",
    }

    # Numeric fields that can be aggregated with ranges
    NUMERIC_FIELDS = {"power", "temperature", "humidity", "current", "illuminance"}

    # Topic prefix replacements to shorten common prefixes
    TOPIC_REPLACEMENTS = {
        "zigbee2mqtt/": "z2m/",
    }

    def __init__(self, config: Config):
        self.config = config

    def build(
        self,
        messages_snapshot: str,
        kb: KnowledgeBase,
        trigger_results: Optional[List[TriggerResult]] = None,
        trigger_reason: str = ""
    ) -> str:
        """Build an optimized prompt for AI analysis.
        
        Args:
            messages_snapshot: Raw MQTT messages as newline-separated string
            kb: KnowledgeBase with rules, patterns, and rulebook
            trigger_results: Optional list of TriggerResult objects with trigger context.
                            Multiple triggers may fire between AI checks.
            trigger_reason: Human-readable trigger reason string
            
        Returns:
            Optimized prompt string
        """
        # Normalize to list
        if trigger_results is None:
            trigger_results = []
        elif isinstance(trigger_results, TriggerResult):
            trigger_results = [trigger_results]
        
        # Extract trigger topics from all trigger_results
        trigger_topics = self._extract_trigger_topics(trigger_results, trigger_reason)
        # Use first topic for backwards compatibility where single topic is expected
        trigger_topic = trigger_topics[0] if trigger_topics else None

        # Filter rules and patterns by relevance (using all trigger topics)
        relevant_rules = self._filter_relevant_rules(kb.learned_rules, trigger_topic, trigger_topics=trigger_topics)

        # Build set of existing rule patterns for message annotation
        existing_patterns = self._build_existing_patterns_set(kb.learned_rules)

        # Compress messages with annotations for existing rules
        compressed_messages = self._compress_messages(
            messages_snapshot, trigger_topic, existing_patterns=existing_patterns,
            trigger_topics=trigger_topics
        )
        relevant_patterns = self._filter_relevant_patterns(
            kb.pending_patterns, trigger_topic, trigger_topics=trigger_topics
        )
        rejected_patterns = kb.rejected_patterns

        # Build prompt sections
        safety_alert = self._build_safety_alert(trigger_reason)
        demo_instruction = self._build_demo_instruction()
        # Pass both enabled rules (for display) and ALL rules (for skip patterns)
        all_rules = kb.learned_rules.get("rules", [])
        rules_section = self._format_rules(relevant_rules, trigger_topic, all_rules, trigger_topics=trigger_topics)
        patterns_section = self._format_patterns(relevant_patterns)
        rejected_section = self._format_rejected(rejected_patterns)
        
        # Add trigger context section if multiple triggers
        trigger_context = self._build_trigger_context(trigger_results)

        prompt = (
            f"{safety_alert}"
            f"You are a home automation AI with pattern learning. {demo_instruction}"
            "Use send_mqtt_message tool for MQTT actions - no shell commands.\n\n"
            f"{COMPACT_RULEBOOK}\n"
            f"{trigger_context}"
            f"{rules_section}"
            f"{patterns_section}"
            f"{rejected_section}\n"
            "## MQTT Messages (analyze for patterns):\n"
            f"{compressed_messages}\n"
        )

        return prompt

    def build_compact(
        self,
        messages_snapshot: str,
        kb: KnowledgeBase,
        trigger_results: Optional[List[TriggerResult]] = None,
        trigger_reason: str = ""
    ) -> str:
        """Build an extra-compact prompt for small context models.
        
        Targets ~2000 tokens for Ollama/Groq.
        """
        # Normalize to list
        if trigger_results is None:
            trigger_results = []
        elif isinstance(trigger_results, TriggerResult):
            trigger_results = [trigger_results]
        
        trigger_topics = self._extract_trigger_topics(trigger_results, trigger_reason)
        trigger_topic = trigger_topics[0] if trigger_topics else None

        # Only include directly matching rules
        relevant_rules = self._filter_relevant_rules(
            kb.learned_rules, trigger_topic, strict=True, trigger_topics=trigger_topics
        )

        # Build set of existing rule patterns for message annotation
        existing_patterns = self._build_existing_patterns_set(kb.learned_rules)
        
        # Compress messages with annotations
        compressed_messages = self._compress_messages(
            messages_snapshot, trigger_topic, existing_patterns=existing_patterns,
            trigger_topics=trigger_topics
        )

        # Build minimal prompt
        safety = ""
        if any(x in trigger_reason.lower() for x in ["temperature", "smoke", "water", "leak"]):
            safety = "SAFETY ALERT: Check for dangerous conditions!\n\n"

        demo = self._build_demo_instruction()

        rules_summary = ""
        if relevant_rules:
            rules_summary = "EXISTING RULES (DO NOT re-learn these):\n"
            for rule in relevant_rules[:5]:
                trigger = rule.get("trigger", {})
                action = rule.get("action", {})
                enabled = "ON" if rule.get("enabled", True) else "OFF"
                rules_summary += (
                    f"- [{enabled}] {rule.get('id', '?')}: "
                    f"{trigger.get('topic', '?')}[{trigger.get('field', '?')}="
                    f"{trigger.get('value', '?')}] → {action.get('topic', '?')}\n"
                )

        # Check for security-relevant patterns in messages
        security_alert = self._check_security_patterns(compressed_messages)

        prompt = f"""{safety}{demo}Analyze MQTT for automation patterns.

{rules_summary}
Messages:
{compressed_messages}

Tasks (in order):
1. SECURITY: Check for suspicious patterns (motion sensor + window/door opens within minutes)
   - If found: call raise_alert(severity, reason) where severity 0.7-1.0 for break-in risk
   - Perimeter motion followed by entry point breach = raise_alert(0.9, "Motion + entry breach")
2. LEARN: If trigger event followed by /set action within 30s → record_pattern_observation
   - Use the trigger VALUE that CAUSED the action (e.g., occupancy:true, NOT false)
   - Skip if rule already exists for this trigger→action
   - DO NOT send any MQTT messages during learning
3. EXECUTE: ONLY if an ENABLED rule matches the CURRENT trigger → send_mqtt_message
4. CREATE: After 3+ observations of same pattern → create_rule
{security_alert}"""
        return prompt

    def _check_security_patterns(self, messages: str) -> str:
        """Check messages for security-relevant patterns and return alert hint."""
        # Look for patterns like parking/outdoor PIR + window/door contact
        has_perimeter_motion = any(x in messages.lower() for x in [
            "parking", "outdoor", "driveway", "garden", "perimeter"
        ]) and "occupancy" in messages.lower()
        
        has_entry_breach = "contact" in messages.lower() and "false" in messages.lower()
        
        if has_perimeter_motion and has_entry_breach:
            return "\n⚠️ SECURITY: Detected perimeter motion + entry breach pattern. Consider raise_alert()."
        
        return ""

    def _extract_trigger_topics(
        self,
        trigger_results: List[TriggerResult],
        trigger_reason: str
    ) -> List[str]:
        """Extract all triggering topics from trigger results and reason string.
        
        Args:
            trigger_results: List of TriggerResult objects (may have .topic attribute)
            trigger_reason: Human-readable trigger reason string
            
        Returns:
            List of unique trigger topics (preserves order)
        """
        topics = []
        seen = set()
        
        # First, extract from TriggerResult objects (most reliable)
        for tr in trigger_results:
            if tr.topic and tr.topic not in seen:
                topics.append(tr.topic)
                seen.add(tr.topic)
        
        # Also try to extract from trigger_reason string as fallback
        if "topic:" in trigger_reason.lower():
            match = re.search(r'topic:\s*(\S+)', trigger_reason, re.IGNORECASE)
            if match and match.group(1) not in seen:
                topics.append(match.group(1))
                seen.add(match.group(1))
        
        # Look for zigbee2mqtt or similar patterns in reason
        for match in re.finditer(r'(zigbee2mqtt/\S+|homie/\S+|ring/\S+)', trigger_reason):
            topic = match.group(1)
            if topic not in seen:
                topics.append(topic)
                seen.add(topic)
        
        return topics
    
    def _build_trigger_context(self, trigger_results: List[TriggerResult]) -> str:
        """Build a trigger context section for multiple triggers.
        
        When multiple triggers fire between AI checks, this section explains
        all of them so the AI can properly analyze the full context.
        """
        if not trigger_results:
            return ""
        
        if len(trigger_results) == 1:
            # Single trigger - no special section needed
            return ""
        
        lines = [
            f"\n## ⚠️ MULTIPLE TRIGGERS ({len(trigger_results)}) ⚠️",
            "The following events ALL triggered this AI check (analyze all):"
        ]
        
        for i, tr in enumerate(trigger_results, 1):
            topic = tr.topic or "unknown"
            field = tr.field_name or "?"
            old_val = tr.old_value
            new_val = tr.new_value
            lines.append(f"  {i}. {topic}[{field}]: {old_val} → {new_val}")
        
        lines.append("")  # Empty line for separation
        return '\n'.join(lines) + '\n'

    def _build_existing_patterns_set(
        self,
        learned_rules: Dict[str, Any]
    ) -> set:
        """Build a set of (trigger_topic, trigger_field, action_topic) tuples from existing rules.
        
        This is used to annotate MQTT messages that match existing patterns,
        making it visually clear to the AI that these patterns are already learned.
        """
        patterns = set()
        for rule in learned_rules.get("rules", []):
            trigger = rule.get("trigger", {})
            action = rule.get("action", {})
            trigger_topic = trigger.get("topic", "")
            trigger_field = trigger.get("field", "")
            action_topic = action.get("topic", "")
            if trigger_topic and trigger_field and action_topic:
                patterns.add((trigger_topic, trigger_field, action_topic))
        return patterns

    def _compress_messages(
        self,
        messages_snapshot: str,
        trigger_topic: Optional[str],
        existing_patterns: Optional[set] = None,
        trigger_topics: Optional[List[str]] = None
    ) -> str:
        """Compress MQTT messages with deduplication and counts.

        Args:
            messages_snapshot: Raw messages as newline-separated string
            trigger_topic: Primary topic that triggered analysis (backwards compat)
            existing_patterns: Set of existing rule patterns for annotation
            trigger_topics: List of all trigger topics (when multiple triggers fired)

        Returns:
            Compressed messages string
        """
        lines = messages_snapshot.strip().split('\n')
        if not lines:
            return ""
        
        # Build set of all trigger topics for efficient lookup
        topics_to_highlight = set()
        if trigger_topic:
            topics_to_highlight.add(trigger_topic)
        if trigger_topics:
            topics_to_highlight.update(trigger_topics)

        # First pass: collect all /set topics to identify status feedback
        # Also collect action_topic values from announce messages to mark as automated
        set_topics: set = set()
        auto_action_topics: set = set()
        for line in lines:
            parsed = self._parse_message_line(line)
            if parsed:
                _, topic, payload = parsed
                if topic.endswith('/set'):
                    # Store the base topic (without /set) to identify status feedback
                    set_topics.add(topic[:-4])  # Remove '/set' suffix
                # Track action topics from announce messages
                if topic.startswith("mqtt2ai/action/") and "action_topic" in payload:
                    action_topic = payload.get("action_topic")
                    if action_topic:
                        auto_action_topics.add(action_topic)

        # Parse messages and track by topic
        topic_stats: Dict[str, MessageStats] = {}
        trigger_messages: List[str] = []

        for line in lines:
            if not line.strip():
                continue

            parsed = self._parse_message_line(line)
            if not parsed:
                continue

            timestamp, topic, payload = parsed

            # Check if this is any trigger topic - always keep and mark
            if topic in topics_to_highlight:
                short_topic = self._shorten_topic(topic)
                trigger_messages.append(
                    f"[{timestamp}] {short_topic} {self._format_payload(payload)} (TRIGGER)"
                )
                continue

            # Deduplicate by topic
            if topic in topic_stats:
                stats = topic_stats[topic]
                stats.count += 1
                stats.timestamp = timestamp  # Update to latest
                stats.payload = payload  # Update to latest payload
                
                # Track numeric ranges
                for field_name in self.NUMERIC_FIELDS:
                    if field_name in payload:
                        try:
                            val = float(payload[field_name])
                            if field_name in stats.numeric_ranges:
                                min_val, max_val = stats.numeric_ranges[field_name]
                                stats.numeric_ranges[field_name] = (
                                    min(min_val, val), max(max_val, val)
                                )
                            else:
                                stats.numeric_ranges[field_name] = (val, val)
                        except (TypeError, ValueError):
                            pass
            else:
                topic_stats[topic] = MessageStats(
                    topic=topic,
                    payload=payload,
                    timestamp=timestamp,
                    first_seen=timestamp
                )

        # Build output
        output_lines = []

        # Trigger messages first
        output_lines.extend(trigger_messages)

        # Then deduplicated messages sorted by timestamp (most recent first)
        sorted_stats = sorted(
            topic_stats.values(),
            key=lambda s: s.timestamp,
            reverse=True
        )

        for stats in sorted_stats:
            line = self._format_stats_line(stats)

            # Check if this is an automated action announcement
            # These messages indicate actions taken by the rule engine or AI
            is_auto_announce = stats.topic.startswith("mqtt2ai/action/")
            
            # Check if this topic was the target of an automated action
            # (i.e., it appears in an announce message's action_topic)
            is_auto_action = stats.topic in auto_action_topics

            # Check if this is status feedback (topic has a corresponding /set command)
            # Status feedback should NOT be used as triggers for pattern learning
            is_status_feedback = stats.topic in set_topics and not stats.topic.endswith('/set')

            # Mark automated action announcements and their targets with [AUTO] prefix
            if is_auto_announce or is_auto_action:
                line = "[AUTO] " + line
            # Annotate messages that have existing rules with PREFIX so AI sees it first
            elif existing_patterns:
                for trigger_field in stats.payload.keys():
                    # Check if this topic+field has a rule to any action
                    for pattern in existing_patterns:
                        if pattern[0] == stats.topic and pattern[1] == trigger_field:
                            line = "[SKIP-LEARNED] " + line
                            break
                    else:
                        continue
                    break

            # Mark status feedback messages so AI knows not to use them as triggers
            if is_status_feedback and "[SKIP-LEARNED]" not in line and "[AUTO]" not in line:
                line = "[STATUS] " + line

            output_lines.append(line)

        return '\n'.join(output_lines)

    def _parse_message_line(
        self, line: str
    ) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """Parse a message line into (timestamp, topic, payload).
        
        Expected format: [HH:MM:SS] topic/path {"key": "value"}
        """
        try:
            # Extract timestamp
            ts_match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*', line)
            if not ts_match:
                return None
            
            timestamp = ts_match.group(1)
            rest = line[ts_match.end():]

            # Find JSON start
            json_start = rest.find('{')
            if json_start == -1:
                # No JSON payload
                topic = rest.strip()
                return timestamp, topic, {}

            topic = rest[:json_start].strip()
            json_str = rest[json_start:]

            # Parse and filter JSON
            payload = json.loads(json_str)
            if isinstance(payload, dict):
                # Remove noise fields
                payload = {
                    k: v for k, v in payload.items()
                    if k not in self.REMOVE_FIELDS
                    and v is not None
                    and not isinstance(v, dict)
                }

            return timestamp, topic, payload

        except (json.JSONDecodeError, ValueError):
            return None

    def _format_payload(self, payload: Dict[str, Any]) -> str:
        """Format payload as compact key:value string."""
        if not payload:
            return "{}"
        parts = [f"{k}:{json.dumps(v)}" for k, v in payload.items()]
        return " ".join(parts)

    def _shorten_topic(self, topic: str) -> str:
        """Shorten topic by replacing common prefixes."""
        for prefix, replacement in self.TOPIC_REPLACEMENTS.items():
            if topic.startswith(prefix):
                return replacement + topic[len(prefix):]
        return topic

    def _format_stats_line(self, stats: MessageStats) -> str:
        """Format a MessageStats as a compressed line."""
        topic = self._shorten_topic(stats.topic)
        payload_str = self._format_payload(stats.payload)
        
        # Add count suffix if > 1
        count_suffix = f" ({stats.count}x)" if stats.count > 1 else ""
        
        # Add range info for numeric fields if significantly different
        range_info = ""
        for field_name, (min_val, max_val) in stats.numeric_ranges.items():
            if max_val - min_val > 1:  # Only show if meaningful range
                range_info += f" range:{min_val:.0f}-{max_val:.0f}"

        return f"[{stats.timestamp}] {topic} {payload_str}{count_suffix}{range_info}"

    def _filter_relevant_rules(
        self,
        learned_rules: Dict[str, Any],
        trigger_topic: Optional[str],
        strict: bool = False,
        trigger_topics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Filter rules to those relevant to the trigger(s).
        
        Args:
            learned_rules: Full learned rules dict
            trigger_topic: Primary topic that triggered analysis (backwards compat)
            strict: If True, only exact matches; if False, also include recent rules
            trigger_topics: List of all trigger topics (when multiple triggers fired)
            
        Returns:
            List of relevant rule dicts
        """
        rules = learned_rules.get("rules", [])
        if not rules:
            return []
        
        # Build set of all trigger topics for efficient lookup
        topics_to_match = set()
        if trigger_topic:
            topics_to_match.add(trigger_topic)
        if trigger_topics:
            topics_to_match.update(trigger_topics)

        relevant = []
        other = []
        now = datetime.now()

        for rule in rules:
            if not rule.get("enabled", True):
                continue

            rule_trigger_topic = rule.get("trigger", {}).get("topic", "")
            
            # Check if matches any trigger topic
            if rule_trigger_topic in topics_to_match:
                relevant.append(rule)
                continue

            # Check if recently triggered (within 5 minutes)
            if not strict:
                last_triggered = rule.get("confidence", {}).get("last_triggered")
                if last_triggered:
                    try:
                        triggered_dt = datetime.fromisoformat(last_triggered)
                        if (now - triggered_dt).total_seconds() < 300:
                            relevant.append(rule)
                            continue
                    except ValueError:
                        pass

            other.append(rule)

        # If we have relevant rules, return them
        # Otherwise, return all enabled rules
        if relevant:
            return relevant
        
        # Return all enabled rules if no specific matches (other list has enabled rules)
        return other if not strict else []

    def _filter_relevant_patterns(
        self,
        pending_patterns: Dict[str, Any],
        trigger_topic: Optional[str],
        trigger_topics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Filter patterns to those relevant to the trigger(s)."""
        patterns = pending_patterns.get("patterns", [])
        if not patterns:
            return []
        
        # Build set of all trigger topics for efficient lookup
        topics_to_match = set()
        if trigger_topic:
            topics_to_match.add(trigger_topic)
        if trigger_topics:
            topics_to_match.update(trigger_topics)

        relevant = []
        for pattern in patterns:
            pattern_trigger = pattern.get("trigger_topic", "")
            
            # Include if matches any trigger topic or is being actively tracked
            if pattern_trigger in topics_to_match:
                relevant.append(pattern)
            elif len(pattern.get("observations", [])) >= 2:
                # Include patterns close to becoming rules
                relevant.append(pattern)

        return relevant

    def _format_rules(
        self,
        rules: List[Dict[str, Any]],
        trigger_topic: Optional[str],
        all_rules: Optional[List[Dict[str, Any]]] = None,
        trigger_topics: Optional[List[str]] = None
    ) -> str:
        """Format rules section for prompt.
        
        Args:
            rules: Filtered/enabled rules to display for execution
            trigger_topic: Primary topic that triggered analysis (backwards compat)
            all_rules: ALL rules (enabled and disabled) for SKIP PATTERNS section
            trigger_topics: List of all trigger topics (when multiple triggers fired)
        """
        # Build set of all trigger topics for matching
        topics_to_match = set()
        if trigger_topic:
            topics_to_match.add(trigger_topic)
        if trigger_topics:
            topics_to_match.update(trigger_topics)
        
        # Display section for enabled rules
        if not rules:
            rules_section = "\n## Learned Rules: None yet.\n"
        else:
            lines = ["\n## Learned Rules (execute when triggers match):"]
            for rule in rules:
                rule_id = rule.get("id", "unknown")
                trigger = rule.get("trigger", {})
                action = rule.get("action", {})
                occurrences = rule.get("confidence", {}).get("occurrences", 0)
                
                # Mark if this matches any trigger topic
                rule_topic = trigger.get("topic")
                marker = " ← MATCHES" if rule_topic in topics_to_match else ""
                
                lines.append(
                    f"- {rule_id}: {rule_topic} "
                    f"[{trigger.get('field')}={trigger.get('value')}] "
                    f"→ {action.get('topic')} ({occurrences} triggers){marker}"
                )
            rules_section = '\n'.join(lines) + '\n'

        # Use ALL rules (enabled or disabled) for SKIP PATTERNS to prevent re-learning
        skip_rules = all_rules if all_rules else rules
        if not skip_rules:
            return rules_section

        # Add explicit SKIP PATTERNS section to prevent redundant tool calls
        skip_lines = [
            "\n## ⚠️ SKIP PATTERNS - RULES ALREADY EXIST ⚠️",
            "These patterns are ALREADY LEARNED. Calling record_pattern_observation or create_rule for these is WASTEFUL:",
        ]
        for rule in skip_rules:
            trigger = rule.get("trigger", {})
            action = rule.get("action", {})
            skip_lines.append(
                f"  ✗ {trigger.get('topic')}[{trigger.get('field')}] -> {action.get('topic')}"
            )
        skip_lines.append("")  # Empty line for separation

        return rules_section + '\n'.join(skip_lines) + '\n'

    def _format_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """Format pending patterns section for prompt."""
        if not patterns:
            return ""

        lines = ["\n## Pending Patterns (close to becoming rules):"]
        for pattern in patterns:
            obs_count = len(pattern.get("observations", []))
            lines.append(
                f"- {pattern.get('trigger_topic')} [{pattern.get('trigger_field')}] "
                f"→ {pattern.get('action_topic')} ({obs_count}/3 observations)"
            )

        return '\n'.join(lines) + '\n'

    def _format_rejected(self, rejected: Dict[str, Any]) -> str:
        """Format rejected patterns section."""
        patterns = rejected.get("patterns", [])
        if not patterns:
            return ""

        lines = ["\n## Rejected Patterns (DO NOT learn):"]
        for pattern in patterns[:5]:  # Limit to 5
            lines.append(
                f"- {pattern.get('trigger_topic')} → {pattern.get('action_topic')}"
            )

        return '\n'.join(lines) + '\n'

    def _build_safety_alert(self, trigger_reason: str) -> str:
        """Build safety alert section if applicable."""
        trigger_lower = trigger_reason.lower()
        if any(x in trigger_lower for x in ["temperature", "smoke", "water", "leak"]):
            return (
                "**SAFETY ALERT**: Potential safety event detected. "
                "Check for smoke:true, water_leak:true, or temperature>50C "
                "and ACT IMMEDIATELY if found.\n\n"
            )
        return ""

    def _build_demo_instruction(self) -> str:
        """Build demo mode instruction."""
        if self.config.demo_mode:
            return (
                "**🎭 DEMO MODE ENABLED**\n"
                "Your ONLY task: Send ONE joke to the `jokes/` topic using send_mqtt_message, then STOP.\n"
                "DO NOT: record patterns, create rules, analyze messages, or make any other tool calls.\n"
                "After sending the joke, your response is complete.\n\n"
            )
        return ""

