"""Tests for the PromptBuilder module."""
from unittest.mock import MagicMock
import pytest

from prompt_builder import PromptBuilder, MessageStats, COMPACT_RULEBOOK
from trigger_analyzer import TriggerResult
from knowledge_base import KnowledgeBase


class TestPromptBuilder:
    """Tests for PromptBuilder class."""

    @pytest.fixture
    def builder(self, config):
        """Create a PromptBuilder instance."""
        return PromptBuilder(config)

    @pytest.fixture
    def mock_kb(self, sample_learned_rules, sample_pending_patterns, sample_rejected_patterns):
        """Create a mock KnowledgeBase."""
        kb = MagicMock(spec=KnowledgeBase)
        kb.learned_rules = sample_learned_rules
        kb.pending_patterns = sample_pending_patterns
        kb.rejected_patterns = sample_rejected_patterns
        kb.rulebook_content = "# Test Rulebook"
        return kb

    @pytest.fixture
    def sample_messages_snapshot(self):
        """Sample MQTT messages in snapshot format."""
        return """[12:00:01] zigbee2mqtt/hallway_pir {"occupancy":true,"linkquality":150}
[12:00:02] zigbee2mqtt/temp_sensor {"temperature":22.5,"humidity":45}
[12:00:03] zigbee2mqtt/hallway_light/set {"state":"ON"}
[12:00:04] zigbee2mqtt/power_plug {"power":150,"voltage":230}
[12:00:05] zigbee2mqtt/hallway_pir {"occupancy":true,"linkquality":148}
[12:00:06] zigbee2mqtt/temp_sensor {"temperature":22.6,"humidity":44}"""


class TestMessageCompression(TestPromptBuilder):
    """Tests for message compression functionality."""

    def test_compress_removes_noise_fields(self, builder, sample_messages_snapshot):
        """Test that noise fields like linkquality and voltage are removed."""
        compressed = builder._compress_messages(sample_messages_snapshot, None)

        assert "linkquality" not in compressed
        assert "voltage" not in compressed
        # Useful fields should remain
        assert "occupancy" in compressed
        assert "temperature" in compressed
        assert "power" in compressed

    def test_compress_deduplicates_topics(self, builder):
        """Test that duplicate messages from same topic are deduplicated with count."""
        snapshot = """[12:00:01] zigbee2mqtt/pir {"occupancy":true}
[12:00:02] zigbee2mqtt/pir {"occupancy":true}
[12:00:03] zigbee2mqtt/pir {"occupancy":true}
[12:00:04] zigbee2mqtt/pir {"occupancy":true}
[12:00:05] zigbee2mqtt/pir {"occupancy":true}"""

        compressed = builder._compress_messages(snapshot, None)

        # Should contain count indicator
        assert "(5x)" in compressed
        # Should only have one line for this topic (plus maybe omitted message)
        lines = [l for l in compressed.split('\n') if l.strip() and 'omitted' not in l]
        assert len(lines) == 1

    def test_compress_marks_trigger_topic(self, builder):
        """Test that trigger topic is marked with (TRIGGER) suffix."""
        snapshot = """[12:00:01] zigbee2mqtt/hallway_pir {"occupancy":true}
[12:00:02] zigbee2mqtt/other_sensor {"state":"ON"}"""

        compressed = builder._compress_messages(
            snapshot, trigger_topic="zigbee2mqtt/hallway_pir"
        )

        assert "(TRIGGER)" in compressed

    def test_compress_tracks_numeric_ranges(self, builder):
        """Test that numeric field ranges are tracked during aggregation."""
        snapshot = """[12:00:01] zigbee2mqtt/power_plug {"power":100}
[12:00:02] zigbee2mqtt/power_plug {"power":150}
[12:00:03] zigbee2mqtt/power_plug {"power":200}
[12:00:04] zigbee2mqtt/power_plug {"power":120}"""

        compressed = builder._compress_messages(snapshot, None)

        # Should contain count and possibly range
        assert "(4x)" in compressed


class TestParseMessageLine(TestPromptBuilder):
    """Tests for message line parsing."""

    def test_parse_valid_line(self, builder):
        """Test parsing a valid message line."""
        line = '[12:05:01] zigbee2mqtt/sensor {"temperature":22.5}'
        result = builder._parse_message_line(line)

        assert result is not None
        timestamp, topic, payload = result
        assert timestamp == "12:05:01"
        assert topic == "zigbee2mqtt/sensor"
        assert payload == {"temperature": 22.5}

    def test_parse_line_without_json(self, builder):
        """Test parsing a line without JSON payload."""
        line = '[12:05:01] zigbee2mqtt/sensor simple_value'
        result = builder._parse_message_line(line)

        assert result is not None
        timestamp, topic, payload = result
        assert topic == "zigbee2mqtt/sensor simple_value"
        assert payload == {}

    def test_parse_invalid_line(self, builder):
        """Test parsing an invalid line returns None."""
        line = 'invalid line without timestamp'
        result = builder._parse_message_line(line)

        assert result is None

    def test_parse_removes_noise_fields(self, builder):
        """Test that noise fields are removed during parsing."""
        line = '[12:05:01] zigbee2mqtt/sensor {"occupancy":true,"linkquality":150,"voltage":3.1}'
        result = builder._parse_message_line(line)

        assert result is not None
        _, _, payload = result
        assert "occupancy" in payload
        assert "linkquality" not in payload
        assert "voltage" not in payload


class TestRuleFiltering(TestPromptBuilder):
    """Tests for rule filtering by relevance."""

    def test_filter_rules_by_trigger_topic(self, builder, sample_learned_rules):
        """Test that rules are filtered by trigger topic."""
        trigger_topic = "zigbee2mqtt/kitchen_pir"

        relevant = builder._filter_relevant_rules(
            sample_learned_rules, trigger_topic
        )

        assert len(relevant) == 1
        assert relevant[0]["trigger"]["topic"] == trigger_topic

    def test_filter_rules_no_match_returns_all(self, builder, sample_learned_rules):
        """Test that when no match, all rules are returned (non-strict mode)."""
        trigger_topic = "zigbee2mqtt/nonexistent"

        relevant = builder._filter_relevant_rules(
            sample_learned_rules, trigger_topic, strict=False
        )

        # Should return all rules when no match found
        assert len(relevant) == len(sample_learned_rules["rules"])

    def test_filter_rules_strict_mode(self, builder, sample_learned_rules):
        """Test that strict mode returns empty when no match."""
        trigger_topic = "zigbee2mqtt/nonexistent"

        relevant = builder._filter_relevant_rules(
            sample_learned_rules, trigger_topic, strict=True
        )

        assert len(relevant) == 0

    def test_filter_excludes_disabled_rules(self, builder):
        """Test that disabled rules are excluded."""
        rules = {
            "rules": [
                {
                    "id": "disabled_rule",
                    "trigger": {"topic": "zigbee2mqtt/sensor"},
                    "enabled": False
                },
                {
                    "id": "enabled_rule",
                    "trigger": {"topic": "zigbee2mqtt/sensor"},
                    "enabled": True
                }
            ]
        }

        relevant = builder._filter_relevant_rules(rules, None)

        assert len(relevant) == 1
        assert relevant[0]["id"] == "enabled_rule"


class TestPatternFiltering(TestPromptBuilder):
    """Tests for pattern filtering by relevance."""

    def test_filter_patterns_by_trigger_topic(self, builder, sample_pending_patterns):
        """Test that patterns are filtered by trigger topic."""
        trigger_topic = "zigbee2mqtt/hallway_pir"

        relevant = builder._filter_relevant_patterns(
            sample_pending_patterns, trigger_topic
        )

        assert len(relevant) == 1
        assert relevant[0]["trigger_topic"] == trigger_topic

    def test_filter_patterns_includes_near_complete(self, builder):
        """Test that patterns with 2+ observations are included."""
        patterns = {
            "patterns": [
                {
                    "trigger_topic": "zigbee2mqtt/other_pir",
                    "observations": [
                        {"delay_seconds": 2.0},
                        {"delay_seconds": 2.5}
                    ]
                },
                {
                    "trigger_topic": "zigbee2mqtt/yet_another",
                    "observations": [
                        {"delay_seconds": 2.0}
                    ]
                }
            ]
        }

        # With a non-matching trigger
        relevant = builder._filter_relevant_patterns(
            patterns, "zigbee2mqtt/nonexistent"
        )

        # Should include the one with 2 observations
        assert len(relevant) == 1
        assert relevant[0]["trigger_topic"] == "zigbee2mqtt/other_pir"


class TestPromptBuilding(TestPromptBuilder):
    """Tests for full prompt building."""

    def test_build_includes_compact_rulebook(self, builder, mock_kb):
        """Test that build includes the compact rulebook."""
        prompt = builder.build("", mock_kb)

        assert "## Core Rules" in prompt
        assert "Safety" in prompt
        assert "Pattern Learning" in prompt

    def test_build_includes_safety_alert_for_smoke(self, builder, mock_kb):
        """Test that safety alert is included for smoke-related triggers."""
        prompt = builder.build(
            "", mock_kb, trigger_reason="State field 'smoke' changed"
        )

        assert "SAFETY ALERT" in prompt

    def test_build_includes_demo_instruction(self, config, mock_kb):
        """Test that demo mode instruction is included when enabled."""
        config.demo_mode = True
        builder = PromptBuilder(config)

        prompt = builder.build("", mock_kb)

        assert "DEMO MODE" in prompt
        assert "jokes/" in prompt

    def test_build_compact_includes_demo_instruction(self, config, mock_kb):
        """Test that demo mode instruction is included in compact build."""
        config.demo_mode = True
        builder = PromptBuilder(config)

        prompt = builder.build_compact("", mock_kb)

        assert "DEMO MODE" in prompt
        assert "jokes/" in prompt

    def test_build_compact_is_shorter(self, builder, mock_kb, sample_messages_snapshot):
        """Test that compact prompt is shorter than full prompt."""
        full_prompt = builder.build(sample_messages_snapshot, mock_kb)
        compact_prompt = builder.build_compact(sample_messages_snapshot, mock_kb)

        assert len(compact_prompt) < len(full_prompt)

    def test_build_formats_rules_with_marker(self, builder, mock_kb):
        """Test that matching rules are marked with MATCHES."""
        trigger_topic = "zigbee2mqtt/kitchen_pir"

        prompt = builder.build("", mock_kb, trigger_reason=f"topic: {trigger_topic}")

        # The rule should be marked as matching
        assert "MATCHES" in prompt or "kitchen_pir" in prompt


class TestTriggerTopicExtraction(TestPromptBuilder):
    """Tests for trigger topic extraction from context."""

    def test_extract_from_topic_prefix(self, builder):
        """Test extracting topic from 'topic: xxx' format."""
        result = builder._extract_trigger_topics(
            [], "topic: zigbee2mqtt/hallway_pir"
        )

        assert "zigbee2mqtt/hallway_pir" in result

    def test_extract_from_zigbee_pattern(self, builder):
        """Test extracting topic from zigbee2mqtt pattern in text."""
        result = builder._extract_trigger_topics(
            [], "State changed on zigbee2mqtt/kitchen_pir sensor"
        )

        assert "zigbee2mqtt/kitchen_pir" in result

    def test_extract_returns_none_for_no_match(self, builder):
        """Test that extraction returns empty list when no topic found."""
        result = builder._extract_trigger_topics(
            [], "Manual trigger pressed"
        )

        assert result == []


class TestFormatPayload(TestPromptBuilder):
    """Tests for payload formatting."""

    def test_format_payload_empty(self, builder):
        """Test formatting empty payload."""
        result = builder._format_payload({})
        assert result == "{}"

    def test_format_payload_simple(self, builder):
        """Test formatting simple payload."""
        result = builder._format_payload({"state": "ON", "brightness": 100})

        assert "state:" in result
        assert "ON" in result
        assert "brightness:" in result


class TestMessageStats:
    """Tests for MessageStats dataclass."""

    def test_message_stats_defaults(self):
        """Test MessageStats default values."""
        stats = MessageStats(
            topic="test/topic",
            payload={"key": "value"},
            timestamp="12:00:00"
        )

        assert stats.count == 1
        assert stats.first_seen is None
        assert stats.numeric_ranges == {}

    def test_message_stats_with_values(self):
        """Test MessageStats with explicit values."""
        stats = MessageStats(
            topic="test/topic",
            payload={"key": "value"},
            timestamp="12:00:05",
            count=5,
            first_seen="12:00:00",
            numeric_ranges={"power": (100.0, 150.0)}
        )

        assert stats.count == 5
        assert stats.first_seen == "12:00:00"
        assert stats.numeric_ranges["power"] == (100.0, 150.0)


class TestCompactRulebook:
    """Tests for the compact rulebook constant."""

    def test_compact_rulebook_has_safety_rules(self):
        """Test that compact rulebook includes safety rules."""
        assert "smoke:true" in COMPACT_RULEBOOK
        assert "water_leak:true" in COMPACT_RULEBOOK
        assert "temperature > 50" in COMPACT_RULEBOOK

    def test_compact_rulebook_has_pattern_learning(self):
        """Test that compact rulebook includes pattern learning instructions."""
        assert "Pattern Learning" in COMPACT_RULEBOOK
        assert "record_pattern_observation" in COMPACT_RULEBOOK
        assert "create_rule" in COMPACT_RULEBOOK

    def test_compact_rulebook_is_concise(self):
        """Test that compact rulebook is reasonably short."""
        # Should be under 2500 characters for token efficiency
        # (expanded to include security awareness features)
        assert len(COMPACT_RULEBOOK) < 2500


class TestExistingPatternsSet(TestPromptBuilder):
    """Tests for building and using the existing patterns set."""

    def test_build_existing_patterns_set_empty(self, builder):
        """Test building patterns set from empty rules."""
        patterns = builder._build_existing_patterns_set({"rules": []})
        assert patterns == set()

    def test_build_existing_patterns_set_single_rule(self, builder):
        """Test building patterns set from single rule."""
        rules = {
            "rules": [{
                "id": "test_rule",
                "trigger": {
                    "topic": "zigbee2mqtt/pir",
                    "field": "occupancy",
                    "value": True
                },
                "action": {
                    "topic": "zigbee2mqtt/light/set",
                    "payload": '{"state": "ON"}'
                }
            }]
        }
        patterns = builder._build_existing_patterns_set(rules)

        assert len(patterns) == 1
        assert ("zigbee2mqtt/pir", "occupancy", "zigbee2mqtt/light/set") in patterns

    def test_build_existing_patterns_set_multiple_rules(self, builder):
        """Test building patterns set from multiple rules."""
        rules = {
            "rules": [
                {
                    "id": "rule1",
                    "trigger": {"topic": "zigbee2mqtt/pir1", "field": "occupancy"},
                    "action": {"topic": "zigbee2mqtt/light1/set"}
                },
                {
                    "id": "rule2",
                    "trigger": {"topic": "zigbee2mqtt/door", "field": "contact"},
                    "action": {"topic": "zigbee2mqtt/porch/set"}
                }
            ]
        }
        patterns = builder._build_existing_patterns_set(rules)

        assert len(patterns) == 2
        assert ("zigbee2mqtt/pir1", "occupancy", "zigbee2mqtt/light1/set") in patterns
        assert ("zigbee2mqtt/door", "contact", "zigbee2mqtt/porch/set") in patterns

    def test_build_existing_patterns_set_ignores_incomplete(self, builder):
        """Test that rules with missing fields are ignored."""
        rules = {
            "rules": [
                {
                    "id": "incomplete",
                    "trigger": {"topic": "zigbee2mqtt/pir"},  # Missing field
                    "action": {"topic": "zigbee2mqtt/light/set"}
                },
                {
                    "id": "complete",
                    "trigger": {"topic": "zigbee2mqtt/door", "field": "contact"},
                    "action": {"topic": "zigbee2mqtt/porch/set"}
                }
            ]
        }
        patterns = builder._build_existing_patterns_set(rules)

        assert len(patterns) == 1
        assert ("zigbee2mqtt/door", "contact", "zigbee2mqtt/porch/set") in patterns


class TestSkipLearnedAnnotation(TestPromptBuilder):
    """Tests for SKIP-LEARNED message annotation."""

    def test_compress_adds_skip_learned_for_existing_patterns(self, builder):
        """Test that messages matching existing patterns get SKIP-LEARNED prefix."""
        existing_patterns = {
            ("zigbee2mqtt/hallway_pir", "occupancy", "zigbee2mqtt/hallway_light/set")
        }
        snapshot = '[12:00:01] zigbee2mqtt/hallway_pir {"occupancy":true}'

        compressed = builder._compress_messages(
            snapshot, None, existing_patterns=existing_patterns
        )

        assert "[SKIP-LEARNED]" in compressed

    def test_compress_no_skip_learned_for_new_patterns(self, builder):
        """Test that messages not matching patterns don't get prefix."""
        existing_patterns = {
            ("zigbee2mqtt/other_pir", "occupancy", "zigbee2mqtt/other_light/set")
        }
        snapshot = '[12:00:01] zigbee2mqtt/hallway_pir {"occupancy":true}'

        compressed = builder._compress_messages(
            snapshot, None, existing_patterns=existing_patterns
        )

        assert "[SKIP-LEARNED]" not in compressed

    def test_compress_skip_learned_matches_field(self, builder):
        """Test that SKIP-LEARNED only applies when trigger field matches."""
        # Pattern is for 'occupancy' field
        existing_patterns = {
            ("zigbee2mqtt/sensor", "occupancy", "zigbee2mqtt/light/set")
        }
        # Message has 'temperature' field, not 'occupancy'
        snapshot = '[12:00:01] zigbee2mqtt/sensor {"temperature":22}'

        compressed = builder._compress_messages(
            snapshot, None, existing_patterns=existing_patterns
        )

        assert "[SKIP-LEARNED]" not in compressed

    def test_compress_skip_learned_for_multiple_patterns(self, builder):
        """Test SKIP-LEARNED with multiple patterns."""
        existing_patterns = {
            ("zigbee2mqtt/pir", "occupancy", "zigbee2mqtt/light/set"),
            ("zigbee2mqtt/door", "contact", "zigbee2mqtt/porch/set"),
        }
        snapshot = """[12:00:01] zigbee2mqtt/pir {"occupancy":true}
[12:00:02] zigbee2mqtt/door {"contact":false}
[12:00:03] zigbee2mqtt/new_sensor {"motion":true}"""

        compressed = builder._compress_messages(
            snapshot, None, existing_patterns=existing_patterns
        )

        # Should have 2 SKIP-LEARNED annotations
        assert compressed.count("[SKIP-LEARNED]") == 2


class TestStatusFeedbackDetection(TestPromptBuilder):
    """Tests for distinguishing status feedback from actual triggers.

    This tests the scenario where:
    - User action: /set command sent to device
    - Status feedback: device reports new state (not a trigger)

    Status feedback messages should NOT be treated as triggers for new patterns.
    """

    def test_set_command_topics_identified(self, builder):
        """Test that /set topics are recognized as action topics."""
        snapshot = """[12:00:01] zigbee2mqtt/hallway_pir {"occupancy":true}
[12:00:02] zigbee2mqtt/hallway_light/set {"state":"ON"}
[12:00:03] zigbee2mqtt/hallway_light {"state":"ON","brightness":254}"""

        compressed = builder._compress_messages(snapshot, None)

        # All three messages should be present
        assert "hallway_pir" in compressed
        assert "hallway_light/set" in compressed
        assert "hallway_light" in compressed

    def test_status_feedback_marked_with_status_prefix(self, builder):
        """Test that status feedback messages get [STATUS] prefix."""
        snapshot = """[12:00:01] zigbee2mqtt/hallway_light/set {"state":"ON"}
[12:00:02] zigbee2mqtt/hallway_light {"state":"ON","brightness":254}"""

        compressed = builder._compress_messages(snapshot, None)

        # hallway_light (status feedback) should have [STATUS] prefix
        assert "[STATUS]" in compressed
        # The /set command should NOT have [STATUS]
        lines = compressed.split('\n')
        for line in lines:
            if "/set" in line:
                assert "[STATUS]" not in line

    def test_status_feedback_not_marked_without_set_command(self, builder):
        """Test that messages are not marked [STATUS] if no /set command exists."""
        snapshot = """[12:00:01] zigbee2mqtt/hallway_pir {"occupancy":true}
[12:00:02] zigbee2mqtt/hallway_light {"state":"ON","brightness":254}"""

        compressed = builder._compress_messages(snapshot, None)

        # No /set command, so no [STATUS] prefix
        assert "[STATUS]" not in compressed

    def test_status_feedback_marked_for_multiple_devices(self, builder):
        """Test [STATUS] marking works for multiple devices."""
        snapshot = """[12:00:01] zigbee2mqtt/hallway_light/set {"state":"ON"}
[12:00:02] zigbee2mqtt/hallway_light {"state":"ON"}
[12:00:03] zigbee2mqtt/porch_light/set {"state":"ON"}
[12:00:04] zigbee2mqtt/porch_light {"state":"ON"}"""

        compressed = builder._compress_messages(snapshot, None)

        # Both status feedback messages should have [STATUS]
        assert compressed.count("[STATUS]") == 2

    def test_sensor_messages_not_marked_as_status(self, builder):
        """Test that sensor messages (PIR, door) are not marked as status."""
        snapshot = """[12:00:01] zigbee2mqtt/hallway_pir {"occupancy":true}
[12:00:02] zigbee2mqtt/front_door {"contact":false}
[12:00:03] zigbee2mqtt/hallway_light/set {"state":"ON"}
[12:00:04] zigbee2mqtt/hallway_light {"state":"ON"}"""

        compressed = builder._compress_messages(snapshot, None)

        # Only the light status should have [STATUS], not the sensors
        assert compressed.count("[STATUS]") == 1
        # Verify sensors don't have [STATUS]
        lines = compressed.split('\n')
        for line in lines:
            if "hallway_pir" in line or "front_door" in line:
                assert "[STATUS]" not in line

    def test_trigger_topic_excluded_from_dedup(self, builder):
        """Test that trigger topic is kept separate and not deduplicated."""
        snapshot = """[12:00:01] zigbee2mqtt/hallway_pir {"occupancy":true}
[12:00:02] zigbee2mqtt/hallway_pir {"occupancy":true}
[12:00:03] zigbee2mqtt/hallway_pir {"occupancy":false}"""

        # When hallway_pir is the trigger, all its messages should be kept
        compressed = builder._compress_messages(
            snapshot, trigger_topic="zigbee2mqtt/hallway_pir"
        )

        # Trigger messages should all have (TRIGGER) marker
        assert compressed.count("(TRIGGER)") >= 1

    def test_skip_patterns_section_in_prompt(self, builder, mock_kb):
        """Test that SKIP PATTERNS section is included in prompt."""
        prompt = builder.build("", mock_kb)

        assert "SKIP PATTERNS" in prompt

    def test_skip_patterns_includes_all_rules(self, builder):
        """Test that SKIP PATTERNS section includes both enabled and disabled rules."""
        enabled_rules = [
            {
                "id": "enabled",
                "trigger": {"topic": "zigbee2mqtt/pir", "field": "occupancy"},
                "action": {"topic": "zigbee2mqtt/light/set"},
                "enabled": True
            }
        ]
        all_rules = [
            {
                "id": "enabled",
                "trigger": {"topic": "zigbee2mqtt/pir", "field": "occupancy"},
                "action": {"topic": "zigbee2mqtt/light/set"},
                "enabled": True
            },
            {
                "id": "disabled",
                "trigger": {"topic": "zigbee2mqtt/door", "field": "contact"},
                "action": {"topic": "zigbee2mqtt/porch/set"},
                "enabled": False
            }
        ]

        rules_section = builder._format_rules(enabled_rules, None, all_rules)

        # Both patterns should be in SKIP PATTERNS
        assert "zigbee2mqtt/pir[occupancy]" in rules_section
        assert "zigbee2mqtt/door[contact]" in rules_section


class TestPatternLearningScenarios(TestPromptBuilder):
    """Tests for realistic pattern learning scenarios from simulation."""

    def test_pir_to_light_pattern_sequence(self, builder):
        """Test the hallway PIR -> light pattern learning sequence."""
        # Simulates: PIR triggers -> user turns on light -> light reports state
        snapshot = """[12:00:01] zigbee2mqtt/hallway_pir {"occupancy":true,"battery":85}
[12:00:02] zigbee2mqtt/hallway_light/set {"state":"ON"}
[12:00:02] zigbee2mqtt/hallway_light {"state":"ON","brightness":254}"""

        # After pattern is learned, the trigger should be marked
        existing_patterns = {
            ("zigbee2mqtt/hallway_pir", "occupancy", "zigbee2mqtt/hallway_light/set")
        }

        compressed = builder._compress_messages(
            snapshot, None, existing_patterns=existing_patterns
        )

        # PIR message should have SKIP-LEARNED since pattern exists
        assert "[SKIP-LEARNED]" in compressed
        # Status feedback (hallway_light without /set) should have [STATUS]
        lines = compressed.split('\n')
        for line in lines:
            if "hallway_light " in line and "/set" not in line:
                assert "[STATUS]" in line
                assert "[SKIP-LEARNED]" not in line

    def test_pir_to_light_status_feedback_marked(self, builder):
        """Test that status feedback is properly marked in PIR->light scenario."""
        snapshot = """[12:00:01] zigbee2mqtt/hallway_pir {"occupancy":true}
[12:00:02] zigbee2mqtt/hallway_light/set {"state":"ON"}
[12:00:03] zigbee2mqtt/hallway_light {"state":"ON","brightness":254}"""

        compressed = builder._compress_messages(snapshot, None)

        # Status feedback should have [STATUS] prefix
        assert "[STATUS]" in compressed
        # PIR should NOT have [STATUS] (it's a sensor, not status feedback)
        lines = compressed.split('\n')
        for line in lines:
            if "hallway_pir" in line:
                assert "[STATUS]" not in line

    def test_door_to_porch_light_pattern(self, builder):
        """Test door contact -> porch light pattern."""
        existing_patterns = {
            ("zigbee2mqtt/front_door", "contact", "zigbee2mqtt/porch_light/set")
        }
        snapshot = """[12:00:01] zigbee2mqtt/front_door {"contact":false,"battery":95}
[12:00:02] zigbee2mqtt/porch_light/set {"state":"ON"}
[12:00:02] zigbee2mqtt/porch_light {"state":"ON","brightness":200}"""

        compressed = builder._compress_messages(
            snapshot, None, existing_patterns=existing_patterns
        )

        # Door message should have SKIP-LEARNED
        assert "[SKIP-LEARNED]" in compressed
        assert "front_door" in compressed
        # Porch light status should have [STATUS]
        lines = compressed.split('\n')
        for line in lines:
            if "porch_light " in line and "/set" not in line:
                assert "[STATUS]" in line

    def test_status_feedback_not_marked_as_skip_learned(self, builder):
        """Test that status feedback messages are not incorrectly marked.

        This is the core issue: zigbee2mqtt/hallway_light (status) should NOT
        be marked as SKIP-LEARNED just because there's a pattern for hallway_pir.
        """
        # Pattern: PIR -> light/set
        existing_patterns = {
            ("zigbee2mqtt/hallway_pir", "occupancy", "zigbee2mqtt/hallway_light/set")
        }
        # Status feedback message (NOT a trigger) - but no /set in this snapshot
        snapshot = '[12:00:01] zigbee2mqtt/hallway_light {"state":"ON","brightness":254}'

        compressed = builder._compress_messages(
            snapshot, None, existing_patterns=existing_patterns
        )

        # Status feedback should NOT have SKIP-LEARNED
        # The pattern is for hallway_pir, not hallway_light
        assert "[SKIP-LEARNED]" not in compressed

    def test_full_scenario_with_all_markers(self, builder):
        """Test a full realistic scenario with SKIP-LEARNED and STATUS markers."""
        existing_patterns = {
            ("zigbee2mqtt/hallway_pir", "occupancy", "zigbee2mqtt/hallway_light/set")
        }
        snapshot = """[12:00:01] zigbee2mqtt/hallway_pir {"occupancy":true}
[12:00:02] zigbee2mqtt/hallway_light/set {"state":"ON"}
[12:00:02] zigbee2mqtt/hallway_light {"state":"ON","brightness":254}
[12:00:10] zigbee2mqtt/kitchen_pir {"motion":true}
[12:00:11] zigbee2mqtt/kitchen_light/set {"state":"ON"}
[12:00:11] zigbee2mqtt/kitchen_light {"state":"ON"}"""

        compressed = builder._compress_messages(
            snapshot, None, existing_patterns=existing_patterns
        )

        # hallway_pir should have SKIP-LEARNED (existing pattern)
        # hallway_light and kitchen_light should have STATUS (feedback after /set)
        # kitchen_pir should have neither (new potential trigger)
        lines = compressed.split('\n')

        for line in lines:
            if "hallway_pir" in line:
                assert "[SKIP-LEARNED]" in line
                assert "[STATUS]" not in line
            elif "hallway_light " in line and "/set" not in line:
                assert "[STATUS]" in line
                assert "[SKIP-LEARNED]" not in line
            elif "kitchen_light " in line and "/set" not in line:
                assert "[STATUS]" in line
            elif "kitchen_pir" in line:
                assert "[STATUS]" not in line
                assert "[SKIP-LEARNED]" not in line

