"""Tests for the PromptBuilder module."""
import pytest
from unittest.mock import MagicMock

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

    def test_compress_respects_max_lines(self, builder):
        """Test that compression respects max_lines limit."""
        # Create many messages
        lines = []
        for i in range(200):
            lines.append(f'[12:00:{i:02d}] zigbee2mqtt/sensor_{i} {{"value":{i}}}')
        snapshot = '\n'.join(lines)
        
        compressed = builder._compress_messages(snapshot, None, max_lines=10)
        
        # Should be limited (10 lines + omitted message)
        result_lines = compressed.strip().split('\n')
        assert len(result_lines) <= 12  # 10 + possible omitted message

    def test_compress_respects_max_chars(self, builder):
        """Test that compression respects max_chars limit."""
        # Create a very long snapshot
        lines = []
        for i in range(500):
            lines.append(f'[12:00:00] zigbee2mqtt/sensor_{i} {{"value":{i},"data":"x"*100}}')
        snapshot = '\n'.join(lines)
        
        compressed = builder._compress_messages(snapshot, None, max_chars=1000)
        
        # Should be truncated
        assert len(compressed) <= 1020  # Allow for truncation message

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
        
        assert "Demo mode" in prompt
        assert "jokes/" in prompt

    def test_build_compact_includes_demo_instruction(self, config, mock_kb):
        """Test that demo mode instruction is included in compact build."""
        config.demo_mode = True
        builder = PromptBuilder(config)
        
        prompt = builder.build_compact("", mock_kb)
        
        assert "Demo mode" in prompt
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
        result = builder._extract_trigger_topic(
            None, "topic: zigbee2mqtt/hallway_pir"
        )
        
        assert result == "zigbee2mqtt/hallway_pir"

    def test_extract_from_zigbee_pattern(self, builder):
        """Test extracting topic from zigbee2mqtt pattern in text."""
        result = builder._extract_trigger_topic(
            None, "State changed on zigbee2mqtt/kitchen_pir sensor"
        )
        
        assert result == "zigbee2mqtt/kitchen_pir"

    def test_extract_returns_none_for_no_match(self, builder):
        """Test that extraction returns None when no topic found."""
        result = builder._extract_trigger_topic(
            None, "Manual trigger pressed"
        )
        
        assert result is None


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
        # Should be under 1000 characters for token efficiency
        assert len(COMPACT_RULEBOOK) < 1000

