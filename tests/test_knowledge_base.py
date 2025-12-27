"""Tests for the KnowledgeBase module."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_base import KnowledgeBase
from config import Config


class TestKnowledgeBaseInit:
    """Tests for KnowledgeBase initialization."""

    def test_init_with_config(self, config):
        """Test initialization with a config."""
        kb = KnowledgeBase(config)

        assert kb.config == config
        assert kb.learned_rules == {"rules": []}
        assert kb.pending_patterns == {"patterns": []}
        assert kb.rejected_patterns == {"patterns": []}
        assert kb.rulebook_content == ""

    def test_init_empty_state(self, config):
        """Test that initial state is empty."""
        kb = KnowledgeBase(config)

        assert len(kb.learned_rules.get("rules", [])) == 0
        assert len(kb.pending_patterns.get("patterns", [])) == 0
        assert len(kb.rejected_patterns.get("patterns", [])) == 0


class TestKnowledgeBaseLoadAll:
    """Tests for load_all method."""

    def test_load_all_with_no_files(self, config_with_temp_files, temp_dir):
        """Test load_all when no files exist."""
        kb = KnowledgeBase(config_with_temp_files)
        kb.load_all()

        # Should use defaults
        assert kb.learned_rules == {"rules": []}
        assert kb.pending_patterns == {"patterns": []}
        assert kb.rejected_patterns == {"patterns": []}
        assert kb.rulebook_content == ""

    def test_load_learned_rules(self, config_with_temp_files, sample_learned_rules):
        """Test loading learned rules from file."""
        # Create the rules file
        with open(config_with_temp_files.learned_rules_file, "w", encoding="utf-8") as f:
            json.dump(sample_learned_rules, f)

        kb = KnowledgeBase(config_with_temp_files)
        kb.load_all()

        assert kb.learned_rules == sample_learned_rules
        assert len(kb.learned_rules["rules"]) == 1
        assert kb.learned_rules["rules"][0]["id"] == "pir_kitchen_light"

    def test_load_pending_patterns(self, config_with_temp_files, sample_pending_patterns):
        """Test loading pending patterns from file."""
        with open(config_with_temp_files.pending_patterns_file, "w", encoding="utf-8") as f:
            json.dump(sample_pending_patterns, f)

        kb = KnowledgeBase(config_with_temp_files)
        kb.load_all()

        assert kb.pending_patterns == sample_pending_patterns
        assert len(kb.pending_patterns["patterns"]) == 1

    def test_load_rejected_patterns(self, config_with_temp_files, sample_rejected_patterns):
        """Test loading rejected patterns from file."""
        with open(config_with_temp_files.rejected_patterns_file, "w", encoding="utf-8") as f:
            json.dump(sample_rejected_patterns, f)

        kb = KnowledgeBase(config_with_temp_files)
        kb.load_all()

        assert kb.rejected_patterns == sample_rejected_patterns
        assert len(kb.rejected_patterns["patterns"]) == 1

    def test_load_rulebook(self, config_with_temp_files, sample_rulebook_content):
        """Test loading rulebook from file."""
        with open(config_with_temp_files.rulebook_file, "w", encoding="utf-8") as f:
            f.write(sample_rulebook_content)

        kb = KnowledgeBase(config_with_temp_files)
        kb.load_all()

        assert kb.rulebook_content == sample_rulebook_content
        assert "Home Automation Rulebook" in kb.rulebook_content

    def test_load_all_files(
        self,
        config_with_temp_files,
        sample_learned_rules,
        sample_pending_patterns,
        sample_rejected_patterns,
        sample_rulebook_content
    ):
        """Test loading all files together."""
        # Create all files
        with open(config_with_temp_files.learned_rules_file, "w", encoding="utf-8") as f:
            json.dump(sample_learned_rules, f)
        with open(config_with_temp_files.pending_patterns_file, "w", encoding="utf-8") as f:
            json.dump(sample_pending_patterns, f)
        with open(config_with_temp_files.rejected_patterns_file, "w", encoding="utf-8") as f:
            json.dump(sample_rejected_patterns, f)
        with open(config_with_temp_files.rulebook_file, "w", encoding="utf-8") as f:
            f.write(sample_rulebook_content)

        kb = KnowledgeBase(config_with_temp_files)
        kb.load_all()

        assert kb.learned_rules == sample_learned_rules
        assert kb.pending_patterns == sample_pending_patterns
        assert kb.rejected_patterns == sample_rejected_patterns
        assert kb.rulebook_content == sample_rulebook_content

    def test_reload_picks_up_changes(self, config_with_temp_files):
        """Test that reload picks up file changes."""
        kb = KnowledgeBase(config_with_temp_files)

        # First load - empty
        kb.load_all()
        assert kb.learned_rules == {"rules": []}

        # Modify the file
        new_rules = {
            "rules": [
                {"id": "new_rule", "trigger": {}, "action": {}}
            ]
        }
        with open(config_with_temp_files.learned_rules_file, "w", encoding="utf-8") as f:
            json.dump(new_rules, f)

        # Reload
        kb.load_all()
        assert len(kb.learned_rules["rules"]) == 1
        assert kb.learned_rules["rules"][0]["id"] == "new_rule"

    def test_load_invalid_json_uses_default(self, config_with_temp_files):
        """Test that invalid JSON files result in default values."""
        # Create an invalid JSON file
        with open(config_with_temp_files.learned_rules_file, "w", encoding="utf-8") as f:
            f.write("not valid json {{{")

        kb = KnowledgeBase(config_with_temp_files)
        kb.load_all()

        # Should use default
        assert kb.learned_rules == {"rules": []}

    def test_load_missing_rulebook(self, config_with_temp_files):
        """Test that missing rulebook doesn't crash."""
        # Point to non-existent rulebook
        config_with_temp_files.rulebook_file = "/nonexistent/path/rulebook.md"

        kb = KnowledgeBase(config_with_temp_files)
        kb.load_all()  # Should not raise

        assert kb.rulebook_content == ""


class TestKnowledgeBaseRulesAccess:
    """Tests for accessing rules data."""

    def test_access_rules_list(self, config_with_temp_files, sample_learned_rules):
        """Test accessing the rules list."""
        with open(config_with_temp_files.learned_rules_file, "w", encoding="utf-8") as f:
            json.dump(sample_learned_rules, f)

        kb = KnowledgeBase(config_with_temp_files)
        kb.load_all()

        rules = kb.learned_rules.get("rules", [])
        assert len(rules) == 1

        rule = rules[0]
        assert rule["id"] == "pir_kitchen_light"
        assert rule["trigger"]["topic"] == "zigbee2mqtt/kitchen_pir"
        assert rule["trigger"]["field"] == "occupancy"
        assert rule["enabled"] is True

    def test_access_patterns_list(self, config_with_temp_files, sample_pending_patterns):
        """Test accessing the pending patterns list."""
        with open(config_with_temp_files.pending_patterns_file, "w", encoding="utf-8") as f:
            json.dump(sample_pending_patterns, f)

        kb = KnowledgeBase(config_with_temp_files)
        kb.load_all()

        patterns = kb.pending_patterns.get("patterns", [])
        assert len(patterns) == 1

        pattern = patterns[0]
        assert pattern["trigger_topic"] == "zigbee2mqtt/hallway_pir"
        assert len(pattern["observations"]) == 2


class TestKnowledgeBaseEmptyStructures:
    """Tests for empty data structures."""

    def test_empty_rules_structure(self, config):
        """Test empty rules have correct structure."""
        kb = KnowledgeBase(config)
        assert "rules" in kb.learned_rules
        assert isinstance(kb.learned_rules["rules"], list)

    def test_empty_patterns_structure(self, config):
        """Test empty patterns have correct structure."""
        kb = KnowledgeBase(config)
        assert "patterns" in kb.pending_patterns
        assert isinstance(kb.pending_patterns["patterns"], list)

    def test_empty_rejected_structure(self, config):
        """Test empty rejected patterns have correct structure."""
        kb = KnowledgeBase(config)
        assert "patterns" in kb.rejected_patterns
        assert isinstance(kb.rejected_patterns["patterns"], list)


