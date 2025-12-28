"""Tests for the KnowledgeBase module."""
import json
import os
import sys

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

    def test_load_all_files(
        self,
        config_with_temp_files,
        sample_learned_rules,
        sample_pending_patterns,
        sample_rejected_patterns,
    ):
        """Test loading all files together."""
        # Create all files
        with open(config_with_temp_files.learned_rules_file, "w", encoding="utf-8") as f:
            json.dump(sample_learned_rules, f)
        with open(config_with_temp_files.pending_patterns_file, "w", encoding="utf-8") as f:
            json.dump(sample_pending_patterns, f)
        with open(config_with_temp_files.rejected_patterns_file, "w", encoding="utf-8") as f:
            json.dump(sample_rejected_patterns, f)

        kb = KnowledgeBase(config_with_temp_files)
        kb.load_all()

        assert kb.learned_rules == sample_learned_rules
        assert kb.pending_patterns == sample_pending_patterns
        assert kb.rejected_patterns == sample_rejected_patterns

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


class TestKnowledgeBaseMtimeOptimization:
    """Tests for mtime-based caching optimization."""

    def test_load_all_skips_when_unchanged(self, config_with_temp_files, mocker):
        """Test that load_all skips I/O when files haven't changed."""
        # Create initial rules file
        initial_rules = {"rules": [{"id": "test_rule"}]}
        with open(config_with_temp_files.learned_rules_file, "w", encoding="utf-8") as f:
            json.dump(initial_rules, f)

        kb = KnowledgeBase(config_with_temp_files)

        # First load - should read files
        kb.load_all()
        assert kb.learned_rules == initial_rules

        # Spy on the _do_load method
        do_load_spy = mocker.spy(kb, '_do_load')

        # Second load without changes - should skip I/O
        kb.load_all()

        # _do_load should NOT have been called
        do_load_spy.assert_not_called()

    def test_load_all_reloads_when_file_modified(self, config_with_temp_files):
        """Test that load_all reloads when files are modified."""
        import time

        # Create initial rules file
        initial_rules = {"rules": [{"id": "initial_rule"}]}
        with open(config_with_temp_files.learned_rules_file, "w", encoding="utf-8") as f:
            json.dump(initial_rules, f)

        kb = KnowledgeBase(config_with_temp_files)
        kb.load_all()
        assert kb.learned_rules["rules"][0]["id"] == "initial_rule"

        # Wait a moment to ensure mtime changes
        time.sleep(0.1)

        # Modify the file
        updated_rules = {"rules": [{"id": "updated_rule"}]}
        with open(config_with_temp_files.learned_rules_file, "w", encoding="utf-8") as f:
            json.dump(updated_rules, f)

        # Should reload and pick up changes
        kb.load_all()
        assert kb.learned_rules["rules"][0]["id"] == "updated_rule"

    def test_force_reload_always_reloads(self, config_with_temp_files, mocker):
        """Test that force_reload always reads files regardless of mtime."""
        initial_rules = {"rules": [{"id": "test_rule"}]}
        with open(config_with_temp_files.learned_rules_file, "w", encoding="utf-8") as f:
            json.dump(initial_rules, f)

        kb = KnowledgeBase(config_with_temp_files)

        # First load
        kb.force_reload()

        # Spy on _do_load
        do_load_spy = mocker.spy(kb, '_do_load')

        # force_reload should always call _do_load even when files unchanged
        kb.force_reload()
        do_load_spy.assert_called_once()

    def test_mtime_tracking_initialized_empty(self, config):
        """Test that mtime tracking starts empty."""
        kb = KnowledgeBase(config)
        assert kb._file_mtimes == {}

    def test_mtime_updated_after_load(self, config_with_temp_files):
        """Test that mtimes are updated after loading."""
        # Create a rules file
        with open(config_with_temp_files.learned_rules_file, "w", encoding="utf-8") as f:
            json.dump({"rules": []}, f)

        kb = KnowledgeBase(config_with_temp_files)
        assert kb._file_mtimes == {}

        kb.load_all()

        # Should have recorded mtime for learned_rules_file
        assert config_with_temp_files.learned_rules_file in kb._file_mtimes
        assert kb._file_mtimes[config_with_temp_files.learned_rules_file] > 0

    def test_get_file_mtime_returns_zero_for_missing_file(self, config):
        """Test that _get_file_mtime returns 0 for non-existent files."""
        kb = KnowledgeBase(config)
        mtime = kb._get_file_mtime("/nonexistent/path/file.json")
        assert mtime == 0.0

    def test_needs_reload_true_on_first_call(self, config_with_temp_files):
        """Test that _needs_reload returns True on first call when files exist."""
        # Create at least one file so mtime differs from cached 0
        with open(config_with_temp_files.learned_rules_file, "w", encoding="utf-8") as f:
            json.dump({"rules": []}, f)

        kb = KnowledgeBase(config_with_temp_files)
        # Files exist but no cached mtimes, so should need reload
        assert kb._needs_reload() is True

    def test_needs_reload_false_after_load(self, config_with_temp_files):
        """Test that _needs_reload returns False immediately after load."""
        # Create files
        with open(config_with_temp_files.learned_rules_file, "w", encoding="utf-8") as f:
            json.dump({"rules": []}, f)

        kb = KnowledgeBase(config_with_temp_files)
        kb.load_all()

        # Should return False since nothing changed
        assert kb._needs_reload() is False


