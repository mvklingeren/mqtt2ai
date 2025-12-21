"""Knowledge Base module for the MQTT AI Daemon.

This module manages loading and reloading of learned rules, pending patterns,
rejected patterns, and the rulebook content from disk.
"""
import logging
from typing import Any, Dict

from config import Config
from utils import load_json_file


class KnowledgeBase:  # pylint: disable=too-few-public-methods
    """Manages loading and reloading of rules and patterns."""

    def __init__(self, config: Config):
        self.config = config
        self.learned_rules: Dict[str, Any] = {"rules": []}
        self.pending_patterns: Dict[str, Any] = {"patterns": []}
        self.rejected_patterns: Dict[str, Any] = {"patterns": []}
        self.rulebook_content: str = ""

    def load_all(self):
        """Reload all configuration files from disk."""
        self.learned_rules = load_json_file(
            self.config.learned_rules_file, {"rules": []}
        )
        self.pending_patterns = load_json_file(
            self.config.pending_patterns_file, {"patterns": []}
        )
        self.rejected_patterns = load_json_file(
            self.config.rejected_patterns_file, {"patterns": []}
        )

        try:
            with open(self.config.rulebook_file, "r", encoding="utf-8") as f:
                self.rulebook_content = f.read()
        except FileNotFoundError:
            logging.error(
                "Rulebook file '%s' not found.", self.config.rulebook_file
            )
            # Don't exit, just continue with empty rulebook
            self.rulebook_content = ""

        logging.debug(
            "Loaded %d rules, %d patterns.",
            len(self.learned_rules.get('rules', [])),
            len(self.pending_patterns.get('patterns', []))
        )
