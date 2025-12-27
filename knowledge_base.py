"""Knowledge Base module for the MQTT AI Daemon.

This module manages loading and reloading of learned rules, pending patterns,
rejected patterns, and the rulebook content from disk.
"""
import logging
import os
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
        self._file_mtimes: Dict[str, float] = {}

    def _get_file_mtime(self, path: str) -> float:
        """Get the modification time of a file, or 0 if it doesn't exist."""
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    def _needs_reload(self) -> bool:
        """Check if any knowledge base files have been modified."""
        files = [
            self.config.learned_rules_file,
            self.config.pending_patterns_file,
            self.config.rejected_patterns_file,
            self.config.rulebook_file,
        ]
        for path in files:
            current_mtime = self._get_file_mtime(path)
            if current_mtime != self._file_mtimes.get(path, 0):
                return True
        return False

    def _update_mtimes(self):
        """Update stored modification times for all files."""
        files = [
            self.config.learned_rules_file,
            self.config.pending_patterns_file,
            self.config.rejected_patterns_file,
            self.config.rulebook_file,
        ]
        for path in files:
            self._file_mtimes[path] = self._get_file_mtime(path)

    def load_all(self):
        """Reload all configuration files from disk if they have changed.

        Uses modification time (mtime) tracking to skip file I/O when
        the underlying files haven't been modified since the last load.
        """
        if not self._needs_reload():
            logging.debug("Knowledge base files unchanged, skipping reload")
            return

        self._do_load()

    def force_reload(self):
        """Force reload all configuration files regardless of mtime."""
        self._do_load()

    def _do_load(self):
        """Actually load all configuration files from disk."""
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

        # Update mtimes after successful load
        self._update_mtimes()

        logging.debug(
            "Loaded %d rules, %d patterns.",
            len(self.learned_rules.get('rules', [])),
            len(self.pending_patterns.get('patterns', []))
        )
