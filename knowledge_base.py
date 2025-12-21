import json
import logging
from typing import Dict, Any

from config import Config


class KnowledgeBase:
    """Manages loading and reloading of rules and patterns."""
    def __init__(self, config: Config):
        self.config = config
        self.learned_rules: Dict = {"rules": []}
        self.pending_patterns: Dict = {"patterns": []}
        self.rejected_patterns: Dict = {"patterns": []}
        self.rulebook_content: str = ""

    def load_all(self):
        """Reloads all configuration files from disk."""
        self.learned_rules = self._load_json(self.config.learned_rules_file, {"rules": []})
        self.pending_patterns = self._load_json(self.config.pending_patterns_file, {"patterns": []})
        self.rejected_patterns = self._load_json(self.config.rejected_patterns_file, {"patterns": []})
        
        try:
            with open(self.config.rulebook_file, "r") as f:
                self.rulebook_content = f.read()
        except FileNotFoundError:
            logging.error(f"Rulebook file '{self.config.rulebook_file}' not found.")
            # Don't exit, just continue with empty rulebook
            self.rulebook_content = ""
            
        logging.debug(f"Loaded {len(self.learned_rules.get('rules', []))} rules, {len(self.pending_patterns.get('patterns', []))} patterns.")

    def _load_json(self, filepath: str, default: Any) -> Any:
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default

