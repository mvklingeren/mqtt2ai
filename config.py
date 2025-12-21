"""Configuration module for the MQTT AI Daemon.

This module provides the Config dataclass which holds all configuration
settings for the daemon, including MQTT connection details, AI provider
settings, and filtering options.
"""
import argparse
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:  # pylint: disable=too-many-instance-attributes
    """Configuration settings for the MQTT AI Daemon."""

    mqtt_host: str = "192.168.1.245"
    mqtt_port: str = "1883"
    mqtt_topics: List[str] = field(
        default_factory=lambda: [
            "zigbee2mqtt/#",
            "jokes/#"
        ]
    )
    max_messages: int = 500  # Keep last 500 messages in buffer
    ai_check_interval: int = 300  # 5 minutes
    ai_check_threshold: int = 200  # Trigger AI after 200 messages

    # Files
    rulebook_file: str = "rulebook.md"
    filtered_triggers_file: str = "filtered_triggers.json"
    learned_rules_file: str = "learned_rules.json"
    pending_patterns_file: str = "pending_patterns.json"
    rejected_patterns_file: str = "rejected_patterns.json"

    # AI Provider
    ai_provider: str = "codex-openai"  # "gemini", "claude", or "codex-openai"

    # Gemini
    gemini_command: str = "/opt/homebrew/bin/gemini"
    gemini_model: str = "gemini-2.5-flash"

    # Claude
    claude_command: str = "/Users/mvklingeren/.nvm/versions/node/v20.15.0/bin/claude"
    claude_model: str = "claude-3-5-haiku-latest"
    claude_mcp_config: str = ""  # Path to MCP config file for Claude

    # Codex (OpenAI)
    codex_command: str = "codex"  # Assumes npm global install puts it in PATH
    codex_model: str = "gpt-4.1-mini"

    # Filtering & Display
    verbose: bool = False
    compress_output: bool = False  # Compress MQTT payloads in console output
    demo_mode: bool = False
    no_ai: bool = False  # Run without making AI calls (logging only)
    test_ai: bool = False  # Test AI connection before starting daemon
    disable_new_rules: bool = False  # If True, new rules are disabled by default
    skip_printing_seconds: int = 3
    ignore_printing_topics: List[str] = field(
        default_factory=lambda: [
            "zigbee2mqtt/bridge/logging",
            "zigbee2mqtt/bridge/health"
        ]
    )
    ignore_printing_prefixes: List[str] = field(
        default_factory=lambda: [
            "stat/",  # Tasmota logging/status messages
        ]
    )

    @classmethod
    def from_args(cls) -> 'Config':
        """Parse command-line arguments and return a Config instance."""
        parser = argparse.ArgumentParser(
            description="MQTT AI Daemon - Smart home automation with AI"
        )

        parser.add_argument(
            "--mqtt-host",
            default=os.environ.get("MQTT_HOST", "192.168.1.245"),
            help="MQTT Broker Host"
        )
        parser.add_argument(
            "--mqtt-port",
            default=os.environ.get("MQTT_PORT", "1883"),
            help="MQTT Broker Port"
        )

        # AI Provider
        parser.add_argument(
            "--ai-provider",
            choices=["gemini", "claude", "codex-openai"],
            default=os.environ.get("AI_PROVIDER", "codex-openai"),
            help="AI provider to use (gemini, claude, or codex-openai)"
        )

        # Gemini CLI
        parser.add_argument(
            "--gemini-command",
            default=os.environ.get("GEMINI_CLI_COMMAND", "/opt/homebrew/bin/gemini"),
            help="Path to Gemini CLI"
        )
        parser.add_argument(
            "--gemini-model",
            default="gemini-2.5-flash",
            help="Gemini Model ID"
        )

        # Claude CLI
        claude_default = "/Users/mvklingeren/.nvm/versions/node/v20.15.0/bin/claude"
        parser.add_argument(
            "--claude-command",
            default=os.environ.get("CLAUDE_CLI_COMMAND", claude_default),
            help="Path to Claude CLI"
        )
        parser.add_argument(
            "--claude-model",
            default="claude-3-5-haiku-latest",
            help="Claude Model ID"
        )
        parser.add_argument(
            "--claude-mcp-config",
            default=os.environ.get("CLAUDE_MCP_CONFIG", ""),
            help="Path to MCP config file for Claude"
        )

        # Codex CLI (OpenAI)
        parser.add_argument(
            "--codex-command",
            default=os.environ.get("CODEX_CLI_COMMAND", "codex"),
            help="Path to Codex CLI"
        )
        parser.add_argument(
            "--codex-model",
            default="gpt-4.1-mini",
            help="Codex/OpenAI Model ID"
        )

        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose logging"
        )
        parser.add_argument(
            "--compress", "-c",
            action="store_true",
            help="Compress MQTT payloads in console output (removes noise fields)"
        )
        parser.add_argument(
            "--demo",
            action="store_true",
            help="Enable demo mode"
        )
        parser.add_argument(
            "--no-ai",
            action="store_true",
            help="Disable AI calls (logging only mode)"
        )
        parser.add_argument(
            "--test-ai",
            action="store_true",
            help="Test AI connection before starting daemon"
        )
        parser.add_argument(
            "--disable-new-rules",
            action="store_true",
            help="New rules are disabled by default"
        )

        args = parser.parse_args()

        c = cls()
        c.mqtt_host = args.mqtt_host
        c.mqtt_port = args.mqtt_port
        c.ai_provider = args.ai_provider
        c.gemini_command = args.gemini_command
        c.gemini_model = args.gemini_model
        c.claude_command = args.claude_command
        c.claude_model = args.claude_model
        c.claude_mcp_config = args.claude_mcp_config
        c.codex_command = args.codex_command
        c.codex_model = args.codex_model
        c.verbose = args.verbose
        c.compress_output = args.compress
        c.demo_mode = args.demo
        c.no_ai = args.no_ai
        c.test_ai = args.test_ai
        c.disable_new_rules = args.disable_new_rules
        return c
