"""Tests for the Config module."""
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_mqtt_host(self):
        """Test default MQTT host value."""
        config = Config()
        assert config.mqtt_host == "192.168.1.245"

    def test_default_mqtt_port(self):
        """Test default MQTT port value."""
        config = Config()
        assert config.mqtt_port == "1883"

    def test_default_mqtt_topics(self):
        """Test default MQTT topics value."""
        config = Config()
        assert config.mqtt_topics == ["zigbee2mqtt/#", "jokes/#", "mqtt2ai/#"]

    def test_default_max_messages(self):
        """Test default max messages value."""
        config = Config()
        assert config.max_messages == 500

    def test_default_ai_check_interval(self):
        """Test default AI check interval (5 minutes)."""
        config = Config()
        assert config.ai_check_interval == 300

    def test_default_ai_check_threshold(self):
        """Test default AI check threshold (200 messages)."""
        config = Config()
        assert config.ai_check_threshold == 200

    def test_default_ai_provider(self):
        """Test default AI provider is codex-openai."""
        config = Config()
        assert config.ai_provider == "codex-openai"

    def test_default_verbose_is_false(self):
        """Test verbose mode is disabled by default."""
        config = Config()
        assert config.verbose is False

    def test_default_demo_mode_is_false(self):
        """Test demo mode is disabled by default."""
        config = Config()
        assert config.demo_mode is False

    def test_default_no_ai_is_false(self):
        """Test no-AI mode is disabled by default."""
        config = Config()
        assert config.no_ai is False

    def test_default_test_ai_is_false(self):
        """Test test-AI mode is disabled by default."""
        config = Config()
        assert config.test_ai is False

    def test_default_disable_new_rules_is_false(self):
        """Test disable_new_rules is disabled by default."""
        config = Config()
        assert config.disable_new_rules is False


class TestConfigFileDefaults:
    """Test default file path configuration."""

    def test_default_rulebook_file(self):
        """Test default rulebook file path."""
        config = Config()
        assert config.rulebook_file == "rulebook.md"

    def test_default_filtered_triggers_file(self):
        """Test default filtered triggers file path."""
        config = Config()
        assert config.filtered_triggers_file == "filtered_triggers.json"

    def test_default_learned_rules_file(self):
        """Test default learned rules file path."""
        config = Config()
        assert config.learned_rules_file == "learned_rules.json"

    def test_default_pending_patterns_file(self):
        """Test default pending patterns file path."""
        config = Config()
        assert config.pending_patterns_file == "pending_patterns.json"

    def test_default_rejected_patterns_file(self):
        """Test default rejected patterns file path."""
        config = Config()
        assert config.rejected_patterns_file == "rejected_patterns.json"


class TestConfigAIProviders:
    """Test AI provider configuration."""

    def test_gemini_model_default(self):
        """Test default Gemini model."""
        config = Config()
        assert config.gemini_model == "gemini-2.5-flash"

    def test_claude_model_default(self):
        """Test default Claude model."""
        config = Config()
        assert config.claude_model == "claude-3-5-haiku-latest"

    def test_codex_model_default(self):
        """Test default Codex model."""
        config = Config()
        assert config.codex_model == "gpt-4o-mini"

    def test_claude_mcp_config_default_empty(self):
        """Test Claude MCP config is empty by default."""
        config = Config()
        assert config.claude_mcp_config == ""


class TestConfigIgnoreLists:
    """Test ignore lists configuration."""

    def test_default_ignore_printing_topics(self):
        """Test default ignored topics for printing."""
        config = Config()
        assert "zigbee2mqtt/bridge/logging" in config.ignore_printing_topics
        assert "zigbee2mqtt/bridge/health" in config.ignore_printing_topics

    def test_default_ignore_printing_prefixes_empty(self):
        """Test default ignored prefixes is empty."""
        config = Config()
        assert config.ignore_printing_prefixes == []

    def test_skip_printing_seconds_default(self):
        """Test default skip printing seconds."""
        config = Config()
        assert config.skip_printing_seconds == 3


class TestConfigFromArgs:
    """Test Config.from_args() method."""

    def test_from_args_with_defaults(self):
        """Test from_args with no arguments uses defaults."""
        with patch("sys.argv", ["test"]):
            config = Config.from_args()
            assert config.mqtt_host == "192.168.1.245"
            assert config.verbose is False

    def test_from_args_with_mqtt_host(self):
        """Test from_args with custom MQTT host."""
        with patch("sys.argv", ["test", "--mqtt-host", "10.0.0.1"]):
            config = Config.from_args()
            assert config.mqtt_host == "10.0.0.1"

    def test_from_args_with_mqtt_port(self):
        """Test from_args with custom MQTT port."""
        with patch("sys.argv", ["test", "--mqtt-port", "1884"]):
            config = Config.from_args()
            assert config.mqtt_port == "1884"

    def test_from_args_with_verbose(self):
        """Test from_args with verbose flag."""
        with patch("sys.argv", ["test", "--verbose"]):
            config = Config.from_args()
            assert config.verbose is True

    def test_from_args_with_short_verbose(self):
        """Test from_args with -v flag."""
        with patch("sys.argv", ["test", "-v"]):
            config = Config.from_args()
            assert config.verbose is True

    def test_from_args_with_demo(self):
        """Test from_args with demo flag."""
        with patch("sys.argv", ["test", "--demo"]):
            config = Config.from_args()
            assert config.demo_mode is True

    def test_from_args_with_no_ai(self):
        """Test from_args with no-ai flag."""
        with patch("sys.argv", ["test", "--no-ai"]):
            config = Config.from_args()
            assert config.no_ai is True

    def test_from_args_with_test_ai(self):
        """Test from_args with test-ai flag."""
        with patch("sys.argv", ["test", "--test-ai"]):
            config = Config.from_args()
            assert config.test_ai is True

    def test_from_args_with_disable_new_rules(self):
        """Test from_args with disable-new-rules flag."""
        with patch("sys.argv", ["test", "--disable-new-rules"]):
            config = Config.from_args()
            assert config.disable_new_rules is True

    def test_from_args_with_ai_provider_claude(self):
        """Test from_args with Claude AI provider."""
        with patch("sys.argv", ["test", "--ai-provider", "claude"]):
            config = Config.from_args()
            assert config.ai_provider == "claude"

    def test_from_args_with_ai_provider_codex(self):
        """Test from_args with Codex AI provider."""
        with patch("sys.argv", ["test", "--ai-provider", "codex-openai"]):
            config = Config.from_args()
            assert config.ai_provider == "codex-openai"

    def test_from_args_with_gemini_model(self):
        """Test from_args with custom Gemini model."""
        with patch("sys.argv", ["test", "--gemini-model", "gemini-pro"]):
            config = Config.from_args()
            assert config.gemini_model == "gemini-pro"

    def test_from_args_with_claude_model(self):
        """Test from_args with custom Claude model."""
        with patch("sys.argv", ["test", "--claude-model", "claude-3-opus"]):
            config = Config.from_args()
            assert config.claude_model == "claude-3-opus"

    def test_from_args_with_codex_model(self):
        """Test from_args with custom Codex model."""
        with patch("sys.argv", ["test", "--codex-model", "gpt-4"]):
            config = Config.from_args()
            assert config.codex_model == "gpt-4"

    def test_from_args_with_claude_mcp_config(self):
        """Test from_args with Claude MCP config path."""
        with patch("sys.argv", ["test", "--claude-mcp-config", "/path/to/mcp.json"]):
            config = Config.from_args()
            assert config.claude_mcp_config == "/path/to/mcp.json"


class TestConfigEnvironmentVariables:
    """Test configuration from environment variables."""

    def test_mqtt_host_from_env(self):
        """Test MQTT host from environment variable."""
        with patch.dict(os.environ, {"MQTT_HOST": "mqtt.example.com"}):
            with patch("sys.argv", ["test"]):
                config = Config.from_args()
                assert config.mqtt_host == "mqtt.example.com"

    def test_mqtt_port_from_env(self):
        """Test MQTT port from environment variable."""
        with patch.dict(os.environ, {"MQTT_PORT": "8883"}):
            with patch("sys.argv", ["test"]):
                config = Config.from_args()
                assert config.mqtt_port == "8883"

    def test_ai_provider_from_env(self):
        """Test AI provider from environment variable."""
        with patch.dict(os.environ, {"AI_PROVIDER": "claude"}):
            with patch("sys.argv", ["test"]):
                config = Config.from_args()
                assert config.ai_provider == "claude"

    def test_gemini_command_from_env(self):
        """Test Gemini command from environment variable."""
        with patch.dict(os.environ, {"GEMINI_CLI_COMMAND": "/usr/local/bin/gemini"}):
            with patch("sys.argv", ["test"]):
                config = Config.from_args()
                assert config.gemini_command == "/usr/local/bin/gemini"

    def test_claude_command_from_env(self):
        """Test Claude command from environment variable."""
        with patch.dict(os.environ, {"CLAUDE_CLI_COMMAND": "/usr/bin/claude"}):
            with patch("sys.argv", ["test"]):
                config = Config.from_args()
                assert config.claude_command == "/usr/bin/claude"

    def test_codex_command_from_env(self):
        """Test Codex command from environment variable."""
        with patch.dict(os.environ, {"CODEX_CLI_COMMAND": "/usr/bin/codex"}):
            with patch("sys.argv", ["test"]):
                config = Config.from_args()
                assert config.codex_command == "/usr/bin/codex"

    def test_claude_mcp_config_from_env(self):
        """Test Claude MCP config from environment variable."""
        with patch.dict(os.environ, {"CLAUDE_MCP_CONFIG": "/etc/mcp/config.json"}):
            with patch("sys.argv", ["test"]):
                config = Config.from_args()
                assert config.claude_mcp_config == "/etc/mcp/config.json"

    def test_cli_args_override_env_vars(self):
        """Test that CLI arguments override environment variables."""
        with patch.dict(os.environ, {"MQTT_HOST": "env.example.com"}):
            with patch("sys.argv", ["test", "--mqtt-host", "cli.example.com"]):
                config = Config.from_args()
                assert config.mqtt_host == "cli.example.com"


class TestConfigDataclass:
    """Test Config dataclass features."""

    def test_config_is_mutable(self):
        """Test that config values can be modified."""
        config = Config()
        config.mqtt_host = "new.host.com"
        assert config.mqtt_host == "new.host.com"

    def test_config_ignore_lists_are_independent(self):
        """Test that ignore lists are independent between instances."""
        config1 = Config()
        config2 = Config()
        config1.ignore_printing_topics.append("new/topic")
        # Due to default_factory, lists should be independent
        assert "new/topic" in config1.ignore_printing_topics
        assert "new/topic" not in config2.ignore_printing_topics

