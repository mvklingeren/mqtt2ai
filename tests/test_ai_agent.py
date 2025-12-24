"""Tests for the AiAgent module."""
import json
import os
import subprocess
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_agent import AiAgent, timestamp
from config import Config
from knowledge_base import KnowledgeBase
from prompt_builder import PromptBuilder


class TestTimestamp:
    """Tests for the timestamp function."""

    def test_timestamp_format(self):
        """Test that timestamp returns correct format."""
        ts = timestamp()
        assert ts.startswith("[")
        assert ts.endswith("]")
        # Format should be [HH:MM:SS]
        assert len(ts) == 10  # [XX:XX:XX]

    def test_timestamp_contains_colons(self):
        """Test that timestamp contains time separators."""
        ts = timestamp()
        # Remove brackets and check format
        time_part = ts[1:-1]
        parts = time_part.split(":")
        assert len(parts) == 3


class TestAiAgentInit:
    """Tests for AiAgent initialization."""

    def test_init_with_config(self, config):
        """Test initialization with a config."""
        agent = AiAgent(config)
        assert agent.config == config

    def test_init_preserves_config_values(self):
        """Test that init preserves all config values."""
        config = Config()
        config.ai_provider = "claude"
        config.claude_model = "test-model"

        agent = AiAgent(config)

        assert agent.config.ai_provider == "claude"
        assert agent.config.claude_model == "test-model"


class TestAiAgentGetCliCommand:
    """Tests for _get_cli_command method."""

    def test_gemini_cli_command(self, config):
        """Test Gemini CLI command generation."""
        config.ai_provider = "gemini"
        config.gemini_command = "/path/to/gemini"
        config.gemini_model = "gemini-pro"

        agent = AiAgent(config)
        cmd, args = agent._get_cli_command()

        assert cmd == "/path/to/gemini"
        assert "--yolo" in args
        assert "--model" in args
        assert "gemini-pro" in args

    def test_claude_cli_command(self, config):
        """Test Claude CLI command generation."""
        config.ai_provider = "claude"
        config.claude_command = "/path/to/claude"
        config.claude_model = "claude-3"

        agent = AiAgent(config)
        cmd, args = agent._get_cli_command()

        assert cmd == "/path/to/claude"
        assert "--dangerously-skip-permissions" in args
        assert "--model" in args
        assert "claude-3" in args

    def test_claude_cli_with_mcp_config(self, config):
        """Test Claude CLI with MCP config."""
        config.ai_provider = "claude"
        config.claude_command = "/path/to/claude"
        config.claude_mcp_config = "/path/to/mcp.json"

        agent = AiAgent(config)
        cmd, args = agent._get_cli_command()

        assert "--mcp-config" in args
        assert "/path/to/mcp.json" in args

    def test_claude_cli_without_mcp_config(self, config):
        """Test Claude CLI without MCP config."""
        config.ai_provider = "claude"
        config.claude_command = "/path/to/claude"
        config.claude_mcp_config = ""

        agent = AiAgent(config)
        cmd, args = agent._get_cli_command()

        assert "--mcp-config" not in args

    def test_codex_cli_command(self, config):
        """Test Codex CLI command generation."""
        config.ai_provider = "codex-openai"
        config.codex_command = "/path/to/codex"
        config.codex_model = "gpt-4"

        agent = AiAgent(config)
        cmd, args = agent._get_cli_command()

        assert cmd == "/path/to/codex"
        assert "exec" in args
        assert "--model" in args
        assert "gpt-4" in args
        assert "--full-auto" in args

    def test_default_provider_uses_gemini(self, config):
        """Test that gemini provider works correctly."""
        config.ai_provider = "gemini"
        agent = AiAgent(config)
        cmd, args = agent._get_cli_command()

        assert "--yolo" in args  # Gemini-specific flag


class TestAiAgentPromptBuilder:
    """Tests for prompt building via PromptBuilder.
    
    Note: The actual _build_prompt method has been moved to PromptBuilder class.
    These tests verify that AiAgent correctly uses the PromptBuilder.
    """

    def test_agent_has_prompt_builder(self, config_with_temp_files):
        """Test that agent has a prompt_builder instance."""
        agent = AiAgent(config_with_temp_files)
        assert hasattr(agent, "prompt_builder")
        assert isinstance(agent.prompt_builder, PromptBuilder)

    def test_prompt_builder_uses_config(self, config_with_temp_files):
        """Test that prompt_builder is initialized with config."""
        agent = AiAgent(config_with_temp_files)
        assert agent.prompt_builder.config == config_with_temp_files

    def test_prompt_builder_demo_mode(self, config_with_temp_files):
        """Test that demo mode is indicated in prompt."""
        config_with_temp_files.demo_mode = True
        agent = AiAgent(config_with_temp_files)
        kb = KnowledgeBase(config_with_temp_files)

        prompt = agent.prompt_builder.build("messages", kb, trigger_reason="test")

        assert "Demo mode" in prompt

    def test_prompt_builder_no_demo_mode(self, config_with_temp_files):
        """Test that demo mode is not mentioned when disabled."""
        config_with_temp_files.demo_mode = False
        agent = AiAgent(config_with_temp_files)
        kb = KnowledgeBase(config_with_temp_files)

        prompt = agent.prompt_builder.build("messages", kb, trigger_reason="test")

        assert "Demo mode" not in prompt

    def test_prompt_builder_contains_learned_rules(self, config_with_temp_files):
        """Test that prompt contains learned rules."""
        agent = AiAgent(config_with_temp_files)
        kb = KnowledgeBase(config_with_temp_files)
        kb.learned_rules = {
            "rules": [{
                "id": "test_rule",
                "trigger": {"topic": "zigbee2mqtt/test", "field": "occupancy", "value": True},
                "action": {"topic": "zigbee2mqtt/light/set"},
                "confidence": {"occurrences": 5},
                "enabled": True
            }]
        }

        prompt = agent.prompt_builder.build("messages", kb, trigger_reason="test")

        assert "Learned Rules" in prompt
        assert "test_rule" in prompt

    def test_prompt_builder_safety_alert_temperature(self, config_with_temp_files):
        """Test that safety alert is included for temperature triggers."""
        agent = AiAgent(config_with_temp_files)
        kb = KnowledgeBase(config_with_temp_files)

        prompt = agent.prompt_builder.build("messages", kb, trigger_reason="temperature spike")

        assert "SAFETY ALERT" in prompt

    def test_prompt_builder_safety_alert_smoke(self, config_with_temp_files):
        """Test that safety alert is included for smoke triggers."""
        agent = AiAgent(config_with_temp_files)
        kb = KnowledgeBase(config_with_temp_files)

        prompt = agent.prompt_builder.build("messages", kb, trigger_reason="smoke detected")

        assert "SAFETY ALERT" in prompt

    def test_prompt_builder_safety_alert_water(self, config_with_temp_files):
        """Test that safety alert is included for water/leak triggers."""
        agent = AiAgent(config_with_temp_files)
        kb = KnowledgeBase(config_with_temp_files)

        prompt = agent.prompt_builder.build("messages", kb, trigger_reason="water leak warning")

        assert "SAFETY ALERT" in prompt

    def test_prompt_builder_no_safety_alert_normal(self, config_with_temp_files):
        """Test that no safety alert for normal triggers."""
        agent = AiAgent(config_with_temp_files)
        kb = KnowledgeBase(config_with_temp_files)

        prompt = agent.prompt_builder.build("messages", kb, trigger_reason="interval check")

        assert "SAFETY ALERT" not in prompt


class TestAiAgentExecuteAiCall:
    """Tests for _execute_ai_call method."""

    def test_execute_ai_call_checks_cli_exists(self, config):
        """Test that CLI existence is checked."""
        config.ai_provider = "gemini"
        config.gemini_command = "/nonexistent/path"
        agent = AiAgent(config)

        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            agent._execute_ai_call("GEMINI", "test prompt")
            mock_which.assert_called_once_with("/nonexistent/path")

    def test_execute_ai_call_runs_subprocess(self, config):
        """Test that subprocess is called when CLI exists."""
        config.ai_provider = "gemini"
        agent = AiAgent(config)

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/path/to/cli"
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="response", returncode=0)
                agent._execute_ai_call("GEMINI", "test prompt")
                mock_run.assert_called_once()

    def test_execute_ai_call_handles_timeout(self, config):
        """Test that timeout is handled gracefully."""
        agent = AiAgent(config)

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/path/to/cli"
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired("cmd", 120)
                # Should not raise
                agent._execute_ai_call("GEMINI", "test prompt")

    def test_execute_ai_call_handles_subprocess_error(self, config):
        """Test that subprocess errors are handled gracefully."""
        agent = AiAgent(config)

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/path/to/cli"
            with patch("subprocess.run") as mock_run:
                error = subprocess.CalledProcessError(1, "cmd")
                error.stderr = "error message"
                mock_run.side_effect = error
                # Should not raise
                agent._execute_ai_call("GEMINI", "test prompt")


class TestAiAgentTestConnection:
    """Tests for test_connection method."""

    def test_test_connection_cli_success(self, config):
        """Test successful CLI connection test."""
        config.ai_provider = "gemini"  # Use CLI-based provider
        agent = AiAgent(config)

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/path/to/cli"
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout="Here's a joke about MQTT...",
                    returncode=0
                )
                success, message = agent.test_connection()

                assert success is True
                assert "joke" in message.lower()

    def test_test_connection_cli_not_found(self, config):
        """Test connection test when CLI not found."""
        config.ai_provider = "gemini"  # Use CLI-based provider
        config.gemini_command = "/nonexistent/cli"
        agent = AiAgent(config)

        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            success, message = agent.test_connection()

            assert success is False
            assert "not found" in message.lower()

    def test_test_connection_cli_timeout(self, config):
        """Test CLI connection test timeout."""
        config.ai_provider = "gemini"  # Use CLI-based provider
        agent = AiAgent(config)

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/path/to/cli"
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired("cmd", 60)
                success, message = agent.test_connection()

                assert success is False
                assert "timed out" in message.lower()

    def test_test_connection_cli_subprocess_error(self, config):
        """Test CLI connection test subprocess error."""
        config.ai_provider = "gemini"  # Use CLI-based provider
        agent = AiAgent(config)

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/path/to/cli"
            with patch("subprocess.run") as mock_run:
                error = subprocess.CalledProcessError(1, "cmd")
                error.stderr = "Authentication failed"
                mock_run.side_effect = error
                success, message = agent.test_connection()

                assert success is False
                assert "failed" in message.lower()


class TestAiAgentOpenAICompatible:
    """Tests for OpenAI-compatible API provider."""

    def test_test_connection_openai_compatible_success(self, config):
        """Test successful connection to OpenAI-compatible API."""
        config.ai_provider = "openai-compatible"
        config.openai_api_base = "http://192.168.1.52:11434/v1"
        config.openai_models = ["llama3.2"]
        agent = AiAgent(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Why did the AI cross the road?"

        mock_openai_class = MagicMock()
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai_class)}):
            with patch("ai_agent.OPENAI_AVAILABLE", True):
                with patch.object(agent, "_test_openai_connection") as mock_test:
                    mock_test.return_value = (True, f"Connected to http://192.168.1.52:11434/v1 using model llama3.2\n\nResponse: Why did the AI cross the road?")
                    success, message = agent.test_connection()

                    assert success is True
                    assert "192.168.1.52" in message
                    assert "llama3.2" in message

    def test_test_connection_openai_compatible_sdk_not_installed(self, config):
        """Test connection when OpenAI SDK not installed."""
        config.ai_provider = "openai-compatible"
        agent = AiAgent(config)

        with patch("ai_agent.OPENAI_AVAILABLE", False):
            success, message = agent.test_connection()

            assert success is False
            assert "not installed" in message.lower()

    def test_test_connection_openai_compatible_api_error(self, config):
        """Test connection when API returns error."""
        config.ai_provider = "openai-compatible"
        agent = AiAgent(config)

        with patch("ai_agent.OPENAI_AVAILABLE", True):
            with patch.object(agent, "_test_openai_connection") as mock_test:
                mock_test.return_value = (False, "OpenAI API error: Connection refused")
                success, message = agent.test_connection()

                assert success is False
                assert "error" in message.lower()

    def test_execute_openai_api_call_sdk_not_available(self, config):
        """Test OpenAI API call when SDK not available."""
        config.ai_provider = "openai-compatible"
        agent = AiAgent(config)

        with patch("ai_agent.OPENAI_AVAILABLE", False):
            # Should not raise, just log error
            agent._execute_openai_api_call("OPENAI-COMPATIBLE", "test prompt")


class TestAiAgentRunAnalysis:
    """Tests for run_analysis method."""

    def test_run_analysis_builds_prompt_and_executes(self, config):
        """Test that run_analysis builds prompt and executes."""
        config.ai_provider = "gemini"  # Use CLI-based provider that uses full prompt
        agent = AiAgent(config)
        kb = KnowledgeBase(config)
        kb.rulebook_content = "Test rulebook"

        with patch.object(agent.prompt_builder, "build") as mock_build:
            mock_build.return_value = "test prompt"
            with patch.object(agent, "_execute_ai_call") as mock_execute:
                agent.run_analysis("messages", kb, "test_reason")

                mock_build.assert_called_once()
                mock_execute.assert_called_once()

    def test_run_analysis_passes_correct_provider(self, config):
        """Test that correct provider is passed to execute."""
        config.ai_provider = "claude"
        agent = AiAgent(config)
        kb = KnowledgeBase(config)

        with patch.object(agent.prompt_builder, "build", return_value="prompt"):
            with patch.object(agent, "_execute_ai_call") as mock_execute:
                agent.run_analysis("messages", kb, "reason")

                # First argument should be provider in uppercase
                call_args = mock_execute.call_args[0]
                assert call_args[0] == "CLAUDE"
    
    def test_run_analysis_uses_compact_prompt_for_openai(self, config):
        """Test that openai-compatible uses compact prompt."""
        config.ai_provider = "openai-compatible"
        agent = AiAgent(config)
        kb = KnowledgeBase(config)

        with patch.object(agent.prompt_builder, "build_compact") as mock_compact:
            mock_compact.return_value = "compact prompt"
            with patch.object(agent, "_execute_ai_call"):
                agent.run_analysis("messages", kb, "reason")

                mock_compact.assert_called_once()

