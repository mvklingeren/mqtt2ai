# ai_providers/claude_provider.py
"""Claude AI provider for the MQTT AI Daemon."""
import json
import logging
from typing import TYPE_CHECKING

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

if TYPE_CHECKING:
    from mqtt2ai.core.config import Config
    from mqtt2ai.ai.agent import AiAgent # For execute_tool_call

from mqtt2ai.ai.tool_definitions import OPENAI_TOOLS
from mqtt2ai.ai.providers.base import AiProvider

class ClaudeProvider(AiProvider):
    """Handles interaction with Anthropic Claude AI provider."""

    def __init__(self, config: 'Config', ai_agent: 'AiAgent'):
        super().__init__(config, ai_agent)
        self.client = None
        if ANTHROPIC_AVAILABLE:
            self.client = Anthropic(api_key=self.config.claude_api_key)

    def _get_claude_tool_definitions(self) -> list:
        """Get Claude-format tool definitions."""
        claude_tools = []
        for tool in OPENAI_TOOLS:
            func = tool["function"]
            claude_tools.append({
                "name": func["name"],
                "description": func["description"],
                "input_schema": func["parameters"]
            })
        return claude_tools

    def execute_call(self, prompt: str, rules_count: int = 0, patterns_count: int = 0):
        """Execute AI call via Anthropic Claude SDK."""
        if not ANTHROPIC_AVAILABLE or not self.client:
            logging.error("Anthropic SDK not installed or client not initialized.")
            return

        try:
            cyan, reset = "\033[96m", "\033[0m"
            purple = "\033[95m"
            orange_bold = "\033[1;38;5;208m"

            logging.info(
                "%s[AI Request] model=%s | prompt=%d chars | rules=%d patterns=%d%s",
                orange_bold, self.config.claude_model, len(prompt),
                rules_count, patterns_count, reset
            )

            system_prompt = "You are a home automation AI assistant..."
            claude_tools = self._get_claude_tool_definitions()
            messages = [{"role": "user", "content": prompt}]

            max_iterations = 10
            for iteration in range(max_iterations):
                response = self.client.messages.create(
                    model=self.config.claude_model,
                    max_tokens=1024,
                    system=system_prompt,
                    tools=claude_tools,
                    messages=messages,
                )

                has_tool_use = False
                tool_results = []

                for block in response.content:
                    if block.type == "tool_use":
                        has_tool_use = True
                        func_name = block.name
                        func_args = block.input if block.input else {}

                        logging.info(
                            "%s[Tool Call] %s(%s)%s",
                            purple, func_name, json.dumps(func_args), reset
                        )

                        result = self.ai_agent.execute_tool_call(func_name, func_args)
                        logging.info("%s[Tool Result] %s%s", purple, result, reset)

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })

                    elif block.type == "text" and block.text:
                        logging.info("%sAI Response: %s%s", cyan, block.text.strip(), reset)

                if has_tool_use:
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})
                else:
                    break

                if response.stop_reason == "end_turn":
                    break
            else:
                logging.warning("Max tool call iterations reached")

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Claude API error: %s", e)

    def get_alert_tool_declarations(self) -> list:
        """Get Claude-format tool definitions for alert handling."""
        # For Claude, we might need to convert OPENAI_TOOLS[0] to Claude format
        # For now, just return the converted send_mqtt_message tool
        func = OPENAI_TOOLS[0]["function"]
        return [{
            "name": func["name"],
            "description": func["description"],
            "input_schema": func["parameters"]
        }]

    def test_connection(self) -> tuple[bool, str]:
        """Test connection to Claude API with function calling."""
        if not ANTHROPIC_AVAILABLE or not self.client:
            return False, "Anthropic SDK not installed or client not initialized."

        try:
            # Test basic completion
            response = self.client.messages.create(
                model=self.config.claude_model,
                max_tokens=256,
                messages=[{"role": "user", "content": "Write a very short, funny one-liner joke."}]
            )
            joke = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    joke = block.text.strip()
                    break

            return True, (
                f"Connected to Claude API using model {self.config.claude_model}\n\n"
                f"Joke: {joke}"
            )

        except Exception as e:
            return False, f"Claude API error: {e}"

    def execute_call_for_alert(self, prompt: str, alert_tools: list) -> None:
        """Execute a simplified AI call for alert responses (send_mqtt_message only)."""
        if not ANTHROPIC_AVAILABLE or not self.client:
            logging.error("Anthropic SDK not available for alert AI call")
            return

        try:
            cyan, reset = "\033[96m", "\033[0m"
            purple = "\033[95m"

            system_prompt = "You are a security response AI. Your job is to respond to alerts by activating appropriate devices. Be decisive and act immediately. Use send_mqtt_message to control devices."
            messages = [{"role": "user", "content": prompt}]

            response = self.client.messages.create(
                model=self.config.claude_model,
                max_tokens=1024,
                system=system_prompt,
                tools=alert_tools,
                messages=messages,
            )

            for block in response.content:
                if block.type == "tool_use":
                    func_name = block.name
                    func_args = block.input if block.input else {}

                    logging.info(
                        "%s[Alert Tool] %s(%s)%s",
                        purple, func_name, json.dumps(func_args), reset
                    )

                    result = self.ai_agent.execute_tool_call(func_name, func_args)
                    logging.info("%s[Alert Result] %s%s", purple, result, reset)

                elif block.type == "text" and block.text:
                    logging.info("%sAlert AI: %s%s", cyan, block.text.strip(), reset)

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Alert AI call failed: %s", e)

    def execute_telegram_query(self, prompt: str) -> str:
        """Execute a synchronous AI query for Telegram interaction.

        Args:
            prompt: The prompt with user request and device context

        Returns:
            Text response for the user
        """
        if not ANTHROPIC_AVAILABLE or not self.client:
            return "❌ Anthropic SDK not available"

        try:
            cyan, reset = "\033[96m", "\033[0m"
            purple = "\033[95m"

            telegram_tools = self.get_alert_tool_declarations()

            system_prompt = (
                "You are a helpful home automation assistant responding via Telegram. "
                "Be concise (under 200 chars when possible). "
                "Use send_mqtt_message to control devices when the user requests actions. "
                "Confirm actions briefly, e.g., '✅ Living room light turned ON'"
            )

            messages = [{"role": "user", "content": prompt}]
            actions_taken = []
            final_response = ""

            # Allow up to 3 iterations for tool calls
            for _ in range(3):
                response = self.client.messages.create(
                    model=self.config.claude_model,
                    max_tokens=512,
                    system=system_prompt,
                    tools=telegram_tools,
                    messages=messages,
                )

                has_tool_use = False
                tool_results = []

                for block in response.content:
                    if block.type == "tool_use":
                        has_tool_use = True
                        func_name = block.name
                        func_args = block.input if block.input else {}

                        logging.info(
                            "%s[Telegram Tool] %s(%s)%s",
                            purple, func_name, json.dumps(func_args), reset
                        )

                        result = self.ai_agent.execute_tool_call(func_name, func_args)
                        actions_taken.append(f"{func_name}: {result}")

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })

                    elif block.type == "text" and block.text:
                        final_response = block.text.strip()

                if has_tool_use:
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})
                else:
                    break

                if response.stop_reason == "end_turn":
                    break

            # If we have actions but no final response, summarize
            if actions_taken and not final_response:
                final_response = "✅ " + "; ".join(actions_taken)

            if not final_response:
                final_response = "I processed your request but have nothing to report."

            logging.info("%sTelegram Response: %s%s", cyan, final_response[:100], reset)
            return final_response

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Telegram query failed: %s", e)
            return f"❌ Error: {e}"

