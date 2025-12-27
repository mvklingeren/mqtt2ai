# ai_providers/gemini_provider.py
"""Gemini AI provider for the MQTT AI Daemon."""
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

if TYPE_CHECKING:
    from config import Config
    from ai_agent import AiAgent # For execute_tool_call

from tool_definitions import OPENAI_TOOLS
from ai_providers.base_provider import AiProvider

class GeminiProvider(AiProvider):
    """Handles interaction with Google Gemini AI provider."""

    def __init__(self, config: 'Config', ai_agent: 'AiAgent'):
        super().__init__(config, ai_agent)
        self.client = None
        if GEMINI_AVAILABLE:
            self.client = genai.Client(api_key=self.config.gemini_api_key)

    def _get_gemini_tool_declarations(self) -> list:
        """Get Gemini-format tool declarations."""
        gemini_tools = []
        for tool in OPENAI_TOOLS:
            func = tool["function"]
            gemini_tools.append(types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=func["name"],
                        description=func["description"],
                        parameters=func["parameters"]
                    )
                ]
            ))
        return gemini_tools

    def execute_call(self, prompt: str, rules_count: int = 0, patterns_count: int = 0):
        """Execute AI call via Google Gemini SDK."""
        if not GEMINI_AVAILABLE or not self.client:
            logging.error("Google GenAI SDK not installed or client not initialized.")
            return

        try:
            cyan, reset = "\033[96m", "\033[0m"
            purple = "\033[95m"
            orange_bold = "\033[1;38;5;208m"

            gemini_tools = self._get_gemini_tool_declarations()

            logging.info(
                "%s[AI Request] model=%s | prompt=%d chars | rules=%d patterns=%d%s",
                orange_bold, self.config.gemini_model, len(prompt),
                rules_count, patterns_count, reset
            )

            system_instruction = "You are a home automation AI assistant..."

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=gemini_tools,
            )

            max_iterations = 10
            contents = [prompt]

            for iteration in range(max_iterations):
                response = self.client.models.generate_content(
                    model=self.config.gemini_model,
                    contents=contents,
                    config=config,
                )

                if response.candidates and response.candidates[0].content.parts:
                    has_function_call = False
                    for part in response.candidates[0].content.parts:
                        if part.function_call:
                            has_function_call = True
                            func_name = part.function_call.name
                            func_args = (
                                dict(part.function_call.args)
                                if part.function_call.args else {}
                            )

                            logging.info(
                                "%s[Tool Call] %s(%s)%s",
                                purple, func_name, json.dumps(func_args), reset
                            )

                            result = self.ai_agent.execute_tool_call(func_name, func_args)
                            logging.info("%s[Tool Result] %s%s", purple, result, reset)

                            contents.append(response.candidates[0].content)
                            contents.append(
                                types.Content(
                                    role="user",
                                    parts=[types.Part.from_function_response(
                                        name=func_name,
                                        response={"result": result}
                                    )]
                                )
                            )

                    if not has_function_call:
                        if response.text:
                            logging.info("%sAI Response: %s%s", cyan, response.text.strip(), reset)
                        break
                else:
                    break
            else:
                logging.warning("Max tool call iterations reached")

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Gemini API error: %s", e)

    def get_alert_tool_declarations(self) -> list:
        """Get Gemini-format tool declarations for alert handling."""
        # For Gemini, we might need to convert OPENAI_TOOLS[0] to Gemini format
        # For now, just return the converted send_mqtt_message tool
        func = OPENAI_TOOLS[0]["function"]
        return [types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=func["name"],
                    description=func["description"],
                    parameters=func["parameters"]
                )
            ]
        )]

    def test_connection(self) -> tuple[bool, str]:
        """Test connection to Gemini API with function calling."""
        if not GEMINI_AVAILABLE or not self.client:
            return False, "Google GenAI SDK not installed or client not initialized."

        try:
            # Test basic completion
            response = self.client.models.generate_content(
                model=self.config.gemini_model,
                contents="Write a very short, funny one-liner joke.",
            )
            joke = response.text.strip() if response.text else "No response"

            return True, (
                f"Connected to Gemini API using model {self.config.gemini_model}\n\n"
                f"Joke: {joke}"
            )

        except Exception as e:
            return False, f"Gemini API error: {e}"

    def execute_call_for_alert(self, prompt: str, alert_tools: list) -> None:
        """Execute a simplified AI call for alert responses (send_mqtt_message only)."""
        if not GEMINI_AVAILABLE or not self.client:
            logging.error("Google GenAI SDK not available for alert AI call")
            return

        try:
            cyan, reset = "\033[96m", "\033[0m"
            purple = "\033[95m"

            system_instruction = "You are a security response AI. Your job is to respond to alerts by activating appropriate devices. Be decisive and act immediately. Use send_mqtt_message to control devices."
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=alert_tools,
            )
            contents = [prompt]

            response = self.client.models.generate_content(
                model=self.config.gemini_model,
                contents=contents,
                config=config,
            )

            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        func_name = part.function_call.name
                        func_args = (
                            dict(part.function_call.args)
                            if part.function_call.args else {}
                        )

                        logging.info(
                            "%s[Alert Tool] %s(%s)%s",
                            purple, func_name, json.dumps(func_args), reset
                        )

                        result = self.ai_agent.execute_tool_call(func_name, func_args)
                        logging.info("%s[Alert Result] %s%s", purple, result, reset)

                if response.text:
                    logging.info("%sAlert AI: %s%s", cyan, response.text.strip(), reset)

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Alert AI call failed: %s", e)