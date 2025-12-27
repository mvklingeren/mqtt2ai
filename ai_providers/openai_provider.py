# ai_providers/openai_provider.py
"""OpenAI-compatible AI provider for the MQTT AI Daemon."""
import json
import logging
import time
import os
import hashlib
from datetime import datetime
from typing import TYPE_CHECKING, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

if TYPE_CHECKING:
    from config import Config
    from ai_agent import AiAgent # For execute_tool_call

import httpx # Import httpx here for C0415
from tool_definitions import OPENAI_TOOLS, OPENAI_TOOLS_MINIMAL
from ai_providers.base_provider import AiProvider

class OpenAiProvider(AiProvider):
    """Handles interaction with OpenAI-compatible AI providers."""

    def __init__(self, config: 'Config', ai_agent: 'AiAgent'):
        super().__init__(config, ai_agent)
        self.client = None
        if OPENAI_AVAILABLE:
            self.client = OpenAI(
                base_url=self.config.openai_api_base,
                api_key=self.config.openai_api_key,
                timeout=httpx.Timeout(120.0, connect=10.0),
            )

    def _call_with_retry(self, client, model: str, messages: list,
                          tools: list, extra_body: dict = None,
                          max_retries: int = 2, base_delay: float = 1.0):
        """Call the API with retry logic for transient errors."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    extra_body=extra_body,
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                last_error = e
                err_str = str(e)

                is_retryable = any(code in err_str for code in [
                    "400", "429", "500", "502", "503", "504",
                    "rate limit", "timeout", "temporarily"
                ])

                if attempt < max_retries and is_retryable:
                    delay = base_delay * (2 ** attempt)
                    logging.warning(
                        "API call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1, max_retries + 1, err_str[:100], delay
                    )
                    time.sleep(delay)
                else:
                    raise
        raise last_error

    def execute_call(self, prompt: str, rules_count: int = 0, patterns_count: int = 0):
        """Execute AI call via OpenAI-compatible API with function calling."""
        if not OPENAI_AVAILABLE or not self.client:
            logging.error("OpenAI SDK not installed or client not initialized.")
            return

        try:
            is_ollama = "11434" in self.config.openai_api_base
            system_content = (
                "/no_think\n" if is_ollama else ""
            ) + (
                "You are a home automation AI assistant with access to MQTT tools. "
                "Be brief and concise. "
                "Analyze MQTT messages and take actions using the available functions. "
                "When you detect trigger->action patterns, call record_pattern_observation. "
                "After 3+ observations, call create_rule to formalize the automation. "
                "Use send_mqtt_message to control devices directly when needed."
            )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]

            cyan, reset = "\033[96m", "\033[0m"
            purple = "\033[95m"
            orange_bold = "\033[1;38;5;208m"

            max_iterations = 10
            use_tools = True
            extra_body = {"options": {"num_predict": 200}} if is_ollama else None
            is_groq = "groq.com" in self.config.openai_api_base
            tools_to_use = OPENAI_TOOLS_MINIMAL if (is_ollama or is_groq) else OPENAI_TOOLS

            if self.config.demo_mode:
                tools_to_use = [
                    t for t in tools_to_use if t["function"]["name"] == "send_mqtt_message"
                ]

            current_model = self.config.get_next_model()
            prompt_chars = len(prompt)
            prompt_display = (
                f"{prompt_chars/1000:.1f}K" if prompt_chars >= 1000 else str(prompt_chars)
            )
            est_tokens = prompt_chars // 4

            logging.info(
                "%s[AI Request] model=%s | prompt=%s chars (~%d tok) | "
                "rules=%d patterns=%d | tools=%d%s",
                orange_bold, current_model, prompt_display, est_tokens,
                rules_count, patterns_count, len(tools_to_use), reset
            )

            total_prompt_tokens = 0
            total_completion_tokens = 0

            for iteration in range(max_iterations):
                try:
                    if use_tools:
                        response = self._call_with_retry(
                            self.client, current_model, messages, tools_to_use, extra_body
                        )
                    else:
                        response = self.client.chat.completions.create(
                            model=current_model,
                            messages=messages,
                            extra_body=extra_body,
                        )
                except Exception as api_err:
                    err_str = str(api_err)
                    logging.warning("API error: %s", err_str)
                    if "tool" in err_str.lower() and use_tools:
                        logging.warning("Falling back to text-only mode")
                        use_tools = False
                        response = self.client.chat.completions.create(
                            model=current_model,
                            messages=messages,
                            extra_body=extra_body,
                        )
                    else:
                        raise

                message = response.choices[0].message

                if self.config.debug_output:
                    # (Simplified debug output logic for brevity)
                    pass

                if response.usage:
                    total_prompt_tokens += response.usage.prompt_tokens or 0
                    total_completion_tokens += response.usage.completion_tokens or 0
                    logging.info(
                        "%s[AI Tokens] iter %d/%d | %d+%d=%d total%s",
                        orange_bold, iteration + 1, max_iterations,
                        response.usage.prompt_tokens or 0,
                        response.usage.completion_tokens or 0,
                        response.usage.total_tokens or 0, reset
                    )

                if message.tool_calls:
                    assistant_message_to_add = {"role": message.role}
                    if message.content:
                        assistant_message_to_add["content"] = message.content
                    if message.tool_calls:
                        assistant_message_to_add["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ]




                    for tool_call in message.tool_calls:
                        func_name = tool_call.function.name
                        try:
                            func_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            func_args = {}

                        logging.info(
                            "%s[Tool Call] %s(%s)%s",
                            purple, func_name, json.dumps(func_args), reset
                        )
                        
                        result = self.ai_agent.execute_tool_call(func_name, func_args)
                        logging.info("%s[Tool Result] %s%s", purple, result, reset)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })



                    if self.config.demo_mode:
                        # Demo mode logic...
                        break

                    # Context compression logic...
                    if iteration == 0 and len(messages) > 2:
                        messages[1] = {
                            "role": "user",
                            "content": (
                                "[MQTT analysis complete - context above. "
                                "Continue with tool results.]"
                            )
                        }
                else:
                    if message.content:
                        logging.info("%sAI Response: %s%s", cyan, message.content.strip(), reset)
                    break
            else:
                logging.warning("Max tool call iterations reached")

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("OpenAI API error: %s", e)

    def get_alert_tool_declarations(self) -> list:
        """Get tool declarations for alert handling (send_mqtt_message only)."""
        return [OPENAI_TOOLS[0]] # send_mqtt_message only

    def test_connection(self) -> tuple[bool, str]:
        """Test connection to OpenAI-compatible API."""
        if not OPENAI_AVAILABLE or not self.client:
            return False, "OpenAI SDK not installed or client not initialized."

        try:
            current_model = self.config.get_next_model()
            messages = [{"role": "user", "content": "Write a short joke."}]

            response = self.client.chat.completions.create(
                model=current_model,
                messages=messages,
            )
            joke = response.choices[0].message.content.strip()
            return True, f"Connected to {current_model}\nJoke: {joke}"

        except Exception as e:
            return False, f"OpenAI API error: {e}"

    def execute_call_for_alert(self, prompt: str, alert_tools: list) -> None:
        """Execute a simplified AI call for alert responses (send_mqtt_message only)."""
        if not OPENAI_AVAILABLE or not self.client:
            logging.error("OpenAI SDK not available for alert AI call")
            return

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a security response AI. Your job is to respond to alerts "
                        "by activating appropriate devices. Be decisive and act immediately. "
                        "Use send_mqtt_message to control devices."
                    )
                },
                {"role": "user", "content": prompt}
            ]

            current_model = self.config.get_next_model()
            cyan, reset = "\033[96m", "\033[0m"
            purple = "\033[95m"

            response = self.client.chat.completions.create(
                model=current_model,
                messages=messages,
                tools=alert_tools,
                tool_choice="auto",
            )

            message = response.choices[0].message

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        func_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        func_args = {}

                    logging.info(
                        "%s[Alert Tool] %s(%s)%s",
                        purple, func_name, json.dumps(func_args), reset
                    )

                    result = self.ai_agent.execute_tool_call(func_name, func_args)
                    logging.info("%s[Alert Result] %s%s", purple, result, reset)

            if message.content:
                logging.info("%sAlert AI: %s%s", cyan, message.content.strip(), reset)

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Alert AI call failed: %s", e)