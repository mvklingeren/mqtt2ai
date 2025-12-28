"""Telegram message handler for processing user commands via AI."""
import json
import logging
from typing import Optional

from ai_agent import AiAgent
from config import Config
from device_state_tracker import DeviceStateTracker


class TelegramHandler:
    """Handles Telegram message processing through AI."""
    
    def __init__(
        self,
        config: Config,
        ai_agent: AiAgent,
        device_tracker: DeviceStateTracker
    ):
        self.config = config
        self.ai = ai_agent
        self.device_tracker = device_tracker
        
    def handle_message(self, chat_id: int, message: str) -> str:
        """Process a Telegram message and return the response."""
        if self.config.no_ai:
            return (
                "AI is disabled in NO-AI mode. "
                "Restart without --no-ai to enable commands."
            )

        logging.info(
            "Processing Telegram request from %d: %s",
            chat_id, message[:50]
        )

        # Build context with device states for the AI
        device_states = {}
        if self.device_tracker:
            device_states = self.device_tracker.get_all_states()

        # Create a focused prompt for the user's request
        prompt = self._build_prompt(message, device_states)

        # Use the AI agent to process the request synchronously
        try:
            response = self.ai.process_telegram_query(prompt, message)
            return response
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Error processing Telegram message: %s", e)
            return f"❌ Error: {e}"

    def _build_prompt(self, user_message: str, device_states: dict) -> str:
        """Build AI prompt for the user request."""
        lines = [
            "# User Request via Telegram",
            "",
            f"**User says:** {user_message}",
            "",
            "## Available Devices",
        ]

        if device_states:
            for topic, state in sorted(device_states.items())[:30]:
                # Compact state for token efficiency
                state_copy = {k: v for k, v in state.items() if k != '_updated'}
                state_str = json.dumps(state_copy, separators=(',', ':'))
                if len(state_str) > 80:
                    state_str = state_str[:77] + "..."
                lines.append(f"- {topic}: {state_str}")
        else:
            lines.append("(No devices tracked yet)")

        lines.extend([
            "",
            "## Instructions",
            "Respond to the user's request. Use send_mqtt_message to control devices.",
            "Keep responses concise for Telegram (max 200 chars unless detailed info requested).",
            "Use device-friendly names when referring to devices.",
        ])

        return "\n".join(lines)
