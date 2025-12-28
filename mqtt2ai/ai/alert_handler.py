# alert_handler.py
"""Handles security alerts and coordinates AI responses for the MQTT AI Daemon."""
import json
import logging
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mqtt2ai.core.config import Config
    from mqtt2ai.ai.agent import AiAgent
    from mqtt2ai.rules.device_tracker import DeviceStateTracker
    from mqtt2ai.telegram.bot import TelegramBot

from mqtt2ai.ai.tool_definitions import OPENAI_TOOLS
from mqtt2ai.core.constants import MqttTopics

class AlertHandler:
    """Handles security alerts and coordinates AI responses."""

    def __init__(self, config: 'Config', ai_agent: 'AiAgent',
                 device_tracker: Optional['DeviceStateTracker'] = None,
                 telegram_bot: Optional['TelegramBot'] = None):
        self.config = config
        self.ai_agent = ai_agent
        self.device_tracker = device_tracker
        self.telegram_bot = telegram_bot

    def set_telegram_bot(self, telegram_bot: 'TelegramBot') -> None:
        """Set the Telegram bot reference (for deferred initialization).

        Args:
            telegram_bot: The TelegramBot instance
        """
        self.telegram_bot = telegram_bot

    def raise_alert(self, severity: float, reason: str, context: dict = None) -> str:
        """Raise an alert with severity 0.0-1.0."""
        # Clamp severity to valid range
        severity = max(0.0, min(1.0, severity))

        # Color coding for logs
        if severity >= 0.7:
            color = "\033[91m"  # Red
            level_name = "HIGH"
        elif severity >= 0.3:
            color = "\033[93m"  # Yellow
            level_name = "MEDIUM"
        else:
            color = "\033[94m"  # Blue
            level_name = "LOW"
        reset = "\033[0m"

        logging.info(
            "%s[ALERT %s (%.1f)] %s%s",
            color, level_name, severity, reason, reset
        )

        if context:
            logging.debug("Alert context: %s", json.dumps(context))

        # Send via Telegram if available
        self._send_telegram_alert(severity, reason, context)

        # Low priority: just log
        if severity < 0.3:
            return f"Alert logged (severity {severity:.1f}): {reason}"

        # Medium priority: send notification via MQTT and Telegram
        if severity < 0.7:
            notification = {
                "severity": severity,
                "level": level_name,
                "reason": reason,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
            try:
                # Send via MQTT
                self.ai_agent.execute_tool_call(
                    "send_mqtt_message",
                    {"topic": MqttTopics.ALERTS, "payload": json.dumps(notification)}
                )

                return f"Alert notification sent (severity {severity:.1f}): {reason}"
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.error("Failed to send alert notification: %s", e)
                return f"Alert logged but notification failed: {e}"

        # High priority: AI decides action with full device context
        if not self.device_tracker:
            logging.error("Device tracker not available - cannot process high severity alert")
            return f"Alert logged but AI context missing (severity {severity:.1f}): {reason}"

        # Get all device states for context
        device_states = self.device_tracker.get_all_states()

        # Build alert prompt for AI
        alert_prompt = self._build_alert_prompt(severity, reason, context, device_states)

        logging.info(
            "%s[ALERT AI] Triggering async AI response for high-severity alert...%s",
            color, reset
        )

        # Send Telegram notification for high-severity alerts
        self._send_telegram_alert(severity, reason, context)

        # Execute AI call asynchronously to avoid blocking the main AI worker
        threading.Thread(
            target=self._execute_alert_ai_call,
            args=(alert_prompt,),
            daemon=True,
            name="Alert-AI-Worker"
        ).start()

        return f"Alert processing started (severity {severity:.1f}): {reason}"

    def _build_alert_prompt(
        self, severity: float, reason: str, context: dict, device_states: dict
    ) -> str:
        """Build the prompt for an alert AI call."""
        lines = [
            "# SECURITY ALERT",
            f"Severity: {severity:.1f} (HIGH PRIORITY)",
            f"Reason: {reason}",
            "",
        ]

        if context:
            lines.append("## Alert Context")
            lines.append(json.dumps(context, indent=2))
            lines.append("")

        lines.append("## Available Devices and Current States")
        lines.append(
            "Use these to take appropriate action (e.g., activate sirens, turn on lights)."
        )
        lines.append("")

        if device_states:
            for topic, state in sorted(device_states.items()):
                # Compact state representation
                state_str = json.dumps(state, separators=(',', ':'))
                if len(state_str) > 100:
                    state_str = state_str[:97] + "..."
                lines.append(f"- {topic}: {state_str}")
        else:
            lines.append("(No device states available)")

        lines.append("")
        lines.append("## Instructions")
        lines.append("Based on the alert severity and available devices:")
        lines.append("1. Identify appropriate response devices (sirens, lights, notifications)")
        lines.append("2. Use send_mqtt_message to activate them with correct payloads")
        lines.append("3. For sirens/alarms, set state to ON or true")
        lines.append("4. For lights, consider turning them ON at full brightness")
        lines.append("")
        lines.append("Take action NOW to respond to this security alert.")

        return "\n".join(lines)

    def _send_telegram_alert(
        self,
        severity: float,
        reason: str,
        context: Optional[dict] = None
    ) -> None:
        """Send an alert notification via Telegram.

        Args:
            severity: Alert severity (0.0-1.0)
            reason: Alert description
            context: Optional additional context
        """
        if not self.telegram_bot:
            return

        try:
            count = self.telegram_bot.send_alert(severity, reason, context)
            if count > 0:
                logging.info("Telegram alert sent to %d chat(s)", count)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("Failed to send Telegram alert: %s", e)

    def _execute_alert_ai_call(self, prompt: str) -> None:
        """Execute an AI call for alert response."""
        # Delegate to the current AI provider instance to get alert tools
        # and execute the call
        if self.ai_agent.ai_provider_instance:
            alert_tools = self.ai_agent.ai_provider_instance.get_alert_tool_declarations()
            self.ai_agent.ai_provider_instance.execute_call_for_alert(prompt, alert_tools)
        else:
            logging.error("AI provider not initialized, cannot execute alert AI call.")

