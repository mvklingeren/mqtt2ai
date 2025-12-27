# alert_handler.py
"""Handles security alerts and coordinates AI responses for the MQTT AI Daemon."""
import json
import logging
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from config import Config
    from ai_agent import AiAgent
    from daemon import DeviceStateTracker

from tool_definitions import OPENAI_TOOLS

class AlertHandler:
    """Handles security alerts and coordinates AI responses."""

    def __init__(self, config: 'Config', ai_agent: 'AiAgent',
                 device_tracker: Optional['DeviceStateTracker'] = None):
        self.config = config
        self.ai_agent = ai_agent
        self.device_tracker = device_tracker

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

        # Low priority: just log
        if severity < 0.3:
            return f"Alert logged (severity {severity:.1f}): {reason}"

        # Medium priority: send notification via MQTT
        if severity < 0.7:
            notification = {
                "severity": severity,
                "level": level_name,
                "reason": reason,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
            try:
                # Assuming tools.send_mqtt_message is accessible via ai_agent
                # Or could be passed in init if alert_handler shouldn't directly access tools
                self.ai_agent.execute_tool_call(
                    "send_mqtt_message",
                    {"topic": "mqtt2ai/alerts", "payload": json.dumps(notification)}
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

    def _execute_alert_ai_call(self, prompt: str) -> None:
        """Execute an AI call for alert response."""
        # Delegate to the current AI provider instance to get alert tools
        # and execute the call
        if self.ai_agent.ai_provider_instance:
            alert_tools = self.ai_agent.ai_provider_instance.get_alert_tool_declarations()
            self.ai_agent.ai_provider_instance.execute_call_for_alert(prompt, alert_tools)
        else:
            logging.error("AI provider not initialized, cannot execute alert AI call.")
