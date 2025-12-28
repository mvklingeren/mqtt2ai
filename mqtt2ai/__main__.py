#!/usr/bin/env python3
"""Entry point for running mqtt2ai as a module: python -m mqtt2ai

This allows running the daemon with:
    python -m mqtt2ai [options]

Which is equivalent to:
    python mqtt_ai_daemon.py [options]
"""

import signal
import sys
import logging

from mqtt2ai.core.config import Config
from mqtt2ai.core.daemon import MqttAiDaemon
from mqtt2ai.ai.agent import AiAgent
from mqtt2ai.core.event_bus import event_bus


def main():
    """Main entry point for the MQTT AI daemon."""
    config = Config.from_args()

    # Handle test connection mode
    if config.test_ai:
        try:
            # Create a minimal AI agent to test connection
            test_agent = AiAgent(config, event_bus=event_bus)
            success = test_agent.test_connection()
            sys.exit(0 if success else 1)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Connection test failed: %s", e)
            sys.exit(1)

    daemon = MqttAiDaemon(config)

    # Handle SIGTERM for graceful shutdown (e.g., docker stop)
    def handle_signal(signum, frame):  # pylint: disable=unused-argument
        logging.info("Received signal %d, stopping...", signum)
        daemon.running = False

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    daemon.start()

    # In test mode, exit with appropriate code based on test results
    if config.test_mode:
        test_result = daemon.get_test_result()
        if test_result is None:
            # No assertions were validated (likely no assertions in scenario)
            sys.exit(0)
        sys.exit(0 if test_result else 1)


if __name__ == "__main__":
    main()

