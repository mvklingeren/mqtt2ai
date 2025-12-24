#!/usr/bin/env python3
"""Entry point for the MQTT AI Daemon.

This module provides the main entry point for the daemon, handling
argument parsing, AI connection testing, and signal handling.
"""
import os
import sys
import logging
import signal

from config import Config
from daemon import MqttAiDaemon
from ai_agent import AiAgent


def main():
    """Main entry point for the MQTT AI Daemon."""
    config = Config.from_args()

    # Propagate disable_new_rules to environment for child processes and lazy imports
    if config.disable_new_rules:
        os.environ["DISABLE_NEW_RULES"] = "true"

    # Test AI connection if requested
    if config.test_ai:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="[%H:%M:%S]"
        )
        provider = config.ai_provider.upper()
        logging.info("Testing AI connection [%s]...", provider)

        ai_agent = AiAgent(config)
        success, message = ai_agent.test_connection()

        if success:
            green, reset = "\033[92m", "\033[0m"
            print(f"{green}✓ AI connection test PASSED{reset}")
            print(f"AI Response: {message}")
            sys.exit(0)
        else:
            red, reset = "\033[91m", "\033[0m"
            print(f"{red}✗ AI connection test FAILED{reset}")
            print(f"Error: {message}")
            sys.exit(1)

    daemon = MqttAiDaemon(config)

    # Signal Handling
    def handle_signal(signum, _frame):
        logging.info("Signal received, shutting down...")
        daemon.running = False
        # Allow instant exit if blocked on I/O
        if signum == signal.SIGTERM:
            sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    daemon.start()


if __name__ == "__main__":
    main()
