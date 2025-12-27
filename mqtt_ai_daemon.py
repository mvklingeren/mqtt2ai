#!/usr/bin/env python3
"""Entry point for the MQTT AI Daemon.

This module provides the main entry point for the daemon, handling
argument parsing, AI connection testing, and signal handling.
"""
import json
import os
import sys
import logging
import signal
import time

from config import Config
from daemon import MqttAiDaemon
from ai_agent import AiAgent
from event_bus import event_bus # Import event_bus here


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

        ai_agent = AiAgent(config, event_bus=event_bus)
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

    # Track start time for test report
    start_time = time.time()

    daemon.start()

    # Run test validation if in test mode with a simulation file
    if config.test_mode and config.simulation_file:
        from scenario_validator import (
            ScenarioValidator, print_test_report, write_json_report
        )

        # Load scenario to get assertions
        try:
            with open(config.simulation_file, "r", encoding="utf-8") as f:
                scenario = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error("Failed to load scenario for validation: %s", e)
            sys.exit(2)

        assertions = scenario.get("assertions", {})
        scenario_name = scenario.get("name", config.simulation_file)

        if not assertions:
            logging.warning("No assertions defined in scenario, skipping validation")
            sys.exit(0)

        # Run validation
        validator = ScenarioValidator(assertions)
        results = validator.validate()

        # Calculate duration
        duration = time.time() - start_time

        # Print report
        all_passed = print_test_report(results, scenario_name)

        # Write JSON report if requested
        if config.test_report_file:
            write_json_report(results, config.test_report_file, scenario_name, duration)

        # Exit with appropriate code
        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
