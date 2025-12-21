#!/usr/bin/env python3
import sys
import logging
import signal

from config import Config
from daemon import MqttAiDaemon


def main():
    config = Config.from_args()
    daemon = MqttAiDaemon(config)
    
    # Signal Handling
    def handle_signal(signum, frame):
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
