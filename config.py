import argparse
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    mqtt_host: str = "192.168.1.245"
    mqtt_port: str = "1883"
    mqtt_topic: str = "#"
    max_messages: int = 1000
    ai_check_interval: int = 300  # 5 minutes
    ai_check_threshold: int = 500  # messages
    
    # Files
    rulebook_file: str = "rulebook.md"
    filtered_triggers_file: str = "filtered_triggers.json"
    learned_rules_file: str = "learned_rules.json"
    pending_patterns_file: str = "pending_patterns.json"
    rejected_patterns_file: str = "rejected_patterns.json"
    
    # Gemini
    gemini_command: str = "/opt/homebrew/bin/gemini"
    gemini_model: str = "gemini-2.5-flash"
    
    # Filtering & Display
    verbose: bool = False
    demo_mode: bool = False
    skip_printing_seconds: int = 3
    ignore_printing_topics: List[str] = field(default_factory=lambda: ["zigbee2mqtt/bridge/logging", "zigbee2mqtt/bridge/health"])
    ignore_printing_prefixes: List[str] = field(default_factory=list)

    @classmethod
    def from_args(cls) -> 'Config':
        parser = argparse.ArgumentParser(description="MQTT AI Daemon - Smart home automation with AI")
        
        parser.add_argument("--mqtt-host", default=os.environ.get("MQTT_HOST", "192.168.1.245"), help="MQTT Broker Host")
        parser.add_argument("--mqtt-port", default=os.environ.get("MQTT_PORT", "1883"), help="MQTT Broker Port")
        parser.add_argument("--gemini-command", default=os.environ.get("GEMINI_CLI_COMMAND", "/opt/homebrew/bin/gemini"), help="Path to Gemini CLI")
        parser.add_argument("--model", dest="gemini_model", default="gemini-2.5-flash", help="Gemini Model ID")
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
        parser.add_argument("--demo", action="store_true", help="Enable demo mode")
        
        args = parser.parse_args()
        
        c = cls()
        c.mqtt_host = args.mqtt_host
        c.mqtt_port = args.mqtt_port
        c.gemini_command = args.gemini_command
        c.gemini_model = args.gemini_model
        c.verbose = args.verbose
        c.demo_mode = args.demo
        return c

