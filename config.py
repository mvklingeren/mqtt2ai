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
    
    # AI Provider
    ai_provider: str = "gemini"  # "gemini" or "claude"
    
    # Gemini
    gemini_command: str = "/opt/homebrew/bin/gemini"
    gemini_model: str = "gemini-2.5-flash"
    
    # Claude
    claude_command: str = "/Users/mvklingeren/.nvm/versions/node/v20.15.0/bin/claude"
    claude_model: str = "claude-3-5-haiku-latest"
    claude_mcp_config: str = ""  # Path to MCP config file for Claude
    
    # Filtering & Display
    verbose: bool = False
    demo_mode: bool = False
    no_ai: bool = False  # Run without making AI calls (logging only)
    skip_printing_seconds: int = 3
    ignore_printing_topics: List[str] = field(default_factory=lambda: ["zigbee2mqtt/bridge/logging", "zigbee2mqtt/bridge/health"])
    ignore_printing_prefixes: List[str] = field(default_factory=list)

    @classmethod
    def from_args(cls) -> 'Config':
        parser = argparse.ArgumentParser(description="MQTT AI Daemon - Smart home automation with AI")
        
        parser.add_argument("--mqtt-host", default=os.environ.get("MQTT_HOST", "192.168.1.245"), help="MQTT Broker Host")
        parser.add_argument("--mqtt-port", default=os.environ.get("MQTT_PORT", "1883"), help="MQTT Broker Port")
        
        # AI Provider
        parser.add_argument("--ai-provider", choices=["gemini", "claude"], default=os.environ.get("AI_PROVIDER", "gemini"), help="AI provider to use (gemini or claude)")
        
        # Gemini CLI
        parser.add_argument("--gemini-command", default=os.environ.get("GEMINI_CLI_COMMAND", "/opt/homebrew/bin/gemini"), help="Path to Gemini CLI")
        parser.add_argument("--gemini-model", default="gemini-2.5-flash", help="Gemini Model ID")
        
        # Claude CLI
        parser.add_argument("--claude-command", default=os.environ.get("CLAUDE_CLI_COMMAND", "/Users/mvklingeren/.nvm/versions/node/v20.15.0/bin/claude"), help="Path to Claude CLI")
        parser.add_argument("--claude-model", default="claude-3-5-haiku-latest", help="Claude Model ID")
        parser.add_argument("--claude-mcp-config", default=os.environ.get("CLAUDE_MCP_CONFIG", ""), help="Path to MCP config file for Claude")
        
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
        parser.add_argument("--demo", action="store_true", help="Enable demo mode")
        parser.add_argument("--no-ai", action="store_true", help="Disable AI calls (logging only mode)")
        
        args = parser.parse_args()
        
        c = cls()
        c.mqtt_host = args.mqtt_host
        c.mqtt_port = args.mqtt_port
        c.ai_provider = args.ai_provider
        c.gemini_command = args.gemini_command
        c.gemini_model = args.gemini_model
        c.claude_command = args.claude_command
        c.claude_model = args.claude_model
        c.claude_mcp_config = args.claude_mcp_config
        c.verbose = args.verbose
        c.demo_mode = args.demo
        c.no_ai = args.no_ai
        return c

