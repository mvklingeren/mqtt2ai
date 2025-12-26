"""Shared pytest fixtures for MQTT AI Daemon tests."""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def config():
    """Create a default Config instance for testing."""
    return Config()


@pytest.fixture
def config_with_temp_files(temp_dir):
    """Create a Config instance with temp file paths."""
    cfg = Config()
    cfg.rulebook_file = os.path.join(temp_dir, "rulebook.md")
    cfg.filtered_triggers_file = os.path.join(temp_dir, "filtered_triggers.json")
    cfg.learned_rules_file = os.path.join(temp_dir, "learned_rules.json")
    cfg.pending_patterns_file = os.path.join(temp_dir, "pending_patterns.json")
    cfg.rejected_patterns_file = os.path.join(temp_dir, "rejected_patterns.json")
    return cfg


@pytest.fixture
def sample_rulebook_content():
    """Sample rulebook content for testing."""
    return """# Home Automation Rulebook

## Safety Rules
- Monitor smoke detectors
- Alert on water leaks
- Track temperature anomalies

## Automation Rules
- Motion detection triggers lights
"""


@pytest.fixture
def sample_learned_rules():
    """Sample learned rules data for testing."""
    return {
        "rules": [
            {
                "id": "pir_kitchen_light",
                "trigger": {
                    "topic": "zigbee2mqtt/kitchen_pir",
                    "field": "occupancy",
                    "value": True
                },
                "action": {
                    "topic": "zigbee2mqtt/kitchen_light/set",
                    "payload": '{"state": "ON"}'
                },
                "timing": {
                    "avg_delay_seconds": 2.5,
                    "tolerance_seconds": 1.0
                },
                "confidence": {
                    "occurrences": 5,
                    "last_triggered": "2024-01-01T12:00:00"
                },
                "enabled": True
            }
        ]
    }


@pytest.fixture
def sample_pending_patterns():
    """Sample pending patterns data for testing."""
    return {
        "patterns": [
            {
                "trigger_topic": "zigbee2mqtt/hallway_pir",
                "trigger_field": "occupancy",
                "action_topic": "zigbee2mqtt/hallway_light/set",
                "observations": [
                    {"delay_seconds": 2.0, "timestamp": "2024-01-01T10:00:00"},
                    {"delay_seconds": 2.5, "timestamp": "2024-01-01T11:00:00"}
                ]
            }
        ]
    }


@pytest.fixture
def sample_rejected_patterns():
    """Sample rejected patterns data for testing."""
    return {
        "patterns": [
            {
                "trigger_topic": "zigbee2mqtt/door_sensor",
                "trigger_field": "contact",
                "action_topic": "zigbee2mqtt/garage_light/set",
                "reason": "Coincidental - door sensor unrelated to garage",
                "rejected_at": "2024-01-01T09:00:00"
            }
        ]
    }


@pytest.fixture
def sample_mqtt_messages():
    """Sample MQTT messages for testing."""
    return [
        ("zigbee2mqtt/power_plug_1", '{"power": 50, "state": "ON"}'),
        ("zigbee2mqtt/door_sensor", '{"contact": true}'),
        ("zigbee2mqtt/pir_sensor", '{"occupancy": true}'),
        ("zigbee2mqtt/temperature", '{"temperature": 22.5, "humidity": 45}'),
        ("zigbee2mqtt/smoke_detector", '{"smoke": false}'),
    ]


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run for testing external commands.
    
    Note: This is still used by some legacy code paths, but most MQTT
    publishing now uses paho-mqtt instead of subprocess.
    """
    with patch("subprocess.run") as mock:
        mock.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr=""
        )
        yield mock


def create_json_file(filepath: str, data: dict) -> None:
    """Helper to create a JSON file with data."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def create_text_file(filepath: str, content: str) -> None:
    """Helper to create a text file with content."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

