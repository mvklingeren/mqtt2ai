"""Tests for the TelegramHandler class."""
import json
import logging
from unittest.mock import MagicMock, patch
import pytest

from mqtt2ai.telegram.handler import TelegramHandler
from mqtt2ai.core.config import Config
from mqtt2ai.ai.agent import AiAgent
from mqtt2ai.rules.device_tracker import DeviceStateTracker

@pytest.fixture
def mock_config():
    config = MagicMock(spec=Config)
    config.no_ai = False
    return config

@pytest.fixture
def mock_ai_agent():
    return MagicMock(spec=AiAgent)

@pytest.fixture
def mock_device_tracker():
    tracker = MagicMock(spec=DeviceStateTracker)
    tracker.get_all_states.return_value = {
        "device/1": {"state": "on", "_updated": 123},
        "device/2": {"temp": 21.5, "_updated": 456}
    }
    return tracker

@pytest.fixture
def telegram_handler(mock_config, mock_ai_agent, mock_device_tracker):
    return TelegramHandler(mock_config, mock_ai_agent, mock_device_tracker)

def test_handle_message_no_ai(telegram_handler, mock_config):
    """Test handling message when no_ai is enabled."""
    mock_config.no_ai = True
    response = telegram_handler.handle_message(12345, "Hello")
    assert "AI is disabled" in response

def test_handle_message_success(telegram_handler, mock_ai_agent):
    """Test successful message handling."""
    mock_ai_agent.process_telegram_query.return_value = "Response from AI"
    
    response = telegram_handler.handle_message(12345, "Turn on the light")
    
    assert response == "Response from AI"
    mock_ai_agent.process_telegram_query.assert_called_once()
    
    # Verify the prompt construction (indirectly)
    args, _ = mock_ai_agent.process_telegram_query.call_args
    prompt = args[0]
    assert "**User says:** Turn on the light" in prompt
    assert "device/1" in prompt
    assert "device/2" in prompt

def test_handle_message_exception(telegram_handler, mock_ai_agent):
    """Test handling exception from AI agent."""
    mock_ai_agent.process_telegram_query.side_effect = Exception("AI Error")
    
    response = telegram_handler.handle_message(12345, "Hello")
    
    assert "Error: AI Error" in response

def test_build_prompt_with_devices(telegram_handler):
    """Test prompt building with devices."""
    message = "Status report"
    device_states = {
        "sensor/temp": {"value": 22.5},
        "switch/light": {"state": "off"}
    }
    
    prompt = telegram_handler._build_prompt(message, device_states)
    
    assert "**User says:** Status report" in prompt
    assert "sensor/temp" in prompt
    assert '{"value":22.5}' in prompt
    assert "switch/light" in prompt
    assert '{"state":"off"}' in prompt

def test_build_prompt_no_devices(telegram_handler):
    """Test prompt building with no devices."""
    prompt = telegram_handler._build_prompt("Hello", {})
    
    assert "(No devices tracked yet)" in prompt
