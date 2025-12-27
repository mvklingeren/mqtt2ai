# ai_providers/base_provider.py
"""Base class for AI providers."""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from config import Config
    from ai_agent import AiAgent # For execute_tool_call


class AiProvider(ABC):
    """Abstract base class for AI providers."""

    def __init__(self, config: 'Config', ai_agent: 'AiAgent'):
        self.config = config
        self.ai_agent = ai_agent

    @abstractmethod
    def execute_call(self, prompt: str, rules_count: int = 0,
                     patterns_count: int = 0) -> None:
        """Execute an AI call and process its response."""
        raise NotImplementedError

    @abstractmethod
    def test_connection(self) -> tuple[bool, str]:
        """Test the connection to the AI provider."""
        raise NotImplementedError

    @abstractmethod
    def get_alert_tool_declarations(self) -> list:
        """Get tool declarations for alert handling."""
        raise NotImplementedError