"""Runtime context for dependency injection.

This module provides a RuntimeContext class that holds shared dependencies,
enabling proper dependency injection for better testability and reducing
reliance on global state.
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mqtt_client import MqttClient
    from config import Config


@dataclass
class RuntimeContext:
    """Container for runtime dependencies.

    This class holds references to shared components that would otherwise
    be stored as module-level globals. By passing this context explicitly,
    we enable:
    - Easier unit testing with mock dependencies
    - Clearer dependency relationships between modules
    - Reduced implicit coupling

    Attributes:
        mqtt_client: The MQTT client for publishing messages
        device_tracker: Device state tracker for alert context
        ai_agent: AI agent for processing alerts
        config: Application configuration
    """
    mqtt_client: Optional['MqttClient'] = None
    device_tracker: Optional[object] = None  # DeviceStateTracker
    ai_agent: Optional[object] = None  # AiAgent
    config: Optional['Config'] = None


# Module-level context instance (singleton pattern for backward compatibility)
_RUNTIME_CONTEXT: Optional[RuntimeContext] = None


def get_context() -> Optional[RuntimeContext]:
    """Get the global runtime context.

    Returns:
        The RuntimeContext instance, or None if not initialized.
    """
    return _RUNTIME_CONTEXT


def set_context(ctx: RuntimeContext) -> None:
    """Set the global runtime context.

    Args:
        ctx: The RuntimeContext instance to use globally.
    """
    global _RUNTIME_CONTEXT  # pylint: disable=global-statement
    _RUNTIME_CONTEXT = ctx


def create_context(
    mqtt_client: Optional['MqttClient'] = None,
    device_tracker: Optional[object] = None,
    ai_agent: Optional[object] = None,
    config: Optional['Config'] = None
) -> RuntimeContext:
    """Create and set a new runtime context.

    This is a convenience function that creates a RuntimeContext and
    sets it as the global context in one call.

    Args:
        mqtt_client: The MQTT client for publishing messages
        device_tracker: Device state tracker for alert context
        ai_agent: AI agent for processing alerts
        config: Application configuration

    Returns:
        The newly created RuntimeContext instance.
    """
    ctx = RuntimeContext(
        mqtt_client=mqtt_client,
        device_tracker=device_tracker,
        ai_agent=ai_agent,
        config=config
    )
    set_context(ctx)
    return ctx


