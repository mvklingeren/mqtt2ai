"""Device state tracking for MQTT devices.

This module provides the DeviceStateTracker class that maintains
an in-memory cache of device states for topics matching a pattern.
"""
import json
import logging
import re
import threading
import time
from typing import Optional


class DeviceStateTracker:
    """Tracks the last known state of devices matching a topic pattern.

    This provides an in-memory cache of device states that can be used
    by the alert system to give the AI full context about available devices.
    """

    def __init__(self, pattern: str = "zigbee2mqtt/*"):
        """Initialize the tracker with a topic pattern.

        Args:
            pattern: Glob pattern for topics to track (e.g., "zigbee2mqtt/*")
        """
        self.pattern = pattern
        # Convert glob to regex for matching
        # zigbee2mqtt/* -> ^zigbee2mqtt/[^/]+$
        regex_pattern = pattern.replace("*", "[^/]+")
        self._pattern_re = re.compile(f"^{regex_pattern}$")
        self._states: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def should_track(self, topic: str) -> bool:
        """Check if a topic should be tracked.

        Excludes:
        - /set and /get suffixes (commands, not state)
        - bridge/ topics (not device state)
        """
        if topic.endswith("/set") or topic.endswith("/get"):
            return False
        if "/bridge/" in topic:
            return False
        return bool(self._pattern_re.match(topic))

    def update(self, topic: str, payload: str) -> None:
        """Update the state for a device topic.

        Args:
            topic: The MQTT topic
            payload: The raw payload string (JSON expected)
        """
        if not self.should_track(topic):
            return

        try:
            state = json.loads(payload)
            if isinstance(state, dict):
                with self._lock:
                    self._states[topic] = state
                    self._states[topic]['_updated'] = time.time()
        except json.JSONDecodeError:
            pass  # Ignore non-JSON payloads

        # Periodic cleanup (chance based to avoid checking every message)
        if time.time() - self._last_cleanup > 3600:
            self.cleanup()

    def get_all_states(self) -> dict[str, dict]:
        """Get a copy of all tracked device states.

        Returns:
            Dict mapping topic -> last known state
        """
        with self._lock:
            return dict(self._states)

    def get_state(self, topic: str) -> Optional[dict]:
        """Get the last known state for a specific topic.

        Args:
            topic: The MQTT topic

        Returns:
            The last known state dict, or None if not tracked
        """
        with self._lock:
            return self._states.get(topic)

    def get_device_count(self) -> int:
        """Get the number of tracked devices."""
        with self._lock:
            return len(self._states)

    def cleanup(self, max_age_seconds: float = 86400 * 7) -> int:
        """Remove stale devices that haven't updated in a long time.

        Args:
            max_age_seconds: Max age in seconds (default: 7 days)

        Returns:
            Number of removed devices
        """
        now = time.time()
        removed = 0
        with self._lock:
            to_remove = []
            for topic, state in self._states.items():
                last_updated = state.get('_updated', 0)
                if now - last_updated > max_age_seconds:
                    to_remove.append(topic)

            for topic in to_remove:
                del self._states[topic]
                removed += 1

            self._last_cleanup = now

        if removed > 0:
            logging.info("DeviceStateTracker cleanup: removed %d stale devices", removed)
        return removed

