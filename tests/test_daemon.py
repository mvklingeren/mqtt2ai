"""Tests for the daemon module."""
import collections
import logging
import os
import sys
import threading
import time
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daemon import MqttAiDaemon, setup_logging, timestamp
from config import Config


class TestTimestamp:
    """Tests for the timestamp function."""

    def test_timestamp_format(self):
        """Test that timestamp returns correct format."""
        ts = timestamp()
        assert ts.startswith("[")
        assert ts.endswith("]")
        # Format should be [HH:MM:SS]
        assert len(ts) == 10

    def test_timestamp_contains_colons(self):
        """Test that timestamp contains time separators."""
        ts = timestamp()
        time_part = ts[1:-1]
        parts = time_part.split(":")
        assert len(parts) == 3


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_debug_when_verbose(self, config):
        """Test that debug level is set when verbose."""
        config.verbose = True
        # Reset root logger before testing
        root_logger = logging.getLogger()
        original_level = root_logger.level
        original_handlers = root_logger.handlers[:]

        try:
            # Clear handlers to force basicConfig to take effect
            root_logger.handlers = []
            root_logger.setLevel(logging.NOTSET)
            setup_logging(config)
            assert root_logger.level == logging.DEBUG
        finally:
            # Restore original state
            root_logger.level = original_level
            root_logger.handlers = original_handlers

    def test_setup_logging_info_when_not_verbose(self, config):
        """Test that info level is set when not verbose."""
        config.verbose = False
        # Reset root logger before testing
        root_logger = logging.getLogger()
        original_level = root_logger.level
        original_handlers = root_logger.handlers[:]

        try:
            # Clear handlers to force basicConfig to take effect
            root_logger.handlers = []
            root_logger.setLevel(logging.NOTSET)
            setup_logging(config)
            assert root_logger.level == logging.INFO
        finally:
            # Restore original state
            root_logger.level = original_level
            root_logger.handlers = original_handlers


class TestMqttAiDaemonInit:
    """Tests for MqttAiDaemon initialization."""

    def test_init_creates_components(self, config_with_temp_files):
        """Test that init creates all required components."""
        daemon = MqttAiDaemon(config_with_temp_files)

        assert daemon.config == config_with_temp_files
        assert daemon.kb is not None
        assert daemon.mqtt is not None
        assert daemon.ai is not None
        assert daemon.trigger_analyzer is not None

    def test_init_creates_message_deque(self, config_with_temp_files):
        """Test that message deque is created with correct maxlen."""
        config_with_temp_files.max_messages = 500
        daemon = MqttAiDaemon(config_with_temp_files)

        assert daemon.messages_deque.maxlen == 500

    def test_init_sets_running_true(self, config_with_temp_files):
        """Test that running flag is set to True."""
        daemon = MqttAiDaemon(config_with_temp_files)

        assert daemon.running is True

    def test_init_message_count_zero(self, config_with_temp_files):
        """Test that new message count starts at zero."""
        daemon = MqttAiDaemon(config_with_temp_files)

        assert daemon.new_message_count == 0

    def test_init_creates_lock(self, config_with_temp_files):
        """Test that thread lock is created."""
        daemon = MqttAiDaemon(config_with_temp_files)

        assert isinstance(daemon.lock, type(threading.Lock()))

    def test_init_creates_event(self, config_with_temp_files):
        """Test that AI event is created."""
        daemon = MqttAiDaemon(config_with_temp_files)

        assert isinstance(daemon.ai_event, type(threading.Event()))


class TestMqttAiDaemonDetermineTriggerReason:
    """Tests for _determine_trigger_reason method."""

    def test_instant_trigger_reason(self, config_with_temp_files):
        """Test reason for instant trigger."""
        daemon = MqttAiDaemon(config_with_temp_files)

        reason = daemon._determine_trigger_reason(True, False)

        assert reason == "smart_trigger"

    def test_message_count_reason(self, config_with_temp_files):
        """Test reason for message count trigger."""
        daemon = MqttAiDaemon(config_with_temp_files)

        reason = daemon._determine_trigger_reason(False, True)

        assert "message_count" in reason
        assert str(config_with_temp_files.ai_check_threshold) in reason

    def test_interval_reason(self, config_with_temp_files):
        """Test reason for interval trigger."""
        daemon = MqttAiDaemon(config_with_temp_files)

        reason = daemon._determine_trigger_reason(False, False)

        assert "interval" in reason
        assert str(config_with_temp_files.ai_check_interval) in reason


class TestMqttAiDaemonHandleAiCheck:
    """Tests for _handle_ai_check method."""

    def test_handle_ai_check_no_ai_mode(self, config_with_temp_files):
        """Test that no-AI mode logs but doesn't call AI."""
        config_with_temp_files.no_ai = True
        daemon = MqttAiDaemon(config_with_temp_files)

        with patch.object(daemon.ai, "run_analysis") as mock_run:
            daemon._handle_ai_check("test snapshot", "test reason")
            mock_run.assert_not_called()

    def test_handle_ai_check_queues_request(self, config_with_temp_files):
        """Test that AI request is queued in normal mode."""
        config_with_temp_files.no_ai = False
        daemon = MqttAiDaemon(config_with_temp_files)

        # Ensure queue is empty
        assert daemon.ai_queue.empty()

        daemon._handle_ai_check("test snapshot", "test reason")

        # Request should be queued
        assert not daemon.ai_queue.empty()
        request = daemon.ai_queue.get_nowait()
        assert request.snapshot == "test snapshot"
        assert request.reason == "test reason"

    def test_handle_ai_check_skips_when_busy(self, config_with_temp_files):
        """Test that requests are skipped when AI is busy."""
        config_with_temp_files.no_ai = False
        daemon = MqttAiDaemon(config_with_temp_files)

        # Simulate AI being busy
        daemon.ai_busy.set()

        daemon._handle_ai_check("test snapshot", "test reason")

        # Queue should remain empty when AI is busy
        assert daemon.ai_queue.empty()


class TestMqttAiDaemonMessageDeque:
    """Tests for message deque functionality."""

    def test_messages_added_to_deque(self, config_with_temp_files):
        """Test that messages are added to deque."""
        daemon = MqttAiDaemon(config_with_temp_files)

        with daemon.lock:
            daemon.messages_deque.append("[12:00:00] test message")
            daemon.new_message_count += 1

        assert len(daemon.messages_deque) == 1
        assert daemon.new_message_count == 1

    def test_messages_deque_maxlen_enforced(self, config_with_temp_files):
        """Test that deque maxlen is enforced."""
        config_with_temp_files.max_messages = 3
        daemon = MqttAiDaemon(config_with_temp_files)

        with daemon.lock:
            for i in range(5):
                daemon.messages_deque.append(f"message {i}")

        # Should only have last 3 messages
        assert len(daemon.messages_deque) == 3
        assert "message 2" in daemon.messages_deque
        assert "message 4" in daemon.messages_deque

    def test_snapshot_creation(self, config_with_temp_files):
        """Test creating a snapshot from deque."""
        daemon = MqttAiDaemon(config_with_temp_files)

        with daemon.lock:
            daemon.messages_deque.append("msg 1")
            daemon.messages_deque.append("msg 2")
            daemon.messages_deque.append("msg 3")
            snapshot = "\n".join(list(daemon.messages_deque))

        assert "msg 1" in snapshot
        assert "msg 2" in snapshot
        assert "msg 3" in snapshot


class TestMqttAiDaemonTriggerAnalyzer:
    """Tests for trigger analyzer integration."""

    def test_trigger_analyzer_loaded(self, config_with_temp_files, temp_dir):
        """Test that trigger analyzer is properly loaded."""
        # Create a custom config file
        config_path = os.path.join(temp_dir, "filtered_triggers.json")
        config_with_temp_files.filtered_triggers_file = config_path

        daemon = MqttAiDaemon(config_with_temp_files)

        assert daemon.trigger_analyzer is not None
        stats = daemon.trigger_analyzer.get_stats()
        assert "config" in stats


class TestMqttAiDaemonPrintTrigger:
    """Tests for _print_trigger method."""

    def test_print_trigger_outputs_correctly(self, config_with_temp_files, capsys):
        """Test that trigger prints correctly."""
        daemon = MqttAiDaemon(config_with_temp_files)

        # Create a mock trigger result
        class MockTriggerResult:
            should_trigger = True
            reason = "Test trigger"
            field_name = "occupancy"
            old_value = False
            new_value = True

        daemon._print_trigger(MockTriggerResult(), "zigbee2mqtt/pir {\"occupancy\": true}")

        captured = capsys.readouterr()
        assert "SMART TRIGGER" in captured.out
        assert "occupancy" in captured.out


class TestMqttAiDaemonEventSystem:
    """Tests for the event-driven system."""

    def test_ai_event_initially_clear(self, config_with_temp_files):
        """Test that AI event is initially clear."""
        daemon = MqttAiDaemon(config_with_temp_files)

        assert not daemon.ai_event.is_set()

    def test_ai_event_can_be_set(self, config_with_temp_files):
        """Test that AI event can be set."""
        daemon = MqttAiDaemon(config_with_temp_files)

        daemon.ai_event.set()

        assert daemon.ai_event.is_set()

    def test_ai_event_can_be_cleared(self, config_with_temp_files):
        """Test that AI event can be cleared."""
        daemon = MqttAiDaemon(config_with_temp_files)

        daemon.ai_event.set()
        daemon.ai_event.clear()

        assert not daemon.ai_event.is_set()


class TestMqttAiDaemonRunning:
    """Tests for the running state."""

    def test_running_can_be_set_false(self, config_with_temp_files):
        """Test that running can be set to False."""
        daemon = MqttAiDaemon(config_with_temp_files)

        daemon.running = False

        assert daemon.running is False

    def test_running_starts_true(self, config_with_temp_files):
        """Test that running starts as True."""
        daemon = MqttAiDaemon(config_with_temp_files)

        assert daemon.running is True


class TestMqttAiDaemonThreadSafety:
    """Tests for thread safety."""

    def test_lock_is_functional(self, config_with_temp_files):
        """Test that the lock is functional."""
        daemon = MqttAiDaemon(config_with_temp_files)

        acquired = daemon.lock.acquire(blocking=False)
        assert acquired is True
        daemon.lock.release()

    def test_concurrent_deque_access(self, config_with_temp_files):
        """Test concurrent access to message deque."""
        daemon = MqttAiDaemon(config_with_temp_files)
        errors = []

        def writer():
            try:
                for i in range(100):
                    with daemon.lock:
                        daemon.messages_deque.append(f"msg {i}")
                        daemon.new_message_count += 1
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    with daemon.lock:
                        _ = list(daemon.messages_deque)
                        _ = daemon.new_message_count
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestMqttAiDaemonConfiguration:
    """Tests for configuration usage."""

    def test_uses_config_interval(self, config_with_temp_files):
        """Test that daemon uses config interval."""
        config_with_temp_files.ai_check_interval = 600
        daemon = MqttAiDaemon(config_with_temp_files)

        reason = daemon._determine_trigger_reason(False, False)

        assert "600" in reason

    def test_uses_config_threshold(self, config_with_temp_files):
        """Test that daemon uses config threshold."""
        config_with_temp_files.ai_check_threshold = 1000
        daemon = MqttAiDaemon(config_with_temp_files)

        reason = daemon._determine_trigger_reason(False, True)

        assert "1000" in reason

