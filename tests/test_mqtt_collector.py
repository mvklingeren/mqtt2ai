"""Tests for the MQTT Collector module."""
import queue
import threading
import time
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from mqtt_collector import MqttCollector, CollectorCallbacks, timestamp


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


class TestCollectorCallbacks:
    """Tests for CollectorCallbacks dataclass."""

    def test_callbacks_creation(self):
        """Test that callbacks can be created."""
        on_trigger = MagicMock()
        on_shutdown = MagicMock()
        wait_for_ai = MagicMock()

        callbacks = CollectorCallbacks(
            on_trigger=on_trigger,
            on_shutdown=on_shutdown,
            wait_for_ai=wait_for_ai
        )

        assert callbacks.on_trigger is on_trigger
        assert callbacks.on_shutdown is on_shutdown
        assert callbacks.wait_for_ai is wait_for_ai


class TestMqttCollectorInit:
    """Tests for MqttCollector initialization."""

    def test_init_stores_dependencies(self, collector_dependencies):
        """Test that init stores all dependencies."""
        deps = collector_dependencies
        
        collector = MqttCollector(
            config=deps['config'],
            mqtt_client=deps['mqtt'],
            trigger_analyzer=deps['trigger_analyzer'],
            device_tracker=deps['device_tracker'],
            rule_engine=deps['rule_engine'],
            knowledge_base=deps['kb'],
            messages_deque=deps['messages_deque'],
            lock=deps['lock'],
            callbacks=deps['callbacks']
        )

        assert collector.config is deps['config']
        assert collector.mqtt is deps['mqtt']
        assert not collector._running

    def test_init_creates_message_queue(self, collector_dependencies):
        """Test that init creates message queue."""
        deps = collector_dependencies
        
        collector = MqttCollector(
            config=deps['config'],
            mqtt_client=deps['mqtt'],
            trigger_analyzer=deps['trigger_analyzer'],
            device_tracker=deps['device_tracker'],
            rule_engine=deps['rule_engine'],
            knowledge_base=deps['kb'],
            messages_deque=deps['messages_deque'],
            lock=deps['lock'],
            callbacks=deps['callbacks']
        )

        assert collector._mqtt_message_queue is not None
        assert isinstance(collector._mqtt_message_queue, queue.Queue)


class TestMqttCollectorStartStop:
    """Tests for start/stop functionality."""

    def test_start_sets_running(self, collector_dependencies):
        """Test that start sets running flag."""
        deps = collector_dependencies
        deps['mqtt'].subscribe.return_value = True
        
        collector = MqttCollector(
            config=deps['config'],
            mqtt_client=deps['mqtt'],
            trigger_analyzer=deps['trigger_analyzer'],
            device_tracker=deps['device_tracker'],
            rule_engine=deps['rule_engine'],
            knowledge_base=deps['kb'],
            messages_deque=deps['messages_deque'],
            lock=deps['lock'],
            callbacks=deps['callbacks']
        )

        collector.start()

        try:
            assert collector._running is True
            # Give thread time to start
            time.sleep(0.1)
            assert collector.is_alive()
        finally:
            collector.stop()
            time.sleep(0.2)

    def test_stop_clears_running(self, collector_dependencies):
        """Test that stop clears running flag."""
        deps = collector_dependencies
        deps['mqtt'].subscribe.return_value = True
        
        collector = MqttCollector(
            config=deps['config'],
            mqtt_client=deps['mqtt'],
            trigger_analyzer=deps['trigger_analyzer'],
            device_tracker=deps['device_tracker'],
            rule_engine=deps['rule_engine'],
            knowledge_base=deps['kb'],
            messages_deque=deps['messages_deque'],
            lock=deps['lock'],
            callbacks=deps['callbacks']
        )

        collector.start()
        time.sleep(0.1)
        
        collector.stop()

        assert collector._running is False

    def test_double_start_is_safe(self, collector_dependencies):
        """Test that calling start twice is safe."""
        deps = collector_dependencies
        deps['mqtt'].subscribe.return_value = True
        
        collector = MqttCollector(
            config=deps['config'],
            mqtt_client=deps['mqtt'],
            trigger_analyzer=deps['trigger_analyzer'],
            device_tracker=deps['device_tracker'],
            rule_engine=deps['rule_engine'],
            knowledge_base=deps['kb'],
            messages_deque=deps['messages_deque'],
            lock=deps['lock'],
            callbacks=deps['callbacks']
        )

        collector.start()
        first_thread = collector._thread
        
        collector.start()  # Should not create new thread

        assert collector._thread is first_thread
        collector.stop()


class TestMqttCollectorMessageCount:
    """Tests for message count functionality."""

    def test_initial_message_count_zero(self, collector_dependencies):
        """Test that message count starts at zero."""
        deps = collector_dependencies
        
        collector = MqttCollector(
            config=deps['config'],
            mqtt_client=deps['mqtt'],
            trigger_analyzer=deps['trigger_analyzer'],
            device_tracker=deps['device_tracker'],
            rule_engine=deps['rule_engine'],
            knowledge_base=deps['kb'],
            messages_deque=deps['messages_deque'],
            lock=deps['lock'],
            callbacks=deps['callbacks']
        )

        assert collector.get_new_message_count() == 0

    def test_reset_message_count(self, collector_dependencies):
        """Test that reset clears message count."""
        deps = collector_dependencies
        
        collector = MqttCollector(
            config=deps['config'],
            mqtt_client=deps['mqtt'],
            trigger_analyzer=deps['trigger_analyzer'],
            device_tracker=deps['device_tracker'],
            rule_engine=deps['rule_engine'],
            knowledge_base=deps['kb'],
            messages_deque=deps['messages_deque'],
            lock=deps['lock'],
            callbacks=deps['callbacks']
        )

        # Manually increment
        collector._increment_message_count()
        collector._increment_message_count()
        assert collector.get_new_message_count() == 2

        collector.reset_message_count()
        assert collector.get_new_message_count() == 0


class TestMqttCollectorProcessMessage:
    """Tests for message processing."""

    def test_filters_ignored_prefixes(self, collector_dependencies):
        """Test that messages with ignored prefixes are filtered."""
        deps = collector_dependencies
        deps['config'].ignore_printing_prefixes = ['homeassistant/']
        
        collector = MqttCollector(
            config=deps['config'],
            mqtt_client=deps['mqtt'],
            trigger_analyzer=deps['trigger_analyzer'],
            device_tracker=deps['device_tracker'],
            rule_engine=deps['rule_engine'],
            knowledge_base=deps['kb'],
            messages_deque=deps['messages_deque'],
            lock=deps['lock'],
            callbacks=deps['callbacks']
        )

        collector._process_message('homeassistant/sensor/state', '{"value": 1}')

        # Message should be filtered, not added to deque
        assert len(deps['messages_deque']) == 0

    def test_filters_empty_payload(self, collector_dependencies):
        """Test that empty payloads are filtered."""
        deps = collector_dependencies
        
        collector = MqttCollector(
            config=deps['config'],
            mqtt_client=deps['mqtt'],
            trigger_analyzer=deps['trigger_analyzer'],
            device_tracker=deps['device_tracker'],
            rule_engine=deps['rule_engine'],
            knowledge_base=deps['kb'],
            messages_deque=deps['messages_deque'],
            lock=deps['lock'],
            callbacks=deps['callbacks']
        )

        collector._process_message('zigbee2mqtt/sensor', '')

        assert len(deps['messages_deque']) == 0

    def test_stores_message_in_deque(self, collector_dependencies):
        """Test that valid messages are stored in deque."""
        deps = collector_dependencies
        deps['trigger_analyzer'].analyze.return_value = MagicMock(should_trigger=False)
        
        collector = MqttCollector(
            config=deps['config'],
            mqtt_client=deps['mqtt'],
            trigger_analyzer=deps['trigger_analyzer'],
            device_tracker=deps['device_tracker'],
            rule_engine=deps['rule_engine'],
            knowledge_base=deps['kb'],
            messages_deque=deps['messages_deque'],
            lock=deps['lock'],
            callbacks=deps['callbacks']
        )

        collector._process_message('zigbee2mqtt/sensor', '{"state": "ON"}')

        assert len(deps['messages_deque']) == 1
        assert 'zigbee2mqtt/sensor' in deps['messages_deque'][0]

    def test_retained_messages_have_prefix(self, collector_dependencies):
        """Test that retained messages get [RETAINED] prefix."""
        deps = collector_dependencies
        deps['trigger_analyzer'].analyze.return_value = MagicMock(should_trigger=False)
        
        collector = MqttCollector(
            config=deps['config'],
            mqtt_client=deps['mqtt'],
            trigger_analyzer=deps['trigger_analyzer'],
            device_tracker=deps['device_tracker'],
            rule_engine=deps['rule_engine'],
            knowledge_base=deps['kb'],
            messages_deque=deps['messages_deque'],
            lock=deps['lock'],
            callbacks=deps['callbacks']
        )

        collector._process_message('zigbee2mqtt/sensor', '{"state": "ON"}', is_retained=True)

        assert len(deps['messages_deque']) == 1
        assert deps['messages_deque'][0].startswith('[RETAINED]')

    def test_updates_device_tracker(self, collector_dependencies):
        """Test that device tracker is updated."""
        deps = collector_dependencies
        deps['trigger_analyzer'].analyze.return_value = MagicMock(should_trigger=False)
        
        collector = MqttCollector(
            config=deps['config'],
            mqtt_client=deps['mqtt'],
            trigger_analyzer=deps['trigger_analyzer'],
            device_tracker=deps['device_tracker'],
            rule_engine=deps['rule_engine'],
            knowledge_base=deps['kb'],
            messages_deque=deps['messages_deque'],
            lock=deps['lock'],
            callbacks=deps['callbacks']
        )

        collector._process_message('zigbee2mqtt/sensor', '{"state": "ON"}')

        deps['device_tracker'].update.assert_called_once_with(
            'zigbee2mqtt/sensor', '{"state": "ON"}'
        )

    def test_trigger_calls_rule_engine(self, collector_dependencies):
        """Test that triggers are checked against rule engine."""
        deps = collector_dependencies
        mock_trigger = MagicMock()
        mock_trigger.should_trigger = True
        mock_trigger.reason = "state_change"
        mock_trigger.field_name = "state"
        mock_trigger.old_value = "OFF"
        mock_trigger.new_value = "ON"
        
        deps['trigger_analyzer'].analyze.return_value = mock_trigger
        deps['rule_engine'].check_and_execute.return_value = True  # Rule handled
        
        collector = MqttCollector(
            config=deps['config'],
            mqtt_client=deps['mqtt'],
            trigger_analyzer=deps['trigger_analyzer'],
            device_tracker=deps['device_tracker'],
            rule_engine=deps['rule_engine'],
            knowledge_base=deps['kb'],
            messages_deque=deps['messages_deque'],
            lock=deps['lock'],
            callbacks=deps['callbacks']
        )

        collector._process_message('zigbee2mqtt/sensor', '{"state": "ON"}')

        deps['rule_engine'].check_and_execute.assert_called_once()
        # Callback should NOT be called since rule handled it
        deps['callbacks'].on_trigger.assert_not_called()

    def test_unhandled_trigger_calls_callback(self, collector_dependencies):
        """Test that unhandled triggers call the on_trigger callback."""
        deps = collector_dependencies
        mock_trigger = MagicMock()
        mock_trigger.should_trigger = True
        mock_trigger.reason = "state_change"
        mock_trigger.field_name = "state"
        mock_trigger.old_value = "OFF"
        mock_trigger.new_value = "ON"
        
        deps['trigger_analyzer'].analyze.return_value = mock_trigger
        deps['rule_engine'].check_and_execute.return_value = False  # Not handled
        
        collector = MqttCollector(
            config=deps['config'],
            mqtt_client=deps['mqtt'],
            trigger_analyzer=deps['trigger_analyzer'],
            device_tracker=deps['device_tracker'],
            rule_engine=deps['rule_engine'],
            knowledge_base=deps['kb'],
            messages_deque=deps['messages_deque'],
            lock=deps['lock'],
            callbacks=deps['callbacks']
        )

        collector._process_message('zigbee2mqtt/sensor', '{"state": "ON"}')

        # Callback should be called since rule didn't handle it
        deps['callbacks'].on_trigger.assert_called_once()


class TestMqttCollectorCompressLine:
    """Tests for line compression."""

    def test_compress_non_json_returns_original(self, collector_dependencies):
        """Test that non-JSON lines are returned as-is."""
        deps = collector_dependencies
        
        collector = MqttCollector(
            config=deps['config'],
            mqtt_client=deps['mqtt'],
            trigger_analyzer=deps['trigger_analyzer'],
            device_tracker=deps['device_tracker'],
            rule_engine=deps['rule_engine'],
            knowledge_base=deps['kb'],
            messages_deque=deps['messages_deque'],
            lock=deps['lock'],
            callbacks=deps['callbacks']
        )

        result = collector._compress_line("zigbee2mqtt/sensor plain text")
        assert result == "zigbee2mqtt/sensor plain text"

    def test_compress_removes_noise_fields(self, collector_dependencies):
        """Test that noise fields are removed."""
        deps = collector_dependencies
        
        collector = MqttCollector(
            config=deps['config'],
            mqtt_client=deps['mqtt'],
            trigger_analyzer=deps['trigger_analyzer'],
            device_tracker=deps['device_tracker'],
            rule_engine=deps['rule_engine'],
            knowledge_base=deps['kb'],
            messages_deque=deps['messages_deque'],
            lock=deps['lock'],
            callbacks=deps['callbacks']
        )

        # linkquality is a noise field
        line = 'zigbee2mqtt/sensor {"state":"ON","linkquality":120}'
        result = collector._compress_line(line)
        
        assert result is not None
        assert "state" in result
        assert "linkquality" not in result

    def test_compress_returns_none_for_all_noise(self, collector_dependencies):
        """Test that lines with only noise fields return None."""
        deps = collector_dependencies
        
        collector = MqttCollector(
            config=deps['config'],
            mqtt_client=deps['mqtt'],
            trigger_analyzer=deps['trigger_analyzer'],
            device_tracker=deps['device_tracker'],
            rule_engine=deps['rule_engine'],
            knowledge_base=deps['kb'],
            messages_deque=deps['messages_deque'],
            lock=deps['lock'],
            callbacks=deps['callbacks']
        )

        # Only noise fields (from AiAgent.REMOVE_FIELDS)
        line = 'zigbee2mqtt/sensor {"linkquality":120,"voltage":3.2,"battery":85}'
        result = collector._compress_line(line)
        
        assert result is None


class TestMqttCollectorPrintTrigger:
    """Tests for trigger printing."""

    def test_print_trigger_outputs_correctly(self, collector_dependencies, capsys):
        """Test that trigger prints correctly."""
        deps = collector_dependencies
        deps['config'].compress_output = False
        
        collector = MqttCollector(
            config=deps['config'],
            mqtt_client=deps['mqtt'],
            trigger_analyzer=deps['trigger_analyzer'],
            device_tracker=deps['device_tracker'],
            rule_engine=deps['rule_engine'],
            knowledge_base=deps['kb'],
            messages_deque=deps['messages_deque'],
            lock=deps['lock'],
            callbacks=deps['callbacks']
        )

        mock_trigger = MagicMock()
        mock_trigger.reason = "state_change"
        mock_trigger.field_name = "occupancy"
        mock_trigger.old_value = False
        mock_trigger.new_value = True

        collector._print_trigger(mock_trigger, 'zigbee2mqtt/pir {"occupancy":true}')

        captured = capsys.readouterr()
        assert "SMART TRIGGER" in captured.out
        assert "occupancy" in captured.out


# Fixtures
@pytest.fixture
def collector_dependencies():
    """Create mock dependencies for MqttCollector."""
    config = MagicMock()
    config.mqtt_topics = ['zigbee2mqtt/#']
    config.ignore_printing_prefixes = []
    config.ignore_printing_topics = []
    config.verbose = False
    config.compress_output = False
    config.simulation_file = None
    
    return {
        'config': config,
        'mqtt': MagicMock(),
        'trigger_analyzer': MagicMock(),
        'device_tracker': MagicMock(),
        'rule_engine': MagicMock(),
        'kb': MagicMock(),
        'messages_deque': deque(maxlen=100),
        'lock': threading.Lock(),
        'callbacks': CollectorCallbacks(
            on_trigger=MagicMock(),
            on_shutdown=MagicMock(),
            wait_for_ai=MagicMock()
        )
    }

