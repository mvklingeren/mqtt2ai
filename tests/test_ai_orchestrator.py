"""Tests for the AI Orchestrator module."""
import queue
import threading
import time
from unittest.mock import MagicMock, patch

from ai_orchestrator import AiOrchestrator, AiRequest


class TestAiRequest:
    """Tests for the AiRequest dataclass."""

    def test_request_created_with_timestamp(self):
        """Test that request gets created_at timestamp."""
        before = time.time()
        request = AiRequest(snapshot="test", reason="test_reason")
        after = time.time()

        assert before <= request.created_at <= after

    def test_request_not_expired_immediately(self):
        """Test that new request is not expired."""
        request = AiRequest(snapshot="test", reason="test_reason")
        assert not request.is_expired()

    def test_request_expired_after_max_age(self):
        """Test that request expires after max age."""
        request = AiRequest(snapshot="test", reason="test_reason")
        # Manually set created_at to 31 seconds ago
        request.created_at = time.time() - 31.0
        assert request.is_expired(max_age_seconds=30.0)

    def test_request_not_expired_before_max_age(self):
        """Test that request not expired before max age."""
        request = AiRequest(snapshot="test", reason="test_reason")
        # Manually set created_at to 29 seconds ago
        request.created_at = time.time() - 29.0
        assert not request.is_expired(max_age_seconds=30.0)

    def test_request_with_trigger_result(self):
        """Test request can store trigger result."""
        mock_trigger = MagicMock()
        mock_trigger.topic = "test/topic"
        mock_trigger.field_name = "state"

        request = AiRequest(
            snapshot="test",
            reason="smart_trigger",
            trigger_result=mock_trigger
        )

        assert request.trigger_result == mock_trigger


class TestAiOrchestratorInit:
    """Tests for AiOrchestrator initialization."""

    def test_init_with_no_ai_mode(self):
        """Test initialization in no-AI mode."""
        mock_ai = MagicMock()
        mock_kb = MagicMock()

        orchestrator = AiOrchestrator(
            ai_agent=mock_ai,
            knowledge_base=mock_kb,
            no_ai=True
        )

        assert orchestrator.no_ai is True
        assert not orchestrator.is_running()

    def test_init_with_custom_queue_size(self):
        """Test initialization with custom queue size."""
        mock_ai = MagicMock()
        mock_kb = MagicMock()

        orchestrator = AiOrchestrator(
            ai_agent=mock_ai,
            knowledge_base=mock_kb,
            queue_size=50
        )

        assert orchestrator._queue.maxsize == 50

    def test_init_default_queue_size(self):
        """Test initialization with default queue size."""
        mock_ai = MagicMock()
        mock_kb = MagicMock()

        orchestrator = AiOrchestrator(
            ai_agent=mock_ai,
            knowledge_base=mock_kb
        )

        assert orchestrator._queue.maxsize == 20


class TestAiOrchestratorStartStop:
    """Tests for start/stop functionality."""

    def test_start_creates_worker_thread(self):
        """Test that start creates and starts worker thread."""
        mock_ai = MagicMock()
        mock_kb = MagicMock()

        orchestrator = AiOrchestrator(
            ai_agent=mock_ai,
            knowledge_base=mock_kb,
            no_ai=False
        )

        orchestrator.start()

        try:
            assert orchestrator.is_running()
            assert orchestrator._thread is not None
            assert orchestrator._thread.is_alive()
        finally:
            orchestrator.stop()

    def test_start_does_nothing_in_no_ai_mode(self):
        """Test that start does nothing in no-AI mode."""
        mock_ai = MagicMock()
        mock_kb = MagicMock()

        orchestrator = AiOrchestrator(
            ai_agent=mock_ai,
            knowledge_base=mock_kb,
            no_ai=True
        )

        orchestrator.start()

        assert not orchestrator.is_running()
        assert orchestrator._thread is None

    def test_stop_stops_worker_thread(self):
        """Test that stop cleanly stops worker thread."""
        mock_ai = MagicMock()
        mock_kb = MagicMock()

        orchestrator = AiOrchestrator(
            ai_agent=mock_ai,
            knowledge_base=mock_kb,
            no_ai=False
        )

        orchestrator.start()
        assert orchestrator.is_running()

        orchestrator.stop()

        assert not orchestrator.is_running()

    def test_double_start_is_safe(self):
        """Test that calling start twice is safe."""
        mock_ai = MagicMock()
        mock_kb = MagicMock()

        orchestrator = AiOrchestrator(
            ai_agent=mock_ai,
            knowledge_base=mock_kb,
            no_ai=False
        )

        orchestrator.start()
        first_thread = orchestrator._thread

        orchestrator.start()  # Should not create new thread

        assert orchestrator._thread is first_thread
        orchestrator.stop()

    def test_double_stop_is_safe(self):
        """Test that calling stop twice is safe."""
        mock_ai = MagicMock()
        mock_kb = MagicMock()

        orchestrator = AiOrchestrator(
            ai_agent=mock_ai,
            knowledge_base=mock_kb,
            no_ai=False
        )

        orchestrator.start()
        orchestrator.stop()
        orchestrator.stop()  # Should not raise


class TestAiOrchestratorQueueRequest:
    """Tests for queue_request functionality."""

    def test_queue_request_returns_true_when_queued(self):
        """Test that queue_request returns True when request is queued."""
        mock_ai = MagicMock()
        mock_kb = MagicMock()

        orchestrator = AiOrchestrator(
            ai_agent=mock_ai,
            knowledge_base=mock_kb,
            no_ai=False
        )

        result = orchestrator.queue_request("snapshot", "test_reason")

        assert result is True
        assert orchestrator.queue_size() == 1

    def test_queue_request_returns_false_in_no_ai_mode(self):
        """Test that queue_request returns False in no-AI mode."""
        mock_ai = MagicMock()
        mock_kb = MagicMock()

        orchestrator = AiOrchestrator(
            ai_agent=mock_ai,
            knowledge_base=mock_kb,
            no_ai=True
        )

        result = orchestrator.queue_request("snapshot", "test_reason")

        assert result is False
        assert orchestrator.queue_size() == 0

    def test_queue_request_deduplicates_same_trigger(self):
        """Test that identical triggers are deduplicated."""
        mock_ai = MagicMock()
        mock_kb = MagicMock()

        mock_trigger = MagicMock()
        mock_trigger.topic = "test/topic"
        mock_trigger.field_name = "state"

        orchestrator = AiOrchestrator(
            ai_agent=mock_ai,
            knowledge_base=mock_kb,
            no_ai=False
        )

        # First request should be queued
        result1 = orchestrator.queue_request(
            "snapshot1", "reason1", mock_trigger
        )
        # Second identical trigger should be deduplicated
        result2 = orchestrator.queue_request(
            "snapshot2", "reason2", mock_trigger
        )

        assert result1 is True
        assert result2 is False
        assert orchestrator.queue_size() == 1

    def test_queue_request_allows_different_triggers(self):
        """Test that different triggers are not deduplicated."""
        mock_ai = MagicMock()
        mock_kb = MagicMock()

        mock_trigger1 = MagicMock()
        mock_trigger1.topic = "test/topic1"
        mock_trigger1.field_name = "state"

        mock_trigger2 = MagicMock()
        mock_trigger2.topic = "test/topic2"
        mock_trigger2.field_name = "state"

        orchestrator = AiOrchestrator(
            ai_agent=mock_ai,
            knowledge_base=mock_kb,
            no_ai=False
        )

        result1 = orchestrator.queue_request(
            "snapshot1", "reason1", mock_trigger1
        )
        result2 = orchestrator.queue_request(
            "snapshot2", "reason2", mock_trigger2
        )

        assert result1 is True
        assert result2 is True
        assert orchestrator.queue_size() == 2

    def test_queue_request_drops_when_full(self):
        """Test that request is dropped when queue is full."""
        mock_ai = MagicMock()
        mock_kb = MagicMock()

        orchestrator = AiOrchestrator(
            ai_agent=mock_ai,
            knowledge_base=mock_kb,
            no_ai=False,
            queue_size=2
        )

        # Fill the queue
        orchestrator.queue_request("snapshot1", "reason1")
        orchestrator.queue_request("snapshot2", "reason2")

        # This should be dropped
        result = orchestrator.queue_request("snapshot3", "reason3")

        assert result is False
        assert orchestrator.queue_size() == 2


class TestAiOrchestratorWorker:
    """Tests for the worker thread processing."""

    def test_worker_processes_request(self):
        """Test that worker processes queued request."""
        mock_ai = MagicMock()
        mock_kb = MagicMock()

        orchestrator = AiOrchestrator(
            ai_agent=mock_ai,
            knowledge_base=mock_kb,
            no_ai=False
        )

        orchestrator.start()

        try:
            orchestrator.queue_request("test_snapshot", "test_reason")
            orchestrator.wait_for_completion(timeout=5.0)

            # Verify AI was called
            mock_kb.load_all.assert_called()
            mock_ai.run_analysis.assert_called_once()

            call_args = mock_ai.run_analysis.call_args
            assert call_args[0][0] == "test_snapshot"
            assert call_args[0][2] == "test_reason"
        finally:
            orchestrator.stop()

    def test_worker_skips_expired_requests(self):
        """Test that worker skips expired requests."""
        mock_ai = MagicMock()
        mock_kb = MagicMock()

        orchestrator = AiOrchestrator(
            ai_agent=mock_ai,
            knowledge_base=mock_kb,
            no_ai=False
        )

        # Manually create an expired request
        request = AiRequest(snapshot="test", reason="expired")
        request.created_at = time.time() - 60.0  # 60 seconds ago

        orchestrator._queue.put(request)

        orchestrator.start()

        try:
            orchestrator.wait_for_completion(timeout=5.0)

            # AI should not have been called for expired request
            mock_ai.run_analysis.assert_not_called()
        finally:
            orchestrator.stop()

    def test_worker_handles_ai_errors(self):
        """Test that worker handles AI errors gracefully."""
        mock_ai = MagicMock()
        mock_ai.run_analysis.side_effect = Exception("AI error")
        mock_kb = MagicMock()

        orchestrator = AiOrchestrator(
            ai_agent=mock_ai,
            knowledge_base=mock_kb,
            no_ai=False
        )

        orchestrator.start()

        try:
            orchestrator.queue_request("test_snapshot", "test_reason")
            orchestrator.wait_for_completion(timeout=5.0)

            # Should have attempted to call AI
            mock_ai.run_analysis.assert_called_once()

            # Worker should still be running after error
            assert orchestrator.is_running()
        finally:
            orchestrator.stop()


class TestAiOrchestratorWaitForCompletion:
    """Tests for wait_for_completion functionality."""

    def test_wait_for_completion_returns_when_empty(self):
        """Test that wait_for_completion returns when queue is empty."""
        mock_ai = MagicMock()
        mock_kb = MagicMock()

        orchestrator = AiOrchestrator(
            ai_agent=mock_ai,
            knowledge_base=mock_kb,
            no_ai=False
        )

        orchestrator.start()

        try:
            # Queue should be empty, should return immediately
            start = time.time()
            orchestrator.wait_for_completion(timeout=1.0)
            elapsed = time.time() - start

            # Should return quickly (not wait for timeout)
            assert elapsed < 0.5
        finally:
            orchestrator.stop()

    def test_wait_for_completion_noop_in_no_ai_mode(self):
        """Test that wait_for_completion is a no-op in no-AI mode."""
        mock_ai = MagicMock()
        mock_kb = MagicMock()

        orchestrator = AiOrchestrator(
            ai_agent=mock_ai,
            knowledge_base=mock_kb,
            no_ai=True
        )

        # Should return immediately without error
        start = time.time()
        orchestrator.wait_for_completion(timeout=1.0)
        elapsed = time.time() - start

        assert elapsed < 0.1

