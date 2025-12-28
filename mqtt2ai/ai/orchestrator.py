"""AI Orchestrator for async AI request processing.

This module manages the AI request queue and worker thread, decoupling
trigger detection from AI analysis to prevent blocking the main loop.
"""
import queue
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

from mqtt2ai.rules.trigger_analyzer import TriggerResult
from mqtt2ai.ai.agent import AiAgent
from mqtt2ai.rules.knowledge_base import KnowledgeBase


@dataclass
class AiRequest:
    """Represents a request for AI analysis."""
    snapshot: str
    reason: str
    trigger_result: Optional[TriggerResult] = None
    created_at: float = field(default_factory=time.time)

    def is_expired(self, max_age_seconds: float = 30.0) -> bool:
        """Check if this request is too old to be relevant."""
        return (time.time() - self.created_at) > max_age_seconds


class AiOrchestrator:
    """Manages async AI request queue and worker thread.
    
    This class handles:
    - Queuing AI analysis requests from triggers
    - Deduplicating consecutive identical triggers
    - Processing requests in a background worker thread
    - Skipping expired requests to avoid stale analysis
    """

    def __init__(
        self,
        ai_agent: AiAgent,
        knowledge_base: KnowledgeBase,
        no_ai: bool = False,
        queue_size: int = 20
    ):
        """Initialize the AI orchestrator.
        
        Args:
            ai_agent: AiAgent instance for running analysis
            knowledge_base: KnowledgeBase instance for context
            no_ai: If True, log but don't process requests
            queue_size: Maximum queue size before dropping requests
        """
        self.ai = ai_agent
        self.kb = knowledge_base
        self.no_ai = no_ai

        self._queue: queue.Queue[Optional[AiRequest]] = queue.Queue(maxsize=queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_queued_trigger: Optional[tuple[str, str]] = None

    def start(self) -> None:
        """Start the AI worker thread."""
        if self.no_ai:
            logging.debug("AI orchestrator not starting (no_ai mode)")
            return

        if self._running:
            logging.warning("AI orchestrator already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="AI-Worker"
        )
        self._thread.start()
        logging.info("AI orchestrator started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the AI worker thread gracefully.
        
        Args:
            timeout: Maximum time to wait for thread to stop
        """
        if not self._running:
            return

        self._running = False

        if self._thread and self._thread.is_alive():
            logging.debug("Signaling AI worker to stop...")
            self._queue.put(None)  # Shutdown signal
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logging.warning("AI worker thread did not stop in time")
            else:
                logging.debug("AI worker thread stopped")

        self._thread = None

    def queue_request(
        self,
        snapshot: str,
        reason: str,
        trigger_result: Optional[TriggerResult] = None
    ) -> bool:
        """Queue an AI analysis request.
        
        Args:
            snapshot: MQTT message snapshot for analysis
            reason: Reason for triggering the analysis
            trigger_result: Optional trigger details
            
        Returns:
            True if request was queued, False if dropped or deduplicated
        """
        if self.no_ai:
            logging.info(
                "[NO-AI MODE] Would have triggered AI check (reason: %s), %d messages",
                reason,
                len(snapshot.splitlines())
            )
            return False

        # Deduplicate: skip if same topic/field as last queued trigger
        if trigger_result and trigger_result.topic and trigger_result.field_name:
            trigger_key = (trigger_result.topic, trigger_result.field_name)
            if trigger_key == self._last_queued_trigger:
                logging.debug(
                    "Deduplicating trigger for %s[%s]",
                    trigger_result.topic, trigger_result.field_name
                )
                return False
            self._last_queued_trigger = trigger_key

        request = AiRequest(
            snapshot=snapshot,
            reason=reason,
            trigger_result=trigger_result
        )

        try:
            self._queue.put_nowait(request)
            logging.debug("Queued AI request (reason: %s)", reason)
            return True
        except queue.Full:
            logging.warning(
                "AI queue full (size=%d), dropping request (reason: %s)",
                self._queue.maxsize, reason
            )
            return False

    def wait_for_completion(self, timeout: float = 60.0) -> None:
        """Wait for all queued requests to be processed.
        
        Used in simulation mode to ensure deterministic pattern learning
        by waiting for each trigger to be fully processed before continuing.
        
        Args:
            timeout: Maximum time to wait for queue to empty
        """
        if self.no_ai:
            return

        # Wait for queue to be empty
        start_wait = time.time()
        while not self._queue.empty() and self._running:
            if time.time() - start_wait > timeout:
                logging.warning("AI completion wait timed out (queue not empty)")
                return
            time.sleep(0.1)

        # Then join the queue to ensure current task is done
        self._queue.join()

    def is_running(self) -> bool:
        """Check if the worker thread is running."""
        return self._running and self._thread is not None and self._thread.is_alive()

    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    def _worker_loop(self) -> None:
        """Background thread that processes AI requests from the queue."""
        logging.info("AI worker thread started")

        while self._running:
            try:
                # Wait for a request with timeout to allow checking self._running
                request = self._queue.get(timeout=1.0)

                # Wrap everything after get() to ensure task_done() is always called
                try:
                    # None is the shutdown signal
                    if request is None:
                        logging.debug("AI worker received shutdown signal")
                        break

                    # Skip expired requests to avoid processing stale data
                    if request.is_expired():
                        age = time.time() - request.created_at
                        logging.info(
                            "Skipping expired AI request (age: %.1fs, reason: %s)",
                            age, request.reason
                        )
                        continue

                    # Reload knowledge base to get any updates
                    self.kb.load_all()
                    self.ai.run_analysis(
                        request.snapshot,
                        self.kb,
                        request.reason,
                        request.trigger_result
                    )
                finally:
                    self._queue.task_done()

            except queue.Empty:
                # Timeout, just continue to check self._running
                continue
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.error("AI worker error: %s", e)

        logging.info("AI worker thread stopped")

