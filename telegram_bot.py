"""Telegram Bot module for the MQTT AI Daemon.

This module provides bidirectional Telegram integration:
- Receive alerts and notifications from the daemon
- Send commands/questions to the AI for home automation control

Uses python-telegram-bot library (https://python-telegram-bot.org/)
"""
import asyncio
import json
import logging
import queue
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Optional, List, Callable

try:
    from telegram import Update
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        ContextTypes,
        filters
    )
    from telegram.constants import ParseMode
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = None
    ContextTypes = None

if TYPE_CHECKING:
    from config import Config
    from daemon import DeviceStateTracker


class TelegramBot:
    """Handles Telegram bot communication for the MQTT AI Daemon.

    This class manages:
    - Incoming messages from authorized users
    - Outgoing alerts and notifications
    - Integration with the AI agent for query processing
    """

    def __init__(
        self,
        config: 'Config',
        device_tracker: Optional['DeviceStateTracker'] = None,
        on_user_message: Optional[Callable[[int, str], str]] = None
    ):
        """Initialize the Telegram bot.

        Args:
            config: Application configuration with Telegram settings
            device_tracker: Device state tracker for context
            on_user_message: Callback function(chat_id, message) -> response
                           Called when an authorized user sends a message
        """
        self.config = config
        self.device_tracker = device_tracker
        self.on_user_message = on_user_message

        self._application: Optional[Application] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False

        # Queue for outgoing messages (thread-safe)
        self._outgoing_queue: queue.Queue[tuple[int, str]] = queue.Queue()

        # Parse authorized chat IDs
        self._authorized_chat_ids: set[int] = set()
        if config.telegram_chat_ids:
            for chat_id_str in config.telegram_chat_ids.split(","):
                chat_id_str = chat_id_str.strip()
                if chat_id_str:
                    try:
                        self._authorized_chat_ids.add(int(chat_id_str))
                    except ValueError:
                        logging.warning(
                            "Invalid Telegram chat ID: %s", chat_id_str
                        )

        if not self._authorized_chat_ids:
            logging.warning(
                "No authorized Telegram chat IDs configured. "
                "Set TELEGRAM_CHAT_IDS environment variable."
            )

    @property
    def is_available(self) -> bool:
        """Check if Telegram bot library is available."""
        return TELEGRAM_AVAILABLE

    @property
    def is_configured(self) -> bool:
        """Check if Telegram bot is properly configured."""
        return bool(
            self.config.telegram_bot_token and
            self._authorized_chat_ids
        )

    @property
    def authorized_chat_ids(self) -> List[int]:
        """Get list of authorized chat IDs."""
        return list(self._authorized_chat_ids)

    def _is_authorized(self, chat_id: int) -> bool:
        """Check if a chat ID is authorized to use the bot."""
        return chat_id in self._authorized_chat_ids

    def start(self) -> bool:
        """Start the Telegram bot in a background thread.

        Returns:
            True if started successfully, False otherwise
        """
        if not TELEGRAM_AVAILABLE:
            logging.error(
                "python-telegram-bot not installed. "
                "Run: pip install python-telegram-bot"
            )
            return False

        if not self.config.telegram_bot_token:
            logging.warning("Telegram bot token not configured, skipping")
            return False

        if not self._authorized_chat_ids:
            logging.warning(
                "No authorized chat IDs configured, Telegram bot disabled"
            )
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._run_bot,
            daemon=True,
            name="Telegram-Bot"
        )
        self._thread.start()

        logging.info(
            "Telegram bot started (authorized chat IDs: %s)",
            list(self._authorized_chat_ids)
        )
        return True

    def stop(self) -> None:
        """Stop the Telegram bot gracefully."""
        self._running = False

        # Just wait for the thread to stop - it handles its own cleanup
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                logging.warning("Telegram bot thread did not stop in time")

        logging.info("Telegram bot stopped")

    def _run_bot(self) -> None:
        """Run the bot in a new event loop (called from background thread)."""
        try:
            # Create a new event loop for this thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            # Build the application
            self._application = (
                Application.builder()
                .token(self.config.telegram_bot_token)
                .build()
            )

            # Add handlers
            self._application.add_handler(
                CommandHandler("start", self._handle_start)
            )
            self._application.add_handler(
                CommandHandler("status", self._handle_status)
            )
            self._application.add_handler(
                CommandHandler("devices", self._handle_devices)
            )
            self._application.add_handler(
                CommandHandler("help", self._handle_help)
            )
            # Handle all text messages
            self._application.add_handler(
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND,
                    self._handle_message
                )
            )

            # Run the bot
            self._loop.run_until_complete(self._run_polling())

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Telegram bot error: %s", e)
        finally:
            if self._loop:
                self._loop.close()

    async def _run_polling(self) -> None:
        """Run polling with proper initialization."""
        await self._application.initialize()
        await self._application.start()
        await self._application.updater.start_polling(drop_pending_updates=True)

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)

            # Process outgoing message queue
            await self._process_outgoing_queue()

        # Graceful shutdown in correct order:
        # 1. Stop updater (stops polling for updates)
        # 2. Stop application (stops handlers)
        # 3. Shutdown application (closes HTTP client)
        try:
            await self._application.updater.stop()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.debug("Updater stop error (expected during shutdown): %s", e)
        
        try:
            await self._application.stop()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.debug("Application stop error: %s", e)
        
        try:
            await self._application.shutdown()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.debug("Application shutdown error: %s", e)

    async def _process_outgoing_queue(self) -> None:
        """Process queued outgoing messages."""
        while not self._outgoing_queue.empty():
            try:
                chat_id, message = self._outgoing_queue.get_nowait()
                await self._application.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode=ParseMode.MARKDOWN
                )
                self._outgoing_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.error("Failed to send Telegram message: %s", e)

    async def _handle_start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /start command."""
        chat_id = update.effective_chat.id

        if not self._is_authorized(chat_id):
            await update.message.reply_text(
                f"⛔ Unauthorized. Your chat ID is: `{chat_id}`\n"
                "Add this to TELEGRAM_CHAT_IDS to authorize.",
                parse_mode=ParseMode.MARKDOWN
            )
            logging.warning(
                "Unauthorized Telegram access attempt from chat_id: %d",
                chat_id
            )
            return

        await update.message.reply_text(
            "🏠 *MQTT2AI Bot Ready*\n\n"
            "I can help you control your smart home and monitor alerts.\n\n"
            "*Commands:*\n"
            "/status - Show system status\n"
            "/devices - List tracked devices\n"
            "/help - Show this help\n\n"
            "Or just send me a message like:\n"
            "• _Turn on the kitchen light_\n"
            "• _What's the temperature in the living room?_\n"
            "• _Is anyone home?_",
            parse_mode=ParseMode.MARKDOWN
        )

    async def _handle_help(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /help command."""
        chat_id = update.effective_chat.id

        if not self._is_authorized(chat_id):
            await update.message.reply_text("⛔ Unauthorized")
            return

        await update.message.reply_text(
            "🏠 *MQTT2AI Bot Help*\n\n"
            "*Commands:*\n"
            "`/start` - Initialize bot\n"
            "`/status` - System status\n"
            "`/devices` - List devices\n"
            "`/help` - This help\n\n"
            "*Natural Language:*\n"
            "Just type what you want to do:\n"
            "• _Turn on living room light_\n"
            "• _Set bedroom to 21 degrees_\n"
            "• _What sensors are active?_\n"
            "• _Close the garage door_\n\n"
            "The AI will interpret your request and control devices via MQTT.",
            parse_mode=ParseMode.MARKDOWN
        )

    async def _handle_status(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /status command."""
        chat_id = update.effective_chat.id

        if not self._is_authorized(chat_id):
            await update.message.reply_text("⛔ Unauthorized")
            return

        device_count = 0
        if self.device_tracker:
            device_count = self.device_tracker.get_device_count()

        status_msg = (
            "📊 *System Status*\n\n"
            f"🔌 Tracked devices: {device_count}\n"
            f"🤖 AI Provider: {self.config.ai_provider}\n"
            f"📡 MQTT: {self.config.mqtt_host}:{self.config.mqtt_port}\n"
            f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}"
        )

        await update.message.reply_text(status_msg, parse_mode=ParseMode.MARKDOWN)

    async def _handle_devices(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /devices command."""
        chat_id = update.effective_chat.id

        if not self._is_authorized(chat_id):
            await update.message.reply_text("⛔ Unauthorized")
            return

        if not self.device_tracker:
            await update.message.reply_text("Device tracker not available")
            return

        states = self.device_tracker.get_all_states()
        if not states:
            await update.message.reply_text("No devices tracked yet")
            return

        # Group by device type (extract from topic)
        lines = ["📱 *Tracked Devices*\n"]

        # Limit to first 20 devices to avoid message too long
        for i, (topic, state) in enumerate(sorted(states.items())[:20]):
            # Extract device name from topic (last part)
            device_name = topic.split("/")[-1]

            # Get key state info
            state_info = []
            if "state" in state:
                state_info.append(f"state={state['state']}")
            if "occupancy" in state:
                state_info.append(f"occ={'✓' if state['occupancy'] else '✗'}")
            if "contact" in state:
                state_info.append(f"{'closed' if state['contact'] else 'open'}")
            if "temperature" in state:
                state_info.append(f"{state['temperature']}°C")

            state_str = ", ".join(state_info) if state_info else "..."
            lines.append(f"• `{device_name}`: {state_str}")

        if len(states) > 20:
            lines.append(f"\n_...and {len(states) - 20} more devices_")

        await update.message.reply_text(
            "\n".join(lines),
            parse_mode=ParseMode.MARKDOWN
        )

    async def _handle_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming text messages (non-commands)."""
        chat_id = update.effective_chat.id
        message_text = update.message.text

        if not self._is_authorized(chat_id):
            await update.message.reply_text(
                f"⛔ Unauthorized. Chat ID: `{chat_id}`",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        logging.info(
            "Telegram message from %d: %s",
            chat_id, message_text[:50]
        )

        # Show typing indicator
        await context.bot.send_chat_action(
            chat_id=chat_id,
            action="typing"
        )

        # Process via callback if available
        if self.on_user_message:
            try:
                response = self.on_user_message(chat_id, message_text)
                await update.message.reply_text(
                    response,
                    parse_mode=ParseMode.MARKDOWN
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.error("Error processing Telegram message: %s", e)
                await update.message.reply_text(
                    f"❌ Error processing request: {e}"
                )
        else:
            await update.message.reply_text(
                "🤖 AI processing not configured. "
                "Check daemon integration."
            )

    def send_message(self, chat_id: int, message: str) -> bool:
        """Queue a message to be sent to a specific chat.

        This is thread-safe and can be called from any thread.

        Args:
            chat_id: The Telegram chat ID to send to
            message: The message text (Markdown supported)

        Returns:
            True if queued successfully, False otherwise
        """
        if not self._running or not self._application:
            logging.warning("Telegram bot not running, cannot send message")
            return False

        self._outgoing_queue.put((chat_id, message))
        return True

    def broadcast_message(self, message: str) -> int:
        """Send a message to all authorized chat IDs.

        Args:
            message: The message text (Markdown supported)

        Returns:
            Number of messages queued
        """
        count = 0
        for chat_id in self._authorized_chat_ids:
            if self.send_message(chat_id, message):
                count += 1
        return count

    def send_alert(
        self,
        severity: float,
        reason: str,
        context: Optional[dict] = None
    ) -> int:
        """Send an alert notification to all authorized chats.

        Args:
            severity: Alert severity (0.0-1.0)
            reason: Alert description
            context: Optional additional context

        Returns:
            Number of messages queued
        """
        # Choose emoji based on severity
        if severity >= 0.7:
            emoji = "🚨"
            level = "HIGH"
        elif severity >= 0.3:
            emoji = "⚠️"
            level = "MEDIUM"
        else:
            emoji = "ℹ️"
            level = "LOW"

        message = f"{emoji} *Alert ({level})*\n\n{reason}"

        if context:
            context_str = json.dumps(context, indent=2)
            if len(context_str) < 500:
                message += f"\n\n```\n{context_str}\n```"

        message += f"\n\n_Time: {datetime.now().strftime('%H:%M:%S')}_"

        return self.broadcast_message(message)

