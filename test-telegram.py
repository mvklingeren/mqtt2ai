#!/usr/bin/env python3
"""Test script for Telegram bot connectivity.

This script tests sending and receiving Telegram messages using
the credentials from your .env file.

Usage:
    python test-telegram.py           # Send a test message
    python test-telegram.py --listen  # Send and then listen for replies
"""

import asyncio
import os
import sys
from datetime import datetime

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed, using environment variables only")

try:
    from telegram import Update
    from telegram.ext import (
        Application,
        MessageHandler,
        CommandHandler,
        ContextTypes,
        filters
    )
    from telegram.constants import ParseMode
except ImportError:
    print("❌ python-telegram-bot not installed!")
    print("   Run: pip install python-telegram-bot")
    sys.exit(1)


# Get credentials from environment
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_IDS = os.environ.get("TELEGRAM_CHAT_IDS", "")


async def send_test_message():
    """Send a test message to all configured chat IDs."""
    if not BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not set in .env")
        return False
    
    if not CHAT_IDS:
        print("❌ TELEGRAM_CHAT_IDS not set in .env")
        print("   First, message your bot and run with --get-chat-id to find your ID")
        return False
    
    # Parse chat IDs
    chat_ids = []
    for id_str in CHAT_IDS.split(","):
        id_str = id_str.strip()
        if id_str:
            try:
                chat_ids.append(int(id_str))
            except ValueError:
                print(f"⚠️  Invalid chat ID: {id_str}")
    
    if not chat_ids:
        print("❌ No valid chat IDs found")
        return False
    
    print(f"📱 Bot token: {BOT_TOKEN[:10]}...{BOT_TOKEN[-5:]}")
    print(f"💬 Chat IDs: {chat_ids}")
    
    # Create bot application
    app = Application.builder().token(BOT_TOKEN).build()
    
    try:
        await app.initialize()
        
        # Get bot info
        bot_info = await app.bot.get_me()
        print(f"🤖 Bot: @{bot_info.username} ({bot_info.first_name})")
        
        # Send test message to each chat
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = (
            f"🧪 *MQTT2AI Test Message*\n\n"
            f"✅ Telegram bot is working!\n"
            f"⏰ Time: `{timestamp}`\n\n"
            f"_Reply to this message to test receiving._"
        )
        
        for chat_id in chat_ids:
            try:
                await app.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode=ParseMode.MARKDOWN
                )
                print(f"✅ Message sent to chat {chat_id}")
            except Exception as e:
                print(f"❌ Failed to send to {chat_id}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bot error: {e}")
        return False
    finally:
        await app.shutdown()


async def listen_for_messages():
    """Listen for incoming messages and print them."""
    if not BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not set")
        return
    
    print("\n👂 Listening for messages... (Press Ctrl+C to stop)\n")
    
    app = Application.builder().token(BOT_TOKEN).build()
    
    async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        chat_id = update.effective_chat.id
        user = update.effective_user
        print(f"📥 /start from {user.first_name} (chat_id: {chat_id})")
        await update.message.reply_text(
            f"👋 Hello {user.first_name}!\n\n"
            f"Your chat ID is: `{chat_id}`\n\n"
            f"Add this to your `.env` file:\n"
            f"`TELEGRAM_CHAT_IDS={chat_id}`",
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle any text message."""
        chat_id = update.effective_chat.id
        user = update.effective_user
        text = update.message.text
        
        print(f"📥 Message from {user.first_name} (chat_id: {chat_id}): {text}")
        
        # Echo back
        await update.message.reply_text(
            f"📨 Received: _{text}_\n\n"
            f"Chat ID: `{chat_id}`",
            parse_mode=ParseMode.MARKDOWN
        )
    
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    try:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n👋 Stopping...")
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


async def get_chat_id():
    """Start bot and wait for a message to get the chat ID."""
    if not BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not set")
        return
    
    print("🔍 Waiting for a message to detect your chat ID...")
    print("   Send any message to your bot, then press Ctrl+C\n")
    
    app = Application.builder().token(BOT_TOKEN).build()
    
    async def handle_any(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        user = update.effective_user
        print(f"\n✅ Detected chat ID: {chat_id}")
        print(f"   User: {user.first_name} (@{user.username})")
        print(f"\n   Add to .env: TELEGRAM_CHAT_IDS={chat_id}\n")
        
        await update.message.reply_text(
            f"Your chat ID: `{chat_id}`",
            parse_mode=ParseMode.MARKDOWN
        )
    
    app.add_handler(MessageHandler(filters.ALL, handle_any))
    
    try:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n👋 Done")
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Telegram bot connectivity")
    parser.add_argument(
        "--listen", "-l",
        action="store_true",
        help="Listen for incoming messages after sending test"
    )
    parser.add_argument(
        "--get-chat-id",
        action="store_true",
        help="Wait for a message to detect your chat ID"
    )
    parser.add_argument(
        "--receive-only", "-r",
        action="store_true",
        help="Only listen for messages, don't send"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("  MQTT2AI Telegram Bot Tester")
    print("=" * 50 + "\n")
    
    if args.get_chat_id:
        asyncio.run(get_chat_id())
    elif args.receive_only:
        asyncio.run(listen_for_messages())
    else:
        success = asyncio.run(send_test_message())
        
        if success and args.listen:
            asyncio.run(listen_for_messages())
        elif success:
            print("\n💡 Run with --listen to receive replies")
            print("   Run with --get-chat-id if you need to find your chat ID")


if __name__ == "__main__":
    main()

