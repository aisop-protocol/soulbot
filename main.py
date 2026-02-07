import sys
import os
import logging

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.bot.handler import run_telegram_bot

if __name__ == "__main__":
    try:
        run_telegram_bot()
    except KeyboardInterrupt:
        print("Bot stopped by user.")
    except Exception as e:
        print(f"Fatal Error: {e}")
