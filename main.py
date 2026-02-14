"""
SoulBot 统一入口

一个命令启动：
- Streamlit Web 界面 (http://127.0.0.1:8080)
- Telegram Bot (可在 Web 中配置和控制)

使用方法：
    python main.py              # 启动 Web + 自动启动 Telegram (如已配置)
    python main.py --web-only   # 仅启动 Web 界面
    python main.py --bot-only   # 仅启动 Telegram Bot
    python main.py --port 3000  # 指定端口
"""

import sys
import os
import subprocess
import signal
import time
import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 加载 .env 文件
from dotenv import load_dotenv
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
    print(f"📄 Loaded .env from {_env_path}")

# Streamlit app path
STREAMLIT_APP = os.path.join(os.path.dirname(__file__), "streamlit_app", "app.py")

# Windows: "nul" is a reserved device name (like /dev/null on Linux).
# It appears in file explorers but is NOT a real file (isfile=False, mtime=epoch 0).
# If a real nul file is somehow created by Node.js subprocesses, clean it up.
_nul_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nul")
if os.path.isfile(_nul_file):
    try:
        os.remove(_nul_file)
    except OSError:
        pass

# Auto-retry configuration
MAX_RETRIES = 3
RETRY_COOLDOWN = 30  # seconds


def send_crash_notification(error_msg: str):
    """Send final crash notification via Telegram HTTP API (no bot framework needed)"""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return

    try:
        from src.bot.handler import get_last_chat_id
        chat_id = get_last_chat_id()
    except Exception:
        chat_id = None

    if not chat_id:
        print("   No active chat_id cached, skip Telegram notification")
        return

    try:
        import urllib.request
        import json

        text = (
            f"⚠️ SoulBot crashed {MAX_RETRIES} times and stopped.\n\n"
            f"Error: {error_msg[:200]}\n\n"
            f"Please restart manually: python main.py"
        )

        data = json.dumps({
            "chat_id": chat_id,
            "text": text,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=10)
        print(f"   📨 Crash notification sent via Telegram")
    except Exception:
        pass


def run_streamlit(port: int = 8080):
    """运行 Streamlit Web 界面"""
    print("🌐 Starting Streamlit Web Interface...")
    print(f"   URL: http://127.0.0.1:{port}")
    print("")

    # 使用 subprocess 启动 Streamlit
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", STREAMLIT_APP,
         "--server.address", "0.0.0.0",
         "--server.port", str(port),
         "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    return process, port


def run_telegram_bot_thread() -> bool:
    """在线程中运行 Telegram Bot，返回是否启动成功"""
    from streamlit_app.utils.telegram_controller import get_telegram_controller

    controller = get_telegram_controller()
    token = controller.get_bot_token()

    # 如果有 token（来自 .env 或配置文件），自动启动
    if token:
        print("🤖 Starting Telegram Bot...")
        print(f"   Token: {token[:10]}...{token[-5:]}")
        success, msg = controller.start_bot()
        if success:
            print(f"   ✓ {msg}")
            return True
        else:
            print(f"   ✗ {msg}")
            return False
    else:
        print("🤖 Telegram Bot: No token configured")
        print("   Add TELEGRAM_BOT_TOKEN to .env or configure in Web Settings")
        return False


def run_telegram_only():
    """仅运行 Telegram Bot (传统模式)"""
    from src.bot.handler import run_telegram_bot
    print("🤖 Starting Telegram Bot only...")
    run_telegram_bot()


def main():
    parser = argparse.ArgumentParser(description="SoulBot - AI Assistant")
    parser.add_argument("--web-only", action="store_true", help="Only start Web interface")
    parser.add_argument("--bot-only", action="store_true", help="Only start Telegram Bot")
    parser.add_argument("--port", type=int, default=8080, help="Streamlit port (default: 8080)")
    args = parser.parse_args()

    print("=" * 50)
    print("  SoulBot AI Assistant")
    print("=" * 50)
    print("")

    if args.bot_only:
        # 仅 Telegram 模式
        run_telegram_only()
        return

    # 启动 Streamlit
    streamlit_process, port = run_streamlit(args.port)

    telegram_running = False
    if not args.web_only:
        # 等待 Streamlit 启动
        time.sleep(3)
        # 启动 Telegram Bot (如果配置了自动启动)
        telegram_running = run_telegram_bot_thread()

    print("")
    print("=" * 50)
    print("  Services Running:")
    print(f"  - Web UI: http://127.0.0.1:{port}")
    if telegram_running:
        print("  - Telegram: Running ✓")
    else:
        print("  - Telegram: Not running (configure in Settings)")
    print("")
    print("  Press Ctrl+C to stop all services")
    print("=" * 50)
    print("")

    # 处理退出信号
    def signal_handler(sig, frame):
        print("\n🛑 Shutting down...")

        # 停止 Telegram Bot
        try:
            from streamlit_app.utils.telegram_controller import get_telegram_controller
            controller = get_telegram_controller()
            if controller.is_running():
                controller.stop_bot()
                print("   ✓ Telegram Bot stopped")
        except Exception:
            pass

        # 停止 Streamlit
        if streamlit_process:
            streamlit_process.terminate()
            try:
                streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                streamlit_process.kill()
            print("   ✓ Streamlit stopped")

        print("\nGoodbye! 👋")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, signal_handler)

    # 等待 Streamlit 进程
    try:
        # 转发 Streamlit 输出
        for line in streamlit_process.stdout:
            # 过滤一些噪音日志
            if not any(skip in line for skip in [
                "Watching for changes",
                "You can now view",
                "Local URL",
                "Network URL",
                "External URL",
            ]):
                print(line, end="")
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            main()
            break
        except KeyboardInterrupt:
            print("\nBot stopped by user.")
            break
        except Exception as e:
            print(f"\n💥 Crash (attempt {attempt}/{MAX_RETRIES}): {e}")
            logger.error("Crash details:", exc_info=True)
            if attempt < MAX_RETRIES:
                print(f"🔄 Restarting in {RETRY_COOLDOWN}s...")
                time.sleep(RETRY_COOLDOWN)
            else:
                print(f"🛑 All {MAX_RETRIES} retries failed. SoulBot stopped.")
                send_crash_notification(str(e))
