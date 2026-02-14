"""
Telegram Bot 控制器

在 Web 界面中控制 Telegram Bot 的启动/停止
支持从 .env 文件读取配置
"""

import os
import sys
import json
import asyncio
import threading
from pathlib import Path
from typing import Optional
import signal

# 添加项目路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 加载 .env 文件
from dotenv import load_dotenv
_env_path = Path(_project_root) / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


class TelegramController:
    """Telegram Bot 控制器 - 单例模式"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._bot_thread: Optional[threading.Thread] = None
        self._bot_running = False
        self._bot_app = None
        self._stop_event = threading.Event()

        # 配置文件路径
        self._config_dir = Path(_project_root) / "config"
        self._config_dir.mkdir(exist_ok=True)
        self._config_file = self._config_dir / "telegram_config.json"

        # 加载配置
        self._config = self._load_config()

    def _load_config(self) -> dict:
        """加载配置，token 直接从 .env 读取"""
        config = {
            "auto_start": False
        }

        # 从配置文件加载 auto_start 设置
        if self._config_file.exists():
            try:
                with open(self._config_file, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    config["auto_start"] = saved.get("auto_start", False)
            except Exception:
                pass

        # 如果 .env 有 token，默认自动启动
        if os.environ.get("TELEGRAM_BOT_TOKEN"):
            config["auto_start"] = True

        return config

    def save_config(self, config: dict):
        """保存 Telegram 配置"""
        self._config.update(config)
        with open(self._config_file, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2)

    def get_config(self) -> dict:
        """获取当前配置"""
        return self._config.copy()

    def get_bot_token(self) -> str:
        """获取 Bot Token (直接从环境变量/.env)"""
        return os.environ.get("TELEGRAM_BOT_TOKEN", "")

    def set_bot_token(self, token: str):
        """设置 Bot Token (直接写入 .env 文件)"""
        # 更新环境变量
        os.environ["TELEGRAM_BOT_TOKEN"] = token

        # 写入 .env 文件
        self._update_env_file("TELEGRAM_BOT_TOKEN", token)

    def _update_env_file(self, key: str, value: str):
        """更新 .env 文件中的值"""
        env_path = Path(_project_root) / ".env"

        lines = []
        key_found = False

        # 读取现有内容
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

        # 查找并更新 key
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
                new_lines.append(f"{key}={value}\n")
                key_found = True
            else:
                new_lines.append(line)

        # 如果没找到，添加到文件末尾
        if not key_found:
            # 确保前面有换行
            if new_lines and not new_lines[-1].endswith('\n'):
                new_lines.append('\n')
            new_lines.append(f"{key}={value}\n")

        # 写回文件
        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    def is_running(self) -> bool:
        """检查 Bot 是否在运行"""
        return self._bot_running and self._bot_thread is not None and self._bot_thread.is_alive()

    def start_bot(self) -> tuple[bool, str]:
        """启动 Telegram Bot"""
        if self.is_running():
            return False, "Bot is already running"

        token = self.get_bot_token()
        if not token:
            return False, "Bot token not configured"

        # 设置环境变量
        os.environ["TELEGRAM_BOT_TOKEN"] = token

        self._stop_event.clear()
        self._bot_thread = threading.Thread(target=self._run_bot, daemon=True)
        self._bot_thread.start()

        # 等待启动完成（最多 5 秒）
        import time
        for _ in range(10):
            time.sleep(0.5)
            if self._bot_running:
                return True, "Bot started successfully"

        # 线程还在运行说明正在启动中
        if self._bot_thread.is_alive():
            return True, "Bot starting..."
        else:
            return False, "Bot failed to start (check token)"

    def stop_bot(self) -> tuple[bool, str]:
        """停止 Telegram Bot"""
        if not self.is_running():
            return False, "Bot is not running"

        self._stop_event.set()
        self._bot_running = False

        # 停止 Application
        if self._bot_app:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._bot_app.stop(),
                    self._bot_app._loop if hasattr(self._bot_app, '_loop') else asyncio.get_event_loop()
                )
            except Exception:
                pass

        # 等待线程结束
        if self._bot_thread:
            self._bot_thread.join(timeout=5)

        return True, "Bot stopped"

    def _run_bot(self):
        """在独立线程中运行 Bot"""
        try:
            from telegram.ext import Application, CommandHandler, MessageHandler as TGMessageHandler, filters
            from src.bot.handler import start, clear_history, handle_message, error_handler

            token = self.get_bot_token()
            app = Application.builder().token(token).build()
            self._bot_app = app

            # 添加处理器
            app.add_handler(CommandHandler("start", start))
            app.add_handler(CommandHandler("help", start))  # help 使用同样的处理
            app.add_handler(CommandHandler("clear", clear_history))
            app.add_handler(TGMessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
            app.add_error_handler(error_handler)

            # 运行 (Windows 需要 ProactorEventLoop 支持子进程)
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def run_polling():
                await app.initialize()
                await app.start()
                await app.updater.start_polling(drop_pending_updates=True)

                # 标记为运行中
                self._bot_running = True

                # 等待停止信号
                while not self._stop_event.is_set():
                    await asyncio.sleep(1)

                await app.updater.stop()
                await app.stop()
                await app.shutdown()

            loop.run_until_complete(run_polling())

        except Exception as e:
            print(f"Telegram Bot error: {e}")
        finally:
            self._bot_running = False
            self._bot_app = None


# 全局实例
_controller: Optional[TelegramController] = None


def get_telegram_controller() -> TelegramController:
    """获取 Telegram 控制器实例"""
    global _controller
    if _controller is None:
        _controller = TelegramController()
    return _controller
