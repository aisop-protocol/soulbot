"""
AI 工作线程

使用单一持久化线程处理所有 AI 请求，避免事件循环冲突。
"""

import asyncio
import threading
import queue
from typing import Optional, Callable
import sys
import os

# 添加项目路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class AIWorker:
    """AI 工作线程 - 单例模式，所有 AI 请求在同一事件循环中运行"""

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
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._task_queue = queue.Queue()

        # 启动工作线程
        self._start_worker()

    def _start_worker(self):
        """启动工作线程"""
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self):
        """工作线程主循环"""
        # Windows 需要 ProactorEventLoop 才能支持 asyncio.create_subprocess_exec
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        while self._running:
            try:
                # 等待任务
                task = self._task_queue.get(timeout=1)
                if task is None:
                    break

                coro, result_queue = task

                # 运行协程
                try:
                    self._loop.run_until_complete(coro)
                except Exception as e:
                    result_queue.put(("error", str(e)))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"AIWorker error: {e}")

        self._loop.close()

    def submit(self, coro_func: Callable, result_queue: queue.Queue, *args, **kwargs):
        """
        提交异步任务到工作线程

        Args:
            coro_func: 返回协程的函数
            result_queue: 用于返回结果的队列
            *args, **kwargs: 传递给 coro_func 的参数
        """
        # 创建协程（带结果队列）
        coro = coro_func(result_queue, *args, **kwargs)
        self._task_queue.put((coro, result_queue))

    def stop(self):
        """停止工作线程"""
        self._running = False
        self._task_queue.put(None)
        if self._thread:
            self._thread.join(timeout=5)


# 全局实例
_worker: Optional[AIWorker] = None


def get_ai_worker() -> AIWorker:
    """获取 AI 工作线程实例"""
    global _worker
    if _worker is None:
        _worker = AIWorker()
    return _worker


# AISOP & system instruction: delegate to shared implementation in src.config
from src.config import load_aisop, get_system_instruction, clear_instruction_cache


async def stream_ai_response(result_queue: queue.Queue, user_input: str, session_id: Optional[str]):
    """
    流式获取 AI 响应的协程

    通过 result_queue 返回：
    - ("chunk", text) - 文本块
    - ("session_id", id) - session ID
    - ("done", None) - 完成
    - ("error", msg) - 错误
    """
    from src.llm_client import AIClient

    ai_client = AIClient()

    # 获取系统指令（与 Telegram 版本一致）
    system_instruction = get_system_instruction()

    try:
        async for chunk, returned_session_id in ai_client.get_streaming_response(
            user_input,
            session_id=session_id,
            system_prompt=system_instruction if system_instruction else None
        ):
            if chunk:
                result_queue.put(("chunk", chunk))
            if returned_session_id:
                result_queue.put(("session_id", returned_session_id))

        result_queue.put(("done", None))
    except Exception as e:
        result_queue.put(("error", str(e)))
