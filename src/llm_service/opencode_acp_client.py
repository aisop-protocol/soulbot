"""
OpenCode ACP 客户端

通过 ACP 协议连接 OpenCode，支持免费模型：
- opencode/kimi-k2.5-free
- opencode/minimax-m2.1-free
- opencode/trinity-large-preview-free

使用方式：
    from opencode_acp_client import OpenCodeACPClient, OpenCodeConfig

    config = OpenCodeConfig(
        model="opencode/kimi-k2.5-free",
        cwd="/path/to/workspace"
    )
    client = OpenCodeACPClient(config)
    await client.connect()
    response = await client.query("你好")
"""

import asyncio
import json
import os
import shutil
import logging
import subprocess
import threading
import queue
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class OpenCodeConfig:
    """OpenCode ACP 配置"""
    model: str = "opencode/kimi-k2.5-free"
    cwd: str = ""
    timeout: int = 600  # 10分钟超时
    pool_size: int = 10
    pool_idle_timeout: int = 43200  # 12小时
    auto_approve_permissions: bool = True
    show_thoughts: bool = False
    session_dir: Optional[str] = None


class OpenCodeACPClient:
    """OpenCode ACP 客户端 - 使用同步 subprocess + 线程"""

    def __init__(self, config: OpenCodeConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self._msg_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self.session_id: Optional[str] = None
        self._response_chunks: List[str] = []
        self._response_complete: Optional[asyncio.Event] = None
        self._stream_queue: Optional[asyncio.Queue] = None
        self._streaming_mode: bool = False
        self._connected = False
        self._last_used = 0.0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stop_reader = False
        self._write_lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._connected and self.process and self.process.poll() is None

    @property
    def is_idle_timeout(self) -> bool:
        import time
        return time.time() - self._last_used > self.config.pool_idle_timeout

    def _find_opencode_command(self) -> str:
        """查找 opencode 命令"""
        import sys

        cmd = shutil.which("opencode")
        if cmd:
            return cmd

        if sys.platform == "win32":
            cmd = shutil.which("opencode.cmd")
            if cmd:
                return cmd

        raise RuntimeError(
            "OpenCode not found. Install with: npm install -g opencode-ai"
        )

    async def connect(self, session_id: Optional[str] = None) -> bool:
        """
        连接到 OpenCode ACP

        Args:
            session_id: 可选的 session ID，如果提供则恢复现有 session
                       如果不提供则创建新 session

        Returns:
            bool: 连接是否成功
        """
        if self.is_connected:
            logger.info("[OpenCodeACP] Already connected")
            return True

        try:
            cmd = self._find_opencode_command()
            args = [cmd, "acp"]

            logger.info(f"[OpenCodeACP] Starting subprocess: {' '.join(args)}")

            env = os.environ.copy()
            env["HEADLESS"] = "true"
            env["TERM"] = "dumb"
            env["FORCE_COLOR"] = "0"
            env["NO_COLOR"] = "1"
            # 传递 OpenCode 配置（禁用 title/summary 防止挂起）
            if os.environ.get("OPENCODE_CONFIG_CONTENT"):
                env["OPENCODE_CONFIG_CONTENT"] = os.environ["OPENCODE_CONFIG_CONTENT"]

            cwd = self.config.cwd or os.getcwd()
            logger.info(f"[OpenCodeACP] Working directory: {cwd}")

            # 使用同步 subprocess
            import sys
            creation_flags = 0
            if sys.platform == "win32":
                # Windows: 只隐藏窗口，不使用 CREATE_NEW_PROCESS_GROUP (可能影响 stdin)
                creation_flags = subprocess.CREATE_NO_WINDOW

            self.process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=cwd,
                bufsize=0,  # 无缓冲
                creationflags=creation_flags
            )
            logger.info(f"[OpenCodeACP] Subprocess started, pid={self.process.pid}")

            # 保存事件循环引用
            self._loop = asyncio.get_running_loop()
            self._stop_reader = False

            # 启动线程读取器
            self._reader_thread = threading.Thread(target=self._thread_reader, daemon=True)
            self._reader_thread.start()

            # 启动 stderr 读取线程
            self._stderr_thread = threading.Thread(target=self._thread_stderr_reader, daemon=True)
            self._stderr_thread.start()

            # 初始化（可选择恢复现有 session）
            await self._initialize(resume_session_id=session_id)
            self._connected = True
            self._update_last_used()

            logger.info(f"[OpenCodeACP] Connected, session: {self.session_id}")
            return True

        except Exception as e:
            logger.error(f"[OpenCodeACP] Failed to connect: {e}", exc_info=True)
            return False

    def _update_last_used(self):
        import time
        self._last_used = time.time()

    def _thread_stderr_reader(self):
        """线程读取 stderr"""
        try:
            while not self._stop_reader and self.process and self.process.poll() is None:
                line = self.process.stderr.readline()
                if not line:
                    break
                line_str = line.decode('utf-8', errors='replace').strip()
                if line_str:
                    logger.warning(f"[OpenCodeACP STDERR] {line_str}")
        except Exception as e:
            logger.error(f"[OpenCodeACP] Stderr reader error: {e}")

    def _thread_reader(self):
        """线程读取器 - 同步读取 subprocess stdout"""
        logger.info("[OpenCodeACP] Thread reader started")
        line_count = 0
        import time
        try:
            while not self._stop_reader and self.process and self.process.poll() is None:
                try:
                    # 检查进程状态
                    poll_result = self.process.poll()
                    if poll_result is not None:
                        logger.warning(f"[OpenCodeACP] Process exited with code: {poll_result}")
                        break

                    # 记录等待开始
                    wait_start = time.time()
                    logger.debug(f"[OpenCodeACP] Waiting for readline...")
                    line = self.process.stdout.readline()
                    wait_time = time.time() - wait_start
                    if wait_time > 1.0:
                        logger.info(f"[OpenCodeACP] readline waited {wait_time:.1f}s")
                    if not line:
                        logger.info("[OpenCodeACP] Thread reader: EOF")
                        break

                    line_count += 1
                    line_str = line.decode('utf-8', errors='replace').strip()
                    if not line_str:
                        continue

                    logger.info(f"[OpenCodeACP] Line #{line_count}: {line_str[:100]}...")

                    # 解析 JSON
                    message = None
                    try:
                        if line_str.startswith('{'):
                            message = json.loads(line_str)
                        elif '{' in line_str:
                            start = line_str.find('{')
                            end = line_str.rfind('}')
                            if start != -1 and end != -1 and end > start:
                                message = json.loads(line_str[start:end+1])
                    except json.JSONDecodeError as e:
                        logger.warning(f"[OpenCodeACP] JSON decode error: {e}")

                    if message and self._loop:
                        # 在事件循环中调度消息处理
                        asyncio.run_coroutine_threadsafe(
                            self._handle_message(message),
                            self._loop
                        )

                except Exception as e:
                    if not self._stop_reader:
                        logger.error(f"[OpenCodeACP] Thread reader error: {e}")
                    break

        except Exception as e:
            logger.error(f"[OpenCodeACP] Thread reader outer error: {e}")
        finally:
            logger.info(f"[OpenCodeACP] Thread reader stopped after {line_count} lines")
            self._connected = False
            # 通知等待中的流式响应
            if self._stream_queue:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._stream_queue.put(None),
                        self._loop
                    )
                except Exception:
                    pass

    async def _initialize(self, resume_session_id: Optional[str] = None):
        """
        初始化 ACP 会话

        Args:
            resume_session_id: 可选的 session ID，如果提供则恢复现有 session
        """
        logger.info("[OpenCodeACP] Sending initialize request...")

        init_res = await self._send_request("initialize", {
            "clientInfo": {"name": "SoulBot-OpenCode", "version": "1.0"},
            "protocolVersion": 1,
            "capabilities": {
                "fs": {"readTextFile": True, "writeTextFile": True},
                "terminal": True
            }
        }, timeout=30)

        logger.info(f"[OpenCodeACP] Initialize response: {init_res}")

        cwd = self.config.cwd or os.getcwd()
        logger.info(f"[OpenCodeACP] Session CWD: {cwd}")

        if resume_session_id:
            # 先用 session/list 验证 session 是否存在
            session_exists = False
            try:
                list_res = await self._send_request("session/list", {"cwd": cwd}, timeout=15)
                available_sessions = list_res.get("sessions", []) if list_res else []
                available_ids = [s.get("sessionId") or s.get("id") for s in available_sessions]
                logger.info(
                    f"[OpenCodeACP] Available sessions ({len(available_sessions)}): "
                    f"{[sid[:20]+'...' if sid and len(sid)>20 else sid for sid in available_ids]}"
                )
                session_exists = resume_session_id in available_ids
                if not session_exists:
                    logger.warning(
                        f"[OpenCodeACP] Target session {resume_session_id[:20]}... "
                        f"NOT found in {len(available_sessions)} available sessions"
                    )
            except Exception as e:
                logger.warning(f"[OpenCodeACP] session/list failed (will try resume anyway): {e}")
                session_exists = True  # list 失败时仍尝试 resume

            if session_exists:
                # Session 存在（或 list 失败），尝试 resume
                logger.info(f"[OpenCodeACP] Resuming session: {resume_session_id}")
                try:
                    resume_res = await self._send_request("session/resume", {
                        "sessionId": resume_session_id,
                        "cwd": cwd,
                        "mcpServers": []
                    }, timeout=30)
                    self.session_id = resume_session_id
                    logger.info(f"[OpenCodeACP] Session resumed successfully: {self.session_id}")
                except Exception as e:
                    logger.error(f"[OpenCodeACP] RESUME FAILED for {resume_session_id}: {e}")
                    logger.info("[OpenCodeACP] Creating new session instead...")
                    session_res = await self._send_request("session/new", {
                        "cwd": cwd,
                        "mcpServers": []
                    })
                    self.session_id = session_res.get("sessionId")
                    logger.warning(f"[OpenCodeACP] New session created (resume failed): {self.session_id}")
            else:
                # Session 不在列表中，直接创建新 session（避免无谓的 resume 超时）
                logger.warning(
                    f"[OpenCodeACP] Session {resume_session_id[:20]}... not available, creating new session"
                )
                session_res = await self._send_request("session/new", {
                    "cwd": cwd,
                    "mcpServers": []
                })
                self.session_id = session_res.get("sessionId")
                logger.info(f"[OpenCodeACP] New session created: {self.session_id}")
        else:
            # 创建新 session
            logger.info("[OpenCodeACP] Creating new session...")
            session_res = await self._send_request("session/new", {
                "cwd": cwd,
                "mcpServers": []
            })
            self.session_id = session_res.get("sessionId")
            logger.info(f"[OpenCodeACP] Session created: {self.session_id}")

        if self.config.model:
            logger.info(f"[OpenCodeACP] Setting model to: {self.config.model}")
            await self._send_request("session/set_model", {
                "sessionId": self.session_id,
                "modelId": self.config.model
            })

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        列出可用的 sessions

        Returns:
            List of session info dicts with keys:
            - sessionId: str
            - cwd: str
            - title: str
            - updatedAt: str (ISO format)
        """
        if not self.is_connected:
            raise RuntimeError("Not connected")

        cwd = self.config.cwd or os.getcwd()
        result = await self._send_request("session/list", {
            "cwd": cwd
        }, timeout=30)

        sessions = result.get("sessions", [])
        logger.info(f"[OpenCodeACP] Found {len(sessions)} sessions")
        return sessions

    async def _handle_message(self, message: Dict):
        """处理消息"""
        method = message.get('method')
        msg_id = message.get('id')
        logger.info(f"[OpenCodeACP] Handle message: method={method}, id={msg_id}")

        # 处理 RPC 响应
        if 'id' in message and 'method' not in message and message['id'] in self._pending_requests:
            future = self._pending_requests.pop(message['id'])
            if 'error' in message:
                error = message['error']
                error_msg = error.get('message', str(error)) if isinstance(error, dict) else str(error)
                if not future.done():
                    future.set_exception(Exception(error_msg))
            else:
                result = message.get('result')
                if not future.done():
                    future.set_result(result)

                if isinstance(result, dict) and result.get('stopReason') in ('end_turn', 'stop_sequence', 'max_tokens'):
                    logger.info(f"[OpenCodeACP] stopReason={result.get('stopReason')}")
                    if self._streaming_mode and self._stream_queue:
                        await self._stream_queue.put(None)
            return

        # 处理 session/update
        if method in ('session/update', 'session\\update'):
            params = message.get('params', {})
            update_obj = params.get('update', {})
            update_type = update_obj.get('sessionUpdate')
            logger.info(f"[OpenCodeACP] session/update: type={update_type}, streaming={self._streaming_mode}")

            if update_type == 'agent_message_chunk':
                content_obj = update_obj.get('content', {})
                if isinstance(content_obj, dict) and content_obj.get('type') == 'text':
                    text_chunk = content_obj.get('text', '')
                    self._response_chunks.append(text_chunk)
                    if self._streaming_mode and self._stream_queue:
                        await self._stream_queue.put(text_chunk)

            elif update_type == 'agent_thought_chunk':
                if self.config.show_thoughts:
                    content_obj = update_obj.get('content', {})
                    if isinstance(content_obj, dict) and content_obj.get('type') == 'text':
                        thought_chunk = content_obj.get('text', '')
                        self._response_chunks.append(f"\n> [Thought]: {thought_chunk}\n")
                        if self._streaming_mode and self._stream_queue:
                            await self._stream_queue.put(f"\n> [Thought]: {thought_chunk}\n")

            elif update_type in ('agent_message_end', 'session_end', 'turn_end'):
                if self._response_complete:
                    self._response_complete.set()
                if self._streaming_mode and self._stream_queue:
                    await self._stream_queue.put(None)

        elif method:
            await self._handle_request(message)

    async def _handle_request(self, message: Dict):
        """处理 Agent 请求"""
        method = message.get('method')
        msg_id = message.get('id')
        params = message.get('params', {})

        try:
            if method in ['fs/read_text_file', 'read_text_file', 'fs/readTextFile', 'readTextFile']:
                path = params.get('path', '')
                if not os.path.isabs(path):
                    path = os.path.join(self.config.cwd or os.getcwd(), path.lstrip('/\\'))

                if os.path.isdir(path):
                    files = os.listdir(path)
                    await self._send_response(msg_id, {"content": "\n".join(files)})
                else:
                    try:
                        with open(path, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                        if len(content) > 10000:
                            content = content[:10000] + "\n\n[... 内容已截断 ...]"
                        await self._send_response(msg_id, {"content": content})
                    except Exception as e:
                        await self._send_error(msg_id, -32000, str(e))

            elif method in ['fs/write_text_file', 'write_text_file', 'fs/writeTextFile', 'writeTextFile']:
                path = params.get('path', '')
                if not os.path.isabs(path):
                    path = os.path.join(self.config.cwd or os.getcwd(), path.lstrip('/\\'))
                content = params.get('content', '')
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                await self._send_response(msg_id, {"success": True})

            elif method in ['session/request_permission', 'request_permission']:
                if self.config.auto_approve_permissions:
                    options = params.get('options', [])
                    if options:
                        option_id = options[0].get('optionId')
                        await self._send_response(msg_id, {
                            "outcome": {"outcome": "selected", "optionId": option_id}
                        })
                    else:
                        await self._send_error(msg_id, -32603, "No options provided")
                else:
                    await self._send_error(msg_id, -32603, "Permission denied")

            elif method in ['terminal/create', 'create_terminal']:
                command = params.get('command')
                terminal_id = f"term-{self._msg_id}"
                cwd = self.config.cwd or os.getcwd()
                proc = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    cwd=cwd
                )
                output = proc.stdout.decode(errors='replace') + proc.stderr.decode(errors='replace')
                await self._send_response(msg_id, {
                    "terminalId": terminal_id,
                    "output": output,
                    "exitCode": proc.returncode
                })

            else:
                logger.warning(f"[OpenCodeACP] Unknown method: {method}")
                await self._send_error(msg_id, -32601, f"Method not found: {method}")

        except Exception as e:
            logger.error(f"[OpenCodeACP] Error handling {method}: {e}")
            await self._send_error(msg_id, -32603, str(e))

    async def _send_request(self, method: str, params: Optional[Dict] = None, timeout: int = 600) -> Any:
        """发送 RPC 请求"""
        self._msg_id += 1
        msg = {"jsonrpc": "2.0", "id": self._msg_id, "method": method, "params": params or {}}
        content = json.dumps(msg).encode('utf-8')
        logger.info(f"[OpenCodeACP] Sending request #{self._msg_id}: {method}")

        # 检查进程状态
        if self.process.poll() is not None:
            raise RuntimeError(f"Process exited with code {self.process.poll()}")

        with self._write_lock:
            try:
                data = content + b'\n'
                logger.debug(f"[OpenCodeACP] Writing: {data[:200]}...")

                # 尝试两种写入方式
                import sys as _sys
                if _sys.platform == "win32":
                    # Windows: 使用 os.write 直接写入
                    import os as _os
                    bytes_written = _os.write(self.process.stdin.fileno(), data)
                    logger.info(f"[OpenCodeACP] os.write: {bytes_written}/{len(data)} bytes")
                else:
                    self.process.stdin.write(data)
                    self.process.stdin.flush()
                    bytes_written = len(data)
                    logger.info(f"[OpenCodeACP] stdin.write: {bytes_written} bytes")

            except Exception as write_err:
                logger.error(f"[OpenCodeACP] Write error: {write_err}")
                raise

        logger.info(f"[OpenCodeACP] Request #{self._msg_id} sent, waiting for response...")

        future = self._loop.create_future()
        self._pending_requests[self._msg_id] = future
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.error(f"[OpenCodeACP] Request #{self._msg_id} ({method}) timeout")
            raise

    async def _send_response(self, msg_id: int, result: Dict):
        """发送 RPC 响应"""
        msg = {"jsonrpc": "2.0", "id": msg_id, "result": result}
        content = json.dumps(msg).encode('utf-8')
        with self._write_lock:
            self.process.stdin.write(content + b'\n')
            self.process.stdin.flush()

    async def _send_error(self, msg_id: int, code: int, message: str):
        """发送 RPC 错误"""
        msg = {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}
        content = json.dumps(msg).encode('utf-8')
        with self._write_lock:
            self.process.stdin.write(content + b'\n')
            self.process.stdin.flush()

    async def query(self, prompt: str) -> str:
        """发送查询（非流式）"""
        self._response_chunks = []
        self._response_complete = asyncio.Event()
        self._streaming_mode = False
        self._stream_queue = None
        self._update_last_used()

        logger.debug(f"[OpenCodeACP] Sending prompt: {prompt[:100]}...")

        try:
            rpc_result = await self._send_request("session/prompt", {
                "sessionId": self.session_id,
                "prompt": [{"type": "text", "text": prompt}]
            })
        except Exception as e:
            if self._response_chunks:
                logger.warning(f"[OpenCodeACP] RPC error but content received: {e}")
                return ''.join(self._response_chunks)
            raise

        if rpc_result and isinstance(rpc_result, dict):
            content = rpc_result.get('content') or rpc_result.get('text')
            if rpc_result.get('stopReason'):
                self._response_complete.set()
            if content:
                return content

        try:
            await asyncio.wait_for(self._response_complete.wait(), timeout=self.config.timeout)
        except asyncio.TimeoutError:
            logger.warning(f"[OpenCodeACP] Response timeout")

        return ''.join(self._response_chunks)

    async def query_stream(self, prompt: str) -> AsyncIterator[str]:
        """流式查询"""
        self._response_chunks = []
        self._response_complete = asyncio.Event()
        self._stream_queue = asyncio.Queue()
        self._streaming_mode = True
        self._update_last_used()

        logger.info(f"[OpenCodeACP Stream] Sending prompt: {prompt[:100]}...")

        rpc_result_holder = {"result": None, "error": None}

        async def send_request_async():
            try:
                result = await self._send_request("session/prompt", {
                    "sessionId": self.session_id,
                    "prompt": [{"type": "text", "text": prompt}]
                }, timeout=self.config.timeout)
                rpc_result_holder["result"] = result
            except Exception as e:
                logger.error(f"[OpenCodeACP Stream] Request error: {e}")
                rpc_result_holder["error"] = e
                if self._stream_queue:
                    try:
                        await self._stream_queue.put(None)
                    except:
                        pass

            # 检查进程状态
            if self.process and self.process.poll() is not None:
                logger.error(f"[OpenCodeACP Stream] Process died: {self.process.poll()}")

        try:
            send_task = asyncio.create_task(send_request_async())
            chunks_yielded = 0
            max_wait = self.config.timeout
            chunk_timeout = 5.0
            wait_elapsed = 0

            while wait_elapsed < max_wait:
                if not self.is_connected:
                    logger.warning("[OpenCodeACP Stream] Connection lost")
                    break

                try:
                    chunk = await asyncio.wait_for(
                        self._stream_queue.get(),
                        timeout=chunk_timeout
                    )
                    wait_elapsed = 0
                    if chunk is None:
                        logger.info("[OpenCodeACP Stream] Received end signal")
                        break
                    if chunk:
                        chunks_yielded += 1
                        yield chunk
                except asyncio.TimeoutError:
                    wait_elapsed += chunk_timeout
                    # 检查进程状态
                    poll_result = self.process.poll() if self.process else -999
                    logger.info(f"[OpenCodeACP Stream] Waiting... ({wait_elapsed}s, poll={poll_result})")

            await send_task

            if rpc_result_holder["error"]:
                raise rpc_result_holder["error"]

            if chunks_yielded == 0 and self._response_chunks:
                logger.info(f"[OpenCodeACP Stream] Using buffered chunks")
                yield ''.join(self._response_chunks)

            logger.info(f"[OpenCodeACP Stream] Done, yielded {chunks_yielded} chunks")

        finally:
            self._streaming_mode = False
            self._stream_queue = None

    async def disconnect(self):
        """断开连接"""
        self._stop_reader = True
        self._connected = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            finally:
                self.process = None


class OpenCodeConnectionPool:
    """
    OpenCode 连接池

    支持基于用户 ID 的 session 持久化：
    - 为每个用户维护独立的 OpenCode session
    - 跨进程恢复时自动加载历史对话
    """

    def __init__(self, config: OpenCodeConfig):
        self.config = config
        self._pool: List[OpenCodeACPClient] = []
        self._lock = asyncio.Lock()
        # 用户 ID -> session ID 映射
        self._user_sessions: Dict[str, str] = {}
        # session ID -> client 映射（活跃连接）
        self._session_clients: Dict[str, OpenCodeACPClient] = {}

    async def _create_client(self, session_id: Optional[str] = None) -> OpenCodeACPClient:
        """创建新客户端"""
        logger.info(f"[OpenCodePool] Creating client, resume_session={session_id}")
        client = OpenCodeACPClient(self.config)
        await client.connect(session_id=session_id)
        return client

    def get_user_session(self, user_id: str) -> Optional[str]:
        """获取用户的 session ID"""
        return self._user_sessions.get(user_id)

    def set_user_session(self, user_id: str, session_id: str):
        """设置用户的 session ID"""
        self._user_sessions[user_id] = session_id
        logger.info(f"[OpenCodePool] User {user_id} -> Session {session_id}")

    @asynccontextmanager
    async def acquire(self, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """
        获取连接

        Args:
            user_id: 可选的用户 ID，如果提供则尝试恢复该用户的 session
            session_id: 可选的 session ID，优先使用此 ID 恢复 session（来自持久化存储）

        Yields:
            (client, session_id) 元组
        """
        client = None
        resume_session_id = None

        async with self._lock:
            # 清理断开的连接
            self._pool = [c for c in self._pool if c.is_connected and not c.is_idle_timeout]

            # 优先使用传入的 session_id（来自持久化存储）
            if session_id:
                resume_session_id = session_id
                logger.info(f"[OpenCodePool] Using provided session_id: {session_id}")
            # 其次使用内存中的用户 session 映射
            elif user_id:
                resume_session_id = self._user_sessions.get(user_id)
                if resume_session_id:
                    logger.info(f"[OpenCodePool] Using cached session for user {user_id}: {resume_session_id}")

            # 检查是否有活跃的连接（仅当 session 匹配时）
            if resume_session_id and resume_session_id in self._session_clients:
                client = self._session_clients[resume_session_id]
                if not client.is_connected:
                    del self._session_clients[resume_session_id]
                    client = None
                else:
                    logger.info(f"[OpenCodePool] Reusing active connection for session: {resume_session_id}")

            # 如果没有活跃连接，从池中获取空闲连接
            if not client and self._pool:
                # 如果需要恢复特定 session，不使用池中的现有连接
                if not resume_session_id:
                    client = self._pool.pop(0)
                    if not client.is_connected:
                        client = None

        # 如果仍然没有连接，创建新的（并恢复 session）
        if not client:
            logger.info(f"[OpenCodePool] Creating new client, resume_session={resume_session_id}")
            client = await self._create_client(session_id=resume_session_id)

        try:
            # 更新用户 -> session 映射
            if user_id and client.session_id:
                self._user_sessions[user_id] = client.session_id
                self._session_clients[client.session_id] = client

            yield client, client.session_id

            async with self._lock:
                # 归还连接到池中
                if len(self._pool) < self.config.pool_size:
                    self._pool.append(client)
                else:
                    # 池已满，断开连接但保留 session 映射
                    if client.session_id in self._session_clients:
                        del self._session_clients[client.session_id]
                    await client.disconnect()
        except Exception:
            # 出错时断开连接
            if client.session_id in self._session_clients:
                del self._session_clients[client.session_id]
            await client.disconnect()
            raise

    @asynccontextmanager
    async def acquire_for_user(self, user_id: str):
        """
        为特定用户获取连接（带 session 持久化）

        这是 acquire 的便捷方法，确保使用用户的持久化 session。

        Args:
            user_id: 用户 ID

        Yields:
            (client, session_id) 元组
        """
        async with self.acquire(user_id=user_id) as (client, session_id):
            yield client, session_id

    async def close_all(self):
        """关闭所有连接"""
        async with self._lock:
            for client in self._pool:
                await client.disconnect()
            self._pool.clear()
            for client in self._session_clients.values():
                await client.disconnect()
            self._session_clients.clear()
            # 保留 user_sessions 映射，以便重新连接时可以恢复

    def clear_user_sessions(self):
        """清除所有用户 session 映射"""
        self._user_sessions.clear()
        logger.info("[OpenCodePool] User sessions cleared")


def is_opencode_available() -> bool:
    """检查 OpenCode 是否可用"""
    import sys
    cmd = shutil.which("opencode")
    if cmd:
        return True
    if sys.platform == "win32":
        cmd = shutil.which("opencode.cmd")
        if cmd:
            return True
    return False


# 全局连接池
_opencode_pools: Dict[str, OpenCodeConnectionPool] = {}


def get_opencode_pool(bot_data_dir: Optional[str] = None, config: Optional[OpenCodeConfig] = None) -> OpenCodeConnectionPool:
    """获取 OpenCode 连接池"""
    global _opencode_pools
    cache_key = bot_data_dir or "_default_"

    if cache_key not in _opencode_pools:
        if config is None:
            config = OpenCodeConfig()

        if bot_data_dir:
            config.session_dir = os.path.join(bot_data_dir, "OpenCode_Session")
            os.makedirs(config.session_dir, exist_ok=True)

        _opencode_pools[cache_key] = OpenCodeConnectionPool(config)
        logger.info(f"[OpenCodePool] Created pool for: {cache_key}")

    return _opencode_pools[cache_key]


async def cleanup_opencode_pools():
    """清理所有连接池"""
    global _opencode_pools
    for key, pool in list(_opencode_pools.items()):
        await pool.close_all()
    _opencode_pools.clear()
    logger.info("[OpenCodePool] All pools cleaned up")
