"""
CLI Agent Service - Claude CLI 和 Gemini CLI 的统一集成服务

使用 ACP 模式实现最佳性能：
- Claude ACP: 2.47s/query (+ 1.52s init)
- Gemini ACP: 1.71s/query (+ 6.69s init)

特性：
- 连接池管理（复用 ACP 连接）
- 自动重连与故障恢复
- 与 SoulBot 智能路由兼容
- 工具调用支持
"""

import asyncio
import json
import os
import sys
import uuid
import time
import shutil
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, AsyncIterator
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class CLIProvider(str, Enum):
    """CLI 提供商"""
    CLAUDE = "claude"
    GEMINI = "gemini"


@dataclass
class CLIConfig:
    """CLI 配置"""
    provider: CLIProvider
    acp_cmd: str  # ACP 可执行文件路径
    model: str = ""  # 默认模型
    cwd: str = ""  # 工作目录 (Phase 100)
    timeout: int = 600  # 10分钟超时（复杂任务可能需要较长时间）
    max_retries: int = 3

    # 连接池配置
    pool_size: int = 10  # 保持的空闲连接数，并发时可超出
    pool_idle_timeout: int = 43200  # 空闲连接超时（秒），12小时

    # 工具配置
    allowed_tools: List[str] = field(default_factory=list)
    auto_approve_permissions: bool = True

    # Phase 83: Session 存储目录
    session_dir: Optional[str] = None

    # Phase 100: 是否显示思考过程
    show_thoughts: bool = True

@dataclass
class CLIMessage:
    """CLI 消息"""
    role: str  # "user", "assistant", "system"
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_results: Optional[List[Dict]] = None


@dataclass
class CLIResponse:
    """CLI 响应"""
    content: str
    model: str
    provider: CLIProvider
    latency_ms: float
    tool_calls: Optional[List[Dict]] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ACPClientBase(ABC):
    """ACP 客户端基类"""

    def __init__(self, config: CLIConfig):
        self.config = config
        self.cli_name = config.provider.value  # "claude" or "gemini"
        self.process: Optional[asyncio.subprocess.Process] = None
        self._msg_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self.session_id: Optional[str] = None
        self._response_chunks: List[str] = []
        self._response_complete: Optional[asyncio.Event] = None  # Phase 50 Fix: 等待响应完成
        self._stream_queue: Optional[asyncio.Queue] = None  # Phase 83: 流式队列
        self._streaming_mode: bool = False  # Phase 83: 是否处于流式模式
        self._connected = False
        self._last_used = time.time()
        self._terminals: Dict[str, Dict] = {}
        self._auth_methods: List[Dict] = []  # 保存认证方法信息

    @property
    def is_connected(self) -> bool:
        return self._connected and self.process and self.process.returncode is None

    @property
    def is_idle_timeout(self) -> bool:
        return time.time() - self._last_used > self.config.pool_idle_timeout

    def _ensure_session_dir(self) -> Optional[str]:
        """
        Phase 83: 确保 session 目录存在并包含最新的凭据文件

        Returns:
            session_dir 路径，如果未配置则返回 None
        """
        if not self.config.session_dir:
            return None

        session_dir = self.config.session_dir
        os.makedirs(session_dir, exist_ok=True)

        # 确定凭据文件名（Claude 和 Gemini 都使用 .credentials.json）
        credentials_file = ".credentials.json"
        target_credentials = os.path.join(session_dir, credentials_file)

        # 默认凭据位置
        default_claude_dir = os.path.expanduser("~/.claude")
        source_credentials = os.path.join(default_claude_dir, credentials_file)

        # 始终同步最新凭据（用户可能重新登录）
        if os.path.exists(source_credentials):
            # 检查是否需要更新（源文件更新或目标不存在）
            need_update = not os.path.exists(target_credentials)
            if not need_update:
                source_mtime = os.path.getmtime(source_credentials)
                target_mtime = os.path.getmtime(target_credentials)
                need_update = source_mtime > target_mtime

            if need_update:
                shutil.copy(source_credentials, target_credentials)
                logger.info(f"[ACP] Synced credentials to {session_dir}")
        else:
            logger.warning(f"[ACP] No credentials found at {source_credentials}")

        return session_dir

    async def connect(self) -> bool:
        """连接到 ACP Server"""
        if self.is_connected:
            logger.info("[ACP] Already connected")
            return True
        return await self._start_process()

    def _get_abs_path(self, path: str) -> str:
        """获取绝对路径，确保在工作区内 (Phase 100)"""
        if not path:
            return ""
        
        # 优先使用配置的工作区
        base_dir = self.config.cwd or os.getcwd()

        # Phase 100: 处理伪绝对路径 (例如 /success.txt)
        # 在 Windows 上，这种路径会被视为相对于驱动器根目录，但在 AI 眼中通常相对于工作区根目录
        clean_path = path
        if path.startswith('/') or path.startswith('\\'):
            clean_path = path.lstrip('/\\')
        
        # 重新判断是否为真正的绝对路径 (例如 X:\\...)
        if os.path.isabs(clean_path):
            return clean_path
            
        abs_path = os.path.abspath(os.path.join(base_dir, clean_path))
        logger.debug(f"[ACP] Resolved path: {path} -> {abs_path} (base: {base_dir})")
        return abs_path

    async def _start_process(self) -> bool:
        """启动 ACP 进程 (internal)"""
        try:
            # Phase 83: 准备 session 目录
            session_dir = self._ensure_session_dir()

            cmd = self._get_acp_command()
            logger.info(f"[ACP] Starting subprocess: {' '.join(cmd)}")

            # Phase 83: 设置环境变量，强制静默/Headless 模式
            env = os.environ.copy()
            env["HEADLESS"] = "true"
            env["TERM"] = "dumb"
            env["FORCE_COLOR"] = "0"
            env["NO_COLOR"] = "1"
            
            if session_dir:
                env["CLAUDE_CONFIG_DIR"] = session_dir
                logger.info(f"[ACP] Using custom config dir: {session_dir}")

            cwd = self.config.cwd or os.getcwd()
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd
            )
            logger.info(f"[ACP] Subprocess started, pid={self.process.pid}")

            asyncio.create_task(self._read_loop())
            asyncio.create_task(self._read_stderr_loop())

            # 初始化
            logger.info("[ACP] Initializing session...")
            await self._initialize()
            self._connected = True
            self._last_used = time.time()

            logger.info(f"[ACP] {self.config.provider.value} connected, session: {self.session_id}")
            return True

        except Exception as e:
            logger.error(f"[ACP] Failed to connect {self.config.provider.value} (cmd={' '.join(cmd)}): {e}", exc_info=True)
            return False

    @abstractmethod
    def _get_acp_command(self) -> List[str]:
        """获取 ACP 启动命令"""
        pass

    @abstractmethod
    async def _initialize(self) -> None:
        """初始化 ACP 会话"""
        pass

    # 类级别标记，防止重复触发登录
    _login_triggered = False

    def _trigger_auto_login(self):
        """
        自动触发 CLI 登录流程

        当检测到认证错误时，自动运行 `claude /login` 或 `gemini auth login`
        打开浏览器进行 OAuth 登录。

        使用类级别标记防止重复触发。
        """
        import subprocess
        import threading

        # 防止重复触发
        if ACPClientBase._login_triggered:
            logger.debug(f"[{self.cli_name}ACP] Login already triggered, skipping")
            return

        ACPClientBase._login_triggered = True

        def run_login():
            try:
                if self.cli_name == "claude":
                    # Claude: 使用 claude /login 命令
                    claude_cmd = shutil.which("claude")
                    if claude_cmd:
                        logger.info(f"[ClaudeACP] 正在启动登录: {claude_cmd} /login")
                        # 使用 start 命令在新窗口打开（Windows）
                        if sys.platform == "win32":
                            subprocess.Popen(
                                ["cmd", "/c", "start", "Claude Login", claude_cmd, "/login"],
                                creationflags=subprocess.CREATE_NEW_CONSOLE
                            )
                        else:
                            # Unix: 在后台运行
                            subprocess.Popen([claude_cmd, "/login"])
                        logger.info("[ClaudeACP] 登录窗口已打开，请在浏览器中完成登录")
                    else:
                        logger.warning("[ClaudeACP] 未找到 claude 命令，请手动运行: claude /login")

                elif self.cli_name == "gemini":
                    # Gemini: 使用 gemini auth login 命令
                    gemini_cmd = shutil.which("gemini")
                    if gemini_cmd:
                        logger.info(f"[GeminiACP] 正在启动登录: {gemini_cmd} auth login")
                        if sys.platform == "win32":
                            subprocess.Popen(
                                ["cmd", "/c", "start", "Gemini Login", gemini_cmd, "auth", "login"],
                                creationflags=subprocess.CREATE_NEW_CONSOLE
                            )
                        else:
                            subprocess.Popen([gemini_cmd, "auth", "login"])
                        logger.info("[GeminiACP] 登录窗口已打开，请在浏览器中完成登录")
                    else:
                        logger.warning("[GeminiACP] 未找到 gemini 命令，请手动运行: gemini auth login")
            except Exception as e:
                logger.error(f"[{self.cli_name}ACP] 自动登录启动失败: {e}")
            finally:
                # 60秒后重置标记，允许再次触发
                time.sleep(60)
                ACPClientBase._login_triggered = False

        # 在后台线程中运行，不阻塞主流程
        thread = threading.Thread(target=run_login, daemon=True)
        thread.start()

    async def _read_loop(self):
        """读取 stdout"""
        try:
            while self.process and self.process.returncode is None:
                line = await self.process.stdout.readline()
                if not line:
                    break

                line_str = line.decode('utf-8', errors='replace').strip()
                if not line_str:
                    continue

                try:
                    # Phase 100: 更鲁棒的 JSON 提取 (防止混入 ANSI 或其他杂质)
                    if line_str.startswith('{'):
                         message = json.loads(line_str)
                         await self._handle_message(message)
                    elif '{' in line_str:
                         # 尝试从行中提取 JSON 部分
                         start = line_str.find('{')
                         end = line_str.rfind('}')
                         if start != -1 and end != -1 and end > start:
                             json_str = line_str[start:end+1]
                             message = json.loads(json_str)
                             await self._handle_message(message)
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.debug(f"Read loop error: {e}")
        finally:
            self._connected = False
            # Phase 100: 如果读取循环结束（进程退出），通知流结束
            if self._streaming_mode and self._stream_queue:
                try:
                    self._stream_queue.put_nowait(None)
                except Exception:
                    pass

    async def _read_stderr_loop(self):
        """读取 stderr"""
        try:
            while self.process and self.process.returncode is None:
                line = await self.process.stderr.readline()
                if not line:
                    break
                line_text = line.decode('utf-8', errors='replace').strip()
                if line_text:
                    # Phase 100: 过滤已知噪音 (benign warnings)
                    # 这些错误通常由 Windows 下的 node-pty 引起，不影响实际功能执行
                    noise_keywords = [
                        "AttachConsole failed",
                        "conpty_console_list_agent.js",
                        "node:internal/modules/cjs/loader",
                        "TracingChannel.traceSync",
                        "wrapModuleLoad"
                    ]
                    if any(kw in line_text for kw in noise_keywords):
                        logger.debug(f"[{self.config.provider.value} STDERR NOISE] {line_text}")
                    else:
                        logger.warning(f"[{self.config.provider.value} STDERR] {line_text}")
        except Exception:
            pass

    async def _handle_message(self, message: Dict):
        """处理消息"""
        method = message.get('method')

        # Check for update type to filter logs
        is_chunk = False
        if method in ('session/update', 'session\\update'):
            params = message.get('params', {})
            update_obj = params.get('update', {})
            update_type = update_obj.get('sessionUpdate')
            if update_type == 'agent_message_chunk':
                is_chunk = True

        if method and method not in ('session/update', 'session\\update'):
            # Phase 101 Fix: Don't block the read loop with request handling!
            # Spawn a task to handle the request (e.g. file read, terminal execution)
            # This allows the read loop to continue processing other messages (like session/update chunks from a running command)
            asyncio.create_task(self._handle_request(message))
            return

        # Log complete message unless it is a chunk (to reduce noise)
        if not is_chunk:
            logger.debug(f"[ACP Raw] {json.dumps(message, ensure_ascii=False)}")

        # Phase 50 Fix: 优先处理 session/update (Notification)，不作为普通 Request
        # 这是一个 JSON-RPC Notification (无 id)，不应该回复，且包含流式数据需要本地处理
        if method in ('session/update', 'session\\update'):
            # 直接跳到流式处理逻辑，不进 _handle_request
            pass
        elif 'method' in message:
            # 其他 Request 或 Notification -> 转交 _handle_request (工具调用等)
            # 注意：_handle_request 会尝试回复，所以只有真正的 Request (带 ID) 才应该发 Response
            # 但目前架构下 _handle_request 也会处理无 ID 的 Notification (如果未来有的话)，
            # 只是不发送 Response (需要 _handle_request 内部配合，目前 _handle_request 里的 _send_error 会发 id:null)
            # 不过目前 Claude 只发 session/update 作为 Notification，其他都是 Request
            await self._handle_request(message)
            return

        # 2. 处理 RPC 响应 (无 method 字段，且 id 在 pending 中)
        if 'id' in message and 'method' not in message and message['id'] in self._pending_requests:
            future = self._pending_requests.pop(message['id'])
            if 'error' in message:
                error = message['error']
                error_msg = error.get('message', str(error)) if isinstance(error, dict) else str(error)
                # 检测认证错误，自动启动登录流程
                if 'Authentication required' in error_msg:
                    logger.warning(f"[{self.cli_name}ACP] 检测到认证错误，尝试自动启动登录...")
                    self._trigger_auto_login()

                    # 清除连接池缓存，确保下次请求使用新凭据
                    try:
                        from .litellm_acp_provider import clear_claude_pool
                        clear_claude_pool()  # 清除默认池
                        logger.info(f"[{self.cli_name}ACP] 连接池已清除，下次请求将使用新凭据")
                    except Exception as e:
                        logger.debug(f"[{self.cli_name}ACP] 清除连接池失败: {e}")

                    friendly_msg = (
                        f"[{self.cli_name}] 认证失败，已自动打开登录页面。\n"
                        f"请在浏览器中完成登录，然后重试。"
                    )
                    logger.error(f"[{self.cli_name}ACP] {friendly_msg}")
                    future.set_exception(Exception(friendly_msg))
                else:
                    future.set_exception(Exception(str(error)))
            else:
                result = message.get('result')
                future.set_result(result)

                # Phase 83 Fix: 如果 RPC 响应带有 stopReason，也发送流结束标记
                # 有些 ACP 实现不发送 agent_message_end 更新，只在 RPC 响应中返回 stopReason
                if isinstance(result, dict) and result.get('stopReason') in ('end_turn', 'stop_sequence', 'max_tokens'):
                    logger.info(f"[ACP] RPC response has stopReason={result.get('stopReason')}, signaling stream end")
                    if self._streaming_mode and self._stream_queue:
                        try:
                            self._stream_queue.put_nowait(None)  # 发送流结束标记
                        except Exception:
                            pass
            return

        # 收集流式输出 (对于 session/update)
        # 处理两种可能的方法名格式（正斜杠和反斜杠）
        if method in ('session/update', 'session\\update'):
            params = message.get('params', {})
            update_obj = params.get('update', {})
            update_type = update_obj.get('sessionUpdate')

            # 详细日志：记录所有 session update 类型（包括未处理的）
            logger.info(f"[ACP] Received session update: {update_type}, keys={list(update_obj.keys())}")

            # Debug: 打印完整的 update 对象（仅对未知类型）
            known_types = (
                'agent_message_chunk', 'text_message_start', 'text_message_content',
                'content_block_delta', 'agent_message_end', 'session_end', 'turn_end',
                'tool_use_start', 'tool_use_end', 'tool_call', 'tool_call_update',
                'available_commands_update'  # Claude Code 命令列表更新
            )
            if update_type and update_type not in known_types:
                logger.warning(f"[ACP] UNHANDLED update type: {update_type}, full update: {json.dumps(update_obj, ensure_ascii=False)[:500]}")

            if update_type == 'agent_message_chunk':
                content_obj = update_obj.get('content', {})
                if isinstance(content_obj, dict) and content_obj.get('type') == 'text':
                    text_chunk = content_obj.get('text', '')
                    self._response_chunks.append(text_chunk)
                    logger.info(f"[ACP] agent_message_chunk: {len(text_chunk)} chars")
                    # Phase 83: 流式模式下，实时推送 chunk 到队列
                    if self._streaming_mode and self._stream_queue:
                        self._stream_queue.put_nowait(text_chunk)

            elif update_type == 'agent_thought_chunk':
                if self.config.show_thoughts:
                    content_obj = update_obj.get('content', {})
                    if isinstance(content_obj, dict) and content_obj.get('type') == 'text':
                        thought_chunk = content_obj.get('text', '')
                        # 记录思考过程，并在非静默模式下加入回复
                        logger.info(f"[ACP] agent_thought_chunk: {len(thought_chunk)} chars")
                        # 我们将其加入 response，但可以使用特殊标记
                        self._response_chunks.append(f"\n> [Thought]: {thought_chunk}\n")
                        if self._streaming_mode and self._stream_queue:
                            self._stream_queue.put_nowait(f"\n> [Thought]: {thought_chunk}\n")
                else:
                    logger.debug(f"[ACP] agent_thought_chunk received but hidden by config")

            # Phase 83 Fix: 也处理 text_message_start/content 类型的更新
            elif update_type in ('text_message_start', 'text_message_content', 'content_block_delta'):
                # 尝试从不同位置提取文本
                text_chunk = None
                if 'text' in update_obj:
                    text_chunk = update_obj.get('text', '')
                elif 'content' in update_obj:
                    content = update_obj.get('content')
                    if isinstance(content, str):
                        text_chunk = content
                    elif isinstance(content, dict):
                        text_chunk = content.get('text', '')
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                text_chunk = item.get('text', '')
                                break

                if text_chunk:
                    self._response_chunks.append(text_chunk)
                    logger.debug(f"[ACP] {update_type}: {len(text_chunk)} chars")
                    if self._streaming_mode and self._stream_queue:
                        self._stream_queue.put_nowait(text_chunk)

            # Phase 83 Fix: 处理工具使用事件（不结束流，继续等待）
            elif update_type in ('tool_use_start', 'tool_use_end', 'tool_call', 'tool_call_update'):
                # 详细记录工具事件内容，帮助诊断问题
                tool_info = update_obj.get('toolName', update_obj.get('name', 'unknown'))
                tool_result = update_obj.get('result', update_obj.get('content', ''))[:200] if update_type == 'tool_call_update' else ''
                logger.info(f"[ACP] Tool event: {update_type}, tool={tool_info}, result_preview={tool_result[:100]}...")
                logger.debug(f"[ACP] Full tool update: {json.dumps(update_obj, ensure_ascii=False)[:500]}")
                # 注意：不发送结束标记，继续等待 agent_message_chunk

            # Phase 50 Fix: 检测响应完成信号
            elif update_type in ('agent_message_end', 'session_end', 'turn_end'):
                if self._response_complete:
                    self._response_complete.set()
                    logger.info(f"[ACP] Response complete signal received: {update_type}")
                # Phase 83: 流式模式下，发送结束标记 (None)
                if self._streaming_mode and self._stream_queue:
                    self._stream_queue.put_nowait(None)  # None 表示流结束

    async def _handle_request(self, message: Dict):
        """处理 Agent 请求（权限、文件、终端等）"""
        method = message.get('method')
        msg_id = message.get('id')
        params = message.get('params', {})

        try:
            # 文件读取
            if method in [
                'fs/read_text_file', 'read_text_file', 'read_file', 
                'fs/readTextFile', 'readTextFile', 'fs/readFile', 'readFile',
                'fs/read_file'
            ]:
                path = self._get_abs_path(params.get('path', ''))
                if not path:
                    await self._send_error(msg_id, -32602, "Missing 'path' parameter")
                    return

                start_line = params.get('line', params.get('offset', 1))
                limit = params.get('limit', 2000)
                # Phase 83 Fix: 从 30KB 减少到 10KB
                # Claude Code CLI 处理大文件内容后可能会卡住
                max_chars = 10000  # 字符数硬限制，防止发送过大内容
                logger.info(f"[ACP] Handling read_file: {path} (line={start_line}, limit={limit})")
                if os.path.isdir(path):
                    files = os.listdir(path)
                    await self._send_response(msg_id, {"content": "\n".join(files)})
                    logger.info(f"[ACP] Sent directory listing for {path}")
                else:
                    try:
                        with open(path, 'r', encoding='utf-8', errors='replace') as f:
                            lines = f.readlines()

                        # 尊重 line 和 limit 参数
                        total_lines = len(lines)
                        start_idx = max(0, start_line - 1)  # 转为 0-indexed
                        end_idx = min(start_idx + limit, total_lines)
                        selected_lines = lines[start_idx:end_idx]
                        content = ''.join(selected_lines)

                        # 额外的字符数限制
                        truncated = False
                        if len(content) > max_chars:
                            content = content[:max_chars]
                            truncated = True
                            content += f"\n\n[... 内容已截断，原文件 {total_lines} 行，{sum(len(l) for l in lines)} 字符 ...]"

                        await self._send_response(msg_id, {"content": content})
                        trunc_msg = " (TRUNCATED)" if truncated else ""
                        logger.info(f"[ACP] Sent file content for {path} (lines {start_line}-{end_idx}/{total_lines}, {len(content)} chars){trunc_msg}")
                        logger.info(f"[ACP] File response sent, waiting for Claude Code to continue generating...")
                    except Exception as e:
                        logger.error(f"[ACP] Failed to read {path}: {e}")
                        await self._send_error(msg_id, -32000, str(e))

            # 文件写入
            elif method in [
                'fs/write_text_file', 'write_text_file', 'write_file', 
                'fs/writeTextFile', 'writeTextFile', 'fs/writeFile', 'writeFile',
                'fs/write_file'
            ]:
                path = self._get_abs_path(params.get('path', ''))
                if not path:
                    await self._send_error(msg_id, -32602, "Missing 'path' parameter")
                    return
                content = params.get('content', '')
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # 获取写入后的文件元数据以供返回
                info = os.stat(path)
                await self._send_response(msg_id, {
                    "exists": True,
                    "size": info.st_size,
                    "mtime": info.st_mtime,
                    "isFile": True,
                    "isDirectory": False,
                    "content": f"Successfully wrote {info.st_size} bytes to {path}"
                })
                logger.info(f"[ACP] Wrote file and returned metadata: {path}")

            # 目录列表
            elif method in [
                'list_directory', 'fs/list_directory', 'fs/listDirectory',
                'fs/listFiles', 'listFiles', 'ls', 'dir'
            ]:
                path = self._get_abs_path(params.get('path', '.'))
                if os.path.exists(path) and os.path.isdir(path):
                    files = os.listdir(path)
                    await self._send_response(msg_id, {"files": files, "content": "\n".join(files)})
                else:
                    await self._send_error(msg_id, -32001, f"Directory not found: {path}")

            # 文件存在性检查
            elif method in ['fs/exists', 'exists']:
                path = self._get_abs_path(params.get('path', ''))
                exists = os.path.exists(path)
                await self._send_response(msg_id, {"exists": exists})
                logger.debug(f"[ACP] Sent exists={exists} for: {path}")

            # 文件信息 (Stat)
            elif method in [
                'fs/get_file_info', 'fs/stat', 'fs/getFileInfo', 'getFileInfo', 'stat',
                'fs/metadata', 'metadata', 'fs/getMetadata'
            ]:
                path = self._get_abs_path(params.get('path', ''))
                exists = os.path.exists(path)
                
                if not exists:
                    # 对于 get_file_info，如果不存在，返回成功但带 exists=False
                    # 这能防止 Gemini CLI 在写文件前的“存在性检查”中报错 [object Object]
                    await self._send_response(msg_id, {
                        "exists": False,
                        "isFile": False,
                        "isDirectory": False,
                        "type": "file",  # 增加 type 字段以增强兼容性
                        "path": path,
                        "name": os.path.basename(path),
                        "size": 0,
                        "mtime": 0
                    })
                    logger.debug(f"[ACP] Sent not-found metadata for: {path}")
                    return
                
                info = os.stat(path)
                is_file = os.path.isfile(path)
                is_dir = os.path.isdir(path)
                
                await self._send_response(msg_id, {
                    "exists": True,
                    "size": info.st_size,
                    "mtime": info.st_mtime,
                    "ctime": info.st_ctime,
                    "atime": info.st_atime,
                    "isFile": is_file,
                    "isDirectory": is_dir,
                    "type": "file" if is_file else ("directory" if is_dir else "other"),
                    "permissions": oct(info.st_mode)[-3:],
                    # 增加冗余字段以解决兼容性问题
                    "metadata": {
                        "size": info.st_size,
                        "mtime": info.st_mtime,
                        "isFile": is_file,
                        "isDirectory": is_dir
                    }
                })
                logger.debug(f"[ACP] Sent metadata for: {path}")

            # 权限请求 - 自动批准
            elif method in ['session/request_permission', 'request_permission', 'session/requestPermission']:
                if self.config.auto_approve_permissions:
                    # 记录权限请求
                    logger.info(f"[ACP] Auto-approving permission request: {json.dumps(params)}")
                    options = params.get('options', [])
                    if options:
                        option_id = options[0].get('optionId')
                        await self._send_response(msg_id, {
                            "outcome": {"outcome": "selected", "optionId": option_id}
                        })
                    else:
                        # 某些请求可能没有 options，直接返回 approved?
                        # 观察日志，通常有关闭选项
                        await self._send_error(msg_id, -32603, "No options provided in permission request")
                else:
                    await self._send_error(msg_id, -32603, "Permission denied")

            # 终端创建
            elif method in ['terminal/create', 'create_terminal']:
                command = params.get('command')
                terminal_id = str(uuid.uuid4())
                # Phase 101: Use Event for wait_for_exit instead of polling
                self._terminals[terminal_id] = {
                    "output": "", 
                    "exit_code": None,
                    "exit_event": asyncio.Event()
                }

                try:
                    logger.info(f"[ACP] Creating terminal for command: {command}")
                    # Phase 100: 确保终端在正确的工作目录执行
                    cwd = self.config.cwd or os.getcwd()
                    process = await asyncio.create_subprocess_shell(
                        command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=cwd
                    )
                    await self._send_response(msg_id, {"terminalId": terminal_id})

                    async def _wait_for_process():
                        try:
                            stdout, stderr = await process.communicate()
                            output = stdout.decode(errors='replace') + stderr.decode(errors='replace')
                            self._terminals[terminal_id]["output"] = output
                            self._terminals[terminal_id]["exit_code"] = process.returncode
                            self._terminals[terminal_id]["exit_event"].set()
                            logger.info(f"[ACP] Terminal {terminal_id} finished causing exit_code={process.returncode}")
                        except Exception as e:
                            logger.error(f"[ACP] Terminal execution error: {e}")
                            self._terminals[terminal_id]["exit_code"] = 1
                            self._terminals[terminal_id]["output"] = str(e)
                            self._terminals[terminal_id]["exit_event"].set()

                    # 放入后台任务，避免阻塞消息循环
                    asyncio.create_task(_wait_for_process())
                    
                except Exception as e:
                    logger.error(f"[ACP] Failed to create subprocess: {e}")
                    # If start failed, we might have already responded? 
                    # Actually lines 283 sends response. If create_subprocess fails, we haven't sent response yet.
                    # But if we did... 
                    self._terminals[terminal_id]["exit_code"] = 1
                    self._terminals[terminal_id]["output"] = str(e)
                    self._terminals[terminal_id]["exit_event"].set()
                    await self._send_error(msg_id, -32000, str(e))

            # 终端等待
            elif method in ['terminal/wait_for_exit', 'wait_for_exit']:
                terminal_id = params.get('terminalId')
                term_data = self._terminals.get(terminal_id)
                if term_data:
                    # Phase 101: Efficient wait
                    await term_data["exit_event"].wait()
                    await self._send_response(msg_id, {
                        "exitStatus": {"exitCode": term_data["exit_code"], "signal": None}
                    })
                else:
                    await self._send_error(msg_id, -32602, f"Terminal not found: {terminal_id}")

            # 终端输出
            elif method in ['terminal/output', 'get_terminal_output']:
                terminal_id = params.get('terminalId')
                term_data = self._terminals.get(terminal_id)
                if term_data:
                    response = {"output": term_data["output"], "truncated": False}
                    if term_data["exit_code"] is not None:
                        response["exitStatus"] = {"exitCode": term_data["exit_code"], "signal": None}
                    await self._send_response(msg_id, response)
                else:
                    await self._send_error(msg_id, -32602, f"Terminal not found: {terminal_id}")

            # 终端释放
            elif method in ['terminal/release', 'release_terminal']:
                terminal_id = params.get('terminalId')
                if terminal_id in self._terminals:
                    del self._terminals[terminal_id]
                await self._send_response(msg_id, {})

            else:
                logger.warning(f"[ACP] Unknown method: {method}")
                await self._send_error(msg_id, -32601, f"Method not found: {method}")

        except Exception as e:
            logger.error(f"[ACP] Error handling request {method}: {e}", exc_info=True)
            await self._send_error(msg_id, -32603, str(e))

    async def _send_request(self, method: str, params: Optional[Dict] = None, timeout: int = 600) -> Any:
        """发送 RPC 请求"""
        self._msg_id += 1
        msg = {"jsonrpc": "2.0", "id": self._msg_id, "method": method, "params": params or {}}
        content = json.dumps(msg).encode('utf-8')
        logger.debug(f"[ACP] Sending request #{self._msg_id}: {method}")
        self.process.stdin.write(content + b'\n')
        await self.process.stdin.drain()

        future = asyncio.get_running_loop().create_future()
        self._pending_requests[self._msg_id] = future
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            logger.debug(f"[ACP] Request #{self._msg_id} ({method}) completed")
            return result
        except asyncio.TimeoutError:
            logger.error(f"[ACP] Request #{self._msg_id} ({method}) timeout after {timeout}s")
            raise

    async def _send_response(self, msg_id: int, result: Dict):
        """发送 RPC 响应"""
        msg = {"jsonrpc": "2.0", "id": msg_id, "result": result}
        content = json.dumps(msg).encode('utf-8')
        self.process.stdin.write(content + b'\n')
        await self.process.stdin.drain()

    async def _send_error(self, msg_id: int, code: int, message: str):
        """发送 RPC 错误"""
        msg = {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}
        content = json.dumps(msg).encode('utf-8')
        self.process.stdin.write(content + b'\n')
        await self.process.stdin.drain()

    async def resume(self, session_id: str) -> bool:
        """
        Phase 83: Resume 到指定 session

        允许在同一个 CLI 进程中动态切换 session，实现多用户共享连接池。

        Args:
            session_id: 目标 session id

        Returns:
            是否成功
        """
        if session_id == self.session_id:
            logger.debug(f"[ACP] Already on session {session_id}, skip resume")
            return True

        try:
            logger.info(f"[ACP] Resuming to session: {session_id}")
            result = await self._send_request("session/resume", {
                "sessionId": session_id,
                "cwd": self.config.cwd or os.getcwd()
            })

            new_session_id = result.get("sessionId", session_id)
            self.session_id = new_session_id
            self._last_used = time.time()

            logger.info(f"[ACP] Resumed to session: {new_session_id}")
            return True

        except Exception as e:
            logger.warning(f"[ACP] Resume failed: {e}, keeping current session: {self.session_id}")
            return False

    async def query(self, prompt: str) -> str:
        """发送查询（非流式）"""
        self._response_chunks = []
        self._response_complete = asyncio.Event()  # Phase 50 Fix: 创建完成事件
        self._streaming_mode = False  # Phase 83 Fix: 确保非流式模式
        self._stream_queue = None  # Phase 83 Fix: 清理队列引用
        self._last_used = time.time()

        logger.debug(f"[ACP] Sending prompt: {prompt[:100]}...")

        try:
            # 发送请求并获取 RPC 响应
            rpc_result = await self._send_request("session/prompt", {
                "sessionId": self.session_id,
                "prompt": [{"type": "text", "text": prompt}]
            })
        except Exception as e:
            # Phase 100 resilience: 如果 RPC 失败但我们已经收到了一些 chunk，则返回已收到的内容
            # 特别是针对 "Model stream ended without a finish reason" 这种非致命错误
            if self._response_chunks:
                logger.warning(f"[ACP] RPC session/prompt error: {e}, but content was received. Returning partial response.")
                return ''.join(self._response_chunks)
            raise

        # Phase 50 Fix: 检查 RPC 响应是否包含结果
        if rpc_result and isinstance(rpc_result, dict):
            # 某些 ACP 实现可能直接在 RPC 响应中返回结果
            content = rpc_result.get('content') or rpc_result.get('text') or rpc_result.get('result')
            
            # Phase 51 Fix: 如果 RPC 明确返回结束原因，则不再等待流
            if rpc_result.get('stopReason') in ('end_turn', 'stop_sequence', 'max_tokens'):
                logger.info(f"[ACP] RPC Result indicated stopReason: {rpc_result.get('stopReason')}")
                self._response_complete.set()
                
            if content:
                logger.info(f"[ACP] Got response from RPC result: {len(content)} chars")
                return content

        # Phase 50 Fix: 等待流式响应完成（带超时）
        try:
            await asyncio.wait_for(self._response_complete.wait(), timeout=self.config.timeout)
        except asyncio.TimeoutError:
            logger.warning(f"[ACP] Response timeout after {self.config.timeout}s, returning partial response")

        result = ''.join(self._response_chunks)
        logger.debug(f"[ACP] Response received: {len(result)} chars")
        return result

    async def query_stream(self, prompt: str) -> AsyncIterator[str]:
        """
        真正的实时流式查询

        Phase 83 v2: 并行处理 - 边发送边 yield chunks

        关键改进:
        - 使用 asyncio.create_task 发送请求，不阻塞
        - 立即开始从队列读取 chunks 并 yield
        - 真正实现 UI 实时显示效果
        """
        self._response_chunks = []
        self._response_complete = asyncio.Event()
        self._stream_queue = asyncio.Queue()
        self._streaming_mode = True
        self._last_used = time.time()

        logger.info(f"[ACP Stream] Sending prompt: {prompt[:100]}...")

        # 用于存储 RPC 结果
        rpc_result_holder = {"result": None, "error": None}

        async def send_request_async():
            """异步发送请求，不阻塞主流程"""
            try:
                result = await self._send_request("session/prompt", {
                    "sessionId": self.session_id,
                    "prompt": [{"type": "text", "text": prompt}]
                }, timeout=self.config.timeout)
                rpc_result_holder["result"] = result
                logger.info(f"[ACP Stream] RPC prompt request completed")
            except Exception as e:
                rpc_result_holder["error"] = e
                logger.error(f"[ACP Stream] RPC error: {e}")
                # 只在出错时发送结束标记
                if self._stream_queue:
                    try:
                        self._stream_queue.put_nowait(None)
                    except:
                        pass
            # NOTE: 不在 finally 中发送 None！
            # 结束标记由 _handle_message() 在收到 agent_message_end/turn_end 时发送
            # RPC 响应只是确认收到请求，不代表流式响应完成

        try:
            # 启动异步发送任务（不阻塞）
            send_task = asyncio.create_task(send_request_async())

            # 立即开始从队列读取 chunks 并 yield
            chunks_yielded = 0
            max_wait_seconds = 600  # 总超时 10 分钟
            chunk_timeout = 30.0    # 每 30 秒检查一次并记录状态
            wait_elapsed = 0

            while wait_elapsed < max_wait_seconds:
                # 检查进程是否还活着
                if not self.is_connected:
                    logger.warning(f"[ACP Stream] Disconnected while waiting for stream. PID={self.process.pid if self.process else 'None'}")
                    # 尝试读取最后可能残留在 _response_chunks 的内容
                    if self._response_chunks and chunks_yielded == 0:
                        yield ''.join(self._response_chunks)
                    break

                try:
                    chunk = await asyncio.wait_for(
                        self._stream_queue.get(),
                        timeout=chunk_timeout
                    )
                    wait_elapsed = 0  # 收到 chunk 后重置等待时间
                    if chunk is None:  # 结束标记
                        logger.info(f"[ACP Stream] Stream complete, yielded {chunks_yielded} chunks")
                        break
                    if chunk:
                        chunks_yielded += 1
                        yield chunk
                except asyncio.TimeoutError:
                    wait_elapsed += chunk_timeout
                    logger.info(f"[ACP Stream] Still waiting for response... ({int(wait_elapsed)}s elapsed, {chunks_yielded} chunks so far)")
                    
                    # 如果等待时间过长且没有任何 chunk，检查进程状态
                    if wait_elapsed >= 60 and chunks_yielded == 0:
                        if self.process and self.process.returncode is not None:
                            logger.error(f"[ACP Stream] Process exited with code {self.process.returncode}")
                            break

            if wait_elapsed >= max_wait_seconds:
                logger.warning(f"[ACP Stream] Total timeout ({max_wait_seconds}s) after {chunks_yielded} chunks yielded")

            # 等待发送任务完成
            await send_task

            # 检查是否有错误
            if rpc_result_holder["error"]:
                raise rpc_result_holder["error"]

            # 如果没有收到任何 chunk，尝试从 _response_chunks 获取
            if chunks_yielded == 0 and self._response_chunks:
                full_content = ''.join(self._response_chunks)
                if full_content:
                    logger.info(f"[ACP Stream] Fallback: yielding from _response_chunks ({len(full_content)} chars)")
                    yield full_content

        finally:
            self._streaming_mode = False
            self._stream_queue = None

    async def disconnect(self):
        """断开连接"""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except Exception:
                self.process.kill()
            finally:
                self._connected = False
                self.process = None


class ClaudeACPClient(ACPClientBase):
    """Claude ACP 客户端"""

    def _get_acp_command(self) -> List[str]:
        # 如果是 claude 项目的主入口，需要添加 mcp serve
        if "claude-code-acp" in self.config.acp_cmd.lower():
            return [self.config.acp_cmd]
        return [self.config.acp_cmd, "mcp", "serve"]

    async def _initialize(self):
        logger.info("[ClaudeACP] Sending initialize request...")
        # claude-code-acp 需要 protocolVersion 为数字
        # 标准 claude mcp serve 需要 protocolVersion 为字符串
        is_acp_binary = "claude-code-acp" in self.config.acp_cmd.lower()
        protocol_version = 1 if is_acp_binary else "1"
        logger.debug(f"[ClaudeACP] Using {'ACP' if is_acp_binary else 'MCP'} protocol (protocolVersion={protocol_version!r})")

        try:
             init_res = await asyncio.wait_for(
                self._send_request("initialize", {
                    "clientInfo": {"name": "SoulBot", "version": "1.0"},
                    "protocolVersion": protocol_version,
                    "capabilities": {
                        "fs": {"readTextFile": True, "writeTextFile": True},
                        "terminal": True
                    }
                }, timeout=30), timeout=30
             )
        except asyncio.TimeoutError:
             logger.error("[ClaudeACP] Initialize request timed out after 30s")
             raise

        # 检查认证方法（ACP 协议返回的登录提示）
        auth_methods = init_res.get('authMethods', []) if init_res else []
        if auth_methods:
            logger.info(f"[ClaudeACP] Available auth methods: {[m.get('name') for m in auth_methods]}")
            # 保存认证方法信息，供后续错误处理使用
            self._auth_methods = auth_methods

        logger.info("[ClaudeACP] Initialize done, creating session...")

        try:
            session_res = await self._send_request("session/new", {
                "cwd": self.config.cwd or os.getcwd(),
                "mcpServers": []
            })
            self.session_id = session_res['sessionId']
            logger.info(f"[ClaudeACP] Session created: {self.session_id}")
        except Exception as e:
            error_msg = str(e)
            if "Method not found" in error_msg and not is_acp_binary:
                raise RuntimeError(
                    f"Claude CLI (mcp serve) does not support ACP session methods. "
                    f"Please install the ACP package: "
                    f"npm install -g @anthropic-ai/claude-code "
                    f"(ensure 'claude-code-acp' binary is available in PATH). "
                    f"Original error: {error_msg}"
                ) from e
            raise

        # 切换模型
        if self.config.model:
            logger.info(f"[ClaudeACP] Setting model to: {self.config.model}")
            try:
                await self._send_request("session/set_model", {
                    "sessionId": self.session_id,
                    "modelId": self.config.model
                })
            except Exception as e:
                if "Method not found" in str(e):
                    logger.warning(f"[ClaudeACP] session/set_model not supported, skipping model switch")
                else:
                    raise


class GeminiACPClient(ACPClientBase):
    """Gemini ACP 客户端"""

    def _get_acp_command(self) -> List[str]:
        cmd = [self.config.acp_cmd, "--experimental-acp"]
        if self.config.model and self.config.model != "auto":
             # 提取纯模型ID (去除 gemini-acp/ 前缀 if any)
             model_id = self.config.model.split("/")[-1]
             cmd.extend(["--model", model_id])
        return cmd

    async def _initialize(self):
        init_res = await self._send_request("initialize", {
            "clientInfo": {"name": "SoulBot", "version": "1.0"},
            "protocolVersion": 1,
            "capabilities": {
                "fs": {"readTextFile": True, "writeTextFile": True},
                "terminal": True
            }
        })

        # Gemini 需要认证
        auth_methods = init_res.get('authMethods', [])
        if auth_methods:
            auth_method_id = auth_methods[0].get('id', 'oauth-personal')
            await self._send_request("authenticate", {"methodId": auth_method_id})

        # 不在这里创建 session — 由 acquire() 决定是 load 已有 session 还是创建新的
        # 这样可以先尝试 session/load，避免浪费一个新 session
        self.session_id = None
        cwd = self.config.cwd or os.getcwd()
        logger.info(f"[GeminiACP] Initialized, CWD: {cwd} (no session yet, will be created/loaded in acquire)")

    async def resume(self, session_id: str) -> bool:
        """
        Gemini 使用 session/load（而非 session/resume）从磁盘恢复 session

        Gemini CLI 将会话持久化到 ~/.gemini/tmp/<project_hash>/chats/ 目录，
        session/load 方法会读取磁盘文件并恢复完整对话历史到 GeminiChat 内存中。
        """
        if session_id == self.session_id:
            logger.debug(f"[GeminiACP] Already on session {session_id}, skip load")
            return True

        cwd = self.config.cwd or os.getcwd()
        logger.info(f"[GeminiACP] Attempting to load session {session_id[:20]}..., CWD: {cwd}")

        # 先用 session/list 验证 session 是否存在于磁盘
        session_exists = False
        try:
            list_res = await self._send_request("session/list", {"cwd": cwd}, timeout=15)
            available_sessions = list_res.get("sessions", []) if list_res else []
            available_ids = [s.get("sessionId") or s.get("id") for s in available_sessions]
            logger.info(
                f"[GeminiACP] Available sessions ({len(available_sessions)}): "
                f"{[sid[:20]+'...' if sid and len(sid)>20 else sid for sid in available_ids]}"
            )
            session_exists = session_id in available_ids
            if not session_exists:
                logger.warning(
                    f"[GeminiACP] Target session {session_id[:20]}... "
                    f"NOT found in {len(available_sessions)} available sessions"
                )
        except Exception as e:
            logger.warning(f"[GeminiACP] session/list failed (will try load anyway): {e}")
            session_exists = True  # list 失败时仍尝试 load

        if not session_exists:
            # Session 不在列表中，直接返回失败（避免无谓的 load 超时）
            logger.warning(
                f"[GeminiACP] Session {session_id[:20]}... not available on disk, "
                f"will create new session"
            )
            return False

        try:
            logger.info(f"[GeminiACP] Loading session from disk: {session_id}")
            result = await self._send_request("session/load", {
                "sessionId": session_id,
                "cwd": cwd,
                "mcpServers": []
            })

            # session/load 成功后，更新本地 session_id
            self.session_id = session_id
            self._last_used = time.time()
            logger.info(f"[GeminiACP] Session loaded successfully: {session_id}")
            return True

        except Exception as e:
            logger.warning(
                f"[GeminiACP] session/load failed for {session_id}: {e}. "
                f"Keeping current session: {self.session_id}"
            )
            return False


class ACPConnectionPool:
    """ACP 连接池"""

    def __init__(self, config: CLIConfig):
        self.config = config
        self._pool: List[ACPClientBase] = []
        self._lock = asyncio.Lock()
        self._client_class = ClaudeACPClient if config.provider == CLIProvider.CLAUDE else GeminiACPClient

    async def _create_client(self) -> ACPClientBase:
        """创建新客户端"""
        logger.info(f"[ACPPool] Creating new {self.config.provider.value} client...")
        client = self._client_class(self.config)
        logger.info(f"[ACPPool] Connecting client (cmd={self.config.acp_cmd})...")
        success = await client.connect()
        if not success:
            raise ConnectionError(
                f"Failed to connect {self.config.provider.value} ACP client. "
                f"Check logs above for detailed error."
            )
        # Gemini 延迟创建 session（在 acquire 中决定 load 还是 new）
        # Claude 在 _initialize 中就创建了 session
        if client.session_id:
            logger.info(f"[ACPPool] Client connected, session_id={client.session_id}")
        else:
            logger.info(f"[ACPPool] Client connected (session will be created/loaded in acquire)")
        return client

    @asynccontextmanager
    async def acquire(self, session_id: Optional[str] = None):
        """
        获取连接

        Phase 83: 支持动态 session 切换

        Args:
            session_id: 可选 session id
                       - 提供: resume 到该 session（有历史记忆）
                       - 不提供: 使用当前 session（无历史记忆）

        Yields:
            (client, current_session_id): 客户端和当前 session id
        """
        logger.info(f"[ACPPool] Acquiring connection (session_id={session_id})...")
        client = None

        async with self._lock:
            # 清理过期连接
            self._pool = [c for c in self._pool if c.is_connected and not c.is_idle_timeout]
            logger.debug(f"[ACPPool] Pool size after cleanup: {len(self._pool)}, "
                         f"sessions: {[c.session_id[:12] + '...' if c.session_id else 'None' for c in self._pool]}")

            # 优先匹配相同 session_id 的客户端（避免不必要的 resume/load）
            if self._pool and session_id:
                for i, c in enumerate(self._pool):
                    if c.is_connected and c.session_id == session_id:
                        client = self._pool.pop(i)
                        logger.info(f"[ACPPool] Reusing client with matching session_id={session_id[:12]}...")
                        break

            # 如果没找到匹配的，取第一个可用的
            if not client and self._pool:
                client = self._pool.pop(0)
                if not client.is_connected:
                    client = None

        # 如果没有可用连接，创建新连接
        if not client:
            client = await self._create_client()

        try:
            # 确保 client 有一个可用的 session
            # 策略：先尝试 load 已有 session，失败则创建新 session
            if session_id and session_id != client.session_id:
                success = await client.resume(session_id)
                if success:
                    logger.info(f"[ACPPool] Session loaded/resumed: {session_id[:20]}...")
                else:
                    cwd = self.config.cwd or os.getcwd()
                    logger.warning(
                        f"[ACPPool] Resume/load FAILED for session {session_id[:20]}... "
                        f"(provider={self.config.provider.value}, CWD={cwd}). "
                        f"Will create new session (memory lost for this request)"
                    )

            # 如果 client 还没有 session（Gemini 延迟创建 or resume 失败），创建新的
            if not client.session_id:
                cwd = self.config.cwd or os.getcwd()
                logger.info(f"[ACPPool] Creating new session for {self.config.provider.value} (CWD={cwd})...")
                session_res = await client._send_request("session/new", {
                    "cwd": cwd,
                    "mcpServers": []
                })
                client.session_id = session_res['sessionId']
                logger.info(f"[ACPPool] New session created: {client.session_id}")

            # 返回 (client, current_session_id)
            logger.info(f"[ACPPool] Yielding client with session_id={client.session_id}")
            yield client, client.session_id

            # 放回池中
            async with self._lock:
                if len(self._pool) < self.config.pool_size:
                    self._pool.append(client)
                else:
                    await client.disconnect()

        except Exception:
            await client.disconnect()
            raise

    async def close_all(self):
        """关闭所有连接"""
        async with self._lock:
            for client in self._pool:
                await client.disconnect()
            self._pool.clear()


def _npm_install_global(package: str) -> bool:
    """通过 npm 全局安装指定包"""
    import subprocess as sp
    npm_cmd = shutil.which("npm")
    if not npm_cmd:
        logger.warning(f"[AutoInstall] npm not found, cannot install {package}")
        return False

    logger.info(f"[AutoInstall] Running: npm install -g {package} ...")
    try:
        result = sp.run(
            [npm_cmd, "install", "-g", package],
            capture_output=True, text=True, timeout=180
        )
        if result.returncode == 0:
            logger.info(f"[AutoInstall] {package} installed successfully")
            return True
        else:
            stderr = result.stderr.strip()[:300] if result.stderr else ""
            logger.warning(f"[AutoInstall] npm install {package} failed (exit {result.returncode}): {stderr}")
            return False
    except Exception as e:
        logger.warning(f"[AutoInstall] npm install {package} error: {e}")
        return False


def _find_binary_extended(name: str) -> Optional[str]:
    """在 PATH 和常见全局 npm 位置查找二进制文件"""
    cmd = shutil.which(name)
    if cmd:
        return cmd
    # Windows: 全局 npm 安装路径 %APPDATA%/npm
    if sys.platform == "win32":
        for suffix in [".cmd", ".exe", ""]:
            appdata_path = os.path.join(os.environ.get("APPDATA", ""), "npm", f"{name}{suffix}")
            if os.path.exists(appdata_path):
                return appdata_path
    return None


def find_claude_acp_binary() -> Optional[str]:
    """
    查找或自动安装 Claude ACP 二进制文件

    搜索顺序:
    1. 项目本地 node_modules/.bin/claude-code-acp
    2. CWD node_modules/.bin/claude-code-acp
    3. 全局 PATH claude-code-acp
    4. 自动安装 @anthropic-ai/claude-code → 重新检测
    5. 回退到全局 claude 命令 (MCP 模式，功能受限)

    Returns:
        二进制文件路径，找不到则返回 None
    """
    # 项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    bin_name = "claude-code-acp.cmd" if sys.platform == "win32" else "claude-code-acp"

    # 1. 项目本地 node_modules
    local_cmd = os.path.join(project_root, "node_modules", ".bin", bin_name)
    if os.path.exists(local_cmd):
        logger.info(f"[ClaudeACP] Found local claude-code-acp: {local_cmd}")
        return local_cmd

    # 2. CWD node_modules
    cwd_cmd = os.path.join(os.getcwd(), "node_modules", ".bin", bin_name)
    if cwd_cmd != local_cmd and os.path.exists(cwd_cmd):
        logger.info(f"[ClaudeACP] Found CWD claude-code-acp: {cwd_cmd}")
        return cwd_cmd

    # 3. 全局 PATH
    cmd = _find_binary_extended("claude-code-acp")
    if cmd:
        logger.info(f"[ClaudeACP] Found global claude-code-acp: {cmd}")
        return cmd

    # 4. 自动安装
    logger.warning("[ClaudeACP] claude-code-acp not found. Attempting auto-install...")

    # 4a. 安装 @anthropic-ai/claude-code (提供 claude CLI)
    _npm_install_global("@anthropic-ai/claude-code")
    cmd = _find_binary_extended("claude-code-acp")
    if cmd:
        logger.info(f"[ClaudeACP] claude-code-acp available after install: {cmd}")
        return cmd

    # 4b. 安装 @zed-industries/claude-code-acp (提供 claude-code-acp 二进制)
    _npm_install_global("@zed-industries/claude-code-acp")
    cmd = _find_binary_extended("claude-code-acp")
    if cmd:
        logger.info(f"[ClaudeACP] claude-code-acp available after install: {cmd}")
        return cmd

    # 5. 回退到 claude 命令 (MCP 模式，功能受限)
    cmd = _find_binary_extended("claude")
    if cmd:
        logger.warning(
            f"[ClaudeACP] Only 'claude' found (MCP mode, ACP session methods not supported): {cmd}."
        )
        return cmd

    logger.error("[ClaudeACP] No Claude CLI binary found. Install with: npm install -g @anthropic-ai/claude-code")
    return None


class CLIAgentService:
    """
    CLI Agent 统一服务

    与 SoulBot LLMService 并行，提供 CLI 方式的 LLM 调用。
    """

    def __init__(
        self,
        claude_config: Optional[CLIConfig] = None,
        gemini_config: Optional[CLIConfig] = None,
        default_provider: CLIProvider = CLIProvider.CLAUDE
    ):
        self._pools: Dict[CLIProvider, ACPConnectionPool] = {}
        self._default_provider = default_provider

        if claude_config:
            self._pools[CLIProvider.CLAUDE] = ACPConnectionPool(claude_config)

        if gemini_config:
            self._pools[CLIProvider.GEMINI] = ACPConnectionPool(gemini_config)

    async def complete(
        self,
        messages: List[CLIMessage],
        provider: Optional[CLIProvider] = None,
        **kwargs
    ) -> CLIResponse:
        """
        发送完成请求

        Args:
            messages: 消息列表
            provider: CLI 提供商（默认使用 default_provider）

        Returns:
            CLIResponse
        """
        provider = provider or self._default_provider

        if provider not in self._pools:
            raise ValueError(f"Provider {provider.value} not configured")

        pool = self._pools[provider]

        # 构建 prompt
        prompt = self._build_prompt(messages)

        start_time = time.time()

        async with pool.acquire() as (client, _):
            content = await client.query(prompt)

        latency_ms = (time.time() - start_time) * 1000

        return CLIResponse(
            content=content,
            model=pool.config.model or "default",
            provider=provider,
            latency_ms=latency_ms
        )

    def _build_prompt(self, messages: List[CLIMessage]) -> str:
        """构建 prompt"""
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"[System]: {msg.content}")
            elif msg.role == "user":
                parts.append(f"[User]: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"[Assistant]: {msg.content}")
        return "\n\n".join(parts)

    async def stream_complete(
        self,
        messages: List[CLIMessage],
        provider: Optional[CLIProvider] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """流式完成（当前直接返回完整响应）"""
        response = await self.complete(messages, provider, **kwargs)
        yield response.content

    async def close(self):
        """关闭所有连接池"""
        for pool in self._pools.values():
            await pool.close_all()

    @classmethod
    def create_default(cls) -> "CLIAgentService":
        """创建默认配置的服务"""
        from src.config import GEMINI_MODEL, CLAUDE_MODEL, WORKSPACE_DIR, AUTO_APPROVE_PERMISSIONS, SHOW_THOUGHTS

        # 查找 Claude ACP（使用共享检测+自动安装函数）
        claude_cmd = find_claude_acp_binary()

        claude_config = None
        if claude_cmd:
            claude_config = CLIConfig(
                provider=CLIProvider.CLAUDE,
                acp_cmd=claude_cmd,
                model="sonnet",
                cwd=WORKSPACE_DIR,
                auto_approve_permissions=AUTO_APPROVE_PERMISSIONS,
                show_thoughts=SHOW_THOUGHTS,
                pool_size=10
            )

        # 查找 Gemini ACP
        gemini_cmd = shutil.which("gemini")
        gemini_config = None
        if gemini_cmd:
            gemini_config = CLIConfig(
                provider=CLIProvider.GEMINI,
                acp_cmd=gemini_cmd,
                model=GEMINI_MODEL,  # 使用配置的模型
                cwd=WORKSPACE_DIR,
                auto_approve_permissions=AUTO_APPROVE_PERMISSIONS,
                pool_size=10
            )

        default = CLIProvider.CLAUDE if claude_config else CLIProvider.GEMINI

        return cls(
            claude_config=claude_config,
            gemini_config=gemini_config,
            default_provider=default
        )


# ============================================================
# 便捷函数
# ============================================================


def get_session_dir(bot_data_dir: str, provider: CLIProvider) -> str:
    """
    Phase 83: 获取 bot 的 session 存储目录

    根据 provider 类型自动生成目录名：
    - Claude -> {bot_data_dir}/Claude_Session/
    - Gemini -> {bot_data_dir}/Gemini_Session/

    Args:
        bot_data_dir: bot 的 data 目录路径
        provider: CLI 提供商

    Returns:
        session 存储目录的完整路径

    Example:
        >>> session_dir = get_session_dir("/path/to/bot/data", CLIProvider.CLAUDE)
        >>> # Returns: "/path/to/bot/data/Claude_Session"
        >>>
        >>> config = CLIConfig(
        ...     provider=CLIProvider.CLAUDE,
        ...     acp_cmd="claude-code-acp",
        ...     session_dir=session_dir
        ... )
    """
    provider_dir_names = {
        CLIProvider.CLAUDE: "Claude_Session",
        CLIProvider.GEMINI: "Gemini_Session",
    }
    dir_name = provider_dir_names.get(provider, f"{provider.value.title()}_Session")
    return os.path.join(bot_data_dir, dir_name)


async def cli_complete(
    prompt: str,
    provider: CLIProvider = CLIProvider.CLAUDE,
    **kwargs
) -> str:
    """快速调用 CLI"""
    service = CLIAgentService.create_default()
    try:
        response = await service.complete(
            [CLIMessage(role="user", content=prompt)],
            provider=provider,
            **kwargs
        )
        return response.content
    finally:
        await service.close()


# ============================================================
# Session 清理功能
# ============================================================


@dataclass
class SessionInfo:
    """Session 信息"""
    session_id: str
    full_path: str
    created: str
    modified: str
    message_count: int
    summary: str
    first_prompt: str


@dataclass
class CleanupResult:
    """清理结果"""
    total_sessions: int
    expired_sessions: int
    deleted_sessions: int
    kept_important: int
    deleted_list: List[str]
    kept_list: List[str]
    errors: List[str]


def get_expired_sessions(
    session_dir: str,
    max_age_days: int = 30
) -> List[SessionInfo]:
    """
    获取过期的 session 列表

    Args:
        session_dir: session 存储目录
        max_age_days: 最大保留天数，超过则视为过期

    Returns:
        过期的 session 列表
    """
    from datetime import datetime, timezone

    expired = []

    # 查找所有 projects 目录下的 sessions-index.json
    projects_dir = os.path.join(session_dir, "projects")
    if not os.path.exists(projects_dir):
        return expired

    cutoff = datetime.now(timezone.utc).timestamp() - (max_age_days * 86400)

    for project_name in os.listdir(projects_dir):
        project_path = os.path.join(projects_dir, project_name)
        if not os.path.isdir(project_path):
            continue

        index_file = os.path.join(project_path, "sessions-index.json")
        if not os.path.exists(index_file):
            continue

        try:
            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for entry in data.get("entries", []):
                # 解析修改时间
                modified_str = entry.get("modified", "")
                if not modified_str:
                    continue

                try:
                    # ISO 8601 格式: 2026-01-29T23:30:45.205Z
                    modified_dt = datetime.fromisoformat(
                        modified_str.replace("Z", "+00:00")
                    )
                    modified_ts = modified_dt.timestamp()
                except ValueError:
                    continue

                # 检查是否过期
                if modified_ts < cutoff:
                    expired.append(SessionInfo(
                        session_id=entry.get("sessionId", ""),
                        full_path=entry.get("fullPath", ""),
                        created=entry.get("created", ""),
                        modified=modified_str,
                        message_count=entry.get("messageCount", 0),
                        summary=entry.get("summary", ""),
                        first_prompt=entry.get("firstPrompt", ""),
                    ))

        except Exception as e:
            logger.warning(f"[SessionCleanup] Error reading {index_file}: {e}")

    return expired


async def check_session_importance(
    session: SessionInfo,
    llm_service: Any = None
) -> bool:
    """
    使用 LLM 检查 session 是否重要

    Args:
        session: session 信息
        llm_service: LLM 服务实例（可选，如果不提供则直接返回 False）

    Returns:
        True = 重要，应保留；False = 不重要，可删除
    """
    if not llm_service:
        return False

    # 构建提示
    prompt = f"""请判断以下对话 session 是否重要，是否值得保留。

Session 信息:
- 摘要: {session.summary}
- 首条消息: {session.first_prompt[:200] if session.first_prompt else '(无)'}
- 消息数量: {session.message_count}
- 最后修改: {session.modified}

判断标准:
1. 包含重要决策、架构设计、关键配置 → 重要
2. 包含重要的代码实现或bug修复记录 → 重要
3. 只是简单问答、测试、临时对话 → 不重要
4. 摘要为空或无意义 → 不重要

请只回答一个词: IMPORTANT 或 NOT_IMPORTANT
"""

    try:
        response = await llm_service.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=20
        )
        result = response.content.strip().upper()
        return "IMPORTANT" in result
    except Exception as e:
        logger.warning(f"[SessionCleanup] LLM check failed: {e}")
        return False  # 如果 LLM 检查失败，默认不重要


def delete_session(session: SessionInfo, session_dir: str) -> bool:
    """
    删除 session 文件并更新索引

    Args:
        session: session 信息
        session_dir: session 存储目录

    Returns:
        是否成功删除
    """
    try:
        # 删除 session 文件
        if os.path.exists(session.full_path):
            os.remove(session.full_path)
            logger.info(f"[SessionCleanup] Deleted: {session.full_path}")

        # 删除相关的子目录（如 subagents）
        session_subdir = session.full_path.replace(".jsonl", "")
        if os.path.isdir(session_subdir):
            shutil.rmtree(session_subdir)
            logger.info(f"[SessionCleanup] Deleted subdir: {session_subdir}")

        # 更新 sessions-index.json
        project_dir = os.path.dirname(session.full_path)
        index_file = os.path.join(project_dir, "sessions-index.json")

        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 移除该 session
            data["entries"] = [
                e for e in data.get("entries", [])
                if e.get("sessionId") != session.session_id
            ]

            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        logger.error(f"[SessionCleanup] Failed to delete {session.session_id}: {e}")
        return False


async def cleanup_expired_sessions(
    session_dir: str,
    max_age_days: int = 30,
    llm_service: Any = None,
    dry_run: bool = False
) -> CleanupResult:
    """
    清理过期的 session

    Args:
        session_dir: session 存储目录
        max_age_days: 最大保留天数（默认 30 天）
        llm_service: LLM 服务实例（用于检查重要性，可选）
        dry_run: 如果为 True，只返回待删除列表，不实际删除

    Returns:
        CleanupResult: 清理结果

    Example:
        >>> from soulbot.core.llm_service import LLMService
        >>> llm = LLMService()
        >>> result = await cleanup_expired_sessions(
        ...     session_dir="/path/to/Claude_Session",
        ...     max_age_days=30,
        ...     llm_service=llm
        ... )
        >>> print(f"Deleted {result.deleted_sessions} sessions")
    """
    result = CleanupResult(
        total_sessions=0,
        expired_sessions=0,
        deleted_sessions=0,
        kept_important=0,
        deleted_list=[],
        kept_list=[],
        errors=[]
    )

    # 获取过期 session
    expired = get_expired_sessions(session_dir, max_age_days)
    result.expired_sessions = len(expired)

    logger.info(f"[SessionCleanup] Found {len(expired)} expired sessions (>{max_age_days} days)")

    for session in expired:
        # 检查是否重要
        if llm_service:
            is_important = await check_session_importance(session, llm_service)
            if is_important:
                result.kept_important += 1
                result.kept_list.append(
                    f"{session.session_id[:8]}... - {session.summary[:50]}"
                )
                logger.info(
                    f"[SessionCleanup] Keeping important: {session.session_id[:8]}... "
                    f"({session.summary[:30]}...)"
                )
                continue

        # 删除 session
        if dry_run:
            result.deleted_list.append(
                f"{session.session_id[:8]}... - {session.summary[:50]}"
            )
            result.deleted_sessions += 1
        else:
            if delete_session(session, session_dir):
                result.deleted_sessions += 1
                result.deleted_list.append(
                    f"{session.session_id[:8]}... - {session.summary[:50]}"
                )
            else:
                result.errors.append(f"Failed to delete {session.session_id}")

    logger.info(
        f"[SessionCleanup] Complete: deleted={result.deleted_sessions}, "
        f"kept_important={result.kept_important}, errors={len(result.errors)}"
    )

    return result
