"""
Cursor CLI Provider

通过 subprocess 调用 Cursor Agent CLI (`agent -p`) 集成到 LiteLLM。

使用方式：
    import litellm
    from soulbot.core.llm_service.cursor_cli_provider import register_cursor_provider

    # 注册 Cursor 提供商
    register_cursor_provider()

    # 通过 LiteLLM 调用
    response = await litellm.acompletion(
        model="cursor-cli/Auto",  # 或 "cursor-cli/gpt-4.1"
        messages=[{"role": "user", "content": "Hello"}]
    )

    # 使用 session（多轮对话）
    response = await litellm.acompletion(
        model="cursor-cli/Auto",
        messages=[{"role": "user", "content": "Hello"}],
        session_id="abc123"  # 恢复指定会话
    )

Session 管理：
    from soulbot.core.llm_service.cursor_cli_provider import CursorSessionManager

    manager = CursorSessionManager()
    session_id = await manager.create_session()  # 创建新会话
    sessions = await manager.list_sessions()     # 列出所有会话
"""

import asyncio
import json
import os
import subprocess
import shutil
import sys
import time
import logging
import re
from typing import Optional, List, Dict, Any, Iterator, AsyncIterator

import litellm
from litellm import CustomLLM, ModelResponse, Message
from litellm.types.utils import Choices, Usage

logger = logging.getLogger(__name__)


def _get_subprocess_encoding() -> str:
    """获取子进程输出的编码（Windows 兼容）"""
    # Cursor CLI 输出始终是 UTF-8，即使在 Windows 上
    # 只有错误消息可能使用系统编码
    return "utf-8"


def _get_system_encoding() -> str:
    """获取系统编码（用于错误消息）"""
    if sys.platform == "win32":
        import locale
        return locale.getpreferredencoding(False) or "utf-8"
    return "utf-8"


def _safe_decode(data: bytes, encoding: str = None) -> str:
    """安全解码字节数据，优先 UTF-8"""
    # Cursor CLI 主要输出 UTF-8
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        # 回退：尝试系统编码（Windows 错误消息可能是 GBK）
        try:
            sys_encoding = _get_system_encoding()
            return data.decode(sys_encoding)
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")


def _find_cursor_agent_cmd() -> str:
    """查找 Cursor Agent CLI 命令路径"""
    import os
    import sys

    # 尝试多种命令名
    cmd_names = ["cursor-agent", "agent"]
    for cmd_name in cmd_names:
        cmd = shutil.which(cmd_name)
        if cmd:
            return cmd

    # 尝试 Windows 常见位置
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        # 尝试多种可能的文件名
        possible_paths = [
            os.path.join(local_appdata, "cursor-agent", "cursor-agent.cmd"),
            os.path.join(local_appdata, "cursor-agent", "cursor-agent.exe"),
            os.path.join(local_appdata, "cursor-agent", "agent.exe"),
            os.path.join(local_appdata, "cursor-agent", "agent.cmd"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path

    raise RuntimeError(
        "Cursor Agent CLI not found. Install with: "
        "irm 'https://cursor.com/install?win32=true' | iex"
    )


class CursorSessionManager:
    """
    Cursor CLI Session 管理器

    管理 Cursor CLI 的会话（chat sessions），支持：
    - 创建新会话
    - 列出所有会话
    - 获取最近会话 ID
    - 本地 session 映射（存储到 {session_dir}/sessions.json）

    注意：Cursor CLI 的实际 session 数据由 Cursor 内部管理，
    我们只在本地维护一个 session 映射文件用于追踪和管理。
    """

    def __init__(
        self,
        default_workspace: Optional[str] = None,
        session_dir: Optional[str] = None
    ):
        """
        初始化 Session 管理器

        Args:
            default_workspace: 默认工作目录（在调用 LLM 时使用）
            session_dir: Session 映射存储目录（如 {bot_data_dir}/Cursor_Session）
        """
        self._cmd: Optional[str] = None
        self.default_workspace = default_workspace
        self.session_dir = session_dir
        self._sessions_file: Optional[str] = None

        # 确保 session 目录存在
        if session_dir:
            import os
            os.makedirs(session_dir, exist_ok=True)
            self._sessions_file = os.path.join(session_dir, "sessions.json")
            logger.info(f"[CursorSession] Session dir: {session_dir}")

    def _get_cmd(self) -> str:
        if self._cmd is None:
            self._cmd = _find_cursor_agent_cmd()
        return self._cmd

    def _load_sessions(self) -> Dict[str, Any]:
        """加载本地 session 映射"""
        if not self._sessions_file:
            return {}

        import os
        if not os.path.exists(self._sessions_file):
            return {}

        try:
            with open(self._sessions_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[CursorSession] Failed to load sessions: {e}")
            return {}

    def _save_sessions(self, sessions: Dict[str, Any]) -> None:
        """保存本地 session 映射"""
        if not self._sessions_file:
            return

        try:
            with open(self._sessions_file, "w", encoding="utf-8") as f:
                json.dump(sessions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[CursorSession] Failed to save sessions: {e}")

    async def create_session(self, name: Optional[str] = None) -> str:
        """
        创建新的 chat session

        Args:
            name: 会话名称/标签（仅用于本地记录）

        Returns:
            session_id: 新创建的会话 ID
        """
        loop = asyncio.get_event_loop()
        session_id = await loop.run_in_executor(None, self._create_session_sync)

        # 保存到本地映射
        if self._sessions_file:
            sessions = self._load_sessions()
            sessions[session_id] = {
                "name": name or f"Session {len(sessions) + 1}",
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "last_used": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "workspace": self.default_workspace,
                "message_count": 0,
            }
            self._save_sessions(sessions)

        return session_id

    def _create_session_sync(self) -> str:
        """同步创建会话"""
        cmd = [self._get_cmd(), "create-chat"]

        logger.info(f"[CursorSession] Creating session...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create session: {result.stderr}")

        # 解析返回的 session ID
        output = result.stdout.strip()
        logger.info(f"[CursorSession] Created: {output}")

        return output

    def update_session_usage(self, session_id: str) -> None:
        """更新 session 使用记录"""
        if not self._sessions_file:
            return

        sessions = self._load_sessions()
        if session_id in sessions:
            sessions[session_id]["last_used"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            sessions[session_id]["message_count"] = (
                sessions[session_id].get("message_count", 0) + 1
            )
            self._save_sessions(sessions)

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取 session 信息"""
        sessions = self._load_sessions()
        return sessions.get(session_id)

    def get_all_sessions(self) -> Dict[str, Any]:
        """获取所有本地记录的 sessions"""
        return self._load_sessions()

    def get_recent_session(self) -> Optional[str]:
        """获取最近使用的 session ID"""
        sessions = self._load_sessions()
        if not sessions:
            return None

        # 按 last_used 排序
        sorted_sessions = sorted(
            sessions.items(),
            key=lambda x: x[1].get("last_used", ""),
            reverse=True
        )
        return sorted_sessions[0][0] if sorted_sessions else None

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        列出所有会话

        Returns:
            sessions: 会话列表 [{"id": "...", "summary": "...", "updated": "..."}]
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._list_sessions_sync)

    def _list_sessions_sync(self) -> List[Dict[str, Any]]:
        """同步列出会话"""
        cmd = [self._get_cmd(), "ls"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30
        )

        if result.returncode != 0:
            logger.warning(f"[CursorSession] Failed to list sessions: {result.stderr}")
            return []

        # 解析输出
        sessions = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # 尝试解析每行（格式可能因版本不同）
            # 假设格式: "chat_id  summary  timestamp"
            parts = line.split(None, 2)  # 最多分成3部分
            if parts:
                session = {
                    "id": parts[0],
                    "summary": parts[1] if len(parts) > 1 else "",
                    "updated": parts[2] if len(parts) > 2 else "",
                }
                sessions.append(session)

        return sessions

    async def get_last_session_id(self) -> Optional[str]:
        """获取最近使用的会话 ID"""
        sessions = await self.list_sessions()
        if sessions:
            return sessions[0].get("id")
        return None


# 全局 session 管理器（按 bot_data_dir 缓存）
_session_managers: Dict[str, CursorSessionManager] = {}


def get_cursor_session_manager(
    bot_data_dir: Optional[str] = None,
    workspace: Optional[str] = None
) -> CursorSessionManager:
    """
    获取 Cursor Session 管理器

    Args:
        bot_data_dir: Bot 的 data 目录，session 映射存储到 {bot_data_dir}/Cursor_Session/
        workspace: 默认工作目录（在调用 LLM 时使用）

    Returns:
        CursorSessionManager 实例
    """
    import os
    global _session_managers
    cache_key = bot_data_dir or "_default_"

    if cache_key not in _session_managers:
        # 计算 session 目录
        session_dir = None
        if bot_data_dir:
            session_dir = os.path.join(bot_data_dir, "Cursor_Session")

        _session_managers[cache_key] = CursorSessionManager(
            default_workspace=workspace,
            session_dir=session_dir
        )
        if session_dir:
            logger.info(f"[CursorSession] Created manager with session_dir: {session_dir}")

    return _session_managers[cache_key]


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """将消息列表转换为 prompt 字符串"""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            parts.append(f"[System Instructions]\n{content}\n")
        elif role == "user":
            parts.append(f"[User]\n{content}\n")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}\n")

    return "\n".join(parts)


class CursorCLIProvider(CustomLLM):
    """
    Cursor CLI LiteLLM 提供商

    模型格式: cursor-cli/{model_id}
    例如: cursor-cli/Auto, cursor-cli/gpt-4.1

    注意: 免费用户只能使用 Auto 和 gpt-4.1 模型
    """

    def __init__(self):
        super().__init__()
        self._cmd: Optional[str] = None

    def _get_cmd(self) -> str:
        """获取 Cursor Agent CLI 命令路径（延迟初始化）"""
        if self._cmd is None:
            self._cmd = _find_cursor_agent_cmd()
            logger.debug(f"[CursorCLI] Found agent command at: {self._cmd}")
        return self._cmd

    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        model_response: Optional[ModelResponse] = None,
        print_verbose: Optional[callable] = None,
        encoding: Optional[Any] = None,
        api_key: Optional[str] = None,
        logging_obj: Optional[Any] = None,
        optional_params: Optional[Dict] = None,
        acompletion: bool = False,
        litellm_params: Optional[Dict] = None,
        logger_fn: Optional[callable] = None,
        headers: Optional[Dict] = None,
        timeout: Optional[float] = None,
        client: Optional[Any] = None,
        **kwargs
    ) -> ModelResponse:
        """同步完成"""
        return self._completion_impl(
            model, messages, optional_params or {}, timeout or 300
        )

    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        model_response: Optional[ModelResponse] = None,
        print_verbose: Optional[callable] = None,
        encoding: Optional[Any] = None,
        api_key: Optional[str] = None,
        logging_obj: Optional[Any] = None,
        optional_params: Optional[Dict] = None,
        acompletion: bool = True,
        litellm_params: Optional[Dict] = None,
        logger_fn: Optional[callable] = None,
        headers: Optional[Dict] = None,
        timeout: Optional[float] = None,
        client: Optional[Any] = None,
        **kwargs
    ) -> ModelResponse:
        """异步完成（在线程池中运行同步代码）"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._completion_impl(
                model, messages, optional_params or {}, timeout or 300
            )
        )

    def _completion_impl(
        self,
        model: str,
        messages: List[Dict[str, str]],
        optional_params: Dict,
        timeout: float
    ) -> ModelResponse:
        """实际的完成实现"""
        start_time = time.time()

        # 解析模型 ID (cursor-cli/Auto -> Auto)
        model_id = model.split("/")[-1] if "/" in model else "Auto"
        logger.info(f"[CursorCLI] Starting completion for model={model_id}")

        # 获取可选参数
        workspace = optional_params.get("workspace")
        force = optional_params.get("force", True)  # 默认自动批准工具
        session_id = optional_params.get("session_id")  # Session ID
        continue_last = optional_params.get("continue_session", False)  # 继续上一个会话

        # 转换消息为 prompt
        prompt = _messages_to_prompt(messages)

        # 构建命令
        cmd = [
            self._get_cmd(),
            "-p",  # print mode (非交互式)
            "--model", model_id,
            "--output-format", "text",
        ]

        if force:
            cmd.append("-f")  # 强制允许工具执行

        if workspace:
            cmd.extend(["--workspace", workspace])

        # Session 控制
        if session_id:
            cmd.extend(["--resume", session_id])
            logger.info(f"[CursorCLI] Resuming session: {session_id}")
        elif continue_last:
            cmd.append("--continue")
            logger.info("[CursorCLI] Continuing last session")

        # 使用较小阈值，优先使用临时文件方式确保可靠传递
        MAX_CMD_LENGTH = 500
        import tempfile
        prompt_file_path = None

        try:
            if len(prompt) < MAX_CMD_LENGTH:
                # 短 prompt：直接作为命令行参数
                cmd.append(prompt)
                logger.debug(f"[CursorCLI] Command: {' '.join(cmd[:8])}... (prompt as arg, {len(prompt)} chars)")
                stdin_input = None
            else:
                # 长 prompt：写入临时文件，然后读取文件内容通过 stdin 传递
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.md',
                    encoding='utf-8',
                    delete=False
                ) as f:
                    f.write(prompt)
                    prompt_file_path = f.name

                # 使用 stdin 传递 prompt（不在命令行添加 prompt 参数）
                # Cursor CLI 会从 stdin 读取如果没有提供 prompt 参数
                stdin_input = prompt
                logger.info(f"[CursorCLI] Command: {' '.join(cmd[:8])}... (prompt via stdin, {len(prompt)} chars)")

            # 执行命令
            encoding = _get_subprocess_encoding()
            result = subprocess.run(
                cmd,
                input=stdin_input,
                capture_output=True,
                text=True,
                encoding=encoding,
                errors="replace",
                timeout=timeout
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                logger.error(f"[CursorCLI] Command failed: {error_msg}")
                raise RuntimeError(f"Cursor CLI error: {error_msg}")

            content = result.stdout.strip()

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Cursor CLI timed out after {timeout}s")
        finally:
            # 清理临时文件
            if prompt_file_path and os.path.exists(prompt_file_path):
                try:
                    os.unlink(prompt_file_path)
                except Exception:
                    pass

        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"[CursorCLI] Completed in {latency_ms:.0f}ms")

        # 构建响应
        response = ModelResponse(
            id=f"cursor-cli-{int(time.time())}",
            created=int(time.time()),
            model=f"cursor-cli/{model_id}",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content=content,
                        role="assistant"
                    )
                )
            ],
            usage=Usage(
                prompt_tokens=len(prompt.split()),  # 估算
                completion_tokens=len(content.split()),
                total_tokens=len(prompt.split()) + len(content.split())
            ),
            _response_ms=latency_ms
        )

        # 附加 session_id 到响应（如果使用了 session）
        if session_id:
            response._hidden_params = response._hidden_params or {}
            response._hidden_params["session_id"] = session_id

            # 更新 session 使用记录
            bot_data_dir = optional_params.get("bot_data_dir")
            if bot_data_dir:
                try:
                    manager = get_cursor_session_manager(
                        bot_data_dir=bot_data_dir,
                        workspace=workspace
                    )
                    manager.update_session_usage(session_id)
                except Exception as e:
                    logger.debug(f"[CursorCLI] Failed to update session usage: {e}")

        return response

    def streaming(self, *args, **kwargs) -> Iterator[str]:
        """同步流式（返回完整响应）"""
        response = self.completion(*args, **kwargs)
        yield response.choices[0].message.content

    async def astreaming(
        self,
        model: str,
        messages: List[Dict[str, str]],
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        model_response: Optional[ModelResponse] = None,
        print_verbose: Optional[callable] = None,
        encoding: Optional[Any] = None,
        api_key: Optional[str] = None,
        logging_obj: Optional[Any] = None,
        optional_params: Optional[Dict] = None,
        acompletion: bool = True,
        litellm_params: Optional[Dict] = None,
        logger_fn: Optional[callable] = None,
        headers: Optional[Dict] = None,
        timeout: Optional[float] = None,
        client: Optional[Any] = None,
        **kwargs
    ) -> AsyncIterator[Any]:
        """
        异步流式输出

        使用 --output-format stream-json --stream-partial-output 实现真正的流式
        """
        logger.info(f"[CursorCLI] Starting streaming for model={model}")

        # 解析模型 ID
        model_id = model.split("/")[-1] if "/" in model else "Auto"

        # 获取可选参数
        optional_params = optional_params or {}
        workspace = optional_params.get("workspace")
        force = optional_params.get("force", True)
        session_id = optional_params.get("session_id")
        continue_last = optional_params.get("continue_session", False)

        # 转换消息
        prompt = _messages_to_prompt(messages)
        prompt_tokens = len(prompt.split())
        completion_tokens = 0

        # 构建命令
        cmd = [
            self._get_cmd(),
            "-p",
            "--model", model_id,
            "--output-format", "stream-json",
            "--stream-partial-output",
        ]

        if force:
            cmd.append("-f")

        if workspace:
            cmd.extend(["--workspace", workspace])

        # Session 控制
        if session_id:
            cmd.extend(["--resume", session_id])
            logger.info(f"[CursorCLI] Streaming with session: {session_id}")
        elif continue_last:
            cmd.append("--continue")

        # 使用较小阈值，优先使用 stdin 方式确保可靠传递
        MAX_CMD_LENGTH = 500
        import tempfile
        prompt_file_path = None
        use_stdin = False

        if len(prompt) < MAX_CMD_LENGTH:
            # 短 prompt：直接作为命令行参数
            cmd.append(prompt)
            logger.debug(f"[CursorCLI] Streaming command: {' '.join(cmd[:8])}... (prompt as arg, {len(prompt)} chars)")
        else:
            # 长 prompt：通过 stdin 传递
            use_stdin = True
            # 同时写入临时文件作为备份（便于调试）
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.md',
                encoding='utf-8',
                delete=False
            ) as f:
                f.write(prompt)
                prompt_file_path = f.name
            logger.info(f"[CursorCLI] Streaming command: {' '.join(cmd[:8])}... (prompt via stdin, {len(prompt)} chars)")
            logger.debug(f"[CursorCLI] Streaming command: {' '.join(cmd[:8])}... (prompt in file, {len(prompt)} chars)")

        # 启动进程（确保 ripgrep 在 PATH 中）
        env = os.environ.copy()
        ripgrep_paths = [
            r"C:\ripgrep-15.1.0-x86_64-pc-windows-gnu",
            os.path.join(os.environ.get("USERPROFILE", ""), "scoop", "shims"),
            os.path.join(os.environ.get("ProgramFiles", ""), "ripgrep"),
        ]
        for rg_path in ripgrep_paths:
            if os.path.exists(rg_path) and rg_path not in env.get("PATH", ""):
                env["PATH"] = rg_path + os.pathsep + env.get("PATH", "")
                logger.debug(f"[CursorCLI] Added ripgrep to PATH: {rg_path}")
                break

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if use_stdin else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        try:
            # 如果使用 stdin，发送 prompt
            if use_stdin:
                process.stdin.write(prompt.encode('utf-8'))
                await process.stdin.drain()
                process.stdin.close()
                await process.stdin.wait_closed()

            # 同时读取 stdout 和 stderr
            async def read_stderr():
                stderr_lines = []
                async for line in process.stderr:
                    # 使用安全解码（Windows 兼容）
                    text = _safe_decode(line).strip()
                    if text:
                        stderr_lines.append(text)
                        logger.warning(f"[CursorCLI STDERR] {text}")
                return stderr_lines

            # 启动 stderr 读取任务
            stderr_task = asyncio.create_task(read_stderr())

            # 跟踪已输出的完整内容，用于计算增量
            full_content = ""
            response_session_id = session_id

            # 逐行读取流式输出
            async for line in process.stdout:
                # 使用安全解码（Windows 兼容）
                line_text = _safe_decode(line).strip()
                if not line_text:
                    continue

                logger.debug(f"[CursorCLI STDOUT] {line_text[:200]}")

                try:
                    # 尝试解析 JSON
                    data = json.loads(line_text)
                    text_chunk = ""

                    msg_type = data.get("type", "")

                    # 提取 session_id
                    if data.get("session_id"):
                        response_session_id = data.get("session_id")

                    if msg_type == "assistant":
                        # 解析 assistant 消息: {"type":"assistant","message":{"content":[{"type":"text","text":"..."}]}}
                        message = data.get("message", {})
                        content_list = message.get("content", [])
                        if content_list and isinstance(content_list, list):
                            for item in content_list:
                                if item.get("type") == "text":
                                    text_chunk = item.get("text", "")
                                    break
                    elif msg_type == "result":
                        # 最终结果: {"type":"result","result":"..."} - 跳过，避免重复
                        continue
                    elif msg_type in ("system", "user"):
                        # 跳过系统和用户消息
                        continue
                    else:
                        # 其他格式尝试
                        text_chunk = data.get("text", "") or data.get("content", "") or data.get("delta", "")

                except json.JSONDecodeError:
                    # 如果不是 JSON，直接作为文本
                    text_chunk = line_text

                # 计算增量内容（处理累积式输出）
                if text_chunk:
                    # 过滤 LLM thinking 内容（只检查开头，避免误过滤正常回复）
                    thinking_patterns = [
                        # English - 典型的 thinking 开头
                        "Reading and summarizing",
                        "I need to read the",
                        "I'll use the Read tool",
                        "Let me read the file",
                        "**Reading and",
                    ]
                    is_thinking = any(text_chunk.strip().startswith(pattern) for pattern in thinking_patterns)
                    if is_thinking:
                        logger.debug(f"[CursorCLI] Filtered thinking content: {text_chunk[:50]}...")
                        continue

                    if text_chunk.startswith(full_content):
                        # 累积式输出：新内容包含旧内容，只取增量部分
                        delta = text_chunk[len(full_content):]
                        full_content = text_chunk
                    elif full_content and text_chunk in full_content:
                        # 重复内容，跳过
                        continue
                    elif full_content and full_content in text_chunk:
                        # 新内容包含旧内容（可能是累积式的另一种形式）
                        delta = text_chunk[text_chunk.index(full_content) + len(full_content):]
                        full_content = text_chunk
                    else:
                        # 全新内容（或增量式输出）
                        delta = text_chunk
                        full_content += text_chunk

                    if delta:
                        completion_tokens += len(delta.split())
                        yield {
                            "text": delta,
                            "is_finished": False,
                            "finish_reason": None,
                            "usage": None,
                            "index": 0,
                            "tool_use": None,
                            "provider_specific_fields": {
                                "session_id": response_session_id,
                            } if response_session_id else None,
                        }

            await process.wait()
            await stderr_task  # 等待 stderr 读取完成

        except Exception as e:
            # 尝试终止进程，忽略进程已结束的错误
            try:
                process.kill()
            except ProcessLookupError:
                pass  # 进程已经结束
            # 清理临时文件
            if prompt_file_path and os.path.exists(prompt_file_path):
                try:
                    os.unlink(prompt_file_path)
                except Exception:
                    pass
            raise

        # 发送最终块
        usage_dict = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        yield {
            "text": "",
            "is_finished": True,
            "finish_reason": "stop",
            "usage": usage_dict,
            "index": 0,
            "tool_use": None,
            "provider_specific_fields": {
                "session_id": response_session_id,
            } if response_session_id else None,
        }
        logger.info(f"[CursorCLI] Streaming complete, ~{completion_tokens} tokens, session={response_session_id}")

        # 清理临时文件
        if prompt_file_path and os.path.exists(prompt_file_path):
            try:
                os.unlink(prompt_file_path)
            except Exception:
                pass


# 全局实例
_cursor_cli_provider: Optional[CursorCLIProvider] = None


def register_cursor_provider():
    """
    注册 Cursor CLI 提供商到 LiteLLM

    调用后可以使用：
    - litellm.completion(model="cursor-cli/Auto", ...)
    - litellm.completion(model="cursor-cli/gpt-4.1", ...)
    """
    global _cursor_cli_provider

    _cursor_cli_provider = CursorCLIProvider()

    # 获取现有的 provider map
    existing_map = getattr(litellm, "custom_provider_map", []) or []

    # 检查是否已注册
    for item in existing_map:
        if item.get("provider") == "cursor-cli":
            logger.info("[CursorCLI] Provider already registered")
            return

    # 添加 cursor-cli provider
    existing_map.append({
        "provider": "cursor-cli",
        "custom_handler": _cursor_cli_provider
    })
    litellm.custom_provider_map = existing_map

    logger.debug("Cursor CLI provider registered: cursor-cli")


def is_cursor_cli_available() -> bool:
    """检查 Cursor CLI 是否可用"""
    try:
        _find_cursor_agent_cmd()
        return True
    except RuntimeError:
        return False


__all__ = [
    "CursorCLIProvider",
    "CursorSessionManager",
    "register_cursor_provider",
    "is_cursor_cli_available",
    "get_cursor_session_manager",
]


# ============================================================
# 使用示例
# ============================================================

async def example_usage():
    """使用示例"""
    import litellm

    # 1. 注册 Cursor 提供商
    register_cursor_provider()

    # 2. 通过 LiteLLM 调用 Cursor CLI
    response = await litellm.acompletion(
        model="cursor-cli/Auto",
        messages=[{"role": "user", "content": "What is 2+2?"}]
    )
    print(f"Cursor: {response.choices[0].message.content}")

    # 3. 指定工作目录
    response = await litellm.acompletion(
        model="cursor-cli/Auto",
        messages=[{"role": "user", "content": "读取 README.md"}],
        workspace=r"D:\vscode\openmind\SoulBot"
    )
    print(f"Cursor (with workspace): {response.choices[0].message.content}")

    # 4. 使用 Session（多轮对话）
    # bot_data_dir: Bot 的数据目录，session 映射存储到 {bot_data_dir}/Cursor_Session/
    bot_data_dir = r"D:\vscode\openmind\SoulBot\examples\test_agent\data"
    workspace = r"D:\vscode\openmind\SoulBot"

    session_manager = get_cursor_session_manager(
        bot_data_dir=bot_data_dir,  # Session 映射存储目录
        workspace=workspace          # 默认工作目录
    )

    # 创建新会话（可指定名称）
    session_id = await session_manager.create_session(name="SoulBot开发会话")
    print(f"Created session: {session_id}")
    # Session 映射保存到: {bot_data_dir}/Cursor_Session/sessions.json

    # 第一轮对话
    response = await litellm.acompletion(
        model="cursor-cli/Auto",
        messages=[{"role": "user", "content": "我叫小明，请记住我的名字"}],
        session_id=session_id,
        workspace=workspace,
        bot_data_dir=bot_data_dir  # 用于更新 session 使用记录
    )
    print(f"Response 1: {response.choices[0].message.content}")

    # 第二轮对话（使用同一个 session）
    response = await litellm.acompletion(
        model="cursor-cli/Auto",
        messages=[{"role": "user", "content": "我叫什么名字？"}],
        session_id=session_id,
        workspace=workspace,
        bot_data_dir=bot_data_dir
    )
    print(f"Response 2: {response.choices[0].message.content}")

    # 查看所有 session
    all_sessions = session_manager.get_all_sessions()
    print(f"All sessions: {all_sessions}")

    # 获取最近使用的 session
    recent_id = session_manager.get_recent_session()
    print(f"Recent session: {recent_id}")


if __name__ == "__main__":
    asyncio.run(example_usage())
