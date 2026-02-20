"""
Cursor CLI Provider — Standalone Functions (no litellm)

Provides cursor_query() and cursor_query_stream() for direct Cursor Agent CLI
integration, plus CursorSessionManager for session persistence.

Usage:
    from src.llm_service.cursor_cli_provider import cursor_query, cursor_query_stream

    content, session_id = await cursor_query("Hello", model_id="Auto")

    async for delta, sid in cursor_query_stream("Hello", model_id="Auto"):
        print(delta, end="")
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
import tempfile
from typing import Optional, List, Dict, Any, AsyncIterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoding utilities (Windows-compatible)
# ---------------------------------------------------------------------------
def _get_subprocess_encoding() -> str:
    """Cursor CLI always outputs UTF-8."""
    return "utf-8"


def _get_system_encoding() -> str:
    """System encoding (for error messages on Windows)."""
    if sys.platform == "win32":
        import locale
        return locale.getpreferredencoding(False) or "utf-8"
    return "utf-8"


def _safe_decode(data: bytes, encoding: str = None) -> str:
    """Safely decode bytes, preferring UTF-8."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            sys_encoding = _get_system_encoding()
            return data.decode(sys_encoding)
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# CLI discovery
# ---------------------------------------------------------------------------
def _find_cursor_agent_cmd() -> str:
    """Find the Cursor Agent CLI command path."""
    cmd_names = ["cursor-agent", "agent"]
    for cmd_name in cmd_names:
        cmd = shutil.which(cmd_name)
        if cmd:
            return cmd

    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
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


def is_cursor_cli_available() -> bool:
    """Check if Cursor CLI is available."""
    try:
        _find_cursor_agent_cmd()
        return True
    except RuntimeError:
        return False


# ---------------------------------------------------------------------------
# CursorSessionManager (unchanged)
# ---------------------------------------------------------------------------
class CursorSessionManager:
    """
    Cursor CLI Session manager.

    Manages Cursor CLI chat sessions with local session mapping
    stored in {session_dir}/sessions.json.
    """

    def __init__(
        self,
        default_workspace: Optional[str] = None,
        session_dir: Optional[str] = None
    ):
        self._cmd: Optional[str] = None
        self.default_workspace = default_workspace
        self.session_dir = session_dir
        self._sessions_file: Optional[str] = None

        if session_dir:
            os.makedirs(session_dir, exist_ok=True)
            self._sessions_file = os.path.join(session_dir, "sessions.json")
            logger.info(f"[CursorSession] Session dir: {session_dir}")

    def _get_cmd(self) -> str:
        if self._cmd is None:
            self._cmd = _find_cursor_agent_cmd()
        return self._cmd

    def _load_sessions(self) -> Dict[str, Any]:
        if not self._sessions_file:
            return {}
        if not os.path.exists(self._sessions_file):
            return {}
        try:
            with open(self._sessions_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[CursorSession] Failed to load sessions: {e}")
            return {}

    def _save_sessions(self, sessions: Dict[str, Any]) -> None:
        if not self._sessions_file:
            return
        try:
            with open(self._sessions_file, "w", encoding="utf-8") as f:
                json.dump(sessions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[CursorSession] Failed to save sessions: {e}")

    async def create_session(self, name: Optional[str] = None) -> str:
        loop = asyncio.get_event_loop()
        session_id = await loop.run_in_executor(None, self._create_session_sync)

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
        cmd = [self._get_cmd(), "create-chat"]
        logger.info("[CursorSession] Creating session...")
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", timeout=30
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create session: {result.stderr}")
        output = result.stdout.strip()
        logger.info(f"[CursorSession] Created: {output}")
        return output

    def update_session_usage(self, session_id: str) -> None:
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
        return self._load_sessions().get(session_id)

    def get_all_sessions(self) -> Dict[str, Any]:
        return self._load_sessions()

    def get_recent_session(self) -> Optional[str]:
        sessions = self._load_sessions()
        if not sessions:
            return None
        sorted_sessions = sorted(
            sessions.items(),
            key=lambda x: x[1].get("last_used", ""),
            reverse=True
        )
        return sorted_sessions[0][0] if sorted_sessions else None

    async def list_sessions(self) -> List[Dict[str, Any]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._list_sessions_sync)

    def _list_sessions_sync(self) -> List[Dict[str, Any]]:
        cmd = [self._get_cmd(), "ls"]
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", timeout=30
        )
        if result.returncode != 0:
            logger.warning(f"[CursorSession] Failed to list sessions: {result.stderr}")
            return []

        sessions = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 2)
            if parts:
                sessions.append({
                    "id": parts[0],
                    "summary": parts[1] if len(parts) > 1 else "",
                    "updated": parts[2] if len(parts) > 2 else "",
                })
        return sessions

    async def get_last_session_id(self) -> Optional[str]:
        sessions = await self.list_sessions()
        if sessions:
            return sessions[0].get("id")
        return None


# Global session managers (cached by bot_data_dir)
_session_managers: Dict[str, CursorSessionManager] = {}


def get_cursor_session_manager(
    bot_data_dir: Optional[str] = None,
    workspace: Optional[str] = None
) -> CursorSessionManager:
    """Get or create a CursorSessionManager (cached by bot_data_dir)."""
    global _session_managers
    cache_key = bot_data_dir or "_default_"

    if cache_key not in _session_managers:
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


# ---------------------------------------------------------------------------
# Standalone query functions (replace CursorCLIProvider)
# ---------------------------------------------------------------------------
def _build_cursor_cmd(
    model_id: str = "Auto",
    session_id: Optional[str] = None,
    continue_last: bool = False,
    workspace: Optional[str] = None,
    force: bool = True,
    output_format: str = "text",
    stream_partial: bool = False,
) -> List[str]:
    """Build the Cursor Agent CLI command."""
    cmd = [
        _find_cursor_agent_cmd(),
        "-p",
        "--model", model_id,
        "--output-format", output_format,
    ]

    if stream_partial:
        cmd.append("--stream-partial-output")

    if force:
        cmd.append("-f")

    if workspace:
        cmd.extend(["--workspace", workspace])

    if session_id:
        cmd.extend(["--resume", session_id])
    elif continue_last:
        cmd.append("--continue")

    return cmd


def _get_env_with_ripgrep() -> dict:
    """Get environment with ripgrep on PATH (Windows compatibility)."""
    env = os.environ.copy()
    ripgrep_paths = [
        os.path.join(os.environ.get("ProgramFiles", ""), "ripgrep-15.1.0-x86_64-pc-windows-gnu"),
        os.path.join(os.environ.get("USERPROFILE", ""), "scoop", "shims"),
        os.path.join(os.environ.get("ProgramFiles", ""), "ripgrep"),
    ]
    for rg_path in ripgrep_paths:
        if os.path.exists(rg_path) and rg_path not in env.get("PATH", ""):
            env["PATH"] = rg_path + os.pathsep + env.get("PATH", "")
            break
    return env


async def cursor_query(
    prompt: str,
    model_id: str = "Auto",
    session_id: Optional[str] = None,
    continue_last: bool = False,
    workspace: Optional[str] = None,
    timeout: float = 600,
    bot_data_dir: Optional[str] = None,
) -> tuple:
    """
    Non-streaming Cursor query. Returns (content, session_id).
    """
    start_time = time.time()
    logger.info(f"[CursorCLI] Starting query for model={model_id}")

    cmd = _build_cursor_cmd(
        model_id=model_id,
        session_id=session_id,
        continue_last=continue_last,
        workspace=workspace,
        output_format="text",
    )

    MAX_CMD_LENGTH = 500
    prompt_file_path = None
    stdin_input = None

    try:
        if len(prompt) < MAX_CMD_LENGTH:
            cmd.append(prompt)
        else:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.md', encoding='utf-8', delete=False
            ) as f:
                f.write(prompt)
                prompt_file_path = f.name
            stdin_input = prompt

        encoding = _get_subprocess_encoding()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                input=stdin_input,
                capture_output=True,
                text=True,
                encoding=encoding,
                errors="replace",
                timeout=timeout,
            )
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise RuntimeError(f"Cursor CLI error: {error_msg}")

        content = result.stdout.strip()

    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Cursor CLI timed out after {timeout}s")
    finally:
        if prompt_file_path and os.path.exists(prompt_file_path):
            try:
                os.unlink(prompt_file_path)
            except Exception:
                pass

    latency_ms = (time.time() - start_time) * 1000
    logger.info(f"[CursorCLI] Completed in {latency_ms:.0f}ms")

    # Update session usage
    if session_id and bot_data_dir:
        try:
            manager = get_cursor_session_manager(
                bot_data_dir=bot_data_dir, workspace=workspace
            )
            manager.update_session_usage(session_id)
        except Exception:
            pass

    return content, session_id


async def cursor_query_stream(
    prompt: str,
    model_id: str = "Auto",
    session_id: Optional[str] = None,
    continue_last: bool = False,
    workspace: Optional[str] = None,
) -> AsyncIterator[tuple]:
    """
    Streaming Cursor query. Yields (text_delta, session_id).

    Uses --output-format stream-json --stream-partial-output for real streaming.
    """
    logger.info(f"[CursorCLI] Starting streaming for model={model_id}")

    cmd = _build_cursor_cmd(
        model_id=model_id,
        session_id=session_id,
        continue_last=continue_last,
        workspace=workspace,
        output_format="stream-json",
        stream_partial=True,
    )

    MAX_CMD_LENGTH = 500
    prompt_file_path = None
    use_stdin = False

    if len(prompt) < MAX_CMD_LENGTH:
        cmd.append(prompt)
    else:
        use_stdin = True
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', encoding='utf-8', delete=False
        ) as f:
            f.write(prompt)
            prompt_file_path = f.name

    env = _get_env_with_ripgrep()

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if use_stdin else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    try:
        if use_stdin:
            process.stdin.write(prompt.encode('utf-8'))
            await process.stdin.drain()
            process.stdin.close()
            await process.stdin.wait_closed()

        async def read_stderr():
            stderr_lines = []
            async for line in process.stderr:
                text = _safe_decode(line).strip()
                if text:
                    stderr_lines.append(text)
                    logger.warning(f"[CursorCLI STDERR] {text}")
            return stderr_lines

        stderr_task = asyncio.create_task(read_stderr())

        full_content = ""
        response_session_id = session_id

        # Thinking patterns to filter
        thinking_patterns = [
            "Reading and summarizing",
            "I need to read the",
            "I'll use the Read tool",
            "Let me read the file",
            "**Reading and",
        ]

        async for line in process.stdout:
            line_text = _safe_decode(line).strip()
            if not line_text:
                continue

            try:
                data = json.loads(line_text)
                text_chunk = ""
                msg_type = data.get("type", "")

                if data.get("session_id"):
                    response_session_id = data.get("session_id")

                if msg_type == "assistant":
                    message = data.get("message", {})
                    content_list = message.get("content", [])
                    if content_list and isinstance(content_list, list):
                        for item in content_list:
                            if item.get("type") == "text":
                                text_chunk = item.get("text", "")
                                break
                elif msg_type in ("result", "system", "user"):
                    continue
                else:
                    text_chunk = (
                        data.get("text", "") or
                        data.get("content", "") or
                        data.get("delta", "")
                    )

            except json.JSONDecodeError:
                text_chunk = line_text

            if text_chunk:
                is_thinking = any(
                    text_chunk.strip().startswith(p) for p in thinking_patterns
                )
                if is_thinking:
                    continue

                # Compute delta (handle cumulative output)
                if text_chunk.startswith(full_content):
                    delta = text_chunk[len(full_content):]
                    full_content = text_chunk
                elif full_content and text_chunk in full_content:
                    continue
                elif full_content and full_content in text_chunk:
                    delta = text_chunk[text_chunk.index(full_content) + len(full_content):]
                    full_content = text_chunk
                else:
                    delta = text_chunk
                    full_content += text_chunk

                if delta:
                    yield delta, response_session_id

        await process.wait()
        await stderr_task

    except Exception:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        raise

    finally:
        if prompt_file_path and os.path.exists(prompt_file_path):
            try:
                os.unlink(prompt_file_path)
            except Exception:
                pass

    logger.info(f"[CursorCLI] Streaming complete, session={response_session_id}")


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
__all__ = [
    "CursorSessionManager",
    "get_cursor_session_manager",
    "is_cursor_cli_available",
    "cursor_query",
    "cursor_query_stream",
]
