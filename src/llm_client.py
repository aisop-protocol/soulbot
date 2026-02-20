"""
AI Client — Direct ACP/CLI Pool Access (no litellm)

Routes prompts to Claude, Gemini, OpenCode, or Cursor via subprocess-based
ACP connections. Supports session persistence and cross-provider fallback.
"""

import os
import shutil
import logging
from typing import Optional, Dict, List

from src.config import (
    CLAUDE_CLI, GEMINI_CLI, OPENCODE_CLI,
    CLAUDE_MODEL, GEMINI_MODEL, OPENCODE_MODEL,
    OPENCODE_MODEL_OVERRIDE, SYSTEM_PROMPT, ENABLE_FALLBACK,
    WORKSPACE_DIR, AUTO_APPROVE_PERMISSIONS, SHOW_THOUGHTS,
    logger,
)
from src.llm_service.cli_agent_service import (
    ACPConnectionPool,
    CLIConfig,
    CLIProvider,
    find_claude_acp_binary,
)
from src.llm_service.opencode_acp_client import (
    OpenCodeConnectionPool,
    OpenCodeConfig,
    get_opencode_pool,
)

# ---------------------------------------------------------------------------
# Pool caches (migrated from litellm_acp_provider.py)
# ---------------------------------------------------------------------------
_claude_pools: Dict[str, ACPConnectionPool] = {}
_gemini_pools: Dict[str, ACPConnectionPool] = {}


def _get_claude_pool(bot_data_dir: Optional[str] = None) -> ACPConnectionPool:
    """Get or create a Claude connection pool (cached by bot_data_dir)."""
    global _claude_pools
    cache_key = bot_data_dir or "_default_"

    session_dir = None
    if bot_data_dir:
        session_dir = os.path.join(bot_data_dir, "Claude_Session")

    if cache_key not in _claude_pools:
        cmd = find_claude_acp_binary()
        if not cmd:
            raise RuntimeError(
                "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            )

        config = CLIConfig(
            provider=CLIProvider.CLAUDE,
            acp_cmd=cmd,
            cwd=WORKSPACE_DIR,
            auto_approve_permissions=AUTO_APPROVE_PERMISSIONS,
            show_thoughts=SHOW_THOUGHTS,
            pool_size=10,
            pool_idle_timeout=43200,
            session_dir=session_dir,
        )
        _claude_pools[cache_key] = ACPConnectionPool(config)
        if session_dir:
            logger.info(f"[ClaudePool] Created pool with session_dir: {session_dir}")

    return _claude_pools[cache_key]


def _get_gemini_pool(bot_data_dir: Optional[str] = None) -> ACPConnectionPool:
    """Get or create a Gemini connection pool (cached by bot_data_dir)."""
    global _gemini_pools
    cache_key = bot_data_dir or "_default_"

    session_dir = None
    if bot_data_dir:
        session_dir = os.path.join(bot_data_dir, "Gemini_Session")

    if cache_key not in _gemini_pools:
        cmd = shutil.which("gemini")
        if not cmd:
            local_cmd = os.path.join(os.getcwd(), "node_modules", ".bin", "gemini.cmd")
            if os.path.exists(local_cmd):
                cmd = local_cmd
            else:
                raise RuntimeError(
                    "gemini not found. Install with: npm install -g @google/gemini-cli"
                )

        model_id = GEMINI_MODEL.split("/")[-1] if "/" in GEMINI_MODEL else GEMINI_MODEL

        config = CLIConfig(
            provider=CLIProvider.GEMINI,
            acp_cmd=cmd,
            model=model_id,
            cwd=WORKSPACE_DIR,
            auto_approve_permissions=AUTO_APPROVE_PERMISSIONS,
            show_thoughts=SHOW_THOUGHTS,
            pool_size=10,
            pool_idle_timeout=43200,
            session_dir=session_dir,
        )
        _gemini_pools[cache_key] = ACPConnectionPool(config)
        if session_dir:
            logger.info(f"[GeminiPool] Created pool with session_dir: {session_dir}")

    return _gemini_pools[cache_key]


# ---------------------------------------------------------------------------
# Message formatting (FULL ACCESS version, migrated from litellm_acp_provider)
# ---------------------------------------------------------------------------
def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert a list of chat messages to a single prompt string."""
    parts = []

    has_system = any(msg.get("role") == "system" for msg in messages)
    if not has_system:
        parts.append(
            "[System Instructions]\n"
            "You have FULL ACCESS to the local system. You can:\n"
            "1. Read/Write ANY file.\n"
            "2. Execute ANY command in the terminal.\n"
            "3. Manage project structure.\n\n"
            "Permissions are ALREADY GRANTED. Do not ask for confirmation. Just execute.\n"
        )

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            parts.append(
                f"[System Instructions]\n{content}\n"
                "(Note: You have FULL PERMISSION to access files and run commands. Execute directly.)\n"
            )
        elif role == "user":
            parts.append(f"[User]\n{content}\n")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}\n")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Provider resolution & fallback
# ---------------------------------------------------------------------------
def _get_fallback_model(current_model: str) -> Optional[str]:
    """Return the cross-provider fallback model, or None."""
    if current_model.startswith("opencode-acp/"):
        return CLAUDE_MODEL
    elif current_model.startswith("claude-acp/"):
        return GEMINI_MODEL
    elif current_model.startswith("gemini-acp/"):
        return CLAUDE_MODEL
    return None


def _extract_model_id(model_name: str, provider: str) -> Optional[str]:
    """Extract the model ID to pass to session/set_model."""
    if provider == "opencode":
        if not OPENCODE_MODEL_OVERRIDE:
            return None
        model_id = model_name.replace("opencode-acp/", "", 1) if model_name.startswith("opencode-acp/") else model_name
        if "/" not in model_id:
            model_id = f"opencode/{model_id}"
        return model_id
    elif provider == "claude":
        return model_name.split("/")[-1] if "/" in model_name else "sonnet"
    elif provider == "gemini":
        return model_name.split("/")[-1] if "/" in model_name else "gemini-2.5-flash"
    return None


async def _query_pool(pool, prompt: str, session_id: Optional[str], model_id: Optional[str],
                      user_id: Optional[str] = None, is_opencode: bool = False):
    """Non-streaming query via a pool. Returns (content, session_id)."""
    acquire_kwargs = {"session_id": session_id}
    if is_opencode:
        acquire_kwargs["user_id"] = user_id

    async with pool.acquire(**acquire_kwargs) as (client, current_session_id):
        if model_id and current_session_id:
            try:
                await client._send_request("session/set_model", {
                    "sessionId": current_session_id,
                    "modelId": model_id,
                })
            except Exception:
                pass

        content = await client.query(prompt)
        return content, current_session_id


async def _query_pool_stream(pool, prompt: str, session_id: Optional[str], model_id: Optional[str],
                             user_id: Optional[str] = None, is_opencode: bool = False):
    """Streaming query via a pool. Yields (text_chunk, session_id)."""
    acquire_kwargs = {"session_id": session_id}
    if is_opencode:
        acquire_kwargs["user_id"] = user_id

    async with pool.acquire(**acquire_kwargs) as (client, current_session_id):
        if model_id and current_session_id:
            try:
                await client._send_request("session/set_model", {
                    "sessionId": current_session_id,
                    "modelId": model_id,
                })
            except Exception:
                pass

        async for text_chunk in client.query_stream(prompt):
            if text_chunk:
                yield text_chunk, current_session_id


def _get_pool_for_model(model_name: str):
    """Return (provider_key, pool, is_opencode) for a model name."""
    if model_name.startswith("opencode-acp/"):
        return "opencode", get_opencode_pool(), True
    elif model_name.startswith("claude-acp/"):
        return "claude", _get_claude_pool(), False
    elif model_name.startswith("gemini-acp/"):
        return "gemini", _get_gemini_pool(), False
    return "echo", None, False


# ---------------------------------------------------------------------------
# AIClient
# ---------------------------------------------------------------------------
class AIClient:
    def __init__(self):
        self._setup_provider()

    def _setup_provider(self):
        # Priority: OpenCode > Claude > Gemini
        if OPENCODE_CLI:
            self.model_name = OPENCODE_MODEL
        elif CLAUDE_CLI:
            self.model_name = CLAUDE_MODEL
        elif GEMINI_CLI:
            self.model_name = GEMINI_MODEL
        else:
            self.model_name = "echo"
            logger.warning("No CLI provider enabled. Defaulting to Echo mode.")

        logger.info(f"AI Client configured with model: {self.model_name}")

    async def get_response(self, user_input: str, session_id: str = None,
                           system_prompt: str = None) -> tuple:
        """
        Non-streaming response. Returns (content, session_id).
        """
        if self.model_name == "echo":
            return f"[Echo] {user_input}", session_id

        messages = [{"role": "user", "content": user_input}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        elif SYSTEM_PROMPT:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        prompt = _messages_to_prompt(messages)

        try:
            provider, pool, is_opencode = _get_pool_for_model(self.model_name)
            model_id = _extract_model_id(self.model_name, provider)
            logger.info(f"Sending prompt to {self.model_name} (session_id: {session_id})...")

            content, new_session_id = await _query_pool(
                pool, prompt, session_id, model_id, is_opencode=is_opencode,
            )
            return content, new_session_id or session_id

        except Exception as e:
            logger.warning(f"Error calling LLM ({self.model_name}): {e}")

            if ENABLE_FALLBACK:
                fallback_model = _get_fallback_model(self.model_name)
                if fallback_model and fallback_model != self.model_name:
                    logger.warning(f"[Fallback] Switching to {fallback_model} due to error: {e}")
                    try:
                        fb_provider, fb_pool, fb_is_oc = _get_pool_for_model(fallback_model)
                        fb_model_id = _extract_model_id(fallback_model, fb_provider)
                        fb_content, _ = await _query_pool(
                            fb_pool, prompt, session_id, fb_model_id, is_opencode=fb_is_oc,
                        )
                        return (
                            f"\u26a0\ufe0f *[{self.model_name} unavailable, switched to {fallback_model}]*\n\n{fb_content}",
                            session_id,
                        )
                    except Exception as fallback_error:
                        error_msg = (
                            f"\u26a0\ufe0f **Service Unavailable (All Models Failed)**\n\n"
                            f"**Primary ({self.model_name})**: {e}\n"
                            f"**Fallback ({fallback_model})**: {fallback_error}\n\n"
                            f"Please check quotas or configuration."
                        )
                        logger.error(error_msg)
                        return error_msg, session_id

            return f"Error interacting with AI: {e}", session_id

    async def get_streaming_response(self, user_input: str, session_id: str = None,
                                     system_prompt: str = None, user_id: str = None):
        """
        Streaming response. Yields (content_chunk, session_id).
        """
        messages = [{"role": "user", "content": user_input}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        elif SYSTEM_PROMPT:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        if self.model_name == "echo":
            yield f"[Echo] {user_input}", session_id
            return

        prompt = _messages_to_prompt(messages)

        try:
            provider, pool, is_opencode = _get_pool_for_model(self.model_name)
            model_id = _extract_model_id(self.model_name, provider)
            logger.info(
                f"Sending streaming prompt to {self.model_name} "
                f"(session_id: {session_id}, user_id: {user_id})..."
            )

            async for text_chunk, new_session_id in _query_pool_stream(
                pool, prompt, session_id, model_id,
                user_id=user_id, is_opencode=is_opencode,
            ):
                yield text_chunk, new_session_id

            logger.info("Streaming finished.")

        except Exception as e:
            logger.warning(f"Error calling streaming LLM ({self.model_name}): {e}")

            fallback_model = None
            if ENABLE_FALLBACK:
                fallback_model = _get_fallback_model(self.model_name)
                if fallback_model and fallback_model == self.model_name:
                    fallback_model = None

            if fallback_model:
                logger.warning(f"[Fallback] Switching stream to {fallback_model} due to error: {e}")
                yield (
                    f"\u26a0\ufe0f *[{self.model_name} unavailable, switched to {fallback_model}]*\n\n",
                    None,
                )
                try:
                    fb_provider, fb_pool, fb_is_oc = _get_pool_for_model(fallback_model)
                    fb_model_id = _extract_model_id(fallback_model, fb_provider)
                    async for text_chunk, _ in _query_pool_stream(
                        fb_pool, prompt, session_id, fb_model_id,
                        user_id=user_id, is_opencode=fb_is_oc,
                    ):
                        yield text_chunk, None
                except Exception as fallback_error:
                    error_msg = (
                        f"\u26a0\ufe0f **Service Unavailable (All Models Failed)**\n\n"
                        f"**Primary ({self.model_name})**: {e}\n"
                        f"**Fallback ({fallback_model})**: {fallback_error}\n\n"
                        f"Please check quotas or configuration."
                    )
                    logger.error(error_msg)
                    yield error_msg, None
            else:
                yield f"Error interacting with AI: {e}", None
