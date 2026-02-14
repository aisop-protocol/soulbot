"""
LiteLLM ACP Provider

将 Claude/Gemini CLI 的 ACP 模式集成到 LiteLLM。

使用方式：
    import litellm
    from soulbot.core.llm_service.litellm_acp_provider import register_acp_providers

    # 注册 ACP 提供商
    register_acp_providers()

    # 通过 LiteLLM 调用
    response = await litellm.acompletion(
        model="claude-acp/sonnet",  # 或 "gemini-acp/default"
        messages=[{"role": "user", "content": "Hello"}]
    )
"""

import asyncio
import os
import time
import logging
import shutil
from typing import Optional, List, Dict, Any, Iterator, AsyncIterator, Union

import litellm
from litellm import CustomLLM, ModelResponse, Message
from litellm.types.utils import Choices, Usage

from .cli_agent_service import (
    CLIAgentService,
    CLIProvider,
    CLIConfig,
    CLIMessage,
    ACPConnectionPool,
    ClaudeACPClient,
    GeminiACPClient,
    find_claude_acp_binary
)

logger = logging.getLogger(__name__)


# 全局连接池（按 bot_data_dir 缓存）
_claude_pools: Dict[str, ACPConnectionPool] = {}
_gemini_pools: Dict[str, ACPConnectionPool] = {}



def _get_claude_pool(bot_data_dir: Optional[str] = None) -> ACPConnectionPool:
    """
    获取 Claude 连接池（按 bot_data_dir 缓存）

    Args:
        bot_data_dir: Bot 的 data 目录，session 会存储到 {bot_data_dir}/Claude_Session/
    """
    global _claude_pools
    cache_key = bot_data_dir or "_default_"

    # 计算实际的 session_dir
    session_dir = None
    if bot_data_dir:
        session_dir = os.path.join(bot_data_dir, "Claude_Session")

    if cache_key not in _claude_pools:
        # 使用共享检测+自动安装函数
        cmd = find_claude_acp_binary()
        if not cmd:
            raise RuntimeError(
                "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            )

        from src.config import WORKSPACE_DIR, AUTO_APPROVE_PERMISSIONS, SHOW_THOUGHTS
        config = CLIConfig(
            provider=CLIProvider.CLAUDE,
            acp_cmd=cmd,
            cwd=WORKSPACE_DIR,
            auto_approve_permissions=AUTO_APPROVE_PERMISSIONS,
            show_thoughts=SHOW_THOUGHTS,
            pool_size=10,
            pool_idle_timeout=43200,  # 12小时
            session_dir=session_dir  # Phase 83: 自定义 session 目录
        )
        _claude_pools[cache_key] = ACPConnectionPool(config)
        if session_dir:
            logger.info(f"[ClaudeACPProvider] Created pool with session_dir: {session_dir}")

    return _claude_pools[cache_key]


def _get_gemini_pool(bot_data_dir: Optional[str] = None) -> ACPConnectionPool:
    """
    获取 Gemini 连接池（按 bot_data_dir 缓存）

    Args:
        bot_data_dir: Bot 的 data 目录，session 会存储到 {bot_data_dir}/Gemini_Session/
    """
    global _gemini_pools
    cache_key = bot_data_dir or "_default_"

    # 计算实际的 session_dir
    session_dir = None
    if bot_data_dir:
        session_dir = os.path.join(bot_data_dir, "Gemini_Session")

    if cache_key not in _gemini_pools:
        from src.config import GEMINI_MODEL, WORKSPACE_DIR, AUTO_APPROVE_PERMISSIONS, SHOW_THOUGHTS
        cmd = shutil.which("gemini")
        if not cmd:
            local_cmd = os.path.join(os.getcwd(), "node_modules", ".bin", "gemini.cmd")
            if os.path.exists(local_cmd):
                cmd = local_cmd
            else:
                raise RuntimeError("gemini not found. Install with: npm install -g @google/gemini-cli")

        # 提取纯模型ID (去除 gemini-acp/ 前缀 if any)
        model_id = GEMINI_MODEL.split("/")[-1] if "/" in GEMINI_MODEL else GEMINI_MODEL

        config = CLIConfig(
            provider=CLIProvider.GEMINI,
            acp_cmd=cmd,
            model=model_id,
            cwd=WORKSPACE_DIR,
            auto_approve_permissions=AUTO_APPROVE_PERMISSIONS,
            show_thoughts=SHOW_THOUGHTS,
            pool_size=10,
            pool_idle_timeout=43200,  # 12小时
            session_dir=session_dir  # Phase 83: 自定义 session 目录
        )
        _gemini_pools[cache_key] = ACPConnectionPool(config)
        if session_dir:
            logger.info(f"[GeminiACPProvider] Created pool with session_dir: {session_dir}")

    return _gemini_pools[cache_key]


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """将消息列表转换为 prompt 字符串"""
    parts = []
    
    # 检测是否已有 system prompt
    has_system = any(msg.get("role") == "system" for msg in messages)
    if not has_system:
        # 默认注入能力描述
        parts.append("[System Instructions]\nYou have FULL ACCESS to the local system. You can:\n1. Read/Write ANY file.\n2. Execute ANY command in the terminal.\n3. Manage project structure.\n\nPermissions are ALREADY GRANTED. Do not ask for confirmation. Just execute.\n")

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            parts.append(f"[System Instructions]\n{content}\n(Note: You have FULL PERMISSION to access files and run commands. Execute directly.)\n")
        elif role == "user":
            parts.append(f"[User]\n{content}\n")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}\n")

    return "\n".join(parts)


class ClaudeACPProvider(CustomLLM):
    """
    Claude ACP LiteLLM 提供商

    模型格式: claude-acp/{model_id}
    例如: claude-acp/sonnet, claude-acp/opus, claude-acp/haiku
    """

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
        """同步完成（内部使用异步）"""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果已有事件循环，创建新线程
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._acompletion_impl(model, messages, optional_params or {})
                )
                return future.result(timeout=timeout or 300)
        else:
            return loop.run_until_complete(
                self._acompletion_impl(model, messages, optional_params or {})
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
        """异步完成"""
        return await self._acompletion_impl(model, messages, optional_params or {})

    async def _acompletion_impl(
        self,
        model: str,
        messages: List[Dict[str, str]],
        optional_params: Dict
    ) -> ModelResponse:
        """实际的异步完成实现"""
        start_time = time.time()
        logger.debug(f"[ClaudeACP] Starting completion for model={model}")

        # 解析模型 ID (claude-acp/sonnet -> sonnet)
        model_id = model.split("/")[-1] if "/" in model else "sonnet"

        # 从 extra_body 获取自定义参数（litellm 传递给 custom provider 的方式）
        if not isinstance(optional_params, dict):
             optional_params = {}
        extra_body = optional_params.get("extra_body") or {}
        if isinstance(extra_body, str):
            import json as _json
            try:
                extra_body = _json.loads(extra_body)
            except Exception:
                extra_body = {}

        session_id = (
            extra_body.get("session_id") or
            optional_params.get("session_id")
        )
        bot_data_dir = extra_body.get("bot_data_dir") or optional_params.get("bot_data_dir")
        logger.debug(f"[ClaudeACP] session_id={session_id}, bot_data_dir={bot_data_dir}")

        # 获取连接池（按 bot_data_dir 缓存，自动创建 Claude_Session 子目录）
        try:
            pool = _get_claude_pool(bot_data_dir=bot_data_dir)
        except Exception as e:
            logger.error(f"[ClaudeACP] Failed to get pool: {e}")
            raise

        # 转换消息
        prompt = _messages_to_prompt(messages)
        logger.debug(f"[ClaudeACP] Prompt length: {len(prompt)} chars")

        # 执行查询 (Phase 83: 传递 session_id)
        async with pool.acquire(session_id=session_id) as (client, current_session_id):
            logger.debug(f"[ClaudeACP] Client acquired, session_id={current_session_id}")
            # 设置模型
            # 设置模型
            if model_id and current_session_id:
                try:
                    await client._send_request("session/set_model", {
                        "sessionId": current_session_id,
                        "modelId": model_id
                    })
                except Exception:
                    pass  # 模型切换失败不阻塞

            content = await client.query(prompt)
            # 保存 session_id 供后续使用
            final_session_id = current_session_id

        latency_ms = (time.time() - start_time) * 1000

        # 构建响应 (Phase 83: 包含 session_id)
        response = ModelResponse(
            id=f"claude-acp-{int(time.time())}",
            created=int(time.time()),
            model=f"claude-acp/{model_id}",
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

        # Phase 83: 附加 session_id 到响应
        response._hidden_params = response._hidden_params or {}
        response._hidden_params["session_id"] = final_session_id

        return response

    def streaming(self, *args, **kwargs) -> Iterator[str]:
        """同步流式（不支持，返回完整响应）"""
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
        真正的异步流式输出

        Phase 83: 使用 query_stream() 实现实时流式，而非等待完整响应
        返回 GenericStreamingChunk (TypedDict) 格式供 litellm streaming_handler 使用
        支持 session_id 参数实现会话记忆
        """
        logger.debug(f"[ClaudeACP] Starting streaming for model={model}")

        # 解析模型 ID
        model_id = model.split("/")[-1] if "/" in model else "sonnet"

        # 从 extra_body 获取自定义参数（litellm 传递给 custom provider 的方式）
        if not isinstance(optional_params, dict):
             optional_params = {}
        extra_body = optional_params.get("extra_body") or kwargs.get("extra_body") or {}
        if isinstance(extra_body, str):
            import json as _json
            try:
                extra_body = _json.loads(extra_body)
            except Exception:
                extra_body = {}

        session_id = (
            extra_body.get("session_id") or
            optional_params.get("session_id") or
            kwargs.get("session_id")
        )
        bot_data_dir = optional_params.get("bot_data_dir") or kwargs.get("bot_data_dir")

        # 获取连接池（按 bot_data_dir 缓存，自动创建 Claude_Session 子目录）
        pool = _get_claude_pool(bot_data_dir=bot_data_dir)

        # 转换消息
        prompt = _messages_to_prompt(messages)
        prompt_tokens = len(prompt.split())  # 估算
        completion_tokens = 0

        # Phase 83: 记录当前 session_id
        current_session_id = None

        # 执行流式查询
        async with pool.acquire(session_id=session_id) as (acp_client, current_session_id):
            # 设置模型
            if model_id and acp_client.session_id:
                try:
                    await acp_client._send_request("session/set_model", {
                        "sessionId": acp_client.session_id,
                        "modelId": model_id
                    })
                except Exception:
                    pass

            # 流式获取响应
            async for text_chunk in acp_client.query_stream(prompt):
                if text_chunk:
                    completion_tokens += len(text_chunk.split())
                    # 发送内容块 (GenericStreamingChunk 格式)
                    # Phase 83: 包含 session_id
                    yield {
                        "text": text_chunk,
                        "is_finished": False,
                        "finish_reason": None,
                        "usage": None,
                        "index": 0,
                        "tool_use": None,
                        "provider_specific_fields": {
                            "session_id": current_session_id,
                        },
                    }

        # 发送最终块
        usage_dict = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        # Phase 83: 最终块也包含 session_id
        yield {
            "text": "",
            "is_finished": True,
            "finish_reason": "stop",
            "usage": usage_dict,
            "index": 0,
            "tool_use": None,
            "provider_specific_fields": {
                "session_id": current_session_id,
            },
        }
        logger.info(f"[ClaudeACP] Streaming complete, ~{completion_tokens} tokens, session={current_session_id}")


class GeminiACPProvider(CustomLLM):
    """
    Gemini ACP LiteLLM 提供商

    模型格式: gemini-acp/{model_id}
    例如: gemini-acp/gemini-2.5-flash, gemini-acp/gemini-2.5-pro
    """

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
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._acompletion_impl(model, messages, optional_params or {})
                )
                return future.result(timeout=timeout or 300)
        else:
            return loop.run_until_complete(
                self._acompletion_impl(model, messages, optional_params or {})
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
        """异步完成"""
        return await self._acompletion_impl(model, messages, optional_params or {})

    async def _acompletion_impl(
        self,
        model: str,
        messages: List[Dict[str, str]],
        optional_params: Dict
    ) -> ModelResponse:
        """实际的异步完成实现"""
        start_time = time.time()

        # 解析模型 ID
        model_id = model.split("/")[-1] if "/" in model else "gemini-2.5-flash"

        # 从 extra_body 获取自定义参数（litellm 传递给 custom provider 的方式）
        if not isinstance(optional_params, dict):
             optional_params = {}
        extra_body = optional_params.get("extra_body") or {}
        if isinstance(extra_body, str):
            import json as _json
            try:
                extra_body = _json.loads(extra_body)
            except Exception:
                extra_body = {}

        session_id = (
            extra_body.get("session_id") or
            optional_params.get("session_id")
        )
        bot_data_dir = extra_body.get("bot_data_dir") or optional_params.get("bot_data_dir")

        # 获取连接池（按 bot_data_dir 缓存，自动创建 Gemini_Session 子目录）
        pool = _get_gemini_pool(bot_data_dir=bot_data_dir)

        # 转换消息
        prompt = _messages_to_prompt(messages)

        # 执行查询
        logger.info(f"[GeminiACP] _acompletion requested session_id={session_id}")
        async with pool.acquire(session_id=session_id) as (client, current_session_id):
            logger.info(f"[GeminiACP] _acompletion acquired session_id={current_session_id}, "
                        f"same_as_requested={current_session_id == session_id}")
            # Phase 100: 设置模型 (如果 model 参数有值)
            if model_id and current_session_id:
                try:
                    await client._send_request("session/set_model", {
                        "sessionId": current_session_id,
                        "modelId": model_id
                    })
                except Exception as e:
                    logger.debug(f"[GeminiACP] Model switch failure (might be not supported by CLI version): {e}")
                    pass  # 模型切换失败不阻塞

            content = await client.query(prompt)
            final_session_id = current_session_id

        latency_ms = (time.time() - start_time) * 1000

        # 构建响应
        response = ModelResponse(
            id=f"gemini-acp-{int(time.time())}",
            created=int(time.time()),
            model=f"gemini-acp/{model_id}",
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
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(content.split()),
                total_tokens=len(prompt.split()) + len(content.split())
            ),
            _response_ms=latency_ms
        )

        # 附加 session_id 到响应
        response._hidden_params = response._hidden_params or {}
        response._hidden_params["session_id"] = final_session_id

        return response

    def streaming(self, *args, **kwargs) -> Iterator[str]:
        """同步流式"""
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
        真正的异步流式输出

        Phase 83: 使用 query_stream() 实现实时流式
        """
        logger.info(f"[GeminiACP] Starting streaming for model={model}")

        # 解析模型 ID
        model_id = model.split("/")[-1] if "/" in model else "gemini-2.5-flash"

        # 从 extra_body 获取自定义参数（litellm 传递给 custom provider 的方式）
        if not isinstance(optional_params, dict):
             optional_params = {}
        extra_body = optional_params.get("extra_body") or kwargs.get("extra_body") or {}
        if isinstance(extra_body, str):
            import json as _json
            try:
                extra_body = _json.loads(extra_body)
            except Exception:
                extra_body = {}

        session_id = (
            extra_body.get("session_id") or
            optional_params.get("session_id") or
            kwargs.get("session_id")
        )
        bot_data_dir = optional_params.get("bot_data_dir") or kwargs.get("bot_data_dir")
        logger.info(f"[GeminiACP] Requested session_id={session_id}, bot_data_dir={bot_data_dir}")

        # 获取连接池（按 bot_data_dir 缓存，自动创建 Gemini_Session 子目录）
        pool = _get_gemini_pool(bot_data_dir=bot_data_dir)

        # 转换消息
        prompt = _messages_to_prompt(messages)
        prompt_tokens = len(prompt.split())
        completion_tokens = 0

        # 记录当前 session_id
        current_session_id = None

        # 执行流式查询
        async with pool.acquire(session_id=session_id) as (acp_client, current_session_id):
            logger.info(f"[GeminiACP] Acquired client, actual session_id={current_session_id}, "
                        f"same_as_requested={current_session_id == session_id}")
            async for text_chunk in acp_client.query_stream(prompt):
                if text_chunk:
                    completion_tokens += len(text_chunk.split())
                    yield {
                        "text": text_chunk,
                        "is_finished": False,
                        "finish_reason": None,
                        "usage": None,
                        "index": 0,
                        "tool_use": None,
                        "provider_specific_fields": {
                            "session_id": current_session_id,
                        },
                    }

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
                "session_id": current_session_id,
            },
        }
        logger.info(f"[GeminiACP] Streaming complete, session={current_session_id}")


# 全局实例
_claude_acp_provider: Optional[ClaudeACPProvider] = None
_gemini_acp_provider: Optional[GeminiACPProvider] = None


def register_acp_providers():
    """
    注册 ACP 提供商到 LiteLLM

    调用后可以使用：
    - litellm.completion(model="claude-acp/sonnet", ...)
    - litellm.completion(model="gemini-acp/gemini-2.5-flash", ...)
    - litellm.completion(model="cursor-cli/Auto", ...)
    """
    global _claude_acp_provider, _gemini_acp_provider

    _claude_acp_provider = ClaudeACPProvider()
    _gemini_acp_provider = GeminiACPProvider()

    # 获取现有的 provider map（避免覆盖已注册的 provider）
    existing_map = getattr(litellm, "custom_provider_map", []) or []
    existing_providers = {item.get("provider") for item in existing_map}

    # 添加 claude-acp（如果未注册）
    if "claude-acp" not in existing_providers:
        existing_map.append({"provider": "claude-acp", "custom_handler": _claude_acp_provider})

    # 添加 gemini-acp（如果未注册）
    if "gemini-acp" not in existing_providers:
        existing_map.append({"provider": "gemini-acp", "custom_handler": _gemini_acp_provider})

    # 先将 claude-acp 和 gemini-acp 写入 litellm，确保后续子注册函数能看到它们
    litellm.custom_provider_map = existing_map

    # 尝试注册 OpenCode ACP provider（如果可用）
    opencode_registered = False
    try:
        from .opencode_acp_provider import register_opencode_provider, is_opencode_available
        if is_opencode_available():
            if "opencode-acp" not in existing_providers:
                register_opencode_provider()
                opencode_registered = True
                # 刷新 existing_map（opencode 已追加进 litellm.custom_provider_map）
                existing_map = getattr(litellm, "custom_provider_map", []) or []
                logger.info("OpenCode ACP provider registered: opencode-acp")
    except Exception as e:
        logger.debug(f"OpenCode ACP not available: {e}")

    # 尝试注册 Cursor CLI provider（如果可用）
    cursor_registered = False
    try:
        from .cursor_cli_provider import register_cursor_provider, is_cursor_cli_available
        if is_cursor_cli_available():
            # 先设置 custom_provider_map，再注册 cursor（避免被覆盖）
            litellm.custom_provider_map = existing_map
            register_cursor_provider()
            cursor_registered = True
            logger.info("Cursor CLI provider registered: cursor-cli")
    except Exception as e:
        logger.debug(f"Cursor CLI not available: {e}")

    # 如果 cursor 未注册，直接设置 map
    if not cursor_registered:
        litellm.custom_provider_map = existing_map

    registered = ["claude-acp", "gemini-acp"]
    if opencode_registered:
        registered.append("opencode-acp")
    if cursor_registered:
        registered.append("cursor-cli")
    logger.debug(f"ACP providers registered: {', '.join(registered)}")


async def cleanup_acp_pools():
    """清理所有连接池"""
    global _claude_pools, _gemini_pools

    for key, pool in list(_claude_pools.items()):
        await pool.close_all()
    _claude_pools.clear()

    for key, pool in list(_gemini_pools.items()):
        await pool.close_all()
    _gemini_pools.clear()

    logger.info("[ACP] All connection pools cleaned up")


def clear_claude_pool(bot_data_dir: Optional[str] = None):
    """
    清除 Claude 连接池缓存（认证失败后调用）

    Args:
        bot_data_dir: 要清除的特定 bot_data_dir 的池，None 清除默认池
    """
    global _claude_pools
    cache_key = bot_data_dir or "_default_"

    if cache_key in _claude_pools:
        # 异步清理（在后台）
        import asyncio
        pool = _claude_pools.pop(cache_key)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(pool.close_all())
            else:
                loop.run_until_complete(pool.close_all())
        except Exception as e:
            logger.debug(f"[ACP] Pool cleanup error: {e}")

        logger.info(f"[ACP] Claude pool cleared for: {cache_key}")


# ============================================================
# 使用示例
# ============================================================

async def example_usage():
    """使用示例"""
    import litellm

    # 1. 注册 ACP 提供商
    register_acp_providers()

    # 2. 通过 LiteLLM 调用 Claude ACP
    response = await litellm.acompletion(
        model="claude-acp/sonnet",
        messages=[{"role": "user", "content": "What is 1+1?"}]
    )
    print(f"Claude: {response.choices[0].message.content}")

    # 3. 通过 LiteLLM 调用 Gemini ACP
    response = await litellm.acompletion(
        model="gemini-acp/gemini-2.5-flash",
        messages=[{"role": "user", "content": "What is 2+2?"}]
    )
    print(f"Gemini: {response.choices[0].message.content}")

    # 4. 混合使用（LiteLLM 统一接口）
    models = [
        "claude-acp/sonnet",      # Claude CLI ACP
        "gemini-acp/default",     # Gemini CLI ACP
        "gpt-4",                  # OpenAI API
        "claude-3-sonnet",        # Anthropic API
    ]

    for model in models:
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "Hi"}]
            )
            print(f"{model}: OK")
        except Exception as e:
            print(f"{model}: {e}")

    # 5. 清理
    await cleanup_acp_pools()


if __name__ == "__main__":
    asyncio.run(example_usage())
