"""
OpenCode ACP LiteLLM Provider

将 OpenCode ACP 集成到 LiteLLM。

使用方式：
    import litellm
    from opencode_acp_provider import register_opencode_provider

    register_opencode_provider()

    response = await litellm.acompletion(
        model="opencode-acp/kimi-k2.5-free",
        messages=[{"role": "user", "content": "Hello"}]
    )

可用模型：
    - opencode-acp/kimi-k2.5-free
    - opencode-acp/minimax-m2.1-free
    - opencode-acp/trinity-large-preview-free
"""

import asyncio
import os
import time
import logging
from typing import Optional, List, Dict, Any, Iterator, AsyncIterator

import litellm
from litellm import CustomLLM, ModelResponse, Message
from litellm.types.utils import Choices, Usage

from .opencode_acp_client import (
    OpenCodeACPClient,
    OpenCodeConfig,
    OpenCodeConnectionPool,
    get_opencode_pool,
    is_opencode_available,
    cleanup_opencode_pools,
)

logger = logging.getLogger(__name__)


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """将消息列表转换为 prompt 字符串"""
    parts = []

    has_system = any(msg.get("role") == "system" for msg in messages)
    if not has_system:
        parts.append("[System Instructions]\nYou are a helpful AI assistant.\n")

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


class OpenCodeACPProvider(CustomLLM):
    """
    OpenCode ACP LiteLLM 提供商

    模型格式: opencode-acp/{model_id}
    例如: opencode-acp/kimi-k2.5-free
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
                    self._acompletion_impl(model, messages, optional_params or {}, litellm_params=litellm_params, **kwargs)
                )
                return future.result(timeout=timeout or 300)
        else:
            return loop.run_until_complete(
                self._acompletion_impl(model, messages, optional_params or {}, litellm_params=litellm_params, **kwargs)
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
        return await self._acompletion_impl(model, messages, optional_params or {}, litellm_params=litellm_params, **kwargs)

    async def _acompletion_impl(
        self,
        model: str,
        messages: List[Dict[str, str]],
        optional_params: Dict,
        litellm_params: Optional[Dict] = None,
        **kwargs
    ) -> ModelResponse:
        """实际的异步完成实现"""
        start_time = time.time()
        logger.debug(f"[OpenCodeACP] Starting completion for model={model}")

        # Model override: if disabled, let OpenCode use its own default model
        from src.config import OPENCODE_MODEL_OVERRIDE
        if OPENCODE_MODEL_OVERRIDE:
            # Strip "opencode-acp/" prefix, preserve provider/model format
            # opencode-acp/kimi-k2.5-free -> kimi-k2.5-free -> opencode/kimi-k2.5-free
            # opencode-acp/anthropic/claude-sonnet-4-5 -> anthropic/claude-sonnet-4-5
            model_id = model.replace("opencode-acp/", "", 1) if model.startswith("opencode-acp/") else model
            if "/" not in model_id:
                model_id = f"opencode/{model_id}"
        else:
            model_id = None  # Skip set_model, use OpenCode default

        # 获取参数 (尝试从多个位置获取)
        litellm_params_local = litellm_params or kwargs.pop("litellm_params", None) or {}

        # 从 extra_body 获取自定义参数
        extra_body = optional_params.get("extra_body") or kwargs.get("extra_body") or {}
        if isinstance(extra_body, str):
            import json as _json
            try:
                extra_body = _json.loads(extra_body)
            except:
                extra_body = {}

        session_id = (
            extra_body.get("session_id") or
            optional_params.get("session_id") or
            kwargs.get("session_id") or
            litellm_params_local.get("session_id")
        )
        user_id = (
            extra_body.get("user_id") or
            optional_params.get("user_id") or
            kwargs.get("user_id") or
            litellm_params_local.get("user_id")
        )
        bot_data_dir = optional_params.get("bot_data_dir") or kwargs.get("bot_data_dir")
        # 使用环境变量或父目录作为 OpenCode 工作目录
        default_cwd = os.environ.get("OPENCODE_CWD") or os.path.dirname(os.getcwd())
        cwd = optional_params.get("cwd") or kwargs.get("cwd") or default_cwd

        logger.debug(f"[OpenCodeACP] session_id={session_id}, user_id={user_id}")

        # 创建配置
        config = OpenCodeConfig(
            model=model_id,
            cwd=cwd,
        )

        # 获取连接池
        pool = get_opencode_pool(bot_data_dir=bot_data_dir, config=config)

        # 转换消息
        prompt = _messages_to_prompt(messages)
        logger.debug(f"[OpenCodeACP] Prompt length: {len(prompt)} chars")

        # 执行查询（支持 session_id 和 user_id 进行 session 持久化）
        async with pool.acquire(user_id=user_id, session_id=session_id) as (client, current_session_id):
            # 设置模型
            if model_id and current_session_id:
                try:
                    await client._send_request("session/set_model", {
                        "sessionId": current_session_id,
                        "modelId": model_id
                    })
                except Exception:
                    pass

            content = await client.query(prompt)
            final_session_id = current_session_id

        latency_ms = (time.time() - start_time) * 1000

        # 构建响应
        response = ModelResponse(
            id=f"opencode-acp-{int(time.time())}",
            created=int(time.time()),
            model=f"opencode-acp/{model_id}",
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

        # 附加 session_id
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
        """异步流式输出"""
        logger.debug(f"[OpenCodeACP] Starting streaming for model={model}")

        # Model override: if disabled, let OpenCode use its own default model
        from src.config import OPENCODE_MODEL_OVERRIDE
        if OPENCODE_MODEL_OVERRIDE:
            # Strip "opencode-acp/" prefix, preserve provider/model format
            model_id = model.replace("opencode-acp/", "", 1) if model.startswith("opencode-acp/") else model
            if "/" not in model_id:
                model_id = f"opencode/{model_id}"
        else:
            model_id = None

        # 获取参数
        if not isinstance(optional_params, dict):
            optional_params = {}

        # 从 extra_body 获取自定义参数（litellm 传递给 custom provider 的方式）
        extra_body = optional_params.get("extra_body") or kwargs.get("extra_body") or {}
        if isinstance(extra_body, str):
            import json as _json
            try:
                extra_body = _json.loads(extra_body)
            except:
                extra_body = {}

        # 尝试从多个位置获取 session_id
        session_id = (
            extra_body.get("session_id") or
            optional_params.get("session_id") or
            kwargs.get("session_id") or
            (litellm_params.get("session_id") if litellm_params else None)
        )
        user_id = (
            extra_body.get("user_id") or
            optional_params.get("user_id") or
            kwargs.get("user_id") or
            (litellm_params.get("user_id") if litellm_params else None)
        )
        bot_data_dir = optional_params.get("bot_data_dir") or kwargs.get("bot_data_dir")
        # 使用环境变量或父目录作为 OpenCode 工作目录
        default_cwd = os.environ.get("OPENCODE_CWD") or os.path.dirname(os.getcwd())
        cwd = optional_params.get("cwd") or kwargs.get("cwd") or default_cwd

        logger.debug(f"[OpenCodeACP Stream] session_id={session_id}, user_id={user_id}")

        # 创建配置
        config = OpenCodeConfig(
            model=model_id,
            cwd=cwd,
        )

        # 获取连接池
        pool = get_opencode_pool(bot_data_dir=bot_data_dir, config=config)

        # 转换消息
        prompt = _messages_to_prompt(messages)
        prompt_tokens = len(prompt.split())
        completion_tokens = 0

        current_session_id = None

        # 执行流式查询（支持 session_id 和 user_id 进行 session 持久化）
        async with pool.acquire(user_id=user_id, session_id=session_id) as (acp_client, current_session_id):
            # 设置模型
            if model_id and acp_client.session_id:
                try:
                    await acp_client._send_request("session/set_model", {
                        "sessionId": acp_client.session_id,
                        "modelId": model_id
                    })
                except Exception:
                    pass

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
        logger.info(f"[OpenCodeACP] Streaming complete, ~{completion_tokens} tokens")


# 全局实例
_opencode_acp_provider: Optional[OpenCodeACPProvider] = None


def register_opencode_provider():
    """
    注册 OpenCode ACP 提供商到 LiteLLM

    调用后可以使用：
    - litellm.completion(model="opencode-acp/kimi-k2.5-free", ...)
    """
    global _opencode_acp_provider

    if not is_opencode_available():
        logger.warning("[OpenCodeACP] OpenCode not found, skipping registration")
        return False

    _opencode_acp_provider = OpenCodeACPProvider()

    # 获取现有的 provider map
    existing_map = getattr(litellm, "custom_provider_map", []) or []

    # 检查是否已注册
    for item in existing_map:
        if item.get("provider") == "opencode-acp":
            logger.info("[OpenCodeACP] Provider already registered")
            return True

    # 添加 opencode-acp provider
    existing_map.append({
        "provider": "opencode-acp",
        "custom_handler": _opencode_acp_provider
    })
    litellm.custom_provider_map = existing_map

    logger.info("OpenCode ACP provider registered: opencode-acp")
    return True


__all__ = [
    "OpenCodeACPProvider",
    "register_opencode_provider",
    "is_opencode_available",
    "cleanup_opencode_pools",
]
