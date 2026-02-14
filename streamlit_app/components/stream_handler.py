"""
Streamlit 流式输出处理器
"""

import streamlit as st
import asyncio
from typing import Optional


class StreamHandler:
    """流式输出处理器"""

    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.full_text = ""

    def update(self, chunk: str):
        """更新显示内容"""
        self.full_text += chunk
        self.placeholder.markdown(self.full_text + "▌")

    def finalize(self) -> str:
        """完成显示"""
        self.placeholder.markdown(self.full_text)
        return self.full_text

    def error(self, message: str):
        """显示错误"""
        self.placeholder.error(message)


async def stream_llm_response(
    ai_client,
    user_input: str,
    session_id: Optional[str],
    user_id: str,
    handler: StreamHandler
) -> tuple:
    """
    流式获取 LLM 响应

    Args:
        ai_client: AIClient 实例
        user_input: 用户输入
        session_id: 当前 session_id
        user_id: 用户 ID
        handler: StreamHandler 实例

    Returns:
        (完整响应文本, 新的 session_id)
    """
    new_session_id = None

    try:
        async for chunk, returned_session_id in ai_client.get_streaming_response(
            user_input,
            session_id=session_id,
            user_id=user_id
        ):
            if chunk:
                handler.update(chunk)

            if returned_session_id:
                new_session_id = returned_session_id

        return handler.finalize(), new_session_id

    except Exception as e:
        handler.error(f"Error: {str(e)}")
        return f"Error: {str(e)}", None


def run_async(coro):
    """在 Streamlit 中运行异步函数"""
    return asyncio.run(coro)
