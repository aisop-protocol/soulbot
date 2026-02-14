"""
聊天页面 - 核心功能

关键点：
1. 使用 extra_body 传递 session_id（LiteLLM 要求）
2. 集成 SessionManager 持久化
3. OpenCode 自动加载历史消息
4. 使用持久化 AI 工作线程（解决事件循环冲突）
"""

import streamlit as st
import queue
import sys
import os

# 添加项目路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from streamlit_app.utils.session_state import (
    init_session_state,
    get_session_id,
    get_session_manager,
    save_session_id,
    clear_current_session,
)
from streamlit_app.utils.ai_worker import get_ai_worker, stream_ai_response

# 页面配置
st.set_page_config(
    page_title="Chat - SoulBot",
    page_icon="💬",
    layout="wide"
)

# 初始化
init_session_state()

st.title("AI Chat")

# 侧边栏：会话控制
with st.sidebar:
    # Session 信息
    session_id = get_session_id()
    if session_id:
        provider = st.session_state.get("current_provider", "opencode")
        info = get_session_manager().get_session_info(provider)
        created_at = info.get("created_at") if info else None
        st.success(f"Session: `{created_at or 'Active'}`")
    else:
        st.info("New session (auto-created on first message)")

    # 清除会话按钮
    if st.button("Clear Session", use_container_width=True):
        clear_current_session()
        st.rerun()

    st.markdown("---")
    provider = st.session_state.get("current_provider", "default")
    st.caption(f"Provider: `{provider}`")

# 显示聊天历史（仅用于 UI，实际历史由 OpenCode 管理）
for message in st.session_state.display_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if user_input := st.chat_input("Type your message...", disabled=st.session_state.is_generating):
    # 添加用户消息到显示
    st.session_state.display_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # AI 响应
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # 获取 session_id（unified per-provider）
        session_id = get_session_id()

        st.session_state.is_generating = True

        try:
            # 使用 AI 工作线程
            result_queue = queue.Queue()
            worker = get_ai_worker()
            worker.submit(stream_ai_response, result_queue, user_input, session_id)

            # 收集响应
            full_response = ""
            new_session_id = None

            while True:
                try:
                    # 10分钟超时，复杂任务可能需要较长时间
                    msg_type, msg_data = result_queue.get(timeout=600)

                    if msg_type == "chunk":
                        full_response += msg_data
                        message_placeholder.markdown(full_response + "▌")
                    elif msg_type == "session_id":
                        new_session_id = msg_data
                    elif msg_type == "done":
                        message_placeholder.markdown(full_response)
                        break
                    elif msg_type == "error":
                        message_placeholder.error(f"Error: {msg_data}")
                        full_response = f"Error: {msg_data}"
                        break
                except queue.Empty:
                    message_placeholder.error("Response timeout (10 min)")
                    full_response = "Response timeout"
                    break

            # 保存 session_id（始终保存当前使用的 session）
            if new_session_id:
                if not session_id:
                    # 第一次对话，保存新 session
                    save_session_id(new_session_id)
                elif new_session_id != session_id:
                    # resume 失败，旧 session 已死，保存新的
                    import logging
                    logging.getLogger(__name__).warning(
                        f"[WebChat] Session changed: {session_id[:20]}... -> {new_session_id[:20]}... "
                        f"(saving new session)"
                    )
                    save_session_id(new_session_id)
        finally:
            st.session_state.is_generating = False

        # 保存助手响应到显示列表（在 with 块内保存，确保顺序正确）
        st.session_state.display_messages.append({
            "role": "assistant",
            "content": full_response
        })

    # 强制 rerun 确保状态正确保存
    st.rerun()

# 提示信息
if not st.session_state.display_messages:
    st.info("""
    **Welcome to SoulBot AI Assistant!**

    Features:
    - **Memory**: AI remembers all your conversations
    - **Cross-session**: Conversation history persists after closing browser
    - **Streaming**: Real-time AI response display

    Start typing your question!
    """)
