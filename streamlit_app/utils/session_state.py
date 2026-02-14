"""
Streamlit 会话状态管理

统一 session：每个 provider 全局一个 session，所有用户/接口共享
"""

import streamlit as st
import sys
import os

# 添加项目根目录到路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.bot.session_manager import SessionManager
from src.config import get_current_provider, OPENCODE_MODEL, CLAUDE_MODEL, GEMINI_MODEL


def get_session_manager() -> SessionManager:
    """获取 SessionManager 单例"""
    return SessionManager.get_instance()


def init_session_state():
    """初始化 Streamlit 会话状态"""

    # 当前 provider（从 config 自动检测，与 Telegram handler 一致）
    if "current_provider" not in st.session_state:
        st.session_state.current_provider = get_current_provider()

    # 当前模型（从 config 读取，与 .env 配置一致）
    if "current_model" not in st.session_state:
        provider = st.session_state.get("current_provider", "opencode")
        model_map = {"opencode": OPENCODE_MODEL, "claude": CLAUDE_MODEL, "gemini": GEMINI_MODEL}
        st.session_state.current_model = model_map.get(provider, OPENCODE_MODEL)

    # 页面显示的消息（用于 UI 渲染，非持久化）
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []

    # 是否正在生成响应
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False

    # 用户名称
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""


def get_session_id() -> str:
    """
    获取当前 provider 的 session_id

    优先级：
    1. URL 参数中的 sid
    2. session_state 中的缓存
    3. SessionManager JSON 文件

    Returns:
        session_id 或 None（新会话）
    """
    query_params = st.query_params

    # 1. 优先从 URL 参数获取
    if "sid" in query_params:
        session_id = query_params["sid"]
        st.session_state._acp_session_id = session_id
        return session_id

    # 2. 检查 session_state 缓存
    if "_acp_session_id" in st.session_state:
        return st.session_state._acp_session_id

    # 3. 从 SessionManager 加载（统一 per-provider）
    manager = get_session_manager()
    provider = st.session_state.get("current_provider", "opencode")
    session_id = manager.get_session_id(provider)

    if session_id:
        st.session_state._acp_session_id = session_id
        st.query_params["sid"] = session_id
        return session_id

    return None


def save_session_id(session_id: str):
    """
    保存 session_id

    持久化到：
    1. session_state（当前会话）
    2. URL 参数（刷新保持）
    3. SessionManager JSON 文件（重启保持）
    """
    if not session_id:
        return

    st.session_state._acp_session_id = session_id
    st.query_params["sid"] = session_id

    manager = get_session_manager()
    provider = st.session_state.get("current_provider", "opencode")
    manager.set_session_id(provider, session_id)


def clear_current_session():
    """清除当前 provider 的 session（开始新对话）"""
    # 清除 session_state 缓存
    if "_acp_session_id" in st.session_state:
        del st.session_state._acp_session_id

    # 清除 URL 参数
    if "sid" in st.query_params:
        del st.query_params["sid"]

    # 清除 SessionManager 中的记录
    manager = get_session_manager()
    provider = st.session_state.get("current_provider", "opencode")
    manager.clear_session(provider)

    # 清除页面显示
    st.session_state.display_messages = []


def get_active_providers() -> list:
    """获取有活跃 session 的 provider 列表"""
    manager = get_session_manager()
    return manager.get_active_providers()
