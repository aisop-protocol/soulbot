"""
用户识别管理

为 Streamlit Web 用户生成唯一标识，用于 session 隔离。
"""

import streamlit as st
import uuid
import hashlib
from typing import Optional


class UserManager:
    """Web 用户管理器"""

    # 用于生成 app hash 的标识
    APP_NAME = "soulbot-streamlit"

    @staticmethod
    def get_app_hash() -> str:
        """
        获取应用标识 hash（类似 Telegram Bot 的 bot_token_hash）

        用于隔离不同部署实例的 session
        """
        return hashlib.sha256(UserManager.APP_NAME.encode()).hexdigest()[:12]

    @staticmethod
    def get_user_id() -> str:
        """
        获取当前用户的唯一 ID

        注意：user_id 用于 SessionManager 映射，主要记忆通过 session_id (URL参数) 持久化

        Returns:
            用户唯一标识符
        """
        if "user_id" not in st.session_state:
            # 生成新的 UUID
            st.session_state.user_id = f"web_{uuid.uuid4().hex[:16]}"

        return st.session_state.user_id

    @staticmethod
    def get_user_display_name() -> str:
        """获取用户显示名称"""
        if "user_name" in st.session_state and st.session_state.user_name:
            return st.session_state.user_name
        return f"用户 {UserManager.get_user_id()[:8]}"

    @staticmethod
    def set_user_name(name: str):
        """设置用户显示名称"""
        st.session_state.user_name = name

    @staticmethod
    def get_shareable_link() -> str:
        """
        获取可分享的链接（包含用户 ID）

        用户可以在不同设备上恢复会话
        """
        user_id = UserManager.get_user_id()
        return f"?uid={user_id}"


def get_user_id() -> str:
    """便捷函数：获取用户 ID"""
    return UserManager.get_user_id()


def get_app_hash() -> str:
    """便捷函数：获取应用 hash"""
    return UserManager.get_app_hash()
