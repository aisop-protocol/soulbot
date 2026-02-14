"""
侧边栏组件
"""

import streamlit as st
from streamlit_app.utils.user_manager import UserManager


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("SoulBot")
        st.caption("AI Assistant with Memory")

        st.markdown("---")

        # 用户信息
        user_name = UserManager.get_user_display_name()
        st.markdown(f"**{user_name}**")
        st.caption(f"ID: `{UserManager.get_user_id()[:12]}...`")

        st.markdown("---")

        # 导航
        st.markdown("### Navigation")

        # 注意：Streamlit 多页面应用会自动添加导航
        # 这里可以添加额外的链接或信息

        st.markdown("---")

        # 版本信息
        st.caption("SoulBot v1.0")
        st.caption("Powered by OpenCode ACP")
