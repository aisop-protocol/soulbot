"""
历史记录页面

显示所有 provider 的 session 记录
"""

import streamlit as st
import os
import sys

# 添加项目路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from streamlit_app.utils.session_state import (
    init_session_state,
    get_session_manager,
    clear_current_session,
)

# 页面配置
st.set_page_config(
    page_title="History - SoulBot",
    page_icon="📜",
    layout="wide"
)

init_session_state()

st.title("Chat History")

manager = get_session_manager()

st.markdown("---")

# 显示所有 provider 的 sessions
st.header("Active Sessions")

all_sessions = manager.list_all_sessions()

if all_sessions:
    for provider, entry in all_sessions.items():
        # 兼容旧格式（纯字符串）和新格式（dict）
        if isinstance(entry, dict):
            session_id = entry.get("session_id", "")
            created_at = entry.get("created_at")
        else:
            session_id = entry
            created_at = None

        display_time = created_at or "Unknown time"
        with st.expander(f"{provider.upper()} - Created {display_time}"):
            st.code(f"Created: {display_time}")
            st.caption(f"Provider: {provider}")

            # 操作按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Load Session", key=f"load_{provider}"):
                    st.session_state.display_messages = []
                    manager.set_session_id(provider, session_id)
                    st.switch_page("pages/1_Chat.py")
            with col2:
                if st.button(f"Delete", key=f"del_{provider}"):
                    manager.clear_session(provider)
                    st.rerun()
else:
    st.info("No active sessions. Start chatting to create one.")

st.markdown("---")

# 清除所有会话
st.header("Danger Zone")

with st.expander("Clear All Data"):
    st.warning("This will delete all session records. This cannot be undone!")

    confirm = st.text_input("Type 'DELETE' to confirm")

    if st.button("Confirm Delete All Sessions", type="primary"):
        if confirm == "DELETE":
            manager.clear_session()  # 清除全部
            st.session_state.display_messages = []
            st.success("All sessions deleted")
            st.rerun()
        else:
            st.error("Please type 'DELETE' to confirm")
