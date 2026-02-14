"""
系统监控页面
"""

import streamlit as st
import time
from datetime import datetime
import sys
import os

# 添加项目路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from streamlit_app.utils.session_state import init_session_state, get_session_manager

# 页面配置
st.set_page_config(
    page_title="Monitor - SoulBot",
    page_icon="📊",
    layout="wide"
)

init_session_state()

st.title("System Monitor")

# 自动刷新
auto_refresh = st.toggle("Auto Refresh (5s)", value=False)

# 系统资源
st.header("System Resources")

try:
    import psutil

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cpu_percent = psutil.cpu_percent()
        st.metric("CPU Usage", f"{cpu_percent}%")

    with col2:
        memory = psutil.virtual_memory()
        st.metric("Memory Usage", f"{memory.percent}%")

    with col3:
        disk = psutil.disk_usage(os.path.splitdrive(os.getcwd())[0] + os.sep if os.name == 'nt' else '/')
        st.metric("Disk Usage", f"{disk.percent}%")

    with col4:
        st.metric("Current Time", datetime.now().strftime("%H:%M:%S"))

except ImportError:
    st.warning("Install psutil for system monitoring: `pip install psutil`")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Time", datetime.now().strftime("%H:%M:%S"))

st.markdown("---")

# OpenCode 连接状态
st.header("OpenCode Status")

try:
    from src.llm_service.opencode_acp_client import is_opencode_available, _opencode_pools

    col1, col2, col3 = st.columns(3)

    with col1:
        opencode_status = "Available" if is_opencode_available() else "Not Available"
        if is_opencode_available():
            st.success(f"OpenCode CLI: {opencode_status}")
        else:
            st.error(f"OpenCode CLI: {opencode_status}")

    with col2:
        pool_count = len(_opencode_pools)
        st.metric("Active Connection Pools", pool_count)

    with col3:
        # 显示连接池详情
        if _opencode_pools:
            for key, pool in _opencode_pools.items():
                st.caption(f"Pool: {key[:20]}...")
        else:
            st.caption("No active pools")

except ImportError as e:
    st.warning(f"Cannot load OpenCode module: {e}")

st.markdown("---")

# Session 管理器状态
st.header("Session Manager")

try:
    manager = get_session_manager()
    all_sessions = manager.list_all_sessions()

    st.metric("Total Sessions", len(all_sessions))

    if all_sessions:
        with st.expander("View All Sessions"):
            for key, entry in all_sessions.items():
                created_at = entry.get("created_at", "Unknown") if isinstance(entry, dict) else "Unknown"
                display = created_at or "Unknown"
                st.code(f"{key}: Created {display}")
except Exception as e:
    st.warning(f"Cannot load session info: {e}")

st.markdown("---")

# OpenCode 存储
st.header("OpenCode Storage")

opencode_storage = os.path.expandvars("%APPDATA%/opencode/storage")
if not os.path.exists(opencode_storage):
    # Try Linux/Mac path
    opencode_storage = os.path.expanduser("~/.local/share/opencode/storage")

if os.path.exists(opencode_storage):
    col1, col2 = st.columns(2)

    with col1:
        # 统计 session 数量
        session_dir = os.path.join(opencode_storage, "session")
        if os.path.exists(session_dir):
            session_count = sum(
                len(files) for _, _, files in os.walk(session_dir)
                if files
            )
            st.metric("Session Files", session_count)
        else:
            st.metric("Session Files", 0)

    with col2:
        # 统计消息数量
        message_dir = os.path.join(opencode_storage, "message")
        if os.path.exists(message_dir):
            message_count = sum(
                len(files) for _, _, files in os.walk(message_dir)
                if files
            )
            st.metric("Message Files", message_count)
        else:
            st.metric("Message Files", 0)

    st.caption("Storage: OpenCode local storage")
else:
    st.info("OpenCode storage directory not found")

# 自动刷新
if auto_refresh:
    time.sleep(5)
    st.rerun()
