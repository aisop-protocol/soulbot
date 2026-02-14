"""
Settings - Telegram Bot Configuration
"""

import streamlit as st
import os
import sys

# 添加项目路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from streamlit_app.utils.session_state import init_session_state
from streamlit_app.utils.telegram_controller import get_telegram_controller

# 页面配置
st.set_page_config(
    page_title="Settings - SoulBot",
    page_icon="⚙️",
    layout="wide"
)

init_session_state()

st.title("Settings")

# ==================== Telegram Bot 配置 ====================
st.header("🤖 Telegram Bot")

telegram = get_telegram_controller()
config = telegram.get_config()

col1, col2 = st.columns([3, 1])

with col1:
    # Bot Token 输入 (直接读写 .env)
    current_token = telegram.get_bot_token()

    new_token = st.text_input(
        "Bot Token",
        value=current_token,
        type="password",
        placeholder="Enter your Telegram Bot Token",
        help="Click Save to write to .env file",
        key="bot_token_input"
    )

    if st.button("Save Token", disabled=(new_token == current_token)):
        telegram.set_bot_token(new_token)
        st.success("Token saved to .env!")
        st.rerun()

with col2:
    st.write("")  # 垂直对齐
    st.write("")

    # 状态显示和控制按钮
    is_running = telegram.is_running()

    if is_running:
        st.success("● Running")
        if st.button("Stop Bot", type="secondary", use_container_width=True):
            success, msg = telegram.stop_bot()
            if success:
                st.success(msg)
            else:
                st.error(msg)
            st.rerun()
    else:
        st.error("● Stopped")
        if st.button("Start Bot", type="primary", use_container_width=True):
            if not telegram.get_bot_token():
                st.error("Please enter Bot Token first")
            else:
                with st.spinner("Starting bot..."):
                    success, msg = telegram.start_bot()
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
                st.rerun()

# 自动启动选项
# 首次设置 token 时默认开启，之后尊重用户选择
has_token = bool(telegram.get_bot_token())
default_auto_start = has_token if "auto_start" not in config else config["auto_start"]
auto_start = st.checkbox(
    "Auto-start Bot on launch",
    value=default_auto_start,
    help="Automatically start the Telegram bot when the app launches"
)
if auto_start != config.get("auto_start"):
    telegram.save_config({"auto_start": auto_start})

st.markdown("---")
st.info("For model, provider, and other settings, edit the `.env` file directly.")
