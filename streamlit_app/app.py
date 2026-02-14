"""
SoulBot Streamlit 主入口
"""

import streamlit as st
import sys
import os

# 添加项目路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from streamlit_app.utils.session_state import init_session_state

# 页面配置
st.set_page_config(
    page_title="SoulBot AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化会话状态
init_session_state()

# ==================== 主页内容 ====================
st.title("SoulBot AI Assistant")

st.markdown("""
*Powered by **AISOP** (AI Standard Operating Protocol) — A self-evolving AI runtime that turns
structured aisop files into intelligent behavior.*
""")

st.markdown("---")

# ==================== 核心介绍 ====================

st.markdown("""
### What is SoulBot?

SoulBot is a **personal AI operating system** that runs on the AISOP protocol.
Instead of hardcoded logic, SoulBot executes `.aisop.json` files —
making it **self-fractal, self-upgrading, and infinitely extensible**.

#### How it works

```
You (Telegram / Web) → SoulBot Runtime → Load .aisop.json → AI Executes AISOP
```

The AI **reads, interprets, and executes** aisop files like an operating system runs programs.
Each aisop file defines a workflow (intent routing, tool usage, data validation),
and the AI follows it step by step.
""")

st.markdown("---")

# ==================== 核心能力 ====================
st.markdown("### Core Capabilities")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🧬 Self-Fractal Architecture**
    - AISops can call other aisops
    - `main.aisop.json` routes to `stock.aisop.json`, `weather.aisop.json`, etc.
    - Infinite nesting, like functions calling functions

    **🔄 Self-Upgrading**
    - AI can create new `.aisop.json` files on its own
    - AI can modify existing aisop files to improve them
    - The system evolves through use, not through code releases
    """)

with col2:
    st.markdown("""
    **📱 Remote CLI Programming via Telegram**
    - Control your dev machine from anywhere
    - Ask AI to write code, run commands, manage files
    - Telegram as a remote terminal to your AI-powered CLI

    **🧠 Personal AI System**
    - Cross-session memory — AI remembers all conversations
    - Per-user isolation — independent workspace per user
    - AISOP library grows as you use it
    """)

st.markdown("---")

# ==================== AISOP 生态 ====================
st.markdown("### AISOP Ecosystem")

st.markdown("""
Ask the AI to create specialized aisop files for any domain:

| AISOP | Description |
|-----------|-------------|
| `main.aisop.json` | Core router — intent detection, task delegation |
| `stock_analysis.aisop.json` | Stock analysis — ticker validation, real-time data, reports |
| `weather.aisop.json` | Weather forecast — location-based queries |
| `health.aisop.json` | Health tracking — reminders, metrics, suggestions |
| `code_review.aisop.json` | Code review — static analysis, best practices |
| `daily_report.aisop.json` | Daily summary — aggregate info from multiple sources |
| *...your idea here* | *Tell the AI to create it* |

> **Example:** "Create a `fitness.aisop.json` that tracks my workouts and suggests routines"
> The AI will generate the aisop file, and SoulBot can immediately execute it.
""")

st.markdown("---")

# ==================== 快速入口 ====================
col1, col2 = st.columns(2)
with col1:
    if st.button("💬 Start Chat", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Chat.py")
with col2:
    if st.button("⚙️ Settings", use_container_width=True):
        st.switch_page("pages/2_Settings.py")
