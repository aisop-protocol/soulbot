import os
import json
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Configure Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Silence verbose httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    logger.warning("TELEGRAM_BOT_TOKEN is not set in .env")

# AI Provider Configuration
CLAUDE_CLI = os.getenv("CLAUDE_CLI", "false").lower() == "true"
GEMINI_CLI = os.getenv("GEMINI_CLI", "false").lower() == "true"
OPENCODE_CLI = os.getenv("OPENCODE_CLI", "false").lower() == "true"

if not CLAUDE_CLI and not GEMINI_CLI and not OPENCODE_CLI:
    logger.warning("No AI CLI provider enabled (CLAUDE_CLI=false, GEMINI_CLI=false, OPENCODE_CLI=false). AI will echo messages.")
elif sum([CLAUDE_CLI, GEMINI_CLI, OPENCODE_CLI]) > 1:
    logger.warning("Multiple AI CLI providers enabled. Priority: OpenCode > Claude > Gemini.")


def get_current_provider() -> str:
    """获取当前使用的 AI 提供商（优先级：OpenCode > Claude > Gemini）"""
    if OPENCODE_CLI:
        return "opencode"
    elif CLAUDE_CLI:
        return "claude"
    elif GEMINI_CLI:
        return "gemini"
    else:
        return "default"

# Model Configuration
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-acp/sonnet")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-acp/gemini-2.5-flash")
OPENCODE_MODEL = os.getenv("OPENCODE_MODEL", "opencode-acp/kimi-k2.5-free")
OPENCODE_MODEL_OVERRIDE = os.getenv("OPENCODE_MODEL_OVERRIDE", "true").lower() == "true"
ENABLE_FALLBACK = os.getenv("ENABLE_FALLBACK", "false").lower() == "true"

# Tool & Workspace Configuration
AUTO_APPROVE_PERMISSIONS = os.getenv("AUTO_APPROVE_PERMISSIONS", "true").lower() == "true"
# Resolve WORKSPACE_DIR relative to project root (not CWD), so "aisop" always maps to <project>/aisop
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_workspace_raw = os.getenv("WORKSPACE_DIR", ".")
WORKSPACE_DIR = os.path.abspath(_workspace_raw) if os.path.isabs(_workspace_raw) else os.path.abspath(os.path.join(PROJECT_ROOT, _workspace_raw))

# AISOP Configuration
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "").replace("\\n", "\n")
SHOW_THOUGHTS = os.getenv("SHOW_THOUGHTS", "true").lower() == "true"


# ── AISOP Loading & System Instruction (shared by Telegram handler and Web UI) ──

_aisop_cache: dict = {}
_system_instruction_cache: str = None


def clear_instruction_cache():
    """清除系统指令缓存（配置更新后调用）"""
    global _system_instruction_cache, _aisop_cache
    _system_instruction_cache = None
    _aisop_cache = {}


def load_aisop(name="main"):
    """加载 AISOP 文件（带缓存，路径基于 PROJECT_ROOT）"""
    if name in _aisop_cache:
        return _aisop_cache[name]

    try:
        path = os.path.join(PROJECT_ROOT, "aisop", f"{name}.aisop.json")
        with open(path, "r", encoding="utf-8-sig") as f:
            aisop = json.load(f)
            _aisop_cache[name] = aisop
            return aisop
    except Exception as e:
        logger.error(f"Failed to load aisop {name}: {e}")
        return None


# Backwards compatibility alias
load_blueprint = load_aisop


def get_system_instruction() -> str:
    """
    构建系统指令（带缓存）

    Layer 1: .env SYSTEM_PROMPT
    Layer 2: AISOP
    Layer 3: WORKSPACE_DIR
    """
    global _system_instruction_cache

    if _system_instruction_cache is not None:
        return _system_instruction_cache

    base_prompt = SYSTEM_PROMPT or ""
    aisop = load_aisop("main")
    aisop_str = json.dumps(aisop, ensure_ascii=False, indent=2) if aisop else "{}"
    workspace_line = f"\n\n[WORKSPACE]\nYour AISOP directory is: {WORKSPACE_DIR}"

    if base_prompt and aisop:
        _system_instruction_cache = (
            f"{base_prompt}\n\n"
            f"---\n"
            f"[LOADED AISOP: main.aisop.json]\n"
            f"```json\n{aisop_str}\n```"
            f"{workspace_line}"
        )
    elif base_prompt:
        _system_instruction_cache = base_prompt + workspace_line
    elif aisop:
        _system_instruction_cache = (
            f"[LOADED AISOP: main.aisop.json]\n"
            f"```json\n{aisop_str}\n```"
            f"{workspace_line}"
        )
    else:
        _system_instruction_cache = workspace_line.strip()

    return _system_instruction_cache

