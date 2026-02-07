import os
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

if not CLAUDE_CLI and not GEMINI_CLI:
    logger.warning("No AI CLI provider enabled (CLAUDE_CLI=false, GEMINI_CLI=false). AI will echo messages.")
elif CLAUDE_CLI and GEMINI_CLI:
    logger.warning("Both CLAUDE_CLI and GEMINI_CLI are true. Configuration may be ambiguous (preferring Claude).")

# Model Configuration
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-acp/sonnet")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-acp/gemini-2.5-flash")
ENABLE_FALLBACK = os.getenv("ENABLE_FALLBACK", "false").lower() == "true"

# Tool & Workspace Configuration
AUTO_APPROVE_PERMISSIONS = os.getenv("AUTO_APPROVE_PERMISSIONS", "true").lower() == "true"
WORKSPACE_DIR = os.path.abspath(os.getenv("WORKSPACE_DIR", "."))

# AISOP Configuration
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "").replace("\\n", "\n")
SHOW_THOUGHTS = os.getenv("SHOW_THOUGHTS", "true").lower() == "true"

