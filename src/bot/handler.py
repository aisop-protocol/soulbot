import time
import json
import os
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes, Application, CommandHandler, MessageHandler, filters
from telegram.error import BadRequest
from src.config import TELEGRAM_BOT_TOKEN, logger, WORKSPACE_DIR
from src.llm_client import AIClient
from src.bot.session_manager import SessionManager

# Initialize AI Client and Session Manager
ai_client = AIClient()
session_manager = SessionManager()

# Load Blueprint
def load_blueprint(name="main"):
    try:
        path = os.path.join("blueprints", f"{name}.aisop.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load blueprint {name}: {e}")
        return None

# --- Helpers ---
async def safe_edit_message(context: ContextTypes.DEFAULT_TYPE, chat_id, message_id, text, parse_mode=ParseMode.MARKDOWN, last_text=None):
    """Edits a message safely, falling back to plain text on parse errors."""
    if last_text and text == last_text:
        return last_text
        
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            parse_mode=parse_mode
        )
    except BadRequest as e:
        err_msg = str(e)
        if "Message is not modified" in err_msg:
            return text
        if "Can't parse entities" in err_msg:
            logger.warning(f"Markdown Parse Error. Falling back to plain text. Error: {err_msg}")
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                parse_mode=None
            )
        else:
            raise e
    return text

async def safe_reply_text(update: Update, text, parse_mode=ParseMode.MARKDOWN):
    """Sends a message safely, falling back to plain text on parse errors."""
    try:
        return await update.message.reply_text(text, parse_mode=parse_mode)
    except BadRequest as e:
        if "Can't parse entities" in str(e):
            logger.warning("Markdown Parse Error on reply. Falling back to plain text.")
            return await update.message.reply_text(text, parse_mode=None)
        else:
            raise e

# --- Core Logic ---
CURRENT_BLUEPRINT = load_blueprint("main")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    app_name = CURRENT_BLUEPRINT.get("app_name", "SoulBot") if CURRENT_BLUEPRINT else "SoulBot"
    
    await update.message.reply_html(
        rf"⚡ <b>{app_name} is Online.</b>{os.linesep}"
        rf"Hi {user.mention_html()}! I am running on <b>AISOP Runtime v1.0</b>.{os.linesep}"
        rf"My brain is loaded with: <code>main.aisop.json</code>{os.linesep}{os.linesep}"
        rf"Use /clear to reset your conversation history."
    )

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear the user's conversation history."""
    user_id = update.effective_user.id
    session_manager.clear_session(user_id)
    await update.message.reply_text("🧼 Conversation history cleared!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Core logic: User Input -> AISOP Runtime (LLM + Blueprint) -> User Output"""
    user_message = update.message.text
    user_name = update.effective_user.name
    logger.info(f"Received from {user_name}: {user_message}")

    # 1. Prepare Interface (Typing...)
    placeholder_message = await update.message.reply_text("🧠 accessing neural pathways...")
    
    # 2. Construct AISOP Context Prompt
    blueprint_str = json.dumps(CURRENT_BLUEPRINT, ensure_ascii=False, indent=2) if CURRENT_BLUEPRINT else "{}"
    
    system_instruction = (
        f"[AISOP RUNTIME V3.1]\n"
        f"You are executing the following Blueprint:\n"
        f"```json\n{blueprint_str}\n```\n"
        f"Follow the 'context.system' instructions.\n"
        f"Execute logical steps defined in 'blueprint.content' (Mermaid).\n"
        f"Respond directly to the user.\n"
        f"STAY IN CHARACTER based on the blueprint roles."
    )
    
    # 2. Get Session ID and Stream Response
    user_id = update.effective_user.id
    current_session_id = session_manager.get_session_id(user_id)
    
    full_response = ""
    last_displayed_text = ""
    last_update_time = time.time()
    update_interval = 0.8 # Seconds

    try:
        # Pass the prompt and current session_id
        async for chunk_text, new_session_id in ai_client.get_streaming_response(user_message, session_id=current_session_id, system_prompt=system_instruction):
            if chunk_text:
                full_response += chunk_text
            if new_session_id:
                current_session_id = new_session_id
            
            # Update UI periodically
            if time.time() - last_update_time > update_interval:
                if full_response.strip():
                    try:
                        # Truncate for Telegram limit (Keep the START of the message)
                        display_text = full_response
                        if len(display_text) > 3900:
                            display_text = display_text[:3800] + "\n\n...(继续生成中)"
                        
                        # Use safe helper to avoid "Message not modified" and "Can't parse entities"
                        last_displayed_text = await safe_edit_message(
                            context,
                            chat_id=placeholder_message.chat_id,
                            message_id=placeholder_message.message_id,
                            text=display_text,
                            parse_mode=ParseMode.MARKDOWN,
                            last_text=last_displayed_text
                        )
                    except Exception:
                        pass
                last_update_time = time.time()

        # 4. Final Polish
        final_text = full_response.strip()
        if not final_text:
            final_text = f"⚠️ **Neural Core returned no text.**\n(Chunks received: {full_response.count('') if 'full_response' in locals() else 'N/A'})"
        
        # Split logic (Telegram 4096 limit)
        MAX_LEN = 4000
        chunks = [final_text[i:i+MAX_LEN] for i in range(0, len(final_text), MAX_LEN)]
        
        await safe_edit_message(
            context,
            chat_id=placeholder_message.chat_id,
            message_id=placeholder_message.message_id,
            text=chunks[0],
            parse_mode=ParseMode.MARKDOWN
        )
        
        for extra_chunk in chunks[1:]:
            await safe_reply_text(update, extra_chunk, parse_mode=ParseMode.MARKDOWN)

        # 5. Update Session ID in storage
        if current_session_id:
            session_manager.set_session_id(user_id, current_session_id)

    except Exception as e:
        logger.error(f"Runtime Panic: {e}")
        await context.bot.edit_message_text(
            chat_id=placeholder_message.chat_id,
            message_id=placeholder_message.message_id,
            text=f"🔥 **Runtime Panic**\n\n`{str(e)}`"
        )

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and send a telegram message to notify the developer."""
    logger.warning(f"Update {update} caused error {context.error}")
    # Common network errors can be logged as warnings to avoid spamming the log with Tracebacks
    if "httpx.ReadError" in str(context.error) or "NetworkError" in str(context.error):
         logger.warning("Transient network error detected, bot will retry...")
    else:
         logger.error("Exception while handling an update:", exc_info=context.error)

def run_telegram_bot():
    """Entry Point for the Bot"""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("Snapshot Failed: TELEGRAM_BOT_TOKEN missing.")
        return

    logger.info("Initializing SoulBot Shell...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Add error handler
    application.add_error_handler(error_handler)

    logger.info("SoulBot is Listening via Long Polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
