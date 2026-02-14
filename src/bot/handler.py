import time
import os
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes, Application, CommandHandler, MessageHandler, filters
from telegram.error import BadRequest
from src.config import TELEGRAM_BOT_TOKEN, logger, OPENCODE_CLI, CLAUDE_CLI, GEMINI_CLI, get_current_provider
from src.llm_client import AIClient
from src.bot.session_manager import SessionManager

# Initialize AI Client and Session Manager (singleton)
ai_client = AIClient()
session_manager = SessionManager.get_instance()

# Cache last active chat_id for crash notification (memory only, no disk)
_last_chat_id = None

# Load AISOP (shared with ai_worker.py via src.config)
from src.config import load_aisop, get_system_instruction

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
CURRENT_AISOP = load_aisop("main")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    app_name = "SoulBot"
    provider = get_current_provider()
    provider_display = {
        "opencode": "🟢 OpenCode",
        "claude": "🟣 Claude",
        "gemini": "🔵 Gemini",
        "default": "⚪ Default"
    }.get(provider, provider)

    await update.message.reply_html(
        f"⚡ <b>{app_name} is Online.</b>\n"
        f"Hi {user.mention_html()}! I am running on <b>AISOP Runtime v1.0</b>.\n"
        f"AI Provider: <b>{provider_display}</b>\n"
        f"My brain is loaded with: <code>main.aisop.json</code>\n\n"
        f"Use /clear to reset your conversation history."
    )

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear the current provider's conversation session."""
    provider = get_current_provider()

    # 清除当前 provider 的全局 session
    session_manager.clear_session(provider=provider)

    provider_name = provider.upper() if provider != "default" else "AI"
    await update.message.reply_text(f"🧼 {provider_name} conversation history cleared!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Core logic: User Input -> AISOP Runtime (LLM + AISOP) -> User Output"""
    global _last_chat_id
    _last_chat_id = update.effective_chat.id

    user_message = update.message.text
    user_name = update.effective_user.name
    logger.info(f"Received from {user_name}: ({len(user_message)} chars)")

    # 1. Prepare Interface (Typing...)
    placeholder_message = await update.message.reply_text("🧠 accessing neural pathways...")
    
    # 2. Construct System Prompt (shared logic with Web UI)
    system_instruction = get_system_instruction()

    # 3. Get Session ID (unified per-provider, shared across all users & interfaces)
    user_id = update.effective_user.id
    provider = get_current_provider()
    saved_session_id = session_manager.get_session_id(provider)  # 文件中保存的 session
    current_session_id = saved_session_id  # 会在流式过程中被更新
    logger.info(f"[Handler] User: {user_id}, Provider: {provider}, Session: {saved_session_id[:20] + '...' if saved_session_id else 'NEW'}")
    
    full_response = ""
    last_displayed_text = ""
    last_update_time = time.time()
    update_interval = 0.8 # Seconds
    first_content_shown = False  # 确保第一次内容到达后立即显示（打字机效果）

    try:
        # Pass user_input, session_id, and user_id for provider-specific session persistence
        async for chunk_text, new_session_id in ai_client.get_streaming_response(
            user_message,
            session_id=saved_session_id,
            system_prompt=system_instruction,
            user_id=str(user_id)  # 用于 OpenCode 的 session 持久化
        ):
            if chunk_text:
                full_response += chunk_text
            if new_session_id:
                current_session_id = new_session_id

            # Typewriter effect:
            # - 第一次有内容时立即更新（让用户立刻看到 AI 开始回复）
            # - 之后每 0.8s 更新一次（避免 Telegram 限速）
            should_update = False
            if not first_content_shown and full_response.strip():
                should_update = True
                first_content_shown = True
            elif time.time() - last_update_time > update_interval:
                should_update = True

            if should_update and full_response.strip():
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
            final_text = "⚠️ **Neural Core returned no text.**"
        
        # Split logic (Telegram 4096 limit)
        MAX_LEN = 4000
        chunks = [final_text[i:i+MAX_LEN] for i in range(0, len(final_text), MAX_LEN)]
        
        await safe_edit_message(
            context,
            chat_id=placeholder_message.chat_id,
            message_id=placeholder_message.message_id,
            text=chunks[0],
            parse_mode=ParseMode.MARKDOWN,
            last_text=last_displayed_text  # 避免 "Message is not modified" 错误
        )
        
        for extra_chunk in chunks[1:]:
            await safe_reply_text(update, extra_chunk, parse_mode=ParseMode.MARKDOWN)

        # 5. Update Session ID (unified per-provider)
        # 始终保存当前使用的 session_id：
        # - 之前没有 session → 保存新创建的
        # - session 相同 → resume 成功，无需重复保存
        # - session 不同 → resume 失败，旧 session 已死，保存新的（否则每次重启都会反复尝试恢复已死 session）
        if current_session_id:
            if not saved_session_id:
                logger.info(f"[Handler] New session created, saving: {current_session_id[:20]}... for {provider}")
                session_manager.set_session_id(provider, current_session_id)
            elif current_session_id != saved_session_id:
                logger.warning(
                    f"[Handler] Session changed: {saved_session_id[:20]}... -> {current_session_id[:20]}... "
                    f"(resume/load failed, saving new session)"
                )
                session_manager.set_session_id(provider, current_session_id)

    except Exception as e:
        logger.error(f"Runtime Panic: {e}")
        try:
            await context.bot.edit_message_text(
                chat_id=placeholder_message.chat_id,
                message_id=placeholder_message.message_id,
                text=f"🔥 **Runtime Panic**\n\nPlease try again later."
            )
        except Exception:
            # 忽略错误显示时的异常（如 "Message is not modified"）
            pass

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


def get_last_chat_id():
    """Get cached chat_id for crash notification"""
    return _last_chat_id
