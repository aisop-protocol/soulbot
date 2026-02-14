from src.config import CLAUDE_CLI, GEMINI_CLI, OPENCODE_CLI, CLAUDE_MODEL, GEMINI_MODEL, OPENCODE_MODEL, SYSTEM_PROMPT, ENABLE_FALLBACK, logger
import litellm
from typing import Optional
from src.llm_service.litellm_acp_provider import register_acp_providers


class AIClient:
    def __init__(self):
        self._setup_provider()

    def _setup_provider(self):
        # Register ACP Providers (Claude CLI, Gemini CLI)
        try:
            register_acp_providers()
            logger.info("ACP Providers registered.")
        except Exception as e:
            logger.warning(f"Failed to register ACP providers: {e}")

        # Determine Model String based on CLI flags
        # Priority: OpenCode > Claude > Gemini
        if OPENCODE_CLI:
            self.model_name = OPENCODE_MODEL
        elif CLAUDE_CLI:
            self.model_name = CLAUDE_MODEL
        elif GEMINI_CLI:
            self.model_name = GEMINI_MODEL
        else:
            self.model_name = "echo" # Special internal flag for Echo
            logger.warning("No CLI provider enabled. Defaulting to Echo mode.")
        
        logger.info(f"AI Client configured with model: {self.model_name}")

    def _get_fallback_model(self, current_model: str) -> str:
        """Determines the fallback model (cross-provider) based on the current one."""
        current = current_model.lower()
        if "opencode" in current:
            return "claude-acp/sonnet"
        elif "claude" in current:
            return "gemini-acp/gemini-2.5-flash"
        elif "gemini" in current:
            return "claude-acp/sonnet"
        return None

    async def get_response(self, user_input: str, session_id: str = None, system_prompt: str = None) -> tuple[str, str]:
        """
        Get response and session_id from LLM via LiteLLM.
        Returns (content, session_id)
        """
        if self.model_name == "echo":
             return f"[Echo] {user_input}", session_id

        messages = [{"role": "user", "content": user_input}]
        
        if system_prompt:
             messages.insert(0, {"role": "system", "content": system_prompt})
        elif SYSTEM_PROMPT:
             messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        try:
            logger.info(f"Sending prompt to {self.model_name} (session_id: {session_id})...")
            response = await litellm.acompletion(
                model=self.model_name,
                messages=messages,
                extra_body={"session_id": session_id},
            )
            content = response.choices[0].message.content
            new_session_id = (response._hidden_params.get("session_id") if hasattr(response, "_hidden_params") and response._hidden_params else None) or session_id
            return content, new_session_id
        except Exception as e:
            logger.warning(f"Error calling LLM ({self.model_name}): {e}")
            
            # Fallback Logic
            if ENABLE_FALLBACK:
                fallback_model = self._get_fallback_model(self.model_name)
                # Check if we should fallback (and haven't already tried the fallback model)
                if fallback_model and fallback_model != self.model_name:
                    logger.warning(f"[Fallback] Switching to {fallback_model} due to error: {e}")
                    try:
                        response = await litellm.acompletion(
                            model=fallback_model,
                            messages=messages,
                            extra_body={"session_id": session_id},
                        )
                        fallback_content = response.choices[0].message.content
                        return f"⚠️ *[{self.model_name} unavailable, switched to {fallback_model}]*\n\n{fallback_content}", session_id
                    except Exception as fallback_error:
                        error_msg = (
                            f"⚠️ **Service Unavailable (All Models Failed)**\n\n"
                            f"**Primary ({self.model_name})**: {e}\n"
                            f"**Fallback ({fallback_model})**: {fallback_error}\n\n"
                            f"Please check quotas or configuration."
                        )
                        logger.error(error_msg)
                        return error_msg, session_id

            return f"Error interacting with AI: {e}", session_id

    async def get_streaming_response(self, user_input: str, session_id: str = None, system_prompt: str = None, user_id: str = None):
        """
        Get streaming response from LLM via LiteLLM.
        Yields (content, session_id)

        Args:
            user_input: User message
            session_id: Session ID for continuity
            system_prompt: System prompt
            user_id: User ID for provider-specific session persistence (e.g., OpenCode)
        """
        messages = [{"role": "user", "content": user_input}]

        if system_prompt:
             messages.insert(0, {"role": "system", "content": system_prompt})
        elif SYSTEM_PROMPT:
             messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        if self.model_name == "echo":
             yield f"[Echo] {user_input}", session_id
             return

        try:
            logger.info(f"Sending streaming prompt to {self.model_name} (session_id: {session_id}, user_id: {user_id})...")
            # 通过 extra_body 传递自定义参数（litellm 会传递给 custom provider）
            response = await litellm.acompletion(
                model=self.model_name,
                messages=messages,
                stream=True,
                extra_body={"session_id": session_id, "user_id": user_id},
            )
            chunk_count = 0
            async for chunk in response:
                chunk_count += 1
                content = ""
                new_session_id = None

                # Debug logging for first few chunks (diagnose streaming & session_id)
                if chunk_count <= 3:
                    logger.debug(f"[Stream] Chunk #{chunk_count} type={type(chunk).__name__}")
                    if hasattr(chunk, 'choices') and chunk.choices:
                        delta = chunk.choices[0].delta
                        logger.debug(f"[Stream] delta.content={getattr(delta, 'content', None)!r}, "
                                     f"delta.provider_specific_fields={getattr(delta, 'provider_specific_fields', None)!r}")
                
                # Robust chunk processing
                if isinstance(chunk, dict):
                    content = chunk.get("text") or chunk.get("content", "")
                    # provider_specific_fields or similar
                    psf = chunk.get("provider_specific_fields")
                    if psf and isinstance(psf, dict):
                        new_session_id = psf.get("session_id")
                else:
                    # Standard LiteLLM chunk object (ModelResponseStream)
                    try:
                        if hasattr(chunk, 'choices') and chunk.choices:
                            delta = chunk.choices[0].delta
                            content = getattr(delta, "content", "") or ""
                            # CRITICAL: Extract session_id from delta.provider_specific_fields
                            # LiteLLM's CustomStreamWrapper recreates chunks internally (dict→object),
                            # which LOSES setattr attributes. Only proper Delta Pydantic fields survive.
                            # delta.provider_specific_fields IS preserved; chunk-level attributes are NOT.
                            psf = getattr(delta, "provider_specific_fields", None)
                            if isinstance(psf, dict):
                                new_session_id = psf.get("session_id")
                    except Exception:
                        pass

                    # Fallback: try other locations (may work for some LiteLLM versions)
                    if not new_session_id:
                        try:
                            new_session_id = getattr(chunk, "session_id", None)
                            if not new_session_id and hasattr(chunk, "_hidden_params"):
                                new_session_id = chunk._hidden_params.get("session_id")
                            if not new_session_id and hasattr(chunk, "provider_specific_fields"):
                                psf = chunk.provider_specific_fields
                                if isinstance(psf, dict):
                                    new_session_id = psf.get("session_id")
                        except Exception:
                            pass
                
                if content or new_session_id:
                    if new_session_id and chunk_count <= 3:
                        logger.info(f"[Stream] session_id captured from chunk #{chunk_count}: {new_session_id[:20]}...")
                    yield content, new_session_id

            logger.info(f"Streaming finished. Total chunks: {chunk_count}")
        except Exception as e:
            logger.warning(f"Error calling streaming LLM ({self.model_name}): {e}")
            
            # Fallback Logic
            should_fallback = False
            fallback_model = None
            
            if ENABLE_FALLBACK:
                fallback_model = self._get_fallback_model(self.model_name)
                if fallback_model and fallback_model != self.model_name:
                    should_fallback = True
            
            if should_fallback:
                logger.warning(f"[Fallback] Switching stream to {fallback_model} due to error: {e}")
                # Notify user about the fallback
                yield f"⚠️ *[{self.model_name} unavailable, switched to {fallback_model}]*\n\n", None
                try:
                    response = await litellm.acompletion(
                        model=fallback_model,
                        messages=messages,
                        stream=True,
                        extra_body={"session_id": session_id, "user_id": user_id},
                    )
                    async for chunk in response:
                        content = ""
                        if hasattr(chunk, "choices") and chunk.choices:
                            content = getattr(chunk.choices[0].delta, "content", "") or ""
                        if content:
                            yield content, None
                except Exception as fallback_error:
                    error_msg = (
                        f"⚠️ **Service Unavailable (All Models Failed)**\n\n"
                        f"**Primary ({self.model_name})**: {e}\n"
                        f"**Fallback ({fallback_model})**: {fallback_error}\n\n"
                        f"Please check quotas or configuration."
                    )
                    logger.error(error_msg)
                    yield error_msg, None
            else:
                yield f"Error interacting with AI: {e}", None
