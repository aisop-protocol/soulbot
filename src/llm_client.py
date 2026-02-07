from src.config import CLAUDE_CLI, GEMINI_CLI, CLAUDE_MODEL, GEMINI_MODEL, SYSTEM_PROMPT, ENABLE_FALLBACK, logger
import litellm
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
        if CLAUDE_CLI:
            self.model_name = CLAUDE_MODEL
        elif GEMINI_CLI:
            self.model_name = GEMINI_MODEL
        else:
            self.model_name = "echo" # Special internal flag for Echo
            logger.warning("No CLI provider enabled. Defaulting to Echo mode.")
        
        logger.info(f"AI Client configured with model: {self.model_name}")

    def _get_fallback_model(self, current_model: str) -> str:
        """Determines the fallback model based on the current one."""
        if "gemini" in current_model.lower():
            return "gemini-acp/gemini-2.5-flash"
        elif "claude" in current_model.lower():
            return "claude-acp/sonnet"
        return None

    async def get_response(self, prompt: str, session_id: str = None, system_prompt: str = None) -> tuple[str, str]:
        """
        Get response and session_id from LLM via LiteLLM.
        Returns (content, session_id)
        """
        if self.model_name == "echo":
             return f"[Echo] {prompt}"

        messages = [{"role": "user", "content": prompt}]
        
        if system_prompt:
             messages.insert(0, {"role": "system", "content": system_prompt})
        elif SYSTEM_PROMPT:
             messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        try:
            logger.info(f"Sending prompt to {self.model_name} (session_id: {session_id})...")
            response = await litellm.acompletion(
                model=self.model_name,
                messages=messages,
                session_id=session_id
            )
            content = response.choices[0].message.content
            new_session_id = response._hidden_params.get("session_id") if hasattr(response, "_hidden_params") else session_id
            return content, new_session_id
        except Exception as e:
            logger.warning(f"Error calling LLM ({self.model_name}): {e}")
            
            # Fallback Logic
            if ENABLE_FALLBACK:
                fallback_model = self._get_fallback_model(self.model_name)
                # Check if we should fallback (and haven't already tried the fallback model)
                if fallback_model and fallback_model != self.model_name:
                    logger.warning(f"[Fallback] Switching to {fallback_model} due to error...")
                    try:
                        response = await litellm.acompletion(
                            model=fallback_model,
                            messages=messages
                        )
                        return response.choices[0].message.content, session_id
                    except Exception as fallback_error:
                        error_msg = (
                            f"⚠️ **Service Unavailable (All Models Failed)**\n\n"
                            f"**Primary ({self.model_name})**: {e}\n"
                            f"**Fallback ({fallback_model})**: {fallback_error}\n\n"
                            f"Please check quotas or configuration."
                        )
                        logger.error(error_msg)
                        return error_msg

            return f"Error interacting with AI: {e}", session_id

    async def get_streaming_response(self, prompt: str, session_id: str = None, system_prompt: str = None):
        """
        Get streaming response from LLM via LiteLLM.
        Yields (content, session_id)
        """
        messages = [{"role": "user", "content": prompt}]
        
        if system_prompt:
             messages.insert(0, {"role": "system", "content": system_prompt})
        elif SYSTEM_PROMPT:
             messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        if self.model_name == "echo":
             yield f"[Echo] {prompt}", session_id
             return

        try:
            logger.info(f"Sending streaming prompt to {self.model_name} (session_id: {session_id})...")
            response = await litellm.acompletion(
                model=self.model_name,
                messages=messages,
                stream=True,
                session_id=session_id
            )
            chunk_count = 0
            async for chunk in response:
                chunk_count += 1
                content = ""
                new_session_id = None
                
                # Debug logging for chunk type
                # logger.debug(f"Chunk type: {type(chunk)}")
                
                # Robust chunk processing
                if isinstance(chunk, dict):
                    content = chunk.get("text") or chunk.get("content", "")
                    # provider_specific_fields or similar
                    psf = chunk.get("provider_specific_fields")
                    if psf and isinstance(psf, dict):
                        new_session_id = psf.get("session_id")
                else:
                    # Standard LiteLLM chunk object
                    try:
                        if hasattr(chunk, 'choices') and chunk.choices:
                            delta = chunk.choices[0].delta
                            content = getattr(delta, "content", "") or ""
                    except Exception:
                        pass
                    
                    # Try to extract session_id from various possible locations in LiteLLM object
                    try:
                        # 1. Try generic dictionary-like access if supported
                        if not new_session_id:
                            new_session_id = getattr(chunk, "session_id", None)
                        
                        # 2. Try looking into hidden params or extra fields
                        if not new_session_id and hasattr(chunk, "_hidden_params"):
                            new_session_id = chunk._hidden_params.get("session_id")
                            
                        # 3. Try provider_specific_fields if passed through as attribute
                        if not new_session_id and hasattr(chunk, "provider_specific_fields"):
                            psf = chunk.provider_specific_fields
                            if isinstance(psf, dict):
                                new_session_id = psf.get("session_id")
                    except Exception:
                        pass
                
                if content or new_session_id:
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
                logger.warning(f"[Fallback] Switching stream to {fallback_model} due to error...")
                try:
                    response = await litellm.acompletion(
                        model=fallback_model,
                        messages=messages,
                        stream=True
                    )
                    async for chunk in response:
                        content = chunk.choices[0].delta.content
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
