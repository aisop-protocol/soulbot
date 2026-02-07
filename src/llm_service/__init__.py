"""
LLM Service - ACP Integration Only (Minified)

Provides integration for Claude CLI and Gemini CLI via LiteLLM.
"""

from .litellm_acp_provider import register_acp_providers

__all__ = [
    "register_acp_providers",
]
