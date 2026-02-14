# Streamlit utilities
# Note: These imports require streamlit to be installed
# telegram_controller can be imported directly without streamlit

__all__ = [
    "UserManager",
    "get_user_id",
    "get_app_hash",
    "init_session_state",
    "get_session_manager",
    "get_session_id",
    "save_session_id",
    "clear_current_session",
    "get_active_providers"
]


def __getattr__(name):
    """Lazy import to avoid requiring streamlit for telegram_controller"""
    if name in ("UserManager", "get_user_id", "get_app_hash"):
        from .user_manager import UserManager, get_user_id, get_app_hash
        return {"UserManager": UserManager, "get_user_id": get_user_id, "get_app_hash": get_app_hash}[name]

    if name in ("init_session_state", "get_session_manager", "get_session_id",
                "save_session_id", "clear_current_session", "get_active_providers"):
        from .session_state import (
            init_session_state,
            get_session_manager,
            get_session_id,
            save_session_id,
            clear_current_session,
            get_active_providers
        )
        return {
            "init_session_state": init_session_state,
            "get_session_manager": get_session_manager,
            "get_session_id": get_session_id,
            "save_session_id": save_session_id,
            "clear_current_session": clear_current_session,
            "get_active_providers": get_active_providers
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
