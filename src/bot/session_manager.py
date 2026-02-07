import os
import json
from src.config import logger

class SessionManager:
    def __init__(self, storage_dir="sessions"):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
        self.session_file = os.path.join(self.storage_dir, "session_map.json")
        self.sessions = self._load()

    def _load(self):
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to read session map: {e}")
        return {}

    def _save(self):
        try:
            with open(self.session_file, "w", encoding="utf-8") as f:
                json.dump(self.sessions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session map: {e}")

    def get_session_id(self, user_id):
        # Telegram ID as string
        return self.sessions.get(str(user_id))

    def set_session_id(self, user_id, session_id):
        self.sessions[str(user_id)] = session_id
        self._save()

    def clear_session(self, user_id):
        if str(user_id) in self.sessions:
            del self.sessions[str(user_id)]
            self._save()
            logger.info(f"Session ID cleared for {user_id}")
