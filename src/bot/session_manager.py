import os
import json
import threading
from datetime import datetime
from src.config import logger, PROJECT_ROOT


class SessionManager:
    """
    Unified Session Manager

    每个 CLI provider (claude / gemini / opencode) 全局只有一个 session。
    所有接口（Telegram、Web）和所有用户共享同一个 session，统一上下文。

    跨进程安全：Web (Streamlit) 和 Telegram 运行在不同进程中，
    通过文件系统共享 session 状态。每次读取前检查文件是否被其他进程修改。

    存储格式 (session_map.json):
    {
        "claude": {"session_id": "session-uuid-xxx", "created_at": "2026-02-13 13:48:56"},
        "gemini": {"session_id": "session-uuid-yyy", "created_at": "2026-02-13 14:02:11"},
        "opencode": {"session_id": "session-uuid-zzz", "created_at": "2026-02-13 14:10:33"}
    }
    """

    _instance = None
    _init_lock = threading.Lock()

    @classmethod
    def get_instance(cls, storage_dir: str = "sessions") -> "SessionManager":
        """获取单例（进程内共享同一个实例）"""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls(storage_dir=storage_dir)
        return cls._instance

    def __init__(self, storage_dir: str = "sessions"):
        self._lock = threading.Lock()
        self._file_mtime = 0.0  # 上次读取时的文件修改时间

        # Resolve relative to project root
        if not os.path.isabs(storage_dir):
            storage_dir = os.path.join(PROJECT_ROOT, storage_dir)
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        self.session_file = os.path.join(self.storage_dir, "session_map.json")
        self.sessions, migrated = self._load_from_disk()
        if migrated:
            self._save()
        logger.info(f"[SessionManager] Initialized: {self.session_file}, providers: {list(self.sessions.keys())}")

    def _load_from_disk(self) -> tuple:
        """从磁盘读取 session 文件并更新 mtime 缓存

        Returns:
            (sessions_dict, needs_migration)
        """
        if not os.path.exists(self.session_file):
            self._file_mtime = 0.0
            return {}, False

        try:
            self._file_mtime = os.path.getmtime(self.session_file)
            with open(self.session_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            migrated = {}
            needs_migration = False

            for key, value in data.items():
                # 兼容旧格式1：key 包含 ":" (provider:user_hash)
                provider = key.split(":")[0] if ":" in key else key
                if ":" in key:
                    needs_migration = True
                    logger.info(f"[SessionManager] Migrated old key '{key}' -> provider '{provider}'")

                # 兼容旧格式2：value 是纯字符串而非 dict
                if isinstance(value, str):
                    needs_migration = True
                    value = {"session_id": value, "created_at": None}
                    logger.info(f"[SessionManager] Migrated plain string -> dict for '{provider}'")

                if provider not in migrated:
                    migrated[provider] = value

            if needs_migration:
                logger.info(f"[SessionManager] Migration complete, saving new format")

            return migrated, needs_migration
        except Exception as e:
            logger.error(f"Failed to read session map: {e}")
            return {}, False

    def _sync_from_disk(self):
        """
        检查文件是否被其他进程修改，如果是则重新加载。

        Web (Streamlit subprocess) 和 Telegram (main process) 各自有独立的内存缓存。
        通过比较文件 mtime 来检测跨进程写入，确保读取到最新数据。
        """
        try:
            if not os.path.exists(self.session_file):
                return
            current_mtime = os.path.getmtime(self.session_file)
            if current_mtime > self._file_mtime:
                old_providers = list(self.sessions.keys())
                self.sessions, migrated = self._load_from_disk()
                if migrated:
                    self._save()
                new_providers = list(self.sessions.keys())
                if old_providers != new_providers:
                    logger.info(f"[SessionManager] Reloaded from disk (external change): {old_providers} -> {new_providers}")
        except Exception as e:
            logger.debug(f"[SessionManager] Sync check error: {e}")

    def _save(self):
        """保存到磁盘并更新 mtime 缓存"""
        try:
            tmp_file = self.session_file + ".tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(self.sessions, f, ensure_ascii=False, indent=2)
            os.replace(tmp_file, self.session_file)
            # 更新 mtime 缓存，避免自己写入后再触发 reload
            self._file_mtime = os.path.getmtime(self.session_file)
        except Exception as e:
            logger.error(f"Failed to save session map: {e}")

    def get_session_id(self, provider: str) -> str:
        """
        获取指定 provider 的 session ID

        每次调用前检查文件是否被其他进程修改，确保跨进程一致性。

        Args:
            provider: AI provider ('claude', 'gemini', 'opencode')

        Returns:
            session ID or None
        """
        with self._lock:
            self._sync_from_disk()
            entry = self.sessions.get(provider)
            sid = entry.get("session_id") if isinstance(entry, dict) else entry
            logger.info(f"[SessionManager] get_session_id('{provider}') -> {sid[:20] + '...' if sid else 'None'} "
                        f"[file={self.session_file}, all={list(self.sessions.keys())}]")
            return sid

    def get_session_info(self, provider: str) -> dict:
        """
        获取指定 provider 的完整 session 信息

        Returns:
            {"session_id": "xxx", "created_at": "2026-02-13 13:48:56"} 或 None
        """
        with self._lock:
            self._sync_from_disk()
            entry = self.sessions.get(provider)
            if isinstance(entry, dict):
                return entry.copy()
            elif isinstance(entry, str):
                return {"session_id": entry, "created_at": None}
            return None

    def set_session_id(self, provider: str, session_id: str):
        """
        保存指定 provider 的 session ID

        Args:
            provider: AI provider ('claude', 'gemini', 'opencode')
            session_id: session ID
        """
        with self._lock:
            # 先同步，避免覆盖其他进程的写入
            self._sync_from_disk()
            self.sessions[provider] = {
                "session_id": session_id,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._save()
            logger.info(f"[SessionManager] SET session for '{provider}': {session_id[:20]}... "
                        f"[file={self.session_file}]")

    def clear_session(self, provider: str = None):
        """
        清除 session

        Args:
            provider: 指定 provider 则只清该 provider，None 则清除全部
        """
        with self._lock:
            self._sync_from_disk()
            if provider:
                if provider in self.sessions:
                    del self.sessions[provider]
                    self._save()
                    logger.info(f"[SessionManager] Cleared session for {provider}")
            else:
                self.sessions.clear()
                self._save()
                logger.info(f"[SessionManager] Cleared all sessions")

    def list_all_sessions(self) -> dict:
        """列出所有 provider 的 session"""
        with self._lock:
            self._sync_from_disk()
            return self.sessions.copy()

    def get_active_providers(self) -> list:
        """获取有活跃 session 的 provider 列表"""
        with self._lock:
            self._sync_from_disk()
            return list(self.sessions.keys())
