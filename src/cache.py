"""SQLite-based persistent disk cache."""

import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Optional


class DiskCache:
    """Thread-safe SQLite persistent disk cache with TTL support."""

    def __init__(self, path: str = ".cache/cache.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )

    def _generate_key(self, namespace: str, key: str) -> str:
        combined = f"{namespace}|{key}".encode("utf-8")
        return hashlib.sha256(combined).hexdigest()

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Retrieve value from cache if present and unexpired."""
        cache_key = self._generate_key(namespace, key)
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT value, expires_at FROM cache WHERE cache_key = ?",
            (cache_key,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        val_str, expires_at = row
        if time.time() > expires_at:
            with self.conn:
                self.conn.execute("DELETE FROM cache WHERE cache_key = ?", (cache_key,))
            return None

        return json.loads(val_str)

    def set(self, namespace: str, key: str, value: Any, ttl: int = 86400) -> None:
        """Store value in cache with specified TTL in seconds."""
        cache_key = self._generate_key(namespace, key)
        expires_at = time.time() + ttl
        serialized_val = json.dumps(value)
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO cache (cache_key, value, expires_at)
                VALUES (?, ?, ?)
                """,
                (cache_key, serialized_val, expires_at),
            )

    def close(self) -> None:
        """Close SQLite database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self) -> "DiskCache":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
