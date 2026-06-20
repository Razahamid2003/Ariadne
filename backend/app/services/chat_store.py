"""Chat history store.

Purpose
-------
Saves conversations locally so the UI can offer ChatGPT-style threads without any
external database or cloud service.

What it does
------------
Creates and lists chats, stores messages, renames a chat from its first question,
returns conversation context for follow-ups, and recalls prior evidence for
continuity.

Flow
----
Each turn is appended to its chat; when answering a follow-up, the store provides
the recent messages and prior evidence so the answer generator can stay consistent
with earlier turns.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _title_from_query(query: str) -> str:
    text = " ".join((query or "").strip().split())
    if not text:
        return "New chat"
    return text[:58] + ("..." if len(text) > 58 else "")


class ChatStore:
    """Persist local chat sessions and messages in SQLite."""

    def __init__(self, logs_dir: str | Path):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.logs_dir / "chat_history.db"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT,
                    FOREIGN KEY(chat_id) REFERENCES chats(chat_id)
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_chat ON chat_messages(chat_id, created_at);")
            conn.commit()

    def create_chat(self, title: str | None = None) -> dict[str, Any]:
        chat_id = f"chat-{uuid.uuid4().hex[:12]}"
        now = _utc_now()
        clean_title = title.strip() if title and title.strip() else "New chat"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chats(chat_id, title, created_at, updated_at, deleted) VALUES (?, ?, ?, ?, 0)",
                (chat_id, clean_title, now, now),
            )
            conn.commit()
        return self.get_chat(chat_id, include_messages=False) or {"chat_id": chat_id, "title": clean_title, "created_at": now, "updated_at": now}

    def ensure_chat(self, chat_id: str | None, first_query: str | None = None) -> dict[str, Any]:
        if chat_id:
            existing = self.get_chat(chat_id, include_messages=False)
            if existing:
                return existing
        return self.create_chat(_title_from_query(first_query or ""))


    def rename_chat_if_empty(self, chat_id: str, title: str) -> None:
        clean = _title_from_query(title)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT title, (SELECT COUNT(*) FROM chat_messages m WHERE m.chat_id = chats.chat_id) AS message_count FROM chats WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
            if not row:
                return
            if int(row["message_count"] or 0) == 0 or str(row["title"] or "").lower() == "new chat":
                conn.execute("UPDATE chats SET title = ?, updated_at = ? WHERE chat_id = ?", (clean, _utc_now(), chat_id))
                conn.commit()

    def list_chats(self, limit: int = 80) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.chat_id, c.title, c.created_at, c.updated_at,
                       (SELECT COUNT(*) FROM chat_messages m WHERE m.chat_id = c.chat_id) AS message_count
                FROM chats c
                WHERE c.deleted = 0
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_chat(self, chat_id: str, include_messages: bool = True) -> dict[str, Any] | None:
        with self._connect() as conn:
            chat = conn.execute(
                """
                SELECT c.chat_id, c.title, c.created_at, c.updated_at,
                       (SELECT COUNT(*) FROM chat_messages m WHERE m.chat_id = c.chat_id) AS message_count
                FROM chats c
                WHERE c.chat_id = ? AND c.deleted = 0
                """,
                (chat_id,),
            ).fetchone()
            if not chat:
                return None
            payload = dict(chat)
            if include_messages:
                messages = conn.execute(
                    """
                    SELECT message_id, chat_id, role, content, created_at, payload_json
                    FROM chat_messages
                    WHERE chat_id = ?
                    ORDER BY created_at ASC
                    """,
                    (chat_id,),
                ).fetchall()
                payload["messages"] = [self._message_dict(row) for row in messages]
            return payload

    def delete_chat(self, chat_id: str) -> bool:
        now = _utc_now()
        with self._connect() as conn:
            cur = conn.execute("UPDATE chats SET deleted = 1, updated_at = ? WHERE chat_id = ? AND deleted = 0", (now, chat_id))
            conn.commit()
            return cur.rowcount > 0

    def add_message(self, chat_id: str, role: str, content: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        message_id = f"msg-{uuid.uuid4().hex[:14]}"
        now = _utc_now()
        payload_json = json.dumps(payload or {}, ensure_ascii=False) if payload is not None else None
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chat_messages(message_id, chat_id, role, content, created_at, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
                (message_id, chat_id, role, content or "", now, payload_json),
            )
            conn.execute("UPDATE chats SET updated_at = ? WHERE chat_id = ?", (now, chat_id))
            conn.commit()
        return {"message_id": message_id, "chat_id": chat_id, "role": role, "content": content, "created_at": now, "payload": payload or None}

    def conversation_context(self, chat_id: str, max_messages: int = 8, max_chars: int = 5000) -> str:
        messages = self._recent_messages(chat_id, max_messages=max_messages)
        lines: list[str] = []
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Ariadne"
            content = msg.get("content") or ""
            content = self._compact_answer(content)
            if content:
                lines.append(f"{role}: {content}")
        text = "\n".join(lines)
        if len(text) > max_chars:
            return text[-max_chars:]
        return text

    def prior_evidence(self, chat_id: str, max_items: int = 16) -> list[dict[str, Any]]:
        messages = self._recent_messages(chat_id, max_messages=10)
        evidence: list[dict[str, Any]] = []
        seen: set[str] = set()
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            payload = msg.get("payload") or {}
            for item in payload.get("evidence", []) or []:
                label = str(item.get("citation_label") or "")
                if not label or label in seen:
                    continue
                seen.add(label)
                evidence.append(item)
                if len(evidence) >= max_items:
                    return list(reversed(evidence))
        return list(reversed(evidence))

    def _recent_messages(self, chat_id: str, max_messages: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT message_id, chat_id, role, content, created_at, payload_json
                FROM chat_messages
                WHERE chat_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (chat_id, max_messages),
            ).fetchall()
        return [self._message_dict(row) for row in reversed(rows)]

    @staticmethod
    def _message_dict(row: sqlite3.Row) -> dict[str, Any]:
        payload = None
        raw = row["payload_json"]
        if raw:
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"raw": raw}
        return {
            "message_id": row["message_id"],
            "chat_id": row["chat_id"],
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
            "payload": payload,
        }

    @staticmethod
    def _compact_answer(text: str) -> str:
        value = (text or "").strip()
        # Keep follow-up context useful without stuffing entire source sections.
        markers = ["### Confidence", "### Missing Information"]
        for marker in markers:
            idx = value.find(marker)
            if idx >= 0:
                value = value[:idx].strip()
        return value[:1800]
