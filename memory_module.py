"""
memory_module.py — Per-session conversation memory.
 
Stores the last N turns per session_id in a thread-safe in-memory dict.
Each turn tracks: role, content, emotion, and ISO timestamp.
 
For production at scale, replace the _store dict with a Redis backend —
the MemoryStore interface stays identical.
 
Thread safety: Python's GIL protects simple dict reads/writes.
For true async concurrency, wrap mutations in asyncio.Lock if needed.
"""
 
import logging
from collections import deque
from datetime import datetime
from typing import Optional
 
logger = logging.getLogger("aura.memory")
 
 
class MemoryStore:
    """
    Multi-session conversation memory using in-process deques.
 
    Parameters
    ----------
    max_turns : Maximum user+assistant pairs kept per session.
                Older turns are evicted automatically.
    """
 
    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        # session_id → deque of raw turn dicts
        self._store: dict[str, deque] = {}
 
    # ── Write ──────────────────────────────────────────────────────────────
 
    def add_turn(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        emotion: str = "neutral",
    ) -> None:
        """
        Append one exchange (user + assistant) to the session history.
        Automatically evicts oldest pair when max_turns is exceeded.
        """
        try:
            if session_id not in self._store:
                # maxlen = pairs * 2 messages each
                self._store[session_id] = deque(maxlen=self.max_turns * 2)
 
            ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            buf = self._store[session_id]
 
            buf.append({
                "role":    "user",
                "content": user_msg,
                "emotion": emotion,
                "ts":      ts,
            })
            buf.append({
                "role":    "assistant",
                "content": assistant_msg,
                "emotion": emotion,
                "ts":      ts,
            })
 
        except Exception as exc:
            logger.error("add_turn error (session=%s): %s", session_id, exc)
 
    # ── Read ───────────────────────────────────────────────────────────────
 
    def get_history(self, session_id: str) -> list[dict]:
        """
        Return message history for an LLM prompt array.
 
        Each item: {"role": "user"|"assistant", "content": str}
        Emotion and timestamp are stripped for LLM compatibility.
        """
        try:
            buf = self._store.get(session_id, deque())
            return [
                {"role": t["role"], "content": t["content"]}
                for t in buf
            ]
        except Exception as exc:
            logger.error("get_history error (session=%s): %s", session_id, exc)
            return []
 
    def get_full_history(self, session_id: str) -> list[dict]:
        """Return full turn dicts including emotion and timestamps."""
        try:
            return list(self._store.get(session_id, deque()))
        except Exception as exc:
            logger.error("get_full_history error: %s", exc)
            return []
 
    def get_recent_emotions(self, session_id: str, n: int = 5) -> list[str]:
        """Return the last `n` detected emotions for the session."""
        try:
            buf = self._store.get(session_id, deque())
            user_turns = [t for t in buf if t["role"] == "user"]
            return [t["emotion"] for t in user_turns[-n:]]
        except Exception as exc:
            logger.error("get_recent_emotions error: %s", exc)
            return []
 
    def get_last_assistant_response(self, session_id: str) -> Optional[str]:
        """Return the most recent assistant message (for repetition avoidance)."""
        try:
            buf = self._store.get(session_id, deque())
            for turn in reversed(list(buf)):
                if turn["role"] == "assistant":
                    return turn["content"]
        except Exception as exc:
            logger.error("get_last_assistant_response error: %s", exc)
        return None
 
    # ── Management ─────────────────────────────────────────────────────────
 
    def clear(self, session_id: str) -> None:
        """Delete all memory for a session."""
        try:
            self._store.pop(session_id, None)
        except Exception as exc:
            logger.error("clear error (session=%s): %s", session_id, exc)
 
    def session_count(self) -> int:
        """Return number of active sessions (for monitoring)."""
        return len(self._store)
