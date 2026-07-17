"""Persistent session storage for SoC design state.

Saves and loads SoCDesignState as JSON files, enabling design sessions
to span multiple CLI invocations and be accessed by Claude Code skills.

This is the bridge between the LangGraph execution engine and the
human-facing skill layer (/architect-loop, /architect-assess, /architect-drill).

Usage:
    from embodied_ai_architect.graphs.session_store import SessionStore

    store = SessionStore()
    store.save(state)                    # Save after each step
    state = store.load("soc_abc123")     # Resume a session
    sessions = store.list_sessions()     # List all sessions
    state = store.load_latest()          # Get most recent session
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from embodied_ai_architect.graphs.soc_state import SoCDesignState

logger = logging.getLogger(__name__)

DEFAULT_SESSION_DIR = Path.home() / ".embodied-ai" / "sessions"


class SessionStore:
    """JSON-based session persistence for SoCDesignState.

    Each session is a JSON file named by session_id. The store
    auto-creates the directory and handles serialization of all
    state fields (dicts, lists, strings, numbers — all JSON-safe).
    """

    def __init__(self, session_dir: Optional[str | Path] = None) -> None:
        self._dir = Path(session_dir) if session_dir else DEFAULT_SESSION_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def session_dir(self) -> Path:
        return self._dir

    def save(self, state: SoCDesignState) -> str:
        """Save a design session state to disk.

        Args:
            state: The SoCDesignState dict to persist.

        Returns:
            The session_id used as the file key.
        """
        session_id = state.get("session_id", "unknown")
        path = self._dir / f"{session_id}.json"

        # Add save metadata
        save_data = dict(state)
        save_data["_saved_at"] = datetime.now().isoformat()

        # Atomic write: write to temp file, then rename. Prevents readers
        # from seeing truncated JSON if they poll mid-write.
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        tmp_path.rename(path)

        logger.debug("Saved session %s to %s", session_id, path)
        return session_id

    def load(self, session_id: str) -> Optional[SoCDesignState]:
        """Load a session by ID.

        Args:
            session_id: The session identifier (e.g., "soc_abc123").

        Returns:
            The deserialized state dict, or None if not found.
        """
        path = self._dir / f"{session_id}.json"
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)

        # Remove save metadata
        data.pop("_saved_at", None)
        return data

    def load_latest(self) -> Optional[SoCDesignState]:
        """Load the most recently saved session.

        Ordered by the embedded ``_saved_at`` timestamp (microsecond ISO), which
        `save()` writes on every save. Filesystem ``st_mtime`` is too coarse to
        distinguish rapid successive saves reliably, so it is only a tie-breaker
        (then the filename) for the essentially-impossible case of equal timestamps.

        Returns:
            The deserialized state dict, or None if no sessions exist.
        """
        best_data: Optional[dict] = None
        best_key: Optional[tuple] = None
        for path in self._dir.glob("soc_*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            key = (data.get("_saved_at", ""), path.stat().st_mtime, path.name)
            if best_key is None or key > best_key:
                best_key, best_data = key, data

        if best_data is None:
            return None

        best_data.pop("_saved_at", None)
        return best_data

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all saved sessions with summary info.

        Returns:
            List of dicts with session_id, goal, status, iteration,
            created_at, saved_at — sorted by most recent first.
        """
        sessions = []
        for path in sorted(
            self._dir.glob("soc_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                with open(path) as f:
                    data = json.load(f)
                sessions.append(
                    {
                        "session_id": data.get("session_id", path.stem),
                        "goal": _truncate(data.get("goal", ""), 60),
                        "status": data.get("status", "unknown"),
                        "iteration": data.get("iteration", 0),
                        "max_iterations": data.get("max_iterations", 20),
                        "created_at": data.get("created_at", ""),
                        "saved_at": data.get("_saved_at", ""),
                        "platform": data.get("platform", ""),
                        "use_case": data.get("use_case", ""),
                    }
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read session %s: %s", path, e)

        return sessions

    def delete(self, session_id: str) -> bool:
        """Delete a saved session.

        Returns:
            True if deleted, False if not found.
        """
        path = self._dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def exists(self, session_id: str) -> bool:
        """Check if a session exists on disk."""
        return (self._dir / f"{session_id}.json").exists()


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
