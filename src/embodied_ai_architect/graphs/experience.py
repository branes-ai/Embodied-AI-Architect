"""Experience cache for learning from past SoC design sessions.

Stores completed design episodes in a SQLite database for later retrieval.
Episodes include the goal, constraints, architecture chosen, PPA achieved,
and optimization trace — enabling the system to learn from past designs.

Usage:
    from embodied_ai_architect.graphs.experience import ExperienceCache, DesignEpisode

    cache = ExperienceCache()  # default: ~/.embodied-ai/experience.db
    cache.save(episode)
    similar = cache.search_similar(use_case="delivery_drone")
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class DesignEpisode(BaseModel):
    """A complete SoC design session record."""

    episode_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    goal: str = ""
    use_case: str = ""
    platform: str = ""
    constraints: dict[str, Any] = Field(default_factory=dict)
    architecture_chosen: str = ""
    hardware_selected: str = ""
    ppa_achieved: dict[str, Any] = Field(default_factory=dict)
    constraint_verdicts: dict[str, str] = Field(default_factory=dict)
    outcome_score: float = 0.0  # 1.0 = all PASS, 0.0 = failures remain
    iterations_used: int = 0
    key_decisions: list[str] = Field(default_factory=list)
    lessons_learned: list[str] = Field(default_factory=list)
    optimization_trace: list[dict[str, Any]] = Field(default_factory=list)
    # KPU micro-architecture fields (Phase 3)
    kpu_config_name: Optional[str] = None
    kpu_process_nm: Optional[int] = None
    floorplan_area_mm2: Optional[float] = None
    bandwidth_balanced: Optional[bool] = None
    rtl_modules_generated: int = 0
    rtl_total_cells: int = 0
    problem_fingerprint: str = ""

    def compute_fingerprint(self) -> str:
        """Compute a deterministic fingerprint from the problem definition.

        Fingerprints are based on use_case + platform + sorted constraint keys.
        """
        parts = [self.use_case, self.platform]
        for k in sorted(self.constraints.keys()):
            v = self.constraints[k]
            if v is not None:
                parts.append(f"{k}={v}")
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ExperienceCache:
    """SQLite-backed cache for design episodes.

    Stores episodes as JSON blobs indexed by use_case, platform, and fingerprint.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize the experience cache.

        Args:
            db_path: Path to SQLite database. None = default location.
                     Use ":memory:" for testing.
        """
        if db_path is None:
            db_dir = Path.home() / ".embodied-ai"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "experience.db")

        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                use_case TEXT NOT NULL,
                platform TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                outcome_score REAL NOT NULL,
                iterations_used INTEGER NOT NULL,
                data TEXT NOT NULL
            )
        """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_use_case ON episodes(use_case)
        """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_fingerprint ON episodes(fingerprint)
        """
        )
        self._conn.commit()

    def save(self, episode: DesignEpisode) -> str:
        """Save an episode to the cache.

        Args:
            episode: The design episode to save.

        Returns:
            The episode_id.
        """
        if not episode.problem_fingerprint:
            episode.problem_fingerprint = episode.compute_fingerprint()

        self._conn.execute(
            """
            INSERT OR REPLACE INTO episodes
                (episode_id, timestamp, use_case, platform, fingerprint,
                 outcome_score, iterations_used, data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode.episode_id,
                episode.timestamp,
                episode.use_case,
                episode.platform,
                episode.problem_fingerprint,
                episode.outcome_score,
                episode.iterations_used,
                json.dumps(episode.model_dump()),
            ),
        )
        self._conn.commit()
        return episode.episode_id

    def load(self, episode_id: str) -> Optional[DesignEpisode]:
        """Load an episode by ID.

        Args:
            episode_id: The episode to load.

        Returns:
            The DesignEpisode, or None if not found.
        """
        row = self._conn.execute(
            "SELECT data FROM episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()

        if row is None:
            return None
        return DesignEpisode(**json.loads(row[0]))

    def search_similar(
        self,
        use_case: str = "",
        platform: str = "",
        fingerprint: str = "",
        limit: int = 10,
    ) -> list[DesignEpisode]:
        """Search for similar episodes.

        Exact match on use_case and/or fingerprint, ordered by outcome_score descending.

        Args:
            use_case: Filter by use case (exact match).
            platform: Filter by platform (exact match).
            fingerprint: Filter by problem fingerprint (exact match).
            limit: Maximum results to return.

        Returns:
            List of matching episodes, best score first.
        """
        conditions = []
        params: list[Any] = []

        if use_case:
            conditions.append("use_case = ?")
            params.append(use_case)
        if platform:
            conditions.append("platform = ?")
            params.append(platform)
        if fingerprint:
            conditions.append("fingerprint = ?")
            params.append(fingerprint)

        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        rows = self._conn.execute(
            f"SELECT data FROM episodes WHERE {where} ORDER BY outcome_score DESC LIMIT ?",
            params,
        ).fetchall()

        return [DesignEpisode(**json.loads(row[0])) for row in rows]

    def list_episodes(self, limit: int = 50) -> list[dict[str, Any]]:
        """List episodes in summary form (no full data).

        Returns:
            List of dicts with episode_id, use_case, platform, outcome_score, iterations_used.
        """
        rows = self._conn.execute(
            """
            SELECT episode_id, timestamp, use_case, platform, outcome_score, iterations_used
            FROM episodes ORDER BY timestamp DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [
            {
                "episode_id": r[0],
                "timestamp": r[1],
                "use_case": r[2],
                "platform": r[3],
                "outcome_score": r[4],
                "iterations_used": r[5],
            }
            for r in rows
        ]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
