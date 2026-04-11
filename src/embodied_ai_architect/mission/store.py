"""Mission persistence store (issue #52).

Persists Mission entities as JSON files in `.branes/missions/<id>/manifest.json`.
Mirrors the SessionStore pattern with atomic writes (temp → rename).

Usage:
    from embodied_ai_architect.mission import MissionStore, Mission

    store = MissionStore()
    mission = Mission(name="Drone Perception", goal="30fps at <5W")
    store.save(mission)

    loaded = store.load(mission.id)
    all_missions = store.list_missions()
    store.delete(mission.id)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from embodied_ai_architect.mission.models import Mission

logger = logging.getLogger(__name__)

# Default storage root — `.branes/missions/` in the current working directory.
# Mirrors `.branes/specs/` from the specs subsystem.
DEFAULT_MISSIONS_DIR = Path(".branes") / "missions"


class MissionStore:
    """JSON-based persistence for Mission entities.

    Each mission gets its own directory: `<root>/<mission_id>/manifest.json`.
    Writes are atomic (write to temp file, then rename) to prevent readers
    from seeing truncated JSON.
    """

    def __init__(self, root: Optional[str | Path] = None) -> None:
        self._root = Path(root) if root else DEFAULT_MISSIONS_DIR
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def save(self, mission: Mission) -> str:
        """Save a mission to disk. Returns the mission ID.

        Creates the mission directory if it doesn't exist. Writes
        atomically via a temp file to prevent corrupt reads.
        """
        mission.touch()
        mission_dir = self._root / mission.id
        mission_dir.mkdir(parents=True, exist_ok=True)

        manifest = mission_dir / "manifest.json"
        tmp = manifest.with_suffix(".tmp")

        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(mission.model_dump(), f, indent=2, default=str)
        tmp.rename(manifest)

        logger.debug("Saved mission %s to %s", mission.id, manifest)
        return mission.id

    def load(self, identifier: str) -> Optional[Mission]:
        """Load a mission by ID or name. Returns None if not found.

        Tries exact ID match first (directory lookup). If that fails,
        scans all missions for a matching name field.
        """
        # Try exact ID match
        manifest = self._root / identifier / "manifest.json"
        if manifest.exists():
            with open(manifest, encoding="utf-8") as f:
                data = json.load(f)
            return Mission(**data)

        # Fall back to name search
        return self._find_by_name(identifier)

    def _find_by_name(self, name: str) -> Optional[Mission]:
        """Scan missions for one matching the given name."""
        for manifest in self._list_manifests():
            try:
                with open(manifest, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("name") == name:
                    return Mission(**data)
            except (json.JSONDecodeError, OSError):
                continue
        return None

    def load_latest(self) -> Optional[Mission]:
        """Load the most recently updated mission."""
        missions = self._list_manifests()
        if not missions:
            return None

        # Sort by file modification time (most recent first)
        missions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        with open(missions[0], encoding="utf-8") as f:
            data = json.load(f)
        return Mission(**data)

    def list_missions(self) -> list[dict]:
        """List all missions with summary info.

        Returns a list of dicts with id, name, status, goal, created_at,
        updated_at — sorted by most recently updated first.
        """
        summaries = []
        for manifest in self._list_manifests():
            try:
                with open(manifest, encoding="utf-8") as f:
                    data = json.load(f)
                summaries.append(
                    {
                        "id": data.get("id", manifest.parent.name),
                        "name": data.get("name", ""),
                        "status": data.get("status", "draft"),
                        "goal": data.get("goal", ""),
                        "platform_id": data.get("platform_id"),
                        "use_case": data.get("use_case"),
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                    }
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read mission %s: %s", manifest, e)

        summaries.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return summaries

    def delete(self, identifier: str) -> bool:
        """Delete a mission by ID or name. Returns True if deleted."""
        resolved_id = self._resolve_id(identifier)
        if not resolved_id:
            return False

        mission_dir = self._root / resolved_id
        if not mission_dir.is_dir():
            return False

        for f in mission_dir.iterdir():
            f.unlink()
        mission_dir.rmdir()

        logger.debug("Deleted mission %s", resolved_id)
        return True

    def exists(self, identifier: str) -> bool:
        """Check if a mission exists by ID or name."""
        if (self._root / identifier / "manifest.json").exists():
            return True
        return self._find_by_name(identifier) is not None

    def _resolve_id(self, identifier: str) -> Optional[str]:
        """Resolve an identifier (ID or name) to a mission ID."""
        if (self._root / identifier / "manifest.json").exists():
            return identifier
        mission = self._find_by_name(identifier)
        return mission.id if mission else None

    def _list_manifests(self) -> list[Path]:
        """Find all manifest.json files under the root."""
        return list(self._root.glob("*/manifest.json"))
