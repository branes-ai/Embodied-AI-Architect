"""Bridge between SpecStore and MissionStore (issue #66).

When a spec is created or modified, the bridge syncs the spec data
into the corresponding Mission's `spec` field. This keeps the spec's
event-sourced versioning intact while connecting specs to missions.

The bridge creates a Mission for each spec (using the spec name as
the mission ID) and keeps `mission.spec` in sync with the spec state.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def sync_spec_to_mission(spec_name: str, spec_data: dict[str, Any]) -> None:
    """Sync a spec's current state to its corresponding Mission.

    Creates the mission if it doesn't exist. Updates mission.spec with
    the full spec data (model_dump output from SystemSpec).
    """
    try:
        from embodied_ai_architect.mission.models import Mission
        from embodied_ai_architect.mission.store import MissionStore

        store = MissionStore()
        mission = store.load(spec_name)

        if not mission:
            mission = Mission(
                id=spec_name,
                name=spec_name,
                description=spec_data.get("description", ""),
                goal=spec_data.get("description", ""),
            )

        mission.spec = spec_data

        # Extract constraints from spec if present
        _sync_constraints(mission, spec_data)

        store.save(mission)
        logger.debug("Synced spec '%s' to mission", spec_name)
    except Exception:
        # Sync is best-effort — never block spec operations
        logger.debug("Failed to sync spec to mission", exc_info=True)


def _sync_constraints(mission: Any, spec_data: dict[str, Any]) -> None:
    """Extract constraint-like fields from spec into mission.constraints."""
    constraints: dict[str, Any] = dict(mission.constraints or {})

    # Power constraints
    power = spec_data.get("power") or {}
    if isinstance(power, dict):
        if power.get("power_budget_watts") is not None:
            constraints["max_power_watts"] = power["power_budget_watts"]
        if power.get("compute_power_watts") is not None:
            constraints.setdefault("max_power_watts", power["compute_power_watts"])

    # Perception latency
    perception = spec_data.get("perception") or {}
    if isinstance(perception, dict):
        if perception.get("max_latency_ms") is not None:
            constraints["max_latency_ms"] = perception["max_latency_ms"]

    # Compute constraints
    compute = spec_data.get("compute") or {}
    if isinstance(compute, dict):
        if compute.get("max_tdp_watts") is not None:
            constraints.setdefault("max_power_watts", compute["max_tdp_watts"])

    if constraints:
        mission.constraints = constraints


def load_spec_from_mission(mission_id: str) -> Optional[dict[str, Any]]:
    """Load a spec from a Mission's spec field.

    Returns the spec dict or None if the mission doesn't exist
    or has no spec data.
    """
    try:
        from embodied_ai_architect.mission.store import MissionStore

        store = MissionStore()
        mission = store.load(mission_id)
        if mission and mission.spec:
            return mission.spec
    except Exception:
        logger.debug("Failed to load spec from mission", exc_info=True)
    return None


def migrate_specs_to_missions() -> list[str]:
    """Auto-migrate all existing specs to missions.

    For each spec in SpecStore that doesn't have a corresponding mission,
    create one and sync the spec data.

    Returns list of migrated spec names.
    """
    try:
        from embodied_ai_architect.specs.store import SpecStore

        spec_store = SpecStore()
        specs = spec_store.list_specs()
        migrated = []

        for name in specs.get("specs", {}):
            try:
                spec_obj = spec_store.get(name)
                spec_data = spec_obj.model_dump(exclude_none=True, mode="json")
                sync_spec_to_mission(name, spec_data)
                migrated.append(name)
            except Exception:
                logger.warning("Failed to migrate spec '%s' to mission", name, exc_info=True)

        return migrated
    except Exception:
        logger.debug("Failed to migrate specs", exc_info=True)
        return []
