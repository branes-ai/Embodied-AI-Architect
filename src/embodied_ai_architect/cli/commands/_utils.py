"""Shared utilities for CLI commands."""

from __future__ import annotations

from typing import Any


def load_mission_constraints(
    mission_id: str | None,
    power: float | None = None,
    latency: float | None = None,
    cost: float | None = None,
    area: float | None = None,
    process: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a mission and merge its constraints with explicit CLI flags.

    Explicit flags take precedence over mission values.
    Returns (mission_or_None, merged_constraints_dict).
    The dict has keys like 'max_power_watts', 'max_latency_ms', etc.
    """
    mission = None
    merged: dict[str, Any] = {}

    if mission_id:
        from embodied_ai_architect.mission.store import MissionStore

        store = MissionStore()
        mission = store.load(mission_id)
        if mission:
            mc = mission.constraints or {}
            merged = dict(mc)

    # Explicit flags override mission values
    if power is not None:
        merged["max_power_watts"] = power
    if latency is not None:
        merged["max_latency_ms"] = latency
    if cost is not None:
        merged["max_cost_usd"] = cost
    if area is not None:
        merged["max_area_mm2"] = area
    if process is not None:
        merged["target_process_nm"] = process

    return mission, merged


def save_mission_result(mission: Any, key: str, result: Any) -> None:
    """Append a result to the mission's optimization_history and save."""
    if mission is None:
        return
    from embodied_ai_architect.mission.store import MissionStore

    if not hasattr(mission, "optimization_history"):
        return
    entry = {"command": key}
    if isinstance(result, dict):
        entry.update(result)
    else:
        entry["result"] = str(result)
    mission.optimization_history.append(entry)
    MissionStore().save(mission)


def get_attr_typical(attributes: dict, key: str) -> float | None:
    """Extract the typical value from a min/max/typical attribute dict."""
    val = attributes.get(key)
    if isinstance(val, dict):
        typical = val.get("typical")
        if isinstance(typical, (int, float)) and not isinstance(typical, bool):
            return float(typical)
        return None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return None
