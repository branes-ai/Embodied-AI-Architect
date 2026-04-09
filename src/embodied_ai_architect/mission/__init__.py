"""Mission lifecycle management (issue #52).

The Mission entity owns the full design state across the lifecycle —
from qualification through design, optimization, validation, and
deployment. MissionStore provides persistent JSON storage in
`.branes/missions/<mission_id>/manifest.json`.

Usage:
    from embodied_ai_architect.mission import Mission, MissionStore

    store = MissionStore()
    mission = Mission(name="Drone Perception SoC", goal="30fps detection at <5W")
    store.save(mission)
    loaded = store.load(mission.id)
"""

from embodied_ai_architect.mission.models import Mission, MissionStatus
from embodied_ai_architect.mission.store import MissionStore

__all__ = ["Mission", "MissionStatus", "MissionStore"]
