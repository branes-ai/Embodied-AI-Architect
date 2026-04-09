"""Actuator registry (issue #55).

Stub implementation mirroring the sensor registry from issue #54.
Phase 2 will populate from the embodied-schemas catalog and/or
local YAML definitions.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

# Actuator type categories — used by `branes actuator categories`
ACTUATOR_TYPES = [
    "brushless_motor",
    "stepper_motor",
    "servo",
    "linear_actuator",
    "hydraulic_cylinder",
    "pneumatic_cylinder",
    "piezoelectric",
    "voice_coil",
    "solenoid",
    "gripper",
    "vacuum_gripper",
    "magnetic_gripper",
    "wheel_drive",
    "track_drive",
    "propeller",
    "gimbal",
    "led_illuminator",
    "speaker",
]


class ActuatorDefinition(BaseModel):
    """An actuator in the registry."""

    id: str
    name: str
    actuator_type: str
    vendor: str = ""
    description: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)


class ActuatorRegistry:
    """Stub actuator registry (Phase 1 — returns empty results).

    Phase 2 will load actuator definitions from:
    - embodied-schemas catalog (if installed)
    - Local YAML files in data/actuators/
    """

    def __init__(self) -> None:
        self._actuators: dict[str, ActuatorDefinition] = {}

    def list_actuators(self, actuator_type: Optional[str] = None) -> list[ActuatorDefinition]:
        """List all actuators, optionally filtered by type."""
        actuators = list(self._actuators.values())
        if actuator_type:
            actuators = [a for a in actuators if a.actuator_type == actuator_type]
        return actuators

    def get(self, actuator_id: str) -> Optional[ActuatorDefinition]:
        """Get an actuator by ID."""
        return self._actuators.get(actuator_id)

    def search(self, query: str, top_k: int = 10) -> list[ActuatorDefinition]:
        """Search actuators by keyword. Returns empty until populated."""
        if top_k <= 0:
            return []
        if not self._actuators:
            return []
        query_lower = query.lower()
        scored = []
        for a in self._actuators.values():
            text = f"{a.name} {a.description} {a.actuator_type} {a.vendor}".lower()
            score = sum(1 for word in query_lower.split() if word in text)
            if score > 0:
                scored.append((score, a))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in scored[:top_k]]

    def categories(self) -> list[str]:
        """List available actuator type categories."""
        return list(ACTUATOR_TYPES)
