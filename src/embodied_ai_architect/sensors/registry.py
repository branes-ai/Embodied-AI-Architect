"""Sensor registry (issue #54).

Stub implementation that returns empty results. Phase 2 will populate
this from the embodied-schemas SensorEntry catalog and/or local YAML
definitions (mirroring the platform registry pattern).
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

# Sensor modality categories — matches data/sensors/taxonomy.yaml top-level
# keys. Phase 2 will load these dynamically from the YAML; for now they're
# kept in sync manually (CodeRabbit PR #107).
MODALITIES = [
    "visual",
    "ranging",
    "inertial",
    "position",
    "environmental",
    "force",
    "audio",
    "biological",
]


class SensorDefinition(BaseModel):
    """A sensor in the registry."""

    id: str
    name: str
    modality: str
    vendor: str = ""
    description: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)


class SensorRegistry:
    """Stub sensor registry (Phase 1 — returns empty results).

    Phase 2 will load sensor definitions from:
    - embodied-schemas SensorEntry catalog (if installed)
    - Local YAML files in data/sensors/
    """

    def __init__(self) -> None:
        self._sensors: dict[str, SensorDefinition] = {}

    def list_sensors(self, modality: Optional[str] = None) -> list[SensorDefinition]:
        """List all sensors, optionally filtered by modality."""
        sensors = list(self._sensors.values())
        if modality:
            sensors = [s for s in sensors if s.modality == modality]
        return sensors

    def get(self, sensor_id: str) -> Optional[SensorDefinition]:
        """Get a sensor by ID."""
        return self._sensors.get(sensor_id)

    def search(self, query: str, top_k: int = 10) -> list[SensorDefinition]:
        """Search sensors by keyword. Returns empty until populated."""
        if top_k <= 0:
            return []
        if not self._sensors:
            return []
        query_lower = query.lower()
        scored = []
        for s in self._sensors.values():
            text = f"{s.name} {s.description} {s.modality} {s.vendor}".lower()
            score = sum(1 for word in query_lower.split() if word in text)
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    def categories(self) -> list[str]:
        """List available sensor modality categories."""
        return list(MODALITIES)
