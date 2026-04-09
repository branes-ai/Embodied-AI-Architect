"""Sensor registry and browsing (issue #54).

Stub package — the registry returns empty until Phase 2 populates it
with sensor definitions from the embodied-schemas catalog.

Usage:
    from embodied_ai_architect.sensors import SensorRegistry

    registry = SensorRegistry()
    sensors = registry.list_sensors()  # [] until populated
"""

from embodied_ai_architect.sensors.registry import SensorRegistry

__all__ = ["SensorRegistry"]
