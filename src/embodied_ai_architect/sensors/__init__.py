"""Sensor registry with TF-IDF keyword matching (issues #54, #59).

Loads 80+ sensor definitions from data/sensors/ and provides ranked
search for the `branes sensor` CLI and qualification pipeline.

Usage:
    from embodied_ai_architect.sensors import SensorRegistry

    registry = SensorRegistry()
    results = registry.search("stereo camera for VIO")
"""

from embodied_ai_architect.sensors.registry import (
    SensorDefinition,
    SensorMatchResult,
    SensorRegistry,
)

__all__ = ["SensorDefinition", "SensorMatchResult", "SensorRegistry"]
