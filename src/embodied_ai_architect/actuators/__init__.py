"""Actuator registry and browsing (issue #55).

Stub package — the registry returns empty until Phase 2 populates it.

Usage:
    from embodied_ai_architect.actuators import ActuatorRegistry

    registry = ActuatorRegistry()
    actuators = registry.list_actuators()  # [] until populated
"""

from embodied_ai_architect.actuators.registry import ActuatorRegistry

__all__ = ["ActuatorRegistry"]
