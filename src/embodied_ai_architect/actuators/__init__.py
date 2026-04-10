"""Actuator registry with TF-IDF keyword matching (issues #55, #62).

Loads 80+ actuator definitions from data/actuators/ and provides ranked
search for the `branes actuator` CLI and qualification pipeline.

Usage:
    from embodied_ai_architect.actuators import ActuatorRegistry

    registry = ActuatorRegistry()
    results = registry.search("gripper for fragile objects")
"""

from embodied_ai_architect.actuators.registry import (
    ActuatorDefinition,
    ActuatorMatchResult,
    ActuatorRegistry,
)

__all__ = ["ActuatorDefinition", "ActuatorMatchResult", "ActuatorRegistry"]
