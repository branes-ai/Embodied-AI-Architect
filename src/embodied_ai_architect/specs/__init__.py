"""Spec-of-specs system for embodied AI requirements.

Hierarchical, event-sourced specification management with full provenance.

Example usage:
    # Create a spec from a template
    from embodied_ai_architect.specs import SpecStore
    store = SpecStore()
    spec = store.create("my-drone", template="drone-perception", author="user")

    # Set fields incrementally
    store.set_field("my-drone", "/perception/min_fps", 60, author="user", reason="need 60fps")
    store.set_field("my-drone", "/compute/soc", "Jetson Orin NX", author="user")

    # Commit a snapshot
    store.commit("my-drone", message="initial requirements", author="user")

    # Query provenance
    events = store.why("my-drone", "/perception/min_fps")

    # Validate cross-subsystem consistency
    issues = store.validate("my-drone")
"""

from .events import EventLog, EventOp, SpecEvent, make_event
from .exceptions import (
    InvalidPathError,
    SpecAlreadyExistsError,
    SpecError,
    SpecNotFoundError,
    VersionNotFoundError,
)
from .models import (
    ActuatorSpec,
    AutonomyLevel,
    AutonomySpec,
    CommsSpec,
    ComputeSpec,
    CoolingType,
    EnvironmentalRating,
    PerceptionSpec,
    PlatformType,
    PowerSpec,
    SafetyLevel,
    SafetySpec,
    SensorSpec,
    SystemSpec,
    delete_at_path,
    get_at_path,
    parse_path,
    set_at_path,
)

__all__ = [
    # Models
    "SystemSpec",
    "PerceptionSpec",
    "ComputeSpec",
    "PowerSpec",
    "SensorSpec",
    "ActuatorSpec",
    "CommsSpec",
    "AutonomySpec",
    "SafetySpec",
    # Enums
    "PlatformType",
    "AutonomyLevel",
    "SafetyLevel",
    "CoolingType",
    "EnvironmentalRating",
    # Events
    "SpecEvent",
    "EventOp",
    "EventLog",
    "make_event",
    # Exceptions
    "SpecError",
    "SpecNotFoundError",
    "SpecAlreadyExistsError",
    "InvalidPathError",
    "VersionNotFoundError",
    # Path utilities
    "parse_path",
    "get_at_path",
    "set_at_path",
    "delete_at_path",
]


# Lazy imports for optional components
def __getattr__(name: str):
    if name == "SpecStore":
        from .store import SpecStore

        return SpecStore
    if name == "get_templates":
        from .templates import get_templates

        return get_templates
    if name == "validate_spec":
        from .validation import validate_spec

        return validate_spec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
