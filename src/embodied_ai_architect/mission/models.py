"""Mission entity model (issue #52).

The Mission is the persistent backbone that all subsequent commands
operate on. It captures everything from the initial qualification
through design, optimization, and validation — so the architect can
resume, compare, and audit the full lifecycle.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class MissionStatus(str, Enum):
    """Lifecycle stages a mission progresses through."""

    DRAFT = "draft"
    QUALIFIED = "qualified"
    DESIGNED = "designed"
    OPTIMIZED = "optimized"
    VALIDATED = "validated"


def _now_iso() -> str:
    return datetime.now().isoformat()


def _generate_mission_id() -> str:
    """Generate a unique mission ID (same scheme as session IDs)."""
    import uuid

    return f"mission_{uuid.uuid4().hex[:12]}"


class Mission(BaseModel):
    """Persistent design mission entity.

    Owns the full design state across the lifecycle. Fields are additive —
    a newly created mission has only the inputs populated; later stages
    fill in the design_state, optimization_history, bom, architecture,
    and reports.
    """

    # Identity
    id: str = Field(default_factory=_generate_mission_id)
    name: str = ""
    description: str = ""
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
    status: MissionStatus = MissionStatus.DRAFT

    # Inputs (from qualification)
    goal: str = ""
    platform_id: Optional[str] = None
    use_case: Optional[str] = None
    constraints: dict[str, Any] = Field(default_factory=dict)

    # Components (from selection)
    selected_sensors: list[str] = Field(default_factory=list)
    selected_actuators: list[str] = Field(default_factory=list)
    selected_compute: Optional[str] = None
    selected_models: list[str] = Field(default_factory=list)

    # State
    spec: dict[str, Any] = Field(default_factory=dict)
    design_state: Optional[dict[str, Any]] = None
    optimization_history: list[dict[str, Any]] = Field(default_factory=list)

    # Artifacts
    bom: Optional[dict[str, Any]] = None
    architecture: Optional[dict[str, Any]] = None
    reports: list[str] = Field(default_factory=list)

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = _now_iso()
