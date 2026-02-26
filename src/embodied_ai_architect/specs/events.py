"""Event-sourced provenance for spec mutations.

Every mutation to a spec is recorded as an append-only event in a JSONL file.
Events can be replayed to reconstruct any historical state. Snapshot events
are inserted periodically for bounded replay time.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .models import SystemSpec, delete_at_path, set_at_path


class EventOp(str, Enum):
    """Types of spec mutations."""

    CREATE = "create"
    SET = "set"
    DELETE = "delete"
    TAG = "tag"
    SNAPSHOT = "snapshot"
    IMPORT = "import"


class SpecEvent(BaseModel):
    """A single mutation event in the spec's history."""

    op: EventOp = Field(description="Type of operation")
    path: Optional[str] = Field(default=None, description="JSON pointer path (e.g., /perception/min_fps)")
    value: Any = Field(default=None, description="New value for set ops, full spec for snapshots")
    author: str = Field(description="Who made the change (user, agent name, template)")
    reason: Optional[str] = Field(default=None, description="Why the change was made")
    timestamp: str = Field(description="ISO 8601 timestamp")
    spec_name: str = Field(description="Name of the spec")
    sequence: int = Field(description="Monotonic per-spec counter")


# Auto-snapshot after this many events since last snapshot
SNAPSHOT_INTERVAL = 50


class EventLog:
    """Append-only event log backed by a JSONL file.

    Supports:
    - Appending new events
    - Replaying events to reconstruct spec state
    - Querying field history (provenance)
    - Auto-snapshots for bounded replay time
    """

    def __init__(self, path: Path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event: SpecEvent) -> None:
        """Append an event to the log."""
        with open(self._path, "a") as f:
            f.write(event.model_dump_json() + "\n")

    def read_all(self) -> list[SpecEvent]:
        """Read all events from the log."""
        if not self._path.exists():
            return []
        events = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(SpecEvent.model_validate_json(line))
        return events

    def next_sequence(self) -> int:
        """Return the next sequence number."""
        events = self.read_all()
        if not events:
            return 0
        return events[-1].sequence + 1

    def replay(self, spec_name: str, up_to_sequence: Optional[int] = None) -> SystemSpec:
        """Replay events to reconstruct spec state.

        Starts from the most recent snapshot (if any) for efficiency.
        """
        events = self.read_all()
        if not events:
            raise ValueError(f"No events for spec '{spec_name}'")

        # Find the most recent snapshot before up_to_sequence
        start_idx = 0
        spec: Optional[SystemSpec] = None
        for i, event in enumerate(events):
            if up_to_sequence is not None and event.sequence > up_to_sequence:
                break
            if event.op == EventOp.SNAPSHOT:
                spec = SystemSpec.model_validate(event.value)
                start_idx = i + 1

        # If no snapshot found, start from creation
        if spec is None:
            for i, event in enumerate(events):
                if event.op == EventOp.CREATE:
                    spec = SystemSpec.model_validate(event.value)
                    start_idx = i + 1
                    break

        if spec is None:
            raise ValueError(f"No create event found for spec '{spec_name}'")

        # Apply subsequent events
        for event in events[start_idx:]:
            if up_to_sequence is not None and event.sequence > up_to_sequence:
                break
            spec = _apply_event(spec, event)

        return spec

    def field_history(self, path: str) -> list[SpecEvent]:
        """Get all events that touched a specific path."""
        events = self.read_all()
        return [e for e in events if e.path == path and e.op in (EventOp.SET, EventOp.DELETE)]

    def events_since_last_snapshot(self) -> int:
        """Count events since the last snapshot."""
        events = self.read_all()
        count = 0
        for event in reversed(events):
            if event.op == EventOp.SNAPSHOT:
                break
            count += 1
        return count

    def should_snapshot(self) -> bool:
        """Check if an auto-snapshot is needed."""
        return self.events_since_last_snapshot() >= SNAPSHOT_INTERVAL


def _apply_event(spec: SystemSpec, event: SpecEvent) -> SystemSpec:
    """Apply a single event to a spec, returning a new spec."""
    if event.op == EventOp.SET:
        return set_at_path(spec, event.path, event.value)
    elif event.op == EventOp.DELETE:
        return delete_at_path(spec, event.path)
    elif event.op == EventOp.SNAPSHOT:
        return SystemSpec.model_validate(event.value)
    elif event.op == EventOp.IMPORT:
        return SystemSpec.model_validate(event.value)
    # CREATE, TAG don't change the spec state after creation
    return spec


def make_event(
    op: EventOp,
    spec_name: str,
    sequence: int,
    author: str,
    path: Optional[str] = None,
    value: Any = None,
    reason: Optional[str] = None,
) -> SpecEvent:
    """Create a new SpecEvent with the current timestamp."""
    return SpecEvent(
        op=op,
        path=path,
        value=value,
        author=author,
        reason=reason,
        timestamp=datetime.now(timezone.utc).isoformat(),
        spec_name=spec_name,
        sequence=sequence,
    )
