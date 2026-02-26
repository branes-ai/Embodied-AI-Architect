"""Structured diff between two spec versions.

Produces a list of changes (added, removed, changed) with paths and values.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from .models import SystemSpec


class ChangeType(str, Enum):
    """Type of change between two spec versions."""

    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"


class SpecChange(BaseModel):
    """A single change between two spec versions."""

    change_type: ChangeType = Field(description="Type of change")
    path: str = Field(description="JSON pointer path to the changed field")
    old_value: Any = Field(default=None, description="Previous value (None for added)")
    new_value: Any = Field(default=None, description="New value (None for removed)")


def diff_specs(old: SystemSpec, new: SystemSpec) -> list[SpecChange]:
    """Compute a structured diff between two spec versions.

    Returns a list of SpecChange objects describing what was added, removed,
    or changed between the old and new specs.
    """
    old_data = old.model_dump(exclude_none=True)
    new_data = new.model_dump(exclude_none=True)
    changes: list[SpecChange] = []
    _diff_dicts(old_data, new_data, "", changes)
    return changes


def _diff_dicts(old: dict, new: dict, prefix: str, changes: list[SpecChange]) -> None:
    """Recursively diff two dicts, accumulating SpecChange objects."""
    all_keys = set(old.keys()) | set(new.keys())
    for key in sorted(all_keys):
        path = f"{prefix}/{key}"
        in_old = key in old
        in_new = key in new

        if in_old and not in_new:
            changes.append(
                SpecChange(change_type=ChangeType.REMOVED, path=path, old_value=old[key])
            )
        elif not in_old and in_new:
            changes.append(SpecChange(change_type=ChangeType.ADDED, path=path, new_value=new[key]))
        else:
            old_val = old[key]
            new_val = new[key]
            if isinstance(old_val, dict) and isinstance(new_val, dict):
                _diff_dicts(old_val, new_val, path, changes)
            elif old_val != new_val:
                changes.append(
                    SpecChange(
                        change_type=ChangeType.CHANGED,
                        path=path,
                        old_value=old_val,
                        new_value=new_val,
                    )
                )


def format_diff(changes: list[SpecChange], colorize: bool = True) -> str:
    """Format a diff as a human-readable string.

    Args:
        changes: List of SpecChange objects.
        colorize: If True, use Rich markup for coloring.
    """
    if not changes:
        return "No changes."

    lines = []
    for change in changes:
        if change.change_type == ChangeType.ADDED:
            if colorize:
                lines.append(f"[green]+ {change.path}: {_fmt_val(change.new_value)}[/green]")
            else:
                lines.append(f"+ {change.path}: {_fmt_val(change.new_value)}")
        elif change.change_type == ChangeType.REMOVED:
            if colorize:
                lines.append(f"[red]- {change.path}: {_fmt_val(change.old_value)}[/red]")
            else:
                lines.append(f"- {change.path}: {_fmt_val(change.old_value)}")
        elif change.change_type == ChangeType.CHANGED:
            if colorize:
                lines.append(
                    f"[yellow]~ {change.path}: {_fmt_val(change.old_value)} → "
                    f"{_fmt_val(change.new_value)}[/yellow]"
                )
            else:
                lines.append(
                    f"~ {change.path}: {_fmt_val(change.old_value)} → "
                    f"{_fmt_val(change.new_value)}"
                )
    return "\n".join(lines)


def _fmt_val(val: Any) -> str:
    """Format a value for display."""
    if val is None:
        return "null"
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        non_none = {k: v for k, v in val.items() if v is not None}
        if not non_none:
            return "{}"
        return str(non_none)
    return str(val)
