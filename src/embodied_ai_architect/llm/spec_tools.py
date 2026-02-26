"""LLM tool definitions for spec management.

Provides tools that allow the chat agent to create, read, modify,
and validate system specifications.
"""

import json
import os
import traceback
from typing import Any, Callable


def _error_response(message: str, error: Exception) -> str:
    """Build a structured JSON error response.

    Tracebacks are only included when DEBUG=1 to avoid leaking internals.
    """
    result: dict[str, Any] = {
        "success": False,
        "error": str(error),
        "message": message,
    }
    if os.getenv("DEBUG") == "1":
        result["trace"] = traceback.format_exc()
    return json.dumps(result, indent=2)


def get_spec_tool_definitions() -> list[dict[str, Any]]:
    """Get tool definitions for spec management in Anthropic format."""
    return [
        {
            "name": "list_specs",
            "description": (
                "List all available system specifications. Returns spec names with "
                "metadata (platform type, template, description, creation date)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "create_spec",
            "description": (
                "Create a new system specification, optionally from a predefined template. "
                "Available templates: drone-perception, quadruped-nav, industrial-inspection, "
                "amr-warehouse, edge-camera, biped-humanoid. Templates provide sensible "
                "defaults that can be refined incrementally."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique name for the spec (e.g., 'my-drone')",
                    },
                    "template": {
                        "type": "string",
                        "description": (
                            "Template name to start from (optional). One of: "
                            "drone-perception, quadruped-nav, industrial-inspection, "
                            "amr-warehouse, edge-camera, biped-humanoid"
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description of the system",
                    },
                },
                "required": ["name"],
            },
        },
        {
            "name": "read_spec",
            "description": (
                "Read the current state of a system specification. Returns the full "
                "spec hierarchy with all defined subsystems (perception, compute, power, "
                "sensors, actuators, comms, autonomy, safety). Use a path to read a "
                "specific subsystem or field."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Spec name",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Optional JSON pointer path to read a specific field "
                            "(e.g., '/perception', '/compute/max_tdp_watts'). "
                            "Omit to read the entire spec."
                        ),
                    },
                },
                "required": ["name"],
            },
        },
        {
            "name": "modify_spec",
            "description": (
                "Set a field on a system specification. Use JSON pointer paths to "
                "target specific fields. Subsystem models are auto-created if needed. "
                "Every modification is recorded with the reason for full provenance."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Spec name",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "JSON pointer path (e.g., '/perception/min_fps', "
                            "'/compute/soc', '/power/power_budget_watts')"
                        ),
                    },
                    "value": {
                        "description": (
                            "New value. Strings, numbers, booleans, lists, and objects "
                            "are all supported."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Why this change is being made. Required for provenance tracking."
                        ),
                    },
                },
                "required": ["name", "path", "value", "reason"],
            },
        },
        {
            "name": "validate_spec",
            "description": (
                "Run cross-subsystem consistency checks on a spec. Checks for issues "
                "like: compute TDP exceeding power budget, perception latency vs FPS "
                "mismatch, missing sensors for navigation mode, safety level vs "
                "redundancy requirements. Returns a list of issues with severity, "
                "message, and suggested fixes."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Spec name to validate",
                    },
                },
                "required": ["name"],
            },
        },
        {
            "name": "spec_field_history",
            "description": (
                "Get the provenance of a specific field — who changed it, when, and why. "
                "Useful for understanding the rationale behind a requirement value."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Spec name",
                    },
                    "path": {
                        "type": "string",
                        "description": "JSON pointer path (e.g., '/perception/min_fps')",
                    },
                },
                "required": ["name", "path"],
            },
        },
    ]


def create_spec_tool_executors() -> dict[str, Callable]:
    """Create executor functions for spec tools."""

    def list_specs() -> str:
        """List all specs."""
        try:
            from embodied_ai_architect.specs.store import SpecStore

            store = SpecStore()
            specs = store.list_specs()
            if not specs:
                return json.dumps({"specs": [], "message": "No specs found. Create one first."})
            return json.dumps({"specs": specs}, indent=2, default=str)
        except Exception as e:
            return _error_response("Error listing specs", e)

    def create_spec(
        name: str,
        template: str | None = None,
        description: str | None = None,
    ) -> str:
        """Create a new spec."""
        try:
            from embodied_ai_architect.specs.store import SpecStore

            store = SpecStore()
            spec = store.create(name, template=template, description=description, author="agent")
            result = spec.model_dump(exclude_none=True, mode="json")
            result["_message"] = f"Created spec '{name}'"
            if template:
                result["_message"] += f" from template '{template}'"
            return json.dumps(result, indent=2, default=str)
        except Exception as e:
            return _error_response("Error creating spec", e)

    def read_spec(name: str, path: str | None = None) -> str:
        """Read a spec or a specific field."""
        try:
            from embodied_ai_architect.specs.models import get_at_path
            from embodied_ai_architect.specs.store import SpecStore

            store = SpecStore()
            spec = store.get(name)

            if path:
                value = get_at_path(spec, path)
                if hasattr(value, "model_dump"):
                    return json.dumps(value.model_dump(exclude_none=True, mode="json"), indent=2)
                return json.dumps({"path": path, "value": value}, indent=2, default=str)
            else:
                return json.dumps(
                    spec.model_dump(exclude_none=True, mode="json"), indent=2, default=str
                )
        except Exception as e:
            return _error_response("Error reading spec", e)

    def modify_spec(name: str, path: str, value: Any, reason: str) -> str:
        """Modify a spec field."""
        try:
            from embodied_ai_architect.specs.store import SpecStore

            store = SpecStore()
            spec = store.set_field(name, path, value, author="agent", reason=reason)
            return json.dumps(
                {
                    "status": "ok",
                    "path": path,
                    "value": value,
                    "reason": reason,
                    "spec_summary": spec.summary(),
                },
                indent=2,
                default=str,
            )
        except Exception as e:
            return _error_response("Error modifying spec", e)

    def validate_spec(name: str) -> str:
        """Validate a spec."""
        try:
            from embodied_ai_architect.specs.store import SpecStore

            store = SpecStore()
            issues = store.validate(name)
            if not issues:
                return json.dumps({"status": "ok", "issues": [], "message": "No issues found."})
            return json.dumps(
                {
                    "status": "issues_found",
                    "count": len(issues),
                    "issues": [i.model_dump(mode="json") for i in issues],
                },
                indent=2,
                default=str,
            )
        except Exception as e:
            return _error_response("Error validating spec", e)

    def spec_field_history(name: str, path: str) -> str:
        """Get field provenance."""
        try:
            from embodied_ai_architect.specs.store import SpecStore

            store = SpecStore()
            events = store.why(name, path)
            if not events:
                return json.dumps({"path": path, "events": [], "message": f"No history for {path}"})
            return json.dumps(
                {
                    "path": path,
                    "events": [e.model_dump(mode="json") for e in events],
                },
                indent=2,
                default=str,
            )
        except Exception as e:
            return _error_response("Error getting field history", e)

    return {
        "list_specs": list_specs,
        "create_spec": create_spec,
        "read_spec": read_spec,
        "modify_spec": modify_spec,
        "validate_spec": validate_spec,
        "spec_field_history": spec_field_history,
    }
