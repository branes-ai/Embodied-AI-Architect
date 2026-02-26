"""Content-addressed spec store with event-sourced provenance.

Storage layout:
    .branes/specs/
        specs.json                  # index: {name -> metadata}
        <spec-name>/
            manifest.json           # version chain: [{hash, parent, ...}]
            events.jsonl            # append-only event log
            blobs/
                ab/ab123...json     # content-addressed snapshots (SHA256)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .diff import SpecChange, diff_specs
from .events import EventLog, EventOp, SpecEvent, make_event
from .exceptions import (
    SpecAlreadyExistsError,
    SpecNotFoundError,
    VersionNotFoundError,
)
from .models import SystemSpec, delete_at_path, set_at_path


class SpecStore:
    """Manages specs with content-addressed versioning and event sourcing.

    Args:
        root: Root directory for spec storage. Defaults to .branes/specs
              in the current working directory.
    """

    def __init__(self, root: Optional[Path] = None):
        if root is None:
            root = Path.cwd() / ".branes" / "specs"
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._index_path = self._root / "specs.json"

    @property
    def root(self) -> Path:
        return self._root

    # ── Index Management ───────────────────────────────────────────────

    def _load_index(self) -> dict[str, Any]:
        """Load the specs index."""
        if not self._index_path.exists():
            return {"version": "1.0", "specs": {}}
        with open(self._index_path) as f:
            return json.load(f)

    def _save_index(self, index: dict[str, Any]) -> None:
        """Save the specs index."""
        index["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self._index_path, "w") as f:
            json.dump(index, f, indent=2)

    def _spec_dir(self, name: str) -> Path:
        self._validate_name(name)
        return self._root / name

    @staticmethod
    def _validate_name(name: str) -> None:
        """Reject spec names that could escape the store root."""
        if not name or not name.strip():
            raise ValueError("Spec name must not be empty")
        if ".." in name.split("/") or ".." in name.split("\\"):
            raise ValueError(f"Invalid spec name (path traversal): {name!r}")
        from pathlib import PurePosixPath

        parts = PurePosixPath(name).parts
        if any(p in (".", "..") for p in parts):
            raise ValueError(f"Invalid spec name (relative path): {name!r}")
        if PurePosixPath(name).is_absolute() or name.startswith("/") or name.startswith("\\"):
            raise ValueError(f"Invalid spec name (absolute path): {name!r}")
        if "/" in name or "\\" in name:
            raise ValueError(f"Invalid spec name (contains path separator): {name!r}")

    def _event_log(self, name: str) -> EventLog:
        return EventLog(self._spec_dir(name) / "events.jsonl")

    def _manifest_path(self, name: str) -> Path:
        return self._spec_dir(name) / "manifest.json"

    def _blobs_dir(self, name: str) -> Path:
        return self._spec_dir(name) / "blobs"

    # ── CRUD ───────────────────────────────────────────────────────────

    def create(
        self,
        name: str,
        author: str = "user",
        template: Optional[str] = None,
        description: Optional[str] = None,
    ) -> SystemSpec:
        """Create a new spec, optionally from a template.

        Args:
            name: Unique spec name.
            author: Who is creating the spec.
            template: Template name (e.g., 'drone-perception').
            description: Human-readable description.

        Returns:
            The created SystemSpec.
        """
        index = self._load_index()
        if name in index["specs"]:
            raise SpecAlreadyExistsError(name)

        # Build initial spec
        if template:
            from .templates import get_template

            spec = get_template(template, name=name)
            if description:
                spec = spec.model_copy(update={"description": description})
            event_author = "template"
            event_reason = f"Created from template '{template}'"
        else:
            spec = SystemSpec(name=name, description=description)
            event_author = author
            event_reason = "Created empty spec"

        # Set up storage
        spec_dir = self._spec_dir(name)
        spec_dir.mkdir(parents=True, exist_ok=True)
        self._blobs_dir(name).mkdir(parents=True, exist_ok=True)

        # Write manifest
        manifest = {"versions": [], "tags": {}}
        with open(self._manifest_path(name), "w") as f:
            json.dump(manifest, f, indent=2)

        # Record creation event
        log = self._event_log(name)
        event = make_event(
            op=EventOp.CREATE,
            spec_name=name,
            sequence=0,
            author=event_author,
            value=spec.model_dump(mode="json"),
            reason=event_reason,
        )
        log.append(event)

        # Update index
        index["specs"][name] = {
            "created_at": event.timestamp,
            "author": author,
            "template": template,
            "description": description,
            "platform_type": spec.platform_type.value if spec.platform_type else None,
            "tags": spec.tags,
        }
        self._save_index(index)

        return spec

    def get(self, name: str) -> SystemSpec:
        """Get the current state of a spec by replaying events."""
        self._ensure_exists(name)
        log = self._event_log(name)
        return log.replay(name)

    def get_version(self, name: str, version: str) -> SystemSpec:
        """Get a specific version of a spec by hash or tag.

        Args:
            name: Spec name.
            version: A SHA256 hash prefix or tag name.
        """
        self._ensure_exists(name)
        manifest = self._load_manifest(name)

        # Check tags first
        if version in manifest.get("tags", {}):
            version = manifest["tags"][version]

        # Find version by hash prefix
        matched_hash = self._resolve_version_prefix(name, version, manifest)
        blob_path = self._blob_path(name, matched_hash)
        if blob_path.exists():
            with open(blob_path) as f:
                data = json.load(f)
            return SystemSpec.model_validate(data)

        raise VersionNotFoundError(name, version)

    def list_specs(self) -> dict[str, Any]:
        """List all specs with their metadata."""
        index = self._load_index()
        return index.get("specs", {})

    def delete_spec(self, name: str) -> None:
        """Delete a spec and all its data."""
        self._ensure_exists(name)
        import shutil

        shutil.rmtree(self._spec_dir(name))
        index = self._load_index()
        index["specs"].pop(name, None)
        self._save_index(index)

    # ── Mutations ──────────────────────────────────────────────────────

    def set_field(
        self,
        name: str,
        path: str,
        value: Any,
        author: str = "user",
        reason: Optional[str] = None,
    ) -> SystemSpec:
        """Set a field on a spec at the given path.

        Args:
            name: Spec name.
            path: JSON pointer path (e.g., /perception/min_fps).
            value: New value.
            author: Who is making the change.
            reason: Why the change is being made.

        Returns:
            The updated SystemSpec.
        """
        self._ensure_exists(name)
        log = self._event_log(name)
        spec = log.replay(name)
        new_spec = set_at_path(spec, path, value)

        event = make_event(
            op=EventOp.SET,
            spec_name=name,
            sequence=log.next_sequence(),
            author=author,
            path=path,
            value=value,
            reason=reason,
        )
        log.append(event)

        # Auto-snapshot if needed
        if log.should_snapshot():
            self._auto_snapshot(name, new_spec, author)

        return new_spec

    def delete_field(
        self,
        name: str,
        path: str,
        author: str = "user",
        reason: Optional[str] = None,
    ) -> SystemSpec:
        """Delete a field from a spec.

        Args:
            name: Spec name.
            path: JSON pointer path.
            author: Who is making the change.
            reason: Why the change is being made.

        Returns:
            The updated SystemSpec.
        """
        self._ensure_exists(name)
        log = self._event_log(name)
        spec = log.replay(name)
        new_spec = delete_at_path(spec, path)

        event = make_event(
            op=EventOp.DELETE,
            spec_name=name,
            sequence=log.next_sequence(),
            author=author,
            path=path,
            reason=reason,
        )
        log.append(event)

        return new_spec

    # ── Versioning ─────────────────────────────────────────────────────

    def commit(
        self,
        name: str,
        message: str,
        author: str = "user",
    ) -> str:
        """Create a content-addressed snapshot of the current spec state.

        Args:
            name: Spec name.
            message: Commit message.
            author: Who is committing.

        Returns:
            The SHA256 hash of the snapshot.
        """
        self._ensure_exists(name)
        log = self._event_log(name)
        spec = log.replay(name)

        # Create content-addressed blob
        spec_data = spec.model_dump(mode="json")
        blob_json = json.dumps(spec_data, sort_keys=True, default=str)
        blob_hash = hashlib.sha256(blob_json.encode()).hexdigest()

        blob_path = self._blob_path(name, blob_hash)
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        with open(blob_path, "w") as f:
            f.write(blob_json)

        # Update manifest
        manifest = self._load_manifest(name)
        parent = manifest["versions"][-1]["hash"] if manifest["versions"] else None
        entry = {
            "hash": blob_hash,
            "parent": parent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "author": author,
            "message": message,
        }
        manifest["versions"].append(entry)
        self._save_manifest(name, manifest)

        # Record snapshot event
        event = make_event(
            op=EventOp.SNAPSHOT,
            spec_name=name,
            sequence=log.next_sequence(),
            author=author,
            value=spec_data,
            reason=message,
        )
        log.append(event)

        return blob_hash

    def tag(
        self,
        name: str,
        tag_name: str,
        version: Optional[str] = None,
        message: Optional[str] = None,
        author: str = "user",
    ) -> None:
        """Tag a version with a human-readable name.

        Args:
            name: Spec name.
            tag_name: Tag label (e.g., 'v1.0', 'baseline').
            version: Hash to tag. Defaults to latest committed version.
            message: Optional tag message.
            author: Who is tagging.
        """
        self._ensure_exists(name)
        manifest = self._load_manifest(name)

        if not manifest["versions"]:
            raise ValueError(f"No committed versions for spec '{name}'. Run commit first.")

        if version is None:
            version = manifest["versions"][-1]["hash"]
        else:
            # Verify version exists
            version = self._resolve_version_prefix(name, version, manifest)

        manifest.setdefault("tags", {})[tag_name] = version
        self._save_manifest(name, manifest)

        # Record tag event
        log = self._event_log(name)
        event = make_event(
            op=EventOp.TAG,
            spec_name=name,
            sequence=log.next_sequence(),
            author=author,
            path=None,
            value={"tag": tag_name, "hash": version},
            reason=message,
        )
        log.append(event)

    # ── Queries ────────────────────────────────────────────────────────

    def history(self, name: str) -> list[dict[str, Any]]:
        """Get the version history (commits) for a spec.

        Returns:
            List of version entries from the manifest.
        """
        self._ensure_exists(name)
        manifest = self._load_manifest(name)
        versions = list(manifest.get("versions", []))
        # Annotate with tags
        tags = manifest.get("tags", {})
        tag_by_hash = {}
        for tag_name, hash_val in tags.items():
            tag_by_hash.setdefault(hash_val, []).append(tag_name)
        for v in versions:
            v["tags"] = tag_by_hash.get(v["hash"], [])
        return versions

    def diff(self, name: str, v1: str, v2: str) -> list[SpecChange]:
        """Compute a structured diff between two versions.

        Args:
            name: Spec name.
            v1: First version (hash prefix or tag).
            v2: Second version (hash prefix or tag).

        Returns:
            List of SpecChange objects.
        """
        old_spec = self.get_version(name, v1)
        new_spec = self.get_version(name, v2)
        return diff_specs(old_spec, new_spec)

    def why(self, name: str, path: str) -> list[SpecEvent]:
        """Get the provenance of a field — all events that modified it.

        Args:
            name: Spec name.
            path: JSON pointer path.

        Returns:
            List of SpecEvent objects that touched this field.
        """
        self._ensure_exists(name)
        log = self._event_log(name)
        return log.field_history(path)

    # ── Export / Import ────────────────────────────────────────────────

    def export_spec(self, name: str, fmt: str = "json") -> str:
        """Export a spec to JSON or YAML string.

        Args:
            name: Spec name.
            fmt: 'json' or 'yaml'.

        Returns:
            Serialized spec string.
        """
        spec = self.get(name)
        data = spec.model_dump(exclude_none=True, mode="json")
        if fmt == "yaml":
            try:
                import yaml

                return yaml.dump(data, default_flow_style=False, sort_keys=False)
            except ImportError:
                raise ImportError("PyYAML is required for YAML export: pip install pyyaml")
        return json.dumps(data, indent=2, default=str)

    def import_spec(
        self,
        name: str,
        data: dict[str, Any],
        author: str = "user",
        reason: Optional[str] = None,
    ) -> SystemSpec:
        """Import a spec from a dict (parsed JSON/YAML).

        Creates a new spec if it doesn't exist, or overwrites the current state.

        Args:
            name: Spec name.
            data: Spec data dict.
            author: Who is importing.
            reason: Why the import is happening.
        """
        data["name"] = name
        spec = SystemSpec.model_validate(data)

        index = self._load_index()
        spec_dir = self._spec_dir(name)
        is_new = name not in index["specs"]

        if is_new:
            spec_dir.mkdir(parents=True, exist_ok=True)
            self._blobs_dir(name).mkdir(parents=True, exist_ok=True)
            manifest = {"versions": [], "tags": {}}
            with open(self._manifest_path(name), "w") as f:
                json.dump(manifest, f, indent=2)

        log = self._event_log(name)
        seq = log.next_sequence()

        if is_new:
            event = make_event(
                op=EventOp.CREATE,
                spec_name=name,
                sequence=seq,
                author=author,
                value=spec.model_dump(mode="json"),
                reason=reason or "Imported",
            )
        else:
            event = make_event(
                op=EventOp.IMPORT,
                spec_name=name,
                sequence=seq,
                author=author,
                value=spec.model_dump(mode="json"),
                reason=reason or "Imported (overwrite)",
            )
        log.append(event)

        if is_new:
            index["specs"][name] = {
                "created_at": event.timestamp,
                "author": author,
                "template": None,
                "description": spec.description,
                "platform_type": spec.platform_type.value if spec.platform_type else None,
                "tags": spec.tags,
            }
            self._save_index(index)

        return spec

    # ── Validation ─────────────────────────────────────────────────────

    def validate(self, name: str) -> list:
        """Run cross-subsystem validation checks on a spec.

        Returns:
            List of ValidationIssue objects.
        """
        spec = self.get(name)
        from .validation import validate_spec

        return validate_spec(spec)

    # ── Resolve ────────────────────────────────────────────────────────

    def resolve(self, name: str) -> dict[str, Any]:
        """Flatten a spec to a dict for agent consumption.

        This produces a flat key-value representation suitable for passing
        to agent execute(input_data).
        """
        spec = self.get(name)
        data = spec.model_dump(exclude_none=True, mode="json")
        flat: dict[str, Any] = {}
        _flatten(data, "", flat)
        return flat

    # ── Internal ───────────────────────────────────────────────────────

    def _ensure_exists(self, name: str) -> None:
        """Raise SpecNotFoundError if the spec doesn't exist."""
        index = self._load_index()
        if name not in index.get("specs", {}):
            raise SpecNotFoundError(name)

    def _load_manifest(self, name: str) -> dict[str, Any]:
        """Load the version manifest for a spec."""
        path = self._manifest_path(name)
        if not path.exists():
            return {"versions": [], "tags": {}}
        with open(path) as f:
            return json.load(f)

    def _save_manifest(self, name: str, manifest: dict[str, Any]) -> None:
        """Save the version manifest."""
        with open(self._manifest_path(name), "w") as f:
            json.dump(manifest, f, indent=2)

    def _resolve_version_prefix(self, name: str, prefix: str, manifest: dict[str, Any]) -> str:
        """Resolve a hash prefix to a single full hash.

        Raises:
            VersionNotFoundError: If no version matches the prefix.
            ValueError: If multiple versions match (ambiguous prefix).
        """
        matches = [e["hash"] for e in manifest["versions"] if e["hash"].startswith(prefix)]
        if len(matches) == 0:
            raise VersionNotFoundError(name, prefix)
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous version prefix '{prefix}' for spec '{name}' — "
                f"matches {len(matches)} versions: {', '.join(matches)}"
            )
        return matches[0]

    def _blob_path(self, name: str, blob_hash: str) -> Path:
        """Get the path for a content-addressed blob."""
        return self._blobs_dir(name) / blob_hash[:2] / f"{blob_hash}.json"

    def _auto_snapshot(self, name: str, spec: SystemSpec, author: str) -> None:
        """Insert an automatic snapshot event for bounded replay."""
        log = self._event_log(name)
        event = make_event(
            op=EventOp.SNAPSHOT,
            spec_name=name,
            sequence=log.next_sequence(),
            author=author,
            value=spec.model_dump(mode="json"),
            reason="Auto-snapshot",
        )
        log.append(event)


def _flatten(data: Any, prefix: str, out: dict[str, Any]) -> None:
    """Flatten a nested dict into dot-notation keys."""
    if isinstance(data, dict):
        for key, val in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            _flatten(val, new_prefix, out)
    elif isinstance(data, list):
        out[prefix] = data
    else:
        out[prefix] = data
