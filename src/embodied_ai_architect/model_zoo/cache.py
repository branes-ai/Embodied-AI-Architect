"""Model cache manager with versioning support.

Provides centralized caching for downloaded models with:
- Versioned storage
- Cache invalidation
- Size tracking
- Cleanup utilities
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .providers.base import ModelFormat


@dataclass
class CacheEntry:
    """Metadata for a cached model."""

    model_id: str
    provider: str
    format: str
    path: str
    size_bytes: int
    cached_at: str
    version: Optional[str] = None
    checksum: Optional[str] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        return cls(**data)


class ModelCache:
    """Centralized cache for downloaded models.

    Default location: ~/.cache/branes/models/

    Structure:
        ~/.cache/branes/models/
        ├── cache.json          # Cache index
        ├── ultralytics/
        │   ├── yolov8n.onnx
        │   ├── yolov8s.onnx
        │   └── yolov8n.pt
        ├── torchvision/
        │   ├── resnet18.onnx
        │   └── efficientnet_b0.onnx
        └── huggingface/
            └── ...
    """

    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "branes" / "models"

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the cache.

        Args:
            cache_dir: Custom cache directory (default: ~/.cache/branes/models/)
        """
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self.index_path = self.cache_dir / "cache.json"
        self._entries: dict[str, CacheEntry] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load cache index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path) as f:
                    data = json.load(f)
                    self._entries = {
                        key: CacheEntry.from_dict(entry)
                        for key, entry in data.get("entries", {}).items()
                    }
            except (json.JSONDecodeError, KeyError):
                # Corrupted index, rebuild from disk
                self._rebuild_index()

    def _save_index(self) -> None:
        """Save cache index to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "entries": {key: entry.to_dict() for key, entry in self._entries.items()},
        }
        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2)

    def _rebuild_index(self) -> None:
        """Rebuild cache index from files on disk."""
        self._entries = {}
        if not self.cache_dir.exists():
            return

        for provider_dir in self.cache_dir.iterdir():
            if provider_dir.is_dir() and provider_dir.name != "__pycache__":
                provider = provider_dir.name
                for model_file in provider_dir.iterdir():
                    if model_file.is_file() or model_file.is_dir():
                        # Infer format from extension
                        format_str = self._infer_format(model_file)
                        model_id = model_file.stem

                        key = self._cache_key(model_id, provider, format_str)
                        self._entries[key] = CacheEntry(
                            model_id=model_id,
                            provider=provider,
                            format=format_str,
                            path=str(model_file),
                            size_bytes=self._get_size(model_file),
                            cached_at=datetime.now().isoformat(),
                        )

        self._save_index()

    def _infer_format(self, path: Path) -> str:
        """Infer model format from file extension."""
        suffix = path.suffix.lower()
        format_map = {
            ".pt": ModelFormat.PYTORCH.value,
            ".pth": ModelFormat.PYTORCH.value,
            ".torchscript": ModelFormat.TORCHSCRIPT.value,
            ".onnx": ModelFormat.ONNX.value,
            ".engine": ModelFormat.TENSORRT.value,
            ".mlpackage": ModelFormat.COREML.value,
            ".mlmodel": ModelFormat.COREML.value,
        }
        if path.is_dir() and "openvino" in path.name:
            return ModelFormat.OPENVINO.value
        return format_map.get(suffix, "unknown")

    def _get_size(self, path: Path) -> int:
        """Get size of file or directory in bytes."""
        if path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return path.stat().st_size

    def _cache_key(self, model_id: str, provider: str, format: str) -> str:
        """Generate a cache key."""
        return f"{provider}/{model_id}/{format}"

    def get_provider_dir(self, provider: str) -> Path:
        """Get the cache directory for a provider."""
        provider_dir = self.cache_dir / provider
        provider_dir.mkdir(parents=True, exist_ok=True)
        return provider_dir

    def get(
        self,
        model_id: str,
        provider: str,
        format: ModelFormat,
    ) -> Optional[Path]:
        """Get a cached model path if it exists.

        Args:
            model_id: Model identifier
            provider: Provider name
            format: Model format

        Returns:
            Path to cached model or None if not cached
        """
        key = self._cache_key(model_id, provider, format.value)
        entry = self._entries.get(key)

        if entry is None:
            return None

        path = Path(entry.path)
        if not path.exists():
            # File was deleted, remove from index
            del self._entries[key]
            self._save_index()
            return None

        return path

    def put(
        self,
        model_id: str,
        provider: str,
        format: ModelFormat,
        path: Path,
        version: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> CacheEntry:
        """Add a model to the cache.

        Args:
            model_id: Model identifier
            provider: Provider name
            format: Model format
            path: Path to the model file
            version: Optional version string
            metadata: Optional additional metadata

        Returns:
            CacheEntry for the cached model
        """
        key = self._cache_key(model_id, provider, format.value)

        entry = CacheEntry(
            model_id=model_id,
            provider=provider,
            format=format.value,
            path=str(path),
            size_bytes=self._get_size(path),
            cached_at=datetime.now().isoformat(),
            version=version,
            metadata=metadata or {},
        )

        self._entries[key] = entry
        self._save_index()

        return entry

    def remove(
        self,
        model_id: str,
        provider: str,
        format: Optional[ModelFormat] = None,
    ) -> bool:
        """Remove a model from the cache.

        Args:
            model_id: Model identifier
            provider: Provider name
            format: Optional format (if None, removes all formats)

        Returns:
            True if any models were removed
        """
        removed = False

        if format is not None:
            key = self._cache_key(model_id, provider, format.value)
            if key in self._entries:
                entry = self._entries[key]
                path = Path(entry.path)
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                del self._entries[key]
                removed = True
        else:
            # Remove all formats
            keys_to_remove = [k for k in self._entries if k.startswith(f"{provider}/{model_id}/")]
            for key in keys_to_remove:
                entry = self._entries[key]
                path = Path(entry.path)
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                del self._entries[key]
                removed = True

        if removed:
            self._save_index()

        return removed

    def list(
        self,
        provider: Optional[str] = None,
        format: Optional[ModelFormat] = None,
    ) -> list[CacheEntry]:
        """List cached models.

        Args:
            provider: Optional filter by provider
            format: Optional filter by format

        Returns:
            List of CacheEntry objects
        """
        entries = list(self._entries.values())

        if provider is not None:
            entries = [e for e in entries if e.provider == provider]

        if format is not None:
            entries = [e for e in entries if e.format == format.value]

        return sorted(entries, key=lambda e: (e.provider, e.model_id))

    def total_size(self) -> int:
        """Get total cache size in bytes."""
        return sum(e.size_bytes for e in self._entries.values())

    def clear(self, provider: Optional[str] = None) -> int:
        """Clear the cache.

        Args:
            provider: Optional provider to clear (clears all if None)

        Returns:
            Number of entries removed
        """
        if provider is not None:
            entries = [e for e in self._entries.values() if e.provider == provider]
            provider_dir = self.cache_dir / provider
            if provider_dir.exists():
                shutil.rmtree(provider_dir)
        else:
            entries = list(self._entries.values())
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)

        count = len(entries)

        if provider is not None:
            self._entries = {k: v for k, v in self._entries.items() if v.provider != provider}
        else:
            self._entries = {}

        self._save_index()
        return count

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries
