"""Model registry for persistent storage and querying of model metadata."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from embodied_ai_architect.registry.exceptions import (
    ModelNotFoundError,
    ModelAlreadyExistsError,
)


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""

    # Identity
    id: str
    name: str
    path: str
    format: str  # "pytorch", "jit", "onnx", "state_dict"
    registered_at: str

    # Analysis results
    total_parameters: int
    trainable_parameters: int
    estimated_flops: Optional[int] = None
    estimated_memory_mb: float = 0.0
    input_shape: Optional[list[int]] = None
    output_shape: Optional[list[int]] = None

    # Architecture info
    architecture_type: Optional[str] = None  # "cnn", "transformer", "mlp", etc.
    architecture_family: Optional[str] = None  # "resnet", "yolo", "vit", etc.
    layer_counts: dict[str, int] = field(default_factory=dict)

    # User metadata
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        return cls(**data)

    def format_parameters(self) -> str:
        """Format parameter count for display (e.g., '11.2M')."""
        params = self.total_parameters
        if params >= 1_000_000_000:
            return f"{params / 1_000_000_000:.1f}B"
        elif params >= 1_000_000:
            return f"{params / 1_000_000:.1f}M"
        elif params >= 1_000:
            return f"{params / 1_000:.1f}K"
        return str(params)

    def format_flops(self) -> str:
        """Format FLOPs for display (e.g., '28.6B')."""
        if self.estimated_flops is None:
            return "N/A"
        flops = self.estimated_flops
        if flops >= 1_000_000_000_000:
            return f"{flops / 1_000_000_000_000:.1f}T"
        elif flops >= 1_000_000_000:
            return f"{flops / 1_000_000_000:.1f}B"
        elif flops >= 1_000_000:
            return f"{flops / 1_000_000:.1f}M"
        elif flops >= 1_000:
            return f"{flops / 1_000:.1f}K"
        return str(flops)

    @property
    def architecture_display(self) -> str:
        """Format architecture for display (e.g., 'cnn/yolo')."""
        if self.architecture_type and self.architecture_family:
            return f"{self.architecture_type}/{self.architecture_family}"
        elif self.architecture_type:
            return self.architecture_type
        return "unknown"


class ModelRegistry:
    """
    Persistent registry for PyTorch models.

    Stores model metadata in a JSON file at ~/.branes/models/registry.json
    """

    DEFAULT_PATH = Path.home() / ".branes" / "models" / "registry.json"

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize the registry.

        Args:
            registry_path: Path to registry JSON file (default: ~/.embodied-ai/models/registry.json)
        """
        self.registry_path = registry_path or self.DEFAULT_PATH
        self._models: dict[str, ModelMetadata] = {}
        self._load()

    def _load(self) -> None:
        """Load registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = json.load(f)
                self._models = {
                    model_id: ModelMetadata.from_dict(model_data)
                    for model_id, model_data in data.get("models", {}).items()
                }

    def _save(self) -> None:
        """Save registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "models": {model_id: model.to_dict() for model_id, model in self._models.items()},
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def register(
        self,
        metadata: ModelMetadata,
        overwrite: bool = False,
    ) -> ModelMetadata:
        """
        Register a model in the registry.

        Args:
            metadata: Model metadata from analyzer
            overwrite: If True, overwrite existing model with same ID

        Returns:
            Registered ModelMetadata

        Raises:
            ModelAlreadyExistsError: If model ID exists and overwrite=False
        """
        if metadata.id in self._models and not overwrite:
            raise ModelAlreadyExistsError(
                f"Model '{metadata.id}' already exists. Use --overwrite to replace."
            )

        self._models[metadata.id] = metadata
        self._save()
        return metadata

    def get(self, model_id: str) -> ModelMetadata:
        """
        Get a model by ID.

        Args:
            model_id: Model identifier

        Returns:
            ModelMetadata

        Raises:
            ModelNotFoundError: If model not found
        """
        if model_id not in self._models:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry.")
        return self._models[model_id]

    def list(
        self, filter_fn: Optional[Callable[[ModelMetadata], bool]] = None
    ) -> list[ModelMetadata]:
        """
        List all models, optionally filtered.

        Args:
            filter_fn: Optional filter function

        Returns:
            List of ModelMetadata
        """
        models = list(self._models.values())
        if filter_fn:
            models = [m for m in models if filter_fn(m)]
        return sorted(models, key=lambda m: m.name.lower())

    def remove(self, model_id: str) -> bool:
        """
        Remove a model from the registry.

        Args:
            model_id: Model identifier

        Returns:
            True if removed

        Raises:
            ModelNotFoundError: If model not found
        """
        if model_id not in self._models:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry.")
        del self._models[model_id]
        self._save()
        return True

    def update(self, model_id: str, **kwargs) -> ModelMetadata:
        """
        Update model metadata.

        Args:
            model_id: Model identifier
            **kwargs: Fields to update

        Returns:
            Updated ModelMetadata

        Raises:
            ModelNotFoundError: If model not found
        """
        if model_id not in self._models:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry.")

        model = self._models[model_id]
        model_dict = model.to_dict()
        model_dict.update(kwargs)
        self._models[model_id] = ModelMetadata.from_dict(model_dict)
        self._save()
        return self._models[model_id]

    def query(
        self,
        min_params: Optional[int] = None,
        max_params: Optional[int] = None,
        architecture: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[ModelMetadata]:
        """
        Query models with filters.

        Args:
            min_params: Minimum total parameters
            max_params: Maximum total parameters
            architecture: Filter by architecture type (e.g., "cnn", "transformer")
            tags: Filter by tags (models must have ALL specified tags)

        Returns:
            List of matching ModelMetadata
        """

        def matches(m: ModelMetadata) -> bool:
            if min_params is not None and m.total_parameters < min_params:
                return False
            if max_params is not None and m.total_parameters > max_params:
                return False
            if architecture is not None:
                if m.architecture_type != architecture and m.architecture_family != architecture:
                    return False
            if tags is not None:
                if not all(tag in m.tags for tag in tags):
                    return False
            return True

        return self.list(filter_fn=matches)

    def generate_id(self, name: str) -> str:
        """
        Generate a unique model ID from name.

        Args:
            name: Model name

        Returns:
            Unique slug-based ID
        """
        base_id = _slugify(name)
        if base_id not in self._models:
            return base_id

        # Add suffix for uniqueness
        counter = 2
        while f"{base_id}-{counter}" in self._models:
            counter += 1
        return f"{base_id}-{counter}"

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._models
