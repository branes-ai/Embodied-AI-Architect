"""Model discovery service.

Provides unified search across all model providers and integrates
with the embodied-schemas catalog when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .providers.base import ModelProvider, ModelQuery


@dataclass
class ModelCandidate:
    """A discovered model candidate."""

    id: str
    name: str
    provider: str
    task: str
    parameters: Optional[int] = None
    flops: Optional[int] = None
    accuracy: Optional[float] = None
    input_shape: Optional[tuple[int, ...]] = None
    benchmarked: bool = False
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @classmethod
    def from_provider_info(cls, info: dict[str, Any]) -> "ModelCandidate":
        """Create from provider list_models result."""
        return cls(
            id=info["id"],
            name=info.get("name", info["id"]),
            provider=info.get("provider", "unknown"),
            task=info.get("task", "unknown"),
            parameters=info.get("parameters"),
            flops=info.get("flops"),
            accuracy=info.get("map50") or info.get("top1") or info.get("accuracy"),
            input_shape=info.get("input_shape"),
            benchmarked=info.get("benchmarked", False),
            metadata=info,
        )

    def format_params(self) -> str:
        """Format parameter count for display."""
        if self.parameters is None:
            return "N/A"
        if self.parameters >= 1_000_000_000:
            return f"{self.parameters / 1_000_000_000:.1f}B"
        elif self.parameters >= 1_000_000:
            return f"{self.parameters / 1_000_000:.1f}M"
        elif self.parameters >= 1_000:
            return f"{self.parameters / 1_000:.1f}K"
        return str(self.parameters)

    def format_accuracy(self) -> str:
        """Format accuracy for display."""
        if self.accuracy is None:
            return "N/A"
        return f"{self.accuracy * 100:.1f}%"


class ModelDiscoveryService:
    """Unified model discovery across providers.

    Searches all registered providers and optionally integrates
    with embodied-schemas ModelEntry catalog.
    """

    def __init__(self):
        self._providers: dict[str, ModelProvider] = {}
        self._register_default_providers()

    def _register_default_providers(self) -> None:
        """Register built-in providers."""
        # Import here to avoid circular imports
        from .providers.ultralytics import UltralyticsProvider
        from .providers.torchvision import TorchVisionProvider
        from .providers.huggingface import HuggingFaceProvider
        from .providers.timm import TimmProvider
        from .providers.onnx_zoo import ONNXModelZooProvider
        from .providers.mediapipe import MediaPipeProvider

        self.register_provider(UltralyticsProvider())
        self.register_provider(TorchVisionProvider())
        self.register_provider(HuggingFaceProvider())
        self.register_provider(TimmProvider())
        self.register_provider(ONNXModelZooProvider())
        self.register_provider(MediaPipeProvider())

    def register_provider(self, provider: ModelProvider) -> None:
        """Register a model provider."""
        self._providers[provider.name] = provider

    def discover(
        self,
        query: Optional[ModelQuery] = None,
        providers: Optional[list[str]] = None,
    ) -> list[ModelCandidate]:
        """Discover models matching a query.

        Args:
            query: Filter criteria
            providers: List of provider names to search (all if None)

        Returns:
            List of matching ModelCandidate objects
        """
        candidates = []

        # Filter providers if specified
        search_providers = self._providers
        if providers is not None:
            search_providers = {k: v for k, v in self._providers.items() if k in providers}

        # Search each provider
        for provider in search_providers.values():
            try:
                models = provider.list_models(query)
                for model_info in models:
                    candidates.append(ModelCandidate.from_provider_info(model_info))
            except Exception as e:
                # Log but don't fail on provider errors
                print(f"Warning: Error searching {provider.name}: {e}")

        # Try to enrich from embodied-schemas if available
        candidates = self._enrich_from_schemas(candidates)

        # Sort by parameters (smallest first for edge deployment focus)
        return sorted(candidates, key=lambda c: c.parameters or float("inf"))

    def _enrich_from_schemas(self, candidates: list[ModelCandidate]) -> list[ModelCandidate]:
        """Enrich candidates with data from embodied-schemas."""
        try:
            from embodied_schemas import Registry

            registry = Registry()

            for candidate in candidates:
                # Try to find matching ModelEntry
                try:
                    entry = registry.get_model(candidate.id)
                    if entry:
                        # Update with schema data
                        if entry.parameters and not candidate.parameters:
                            candidate.parameters = entry.parameters
                        if hasattr(entry, "benchmarked") and entry.benchmarked:
                            candidate.benchmarked = True
                except Exception:
                    pass  # Model not in schema registry

        except ImportError:
            pass  # embodied-schemas not available

        return candidates

    def find_by_task(self, task: str) -> list[ModelCandidate]:
        """Find models for a specific task.

        Args:
            task: Task type (detection, classification, segmentation, etc.)

        Returns:
            List of matching models
        """
        return self.discover(ModelQuery(task=task))

    def find_for_constraints(
        self,
        task: str,
        max_params: Optional[int] = None,
        min_accuracy: Optional[float] = None,
    ) -> list[ModelCandidate]:
        """Find models matching deployment constraints.

        Args:
            task: Task type
            max_params: Maximum parameter count
            min_accuracy: Minimum accuracy (mAP, top-1, etc.)

        Returns:
            List of matching models sorted by size
        """
        query = ModelQuery(
            task=task,
            max_params=max_params,
            min_accuracy=min_accuracy,
            benchmarked=True,
        )
        return self.discover(query)

    def find_for_usecase(self, usecase_id: str) -> list[ModelCandidate]:
        """Find recommended models for a use case.

        Args:
            usecase_id: Use case identifier from embodied-schemas

        Returns:
            List of recommended models
        """
        try:
            from embodied_schemas import Registry

            registry = Registry()
            usecase = registry.get_usecase(usecase_id)

            if usecase and hasattr(usecase, "recommended_models"):
                # Get models from recommendations
                candidates = []
                for model_id in usecase.recommended_models:
                    for provider in self._providers.values():
                        try:
                            info = provider.get_model_info(model_id)
                            candidates.append(ModelCandidate.from_provider_info(info))
                            break
                        except Exception:
                            continue
                return candidates

        except ImportError:
            pass  # embodied-schemas not available
        except Exception:
            pass

        return []

    @property
    def providers(self) -> list[str]:
        """List registered provider names."""
        return list(self._providers.keys())


# Module-level convenience function
_discovery_service: Optional[ModelDiscoveryService] = None


def discover(
    task: Optional[str] = None,
    max_params: Optional[int] = None,
    min_accuracy: Optional[float] = None,
    query: Optional[str] = None,
    **kwargs,
) -> list[ModelCandidate]:
    """Discover available models.

    Convenience function for quick model discovery.

    Args:
        task: Filter by task (detection, classification, etc.)
        max_params: Maximum parameter count
        min_accuracy: Minimum accuracy
        query: Free-text search query
        **kwargs: Additional ModelQuery parameters

    Returns:
        List of matching ModelCandidate objects
    """
    global _discovery_service
    if _discovery_service is None:
        _discovery_service = ModelDiscoveryService()

    model_query = ModelQuery(
        task=task,
        max_params=max_params,
        min_accuracy=min_accuracy,
        query=query,
        **kwargs,
    )

    return _discovery_service.discover(model_query)
