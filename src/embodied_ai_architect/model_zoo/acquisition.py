"""Unified model acquisition API.

Provides a simple interface for downloading models from any provider
with automatic caching and format conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .cache import ModelCache
from .discovery import ModelDiscoveryService
from .providers.base import ModelFormat, ModelArtifact


class ModelAcquisition:
    """Unified API for acquiring models.

    Handles:
    - Provider detection from model ID
    - Cache checking and management
    - Download coordination
    - Format conversion
    """

    def __init__(
        self,
        cache: Optional[ModelCache] = None,
        discovery: Optional[ModelDiscoveryService] = None,
    ):
        """Initialize the acquisition service.

        Args:
            cache: Custom cache instance (creates default if None)
            discovery: Custom discovery service (creates default if None)
        """
        self.cache = cache or ModelCache()
        self.discovery = discovery or ModelDiscoveryService()

    def acquire(
        self,
        model_id: str,
        format: ModelFormat = ModelFormat.ONNX,
        provider: Optional[str] = None,
        force_download: bool = False,
    ) -> Path:
        """Acquire a model, downloading if necessary.

        Args:
            model_id: Model identifier (e.g., 'yolov8n', 'resnet18')
            format: Target format (default: ONNX)
            provider: Provider name (auto-detected if None)
            force_download: If True, re-download even if cached

        Returns:
            Path to the model file

        Raises:
            ValueError: If model not found in any provider
            RuntimeError: If download fails
        """
        # Detect provider if not specified
        if provider is None:
            provider = self._detect_provider(model_id)

        # Check cache first
        if not force_download:
            cached_path = self.cache.get(model_id, provider, format)
            if cached_path is not None:
                return cached_path

        # Get provider instance
        provider_instance = self.discovery._providers.get(provider)
        if provider_instance is None:
            raise ValueError(f"Unknown provider: {provider}")

        # Check format support
        if not provider_instance.supports_format(format):
            raise ValueError(
                f"Provider {provider} does not support {format.value} format. "
                f"Supported: {[f.value for f in provider_instance.supported_formats]}"
            )

        # Download
        cache_dir = self.cache.get_provider_dir(provider)
        artifact = provider_instance.download(model_id, format, cache_dir)

        # Update cache index
        self.cache.put(
            model_id=model_id,
            provider=provider,
            format=format,
            path=artifact.path,
            version=artifact.version,
            metadata=artifact.metadata,
        )

        return artifact.path

    def acquire_for_operator(
        self,
        operator_id: str,
        format: ModelFormat = ModelFormat.ONNX,
    ) -> Path:
        """Acquire a model required by an operator.

        Looks up the OperatorEntry in embodied-schemas to find
        the required model.

        Args:
            operator_id: Operator identifier
            format: Target format

        Returns:
            Path to the model file

        Raises:
            ValueError: If operator or required model not found
        """
        try:
            from embodied_schemas import Registry

            registry = Registry()
            operator = registry.get_operator(operator_id)

            if operator is None:
                raise ValueError(f"Operator '{operator_id}' not found in registry")

            if not hasattr(operator, "requires_model") or not operator.requires_model:
                raise ValueError(f"Operator '{operator_id}' does not specify a required model")

            model_id = operator.requires_model
            return self.acquire(model_id, format)

        except ImportError:
            raise ValueError(
                "embodied-schemas not installed. " "Install with: pip install embodied-schemas"
            )

    def _detect_provider(self, model_id: str) -> str:
        """Detect provider from model ID.

        Uses naming conventions and discovery to find the right provider.

        Args:
            model_id: Model identifier

        Returns:
            Provider name

        Raises:
            ValueError: If provider cannot be determined
        """
        model_lower = model_id.lower()

        # YOLO models -> ultralytics
        if any(
            pattern in model_lower
            for pattern in ["yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolo11"]
        ):
            return "ultralytics"

        # ResNet, VGG, EfficientNet, etc. -> torchvision
        torchvision_patterns = [
            "resnet",
            "vgg",
            "densenet",
            "mobilenet",
            "efficientnet",
            "regnet",
            "shufflenet",
            "squeezenet",
            "mnasnet",
            "googlenet",
            "inception",
            "alexnet",
        ]
        if any(pattern in model_lower for pattern in torchvision_patterns):
            return "torchvision"

        # Check all providers for a match
        for provider_name, provider in self.discovery._providers.items():
            try:
                info = provider.get_model_info(model_id)
                if info:
                    return provider_name
            except Exception:
                continue

        raise ValueError(
            f"Cannot determine provider for '{model_id}'. "
            f"Available providers: {list(self.discovery._providers.keys())}"
        )

    def list_cached(self) -> list[dict]:
        """List all cached models."""
        entries = self.cache.list()
        return [
            {
                "model_id": e.model_id,
                "provider": e.provider,
                "format": e.format,
                "path": e.path,
                "size_mb": e.size_bytes / (1024 * 1024),
                "cached_at": e.cached_at,
            }
            for e in entries
        ]

    def clear_cache(self, provider: Optional[str] = None) -> int:
        """Clear cached models.

        Args:
            provider: Optional provider to clear (all if None)

        Returns:
            Number of entries cleared
        """
        return self.cache.clear(provider)


# Module-level singleton and convenience functions
_acquisition: Optional[ModelAcquisition] = None


def _get_acquisition() -> ModelAcquisition:
    """Get or create the global acquisition instance."""
    global _acquisition
    if _acquisition is None:
        _acquisition = ModelAcquisition()
    return _acquisition


def acquire(
    model_id: str,
    format: str = "onnx",
    provider: Optional[str] = None,
    force_download: bool = False,
) -> Path:
    """Acquire a model, downloading if necessary.

    Convenience function for quick model acquisition.

    Args:
        model_id: Model identifier (e.g., 'yolov8n', 'resnet18')
        format: Target format string (onnx, pytorch, torchscript, etc.)
        provider: Provider name (auto-detected if None)
        force_download: If True, re-download even if cached

    Returns:
        Path to the model file

    Example:
        >>> from embodied_ai_architect.model_zoo import acquire
        >>> path = acquire("yolov8n", format="onnx")
        >>> print(path)
        ~/.cache/branes/models/ultralytics/yolov8n.onnx
    """
    # Convert string format to enum
    format_enum = ModelFormat(format.lower())
    return _get_acquisition().acquire(model_id, format_enum, provider, force_download)


def acquire_for_operator(operator_id: str, format: str = "onnx") -> Path:
    """Acquire the model required by an operator.

    Args:
        operator_id: Operator identifier
        format: Target format string

    Returns:
        Path to the model file
    """
    format_enum = ModelFormat(format.lower())
    return _get_acquisition().acquire_for_operator(operator_id, format_enum)
