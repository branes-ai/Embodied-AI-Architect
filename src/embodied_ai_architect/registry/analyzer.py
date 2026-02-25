"""Static analysis of PyTorch models."""

from __future__ import annotations

import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from embodied_ai_architect.registry.model_registry import ModelMetadata, _slugify
from embodied_ai_architect.registry.exceptions import ModelLoadError


class ModelAnalyzer:
    """
    Static analysis of PyTorch models.

    Supports multiple model formats:
    - Full model (torch.save(model))
    - State dict (torch.save(model.state_dict()))
    - TorchScript (torch.jit.save())
    - ONNX (torch.onnx.export())
    """

    def analyze(
        self,
        path: Union[str, Path],
        name: Optional[str] = None,
        input_shape: Optional[tuple[int, ...]] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> ModelMetadata:
        """
        Analyze a model file and return metadata.

        Args:
            path: Path to model file
            name: Display name (default: filename)
            input_shape: Input shape for FLOPs estimation and output shape inference
            description: Optional description
            tags: Optional tags for filtering

        Returns:
            ModelMetadata with analysis results

        Raises:
            ModelLoadError: If model cannot be loaded
            AnalysisError: If analysis fails
        """
        path = Path(path).resolve()
        if not path.exists():
            raise ModelLoadError(f"Model file not found: {path}")

        # Detect format and load
        format_type = self._detect_format(path)
        model_or_dict = self._load_model(path, format_type)

        # Generate name and ID
        if name is None:
            name = path.stem
        model_id = _slugify(name)

        # Analyze based on what we loaded
        if isinstance(model_or_dict, dict):
            # State dict - limited analysis
            return self._analyze_state_dict(
                model_or_dict,
                model_id=model_id,
                name=name,
                path=str(path),
                format_type=format_type,
                description=description,
                tags=tags or [],
            )
        elif format_type == "onnx":
            return self._analyze_onnx(
                model_or_dict,
                model_id=model_id,
                name=name,
                path=str(path),
                description=description,
                tags=tags or [],
            )
        else:
            # Full model or JIT
            return self._analyze_module(
                model_or_dict,
                model_id=model_id,
                name=name,
                path=str(path),
                format_type=format_type,
                input_shape=input_shape,
                description=description,
                tags=tags or [],
            )

    def _detect_format(self, path: Path) -> str:
        """
        Detect model format from file.

        Returns:
            "pytorch", "jit", "onnx", or "state_dict"
        """
        suffix = path.suffix.lower()

        # ONNX by extension
        if suffix == ".onnx":
            return "onnx"

        # Check if it's a zip file (TorchScript uses zip)
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                if "model.json" in names or "code/__torch__" in str(names):
                    return "jit"

        # Try to peek at the pickle to determine if it's a state_dict or full model
        try:
            # Load with weights_only=True first (state_dict)
            data = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(data, dict):
                # Check if it looks like a state_dict (OrderedDict with tensor values)
                if all(isinstance(v, torch.Tensor) for v in list(data.values())[:5]):
                    return "state_dict"
                # Check for checkpoint format
                if "state_dict" in data or "model_state_dict" in data:
                    return "state_dict"
            return "pytorch"
        except Exception:
            # If weights_only fails, it might be a full model
            return "pytorch"

    def _load_model(self, path: Path, format_type: str) -> Union[nn.Module, dict, Any]:
        """
        Safely load model in detected format.

        Returns:
            nn.Module, state dict, or ONNX model
        """
        try:
            if format_type == "jit":
                return torch.jit.load(str(path), map_location="cpu")

            elif format_type == "onnx":
                try:
                    import onnx

                    return onnx.load(str(path))
                except ImportError:
                    raise ModelLoadError(
                        "ONNX model detected but 'onnx' package not installed. "
                        "Install with: pip install onnx"
                    )

            elif format_type == "state_dict":
                data = torch.load(path, map_location="cpu", weights_only=True)
                # Extract state_dict from checkpoint if needed
                if isinstance(data, dict):
                    if "state_dict" in data:
                        return data["state_dict"]
                    if "model_state_dict" in data:
                        return data["model_state_dict"]
                return data

            else:  # pytorch full model
                try:
                    model = torch.load(path, map_location="cpu", weights_only=False)
                    if isinstance(model, nn.Module):
                        model.eval()
                    return model
                except Exception as e:
                    # Try state_dict fallback
                    try:
                        data = torch.load(path, map_location="cpu", weights_only=True)
                        if isinstance(data, dict):
                            return data
                    except Exception:
                        pass
                    raise ModelLoadError(
                        f"Cannot load model: {e}\n"
                        "This may be a pickled model that requires the original class definition. "
                        "Try saving as state_dict or TorchScript instead."
                    )

        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")

    def _analyze_module(
        self,
        model: nn.Module,
        model_id: str,
        name: str,
        path: str,
        format_type: str,
        input_shape: Optional[tuple[int, ...]] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> ModelMetadata:
        """Analyze a loaded nn.Module."""
        # Count parameters
        total_params, trainable_params = self._count_parameters(model)

        # Census layers
        layer_counts = self._census_layers(model)

        # Detect architecture
        arch_type, arch_family = self._detect_architecture(model, layer_counts)

        # Estimate memory (FP32)
        memory_mb = (total_params * 4) / (1024 * 1024)

        # Estimate FLOPs and output shape if input_shape provided
        estimated_flops = None
        output_shape = None
        if input_shape is not None:
            estimated_flops = self._estimate_flops(model, input_shape)
            output_shape = self._infer_output_shape(model, input_shape)

        return ModelMetadata(
            id=model_id,
            name=name,
            path=path,
            format=format_type,
            registered_at=datetime.now().isoformat(),
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            estimated_flops=estimated_flops,
            estimated_memory_mb=round(memory_mb, 2),
            input_shape=list(input_shape) if input_shape else None,
            output_shape=output_shape,
            architecture_type=arch_type,
            architecture_family=arch_family,
            layer_counts=layer_counts,
            description=description,
            tags=tags or [],
        )

    def _analyze_state_dict(
        self,
        state_dict: dict,
        model_id: str,
        name: str,
        path: str,
        format_type: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> ModelMetadata:
        """Analyze a state dict (limited analysis)."""
        # Count parameters from tensors
        total_params = 0
        layer_types: Counter = Counter()

        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                total_params += tensor.numel()
                # Infer layer type from key
                if ".weight" in key or ".bias" in key:
                    # Extract layer name pattern
                    parts = key.rsplit(".", 1)[0]
                    if "conv" in parts.lower():
                        layer_types["Conv2d"] += 1
                    elif "bn" in parts.lower() or "batch" in parts.lower():
                        layer_types["BatchNorm2d"] += 1
                    elif "fc" in parts.lower() or "linear" in parts.lower():
                        layer_types["Linear"] += 1
                    elif "attention" in parts.lower():
                        layer_types["Attention"] += 1

        # Estimate memory
        memory_mb = (total_params * 4) / (1024 * 1024)

        # Guess architecture from layer types
        arch_type = "unknown"
        arch_family = "unknown"
        if layer_types.get("Conv2d", 0) > 5:
            arch_type = "cnn"
        elif layer_types.get("Attention", 0) > 0:
            arch_type = "transformer"
        elif layer_types.get("Linear", 0) > 2:
            arch_type = "mlp"

        return ModelMetadata(
            id=model_id,
            name=name,
            path=path,
            format=format_type,
            registered_at=datetime.now().isoformat(),
            total_parameters=total_params,
            trainable_parameters=total_params,  # Assume all trainable for state_dict
            estimated_flops=None,
            estimated_memory_mb=round(memory_mb, 2),
            input_shape=None,
            output_shape=None,
            architecture_type=arch_type,
            architecture_family=arch_family,
            layer_counts=dict(layer_types),
            description=description,
            tags=tags or [],
        )

    def _analyze_onnx(
        self,
        onnx_model: Any,
        model_id: str,
        name: str,
        path: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> ModelMetadata:
        """Analyze an ONNX model."""
        from onnx import numpy_helper

        # Count parameters
        total_params = 0
        for initializer in onnx_model.graph.initializer:
            tensor = numpy_helper.to_array(initializer)
            total_params += tensor.size

        # Count ops
        layer_types: Counter = Counter()
        for node in onnx_model.graph.node:
            layer_types[node.op_type] += 1

        # Get input/output shapes
        input_shape = None
        output_shape = None
        if onnx_model.graph.input:
            inp = onnx_model.graph.input[0]
            if inp.type.tensor_type.shape.dim:
                input_shape = [
                    d.dim_value if d.dim_value > 0 else -1 for d in inp.type.tensor_type.shape.dim
                ]
        if onnx_model.graph.output:
            out = onnx_model.graph.output[0]
            if out.type.tensor_type.shape.dim:
                output_shape = [
                    d.dim_value if d.dim_value > 0 else -1 for d in out.type.tensor_type.shape.dim
                ]

        # Estimate memory
        memory_mb = (total_params * 4) / (1024 * 1024)

        # Detect architecture
        arch_type = "unknown"
        arch_family = "unknown"
        if layer_types.get("Conv", 0) > 5:
            arch_type = "cnn"
        elif layer_types.get("Attention", 0) > 0 or layer_types.get("MatMul", 0) > 10:
            arch_type = "transformer"

        return ModelMetadata(
            id=model_id,
            name=name,
            path=path,
            format="onnx",
            registered_at=datetime.now().isoformat(),
            total_parameters=total_params,
            trainable_parameters=0,  # ONNX models are inference-only
            estimated_flops=None,
            estimated_memory_mb=round(memory_mb, 2),
            input_shape=input_shape,
            output_shape=output_shape,
            architecture_type=arch_type,
            architecture_family=arch_family,
            layer_counts=dict(layer_types),
            description=description,
            tags=tags or [],
        )

    def _count_parameters(self, model: nn.Module) -> tuple[int, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def _census_layers(self, model: nn.Module) -> dict[str, int]:
        """Count layer types in the model."""
        counts: Counter = Counter()
        for module in model.modules():
            counts[type(module).__name__] += 1
        # Remove the root module itself
        if type(model).__name__ in counts:
            counts[type(model).__name__] -= 1
        # Filter out empty counts and sort by count
        return dict(
            sorted(
                ((k, v) for k, v in counts.items() if v > 0),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    def _detect_architecture(
        self, model: nn.Module, layer_counts: dict[str, int]
    ) -> tuple[str, str]:
        """Detect architecture type and family."""
        model_class = type(model).__name__.lower()
        model_str = str(type(model)).lower()

        # CNN detection
        conv_count = layer_counts.get("Conv2d", 0) + layer_counts.get("Conv1d", 0)
        if conv_count > 5:
            # YOLO variants
            if "yolo" in model_class or "yolo" in model_str:
                return ("cnn", "yolo")
            if "darknet" in model_class or "csp" in model_class:
                return ("cnn", "yolo")

            # ResNet variants
            if "resnet" in model_class or "resnet" in model_str:
                return ("cnn", "resnet")
            if layer_counts.get("Bottleneck", 0) > 0 or layer_counts.get("BasicBlock", 0) > 0:
                return ("cnn", "resnet")

            # EfficientNet
            if "efficient" in model_class:
                return ("cnn", "efficientnet")

            # MobileNet
            if "mobile" in model_class:
                return ("cnn", "mobilenet")

            # VGG
            if "vgg" in model_class:
                return ("cnn", "vgg")

            return ("cnn", "unknown")

        # Transformer detection
        attention_count = layer_counts.get("MultiheadAttention", 0)
        if attention_count > 0 or "transformer" in model_class:
            # ViT
            if "vit" in model_class or "vision" in model_class:
                return ("transformer", "vit")

            # BERT
            if "bert" in model_class:
                return ("transformer", "bert")

            # GPT
            if "gpt" in model_class:
                return ("transformer", "gpt")

            return ("transformer", "unknown")

        # MLP
        linear_count = layer_counts.get("Linear", 0)
        if linear_count > 2 and conv_count == 0:
            return ("mlp", "unknown")

        # RNN/LSTM
        if layer_counts.get("LSTM", 0) > 0:
            return ("rnn", "lstm")
        if layer_counts.get("GRU", 0) > 0:
            return ("rnn", "gru")

        return ("unknown", "unknown")

    def _estimate_flops(self, model: nn.Module, input_shape: tuple[int, ...]) -> Optional[int]:
        """
        Estimate FLOPs for the model.

        Uses torch.profiler if available, falls back to rough estimation.
        """
        try:
            # Try using thop if available
            try:
                from thop import profile

                dummy_input = torch.randn(*input_shape)
                flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
                return int(flops)
            except ImportError:
                pass

            # Fallback: rough estimation based on layer types
            # This is a very rough estimate
            total_flops = 0
            dummy_input = torch.randn(*input_shape)

            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    # FLOPs = 2 * K^2 * Cin * Cout * Hout * Wout
                    # Rough estimate assuming output size similar to input
                    k = module.kernel_size[0] * module.kernel_size[1]
                    flops = 2 * k * module.in_channels * module.out_channels
                    # Estimate output size (rough)
                    flops *= input_shape[2] * input_shape[3] // (module.stride[0] ** 2)
                    total_flops += flops
                elif isinstance(module, nn.Linear):
                    total_flops += 2 * module.in_features * module.out_features

            return total_flops if total_flops > 0 else None

        except Exception:
            return None

    def _infer_output_shape(
        self, model: nn.Module, input_shape: tuple[int, ...]
    ) -> Optional[list[int]]:
        """Infer output shape by running a forward pass."""
        try:
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(*input_shape)
                output = model(dummy_input)
                if isinstance(output, torch.Tensor):
                    return list(output.shape)
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    if isinstance(output[0], torch.Tensor):
                        return list(output[0].shape)
        except Exception:
            pass
        return None
