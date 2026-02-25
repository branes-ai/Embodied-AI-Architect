"""TorchVision provider for classification and detection models.

Supports ResNet, EfficientNet, MobileNet, VGG, and other torchvision models
in PyTorch, TorchScript, and ONNX formats.
"""

from pathlib import Path
from typing import Any, Optional

from .base import ModelProvider, ModelFormat, ModelQuery, ModelArtifact

# TorchVision model catalog with known specifications
# Source: https://pytorch.org/vision/stable/models.html
TORCHVISION_MODELS = {
    # ResNet family
    "resnet18": {
        "name": "ResNet-18",
        "task": "classification",
        "parameters": 11_689_512,
        "flops": 1_800_000_000,
        "top1": 0.6976,
        "top5": 0.8912,
        "input_shape": (1, 3, 224, 224),
        "family": "resnet",
        "weights": "IMAGENET1K_V1",
    },
    "resnet34": {
        "name": "ResNet-34",
        "task": "classification",
        "parameters": 21_797_672,
        "flops": 3_600_000_000,
        "top1": 0.7340,
        "top5": 0.9142,
        "input_shape": (1, 3, 224, 224),
        "family": "resnet",
        "weights": "IMAGENET1K_V1",
    },
    "resnet50": {
        "name": "ResNet-50",
        "task": "classification",
        "parameters": 25_557_032,
        "flops": 4_100_000_000,
        "top1": 0.8034,
        "top5": 0.9510,
        "input_shape": (1, 3, 224, 224),
        "family": "resnet",
        "weights": "IMAGENET1K_V2",
    },
    "resnet101": {
        "name": "ResNet-101",
        "task": "classification",
        "parameters": 44_549_160,
        "flops": 7_800_000_000,
        "top1": 0.8148,
        "top5": 0.9558,
        "input_shape": (1, 3, 224, 224),
        "family": "resnet",
        "weights": "IMAGENET1K_V2",
    },
    "resnet152": {
        "name": "ResNet-152",
        "task": "classification",
        "parameters": 60_192_808,
        "flops": 11_500_000_000,
        "top1": 0.8212,
        "top5": 0.9592,
        "input_shape": (1, 3, 224, 224),
        "family": "resnet",
        "weights": "IMAGENET1K_V2",
    },
    # MobileNet family
    "mobilenet_v2": {
        "name": "MobileNet V2",
        "task": "classification",
        "parameters": 3_504_872,
        "flops": 300_000_000,
        "top1": 0.7200,
        "top5": 0.9046,
        "input_shape": (1, 3, 224, 224),
        "family": "mobilenet",
        "weights": "IMAGENET1K_V2",
    },
    "mobilenet_v3_small": {
        "name": "MobileNet V3 Small",
        "task": "classification",
        "parameters": 2_542_856,
        "flops": 56_000_000,
        "top1": 0.6792,
        "top5": 0.8742,
        "input_shape": (1, 3, 224, 224),
        "family": "mobilenet",
        "weights": "IMAGENET1K_V1",
    },
    "mobilenet_v3_large": {
        "name": "MobileNet V3 Large",
        "task": "classification",
        "parameters": 5_483_032,
        "flops": 217_000_000,
        "top1": 0.7504,
        "top5": 0.9228,
        "input_shape": (1, 3, 224, 224),
        "family": "mobilenet",
        "weights": "IMAGENET1K_V2",
    },
    # EfficientNet family
    "efficientnet_b0": {
        "name": "EfficientNet B0",
        "task": "classification",
        "parameters": 5_288_548,
        "flops": 390_000_000,
        "top1": 0.7742,
        "top5": 0.9356,
        "input_shape": (1, 3, 224, 224),
        "family": "efficientnet",
        "weights": "IMAGENET1K_V1",
    },
    "efficientnet_b1": {
        "name": "EfficientNet B1",
        "task": "classification",
        "parameters": 7_794_184,
        "flops": 700_000_000,
        "top1": 0.7884,
        "top5": 0.9426,
        "input_shape": (1, 3, 240, 240),
        "family": "efficientnet",
        "weights": "IMAGENET1K_V2",
    },
    "efficientnet_b2": {
        "name": "EfficientNet B2",
        "task": "classification",
        "parameters": 9_109_994,
        "flops": 1_000_000_000,
        "top1": 0.8034,
        "top5": 0.9510,
        "input_shape": (1, 3, 288, 288),
        "family": "efficientnet",
        "weights": "IMAGENET1K_V1",
    },
    "efficientnet_b3": {
        "name": "EfficientNet B3",
        "task": "classification",
        "parameters": 12_233_232,
        "flops": 1_800_000_000,
        "top1": 0.8232,
        "top5": 0.9602,
        "input_shape": (1, 3, 300, 300),
        "family": "efficientnet",
        "weights": "IMAGENET1K_V1",
    },
    "efficientnet_b4": {
        "name": "EfficientNet B4",
        "task": "classification",
        "parameters": 19_341_616,
        "flops": 4_400_000_000,
        "top1": 0.8314,
        "top5": 0.9640,
        "input_shape": (1, 3, 380, 380),
        "family": "efficientnet",
        "weights": "IMAGENET1K_V1",
    },
    # VGG family
    "vgg11": {
        "name": "VGG-11",
        "task": "classification",
        "parameters": 132_863_336,
        "flops": 7_600_000_000,
        "top1": 0.6902,
        "top5": 0.8862,
        "input_shape": (1, 3, 224, 224),
        "family": "vgg",
        "weights": "IMAGENET1K_V1",
    },
    "vgg16": {
        "name": "VGG-16",
        "task": "classification",
        "parameters": 138_357_544,
        "flops": 15_500_000_000,
        "top1": 0.7160,
        "top5": 0.9046,
        "input_shape": (1, 3, 224, 224),
        "family": "vgg",
        "weights": "IMAGENET1K_V1",
    },
    "vgg19": {
        "name": "VGG-19",
        "task": "classification",
        "parameters": 143_667_240,
        "flops": 19_600_000_000,
        "top1": 0.7246,
        "top5": 0.9088,
        "input_shape": (1, 3, 224, 224),
        "family": "vgg",
        "weights": "IMAGENET1K_V1",
    },
    # DenseNet family
    "densenet121": {
        "name": "DenseNet-121",
        "task": "classification",
        "parameters": 7_978_856,
        "flops": 2_900_000_000,
        "top1": 0.7508,
        "top5": 0.9236,
        "input_shape": (1, 3, 224, 224),
        "family": "densenet",
        "weights": "IMAGENET1K_V1",
    },
    "densenet169": {
        "name": "DenseNet-169",
        "task": "classification",
        "parameters": 14_149_480,
        "flops": 3_400_000_000,
        "top1": 0.7600,
        "top5": 0.9300,
        "input_shape": (1, 3, 224, 224),
        "family": "densenet",
        "weights": "IMAGENET1K_V1",
    },
    "densenet201": {
        "name": "DenseNet-201",
        "task": "classification",
        "parameters": 20_013_928,
        "flops": 4_300_000_000,
        "top1": 0.7700,
        "top5": 0.9350,
        "input_shape": (1, 3, 224, 224),
        "family": "densenet",
        "weights": "IMAGENET1K_V1",
    },
    # SqueezeNet
    "squeezenet1_0": {
        "name": "SqueezeNet 1.0",
        "task": "classification",
        "parameters": 1_248_424,
        "flops": 830_000_000,
        "top1": 0.5810,
        "top5": 0.8010,
        "input_shape": (1, 3, 224, 224),
        "family": "squeezenet",
        "weights": "IMAGENET1K_V1",
    },
    "squeezenet1_1": {
        "name": "SqueezeNet 1.1",
        "task": "classification",
        "parameters": 1_235_496,
        "flops": 355_000_000,
        "top1": 0.5820,
        "top5": 0.8020,
        "input_shape": (1, 3, 224, 224),
        "family": "squeezenet",
        "weights": "IMAGENET1K_V1",
    },
    # ShuffleNet
    "shufflenet_v2_x0_5": {
        "name": "ShuffleNet V2 x0.5",
        "task": "classification",
        "parameters": 1_366_792,
        "flops": 41_000_000,
        "top1": 0.6049,
        "top5": 0.8155,
        "input_shape": (1, 3, 224, 224),
        "family": "shufflenet",
        "weights": "IMAGENET1K_V1",
    },
    "shufflenet_v2_x1_0": {
        "name": "ShuffleNet V2 x1.0",
        "task": "classification",
        "parameters": 2_278_604,
        "flops": 146_000_000,
        "top1": 0.6942,
        "top5": 0.8880,
        "input_shape": (1, 3, 224, 224),
        "family": "shufflenet",
        "weights": "IMAGENET1K_V1",
    },
    # RegNet (efficient models)
    "regnet_y_400mf": {
        "name": "RegNet Y 400MF",
        "task": "classification",
        "parameters": 4_344_144,
        "flops": 400_000_000,
        "top1": 0.7500,
        "top5": 0.9220,
        "input_shape": (1, 3, 224, 224),
        "family": "regnet",
        "weights": "IMAGENET1K_V2",
    },
    "regnet_y_800mf": {
        "name": "RegNet Y 800MF",
        "task": "classification",
        "parameters": 6_432_512,
        "flops": 800_000_000,
        "top1": 0.7694,
        "top5": 0.9350,
        "input_shape": (1, 3, 224, 224),
        "family": "regnet",
        "weights": "IMAGENET1K_V2",
    },
    "regnet_y_1_6gf": {
        "name": "RegNet Y 1.6GF",
        "task": "classification",
        "parameters": 11_202_430,
        "flops": 1_600_000_000,
        "top1": 0.7858,
        "top5": 0.9432,
        "input_shape": (1, 3, 224, 224),
        "family": "regnet",
        "weights": "IMAGENET1K_V2",
    },
    # ConvNeXt (modern CNN)
    "convnext_tiny": {
        "name": "ConvNeXt Tiny",
        "task": "classification",
        "parameters": 28_589_128,
        "flops": 4_500_000_000,
        "top1": 0.8252,
        "top5": 0.9608,
        "input_shape": (1, 3, 224, 224),
        "family": "convnext",
        "weights": "IMAGENET1K_V1",
    },
    "convnext_small": {
        "name": "ConvNeXt Small",
        "task": "classification",
        "parameters": 50_223_688,
        "flops": 8_700_000_000,
        "top1": 0.8338,
        "top5": 0.9656,
        "input_shape": (1, 3, 224, 224),
        "family": "convnext",
        "weights": "IMAGENET1K_V1",
    },
    # Vision Transformer
    "vit_b_16": {
        "name": "ViT Base 16",
        "task": "classification",
        "parameters": 86_567_656,
        "flops": 17_600_000_000,
        "top1": 0.8107,
        "top5": 0.9545,
        "input_shape": (1, 3, 224, 224),
        "family": "vit",
        "weights": "IMAGENET1K_V1",
    },
    "vit_b_32": {
        "name": "ViT Base 32",
        "task": "classification",
        "parameters": 88_224_232,
        "flops": 4_400_000_000,
        "top1": 0.7590,
        "top5": 0.9290,
        "input_shape": (1, 3, 224, 224),
        "family": "vit",
        "weights": "IMAGENET1K_V1",
    },
    "vit_l_16": {
        "name": "ViT Large 16",
        "task": "classification",
        "parameters": 304_326_632,
        "flops": 61_600_000_000,
        "top1": 0.7970,
        "top5": 0.9450,
        "input_shape": (1, 3, 224, 224),
        "family": "vit",
        "weights": "IMAGENET1K_V1",
    },
    # Swin Transformer
    "swin_t": {
        "name": "Swin Transformer Tiny",
        "task": "classification",
        "parameters": 28_288_354,
        "flops": 4_500_000_000,
        "top1": 0.8114,
        "top5": 0.9544,
        "input_shape": (1, 3, 224, 224),
        "family": "swin",
        "weights": "IMAGENET1K_V1",
    },
    "swin_s": {
        "name": "Swin Transformer Small",
        "task": "classification",
        "parameters": 49_606_258,
        "flops": 8_700_000_000,
        "top1": 0.8318,
        "top5": 0.9642,
        "input_shape": (1, 3, 224, 224),
        "family": "swin",
        "weights": "IMAGENET1K_V1",
    },
    "swin_b": {
        "name": "Swin Transformer Base",
        "task": "classification",
        "parameters": 87_768_224,
        "flops": 15_400_000_000,
        "top1": 0.8358,
        "top5": 0.9664,
        "input_shape": (1, 3, 224, 224),
        "family": "swin",
        "weights": "IMAGENET1K_V1",
    },
}


# Map model names to torchvision module functions
MODEL_LOADERS = {
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "resnet34": ("resnet34", "ResNet34_Weights"),
    "resnet50": ("resnet50", "ResNet50_Weights"),
    "resnet101": ("resnet101", "ResNet101_Weights"),
    "resnet152": ("resnet152", "ResNet152_Weights"),
    "mobilenet_v2": ("mobilenet_v2", "MobileNet_V2_Weights"),
    "mobilenet_v3_small": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
    "efficientnet_b1": ("efficientnet_b1", "EfficientNet_B1_Weights"),
    "efficientnet_b2": ("efficientnet_b2", "EfficientNet_B2_Weights"),
    "efficientnet_b3": ("efficientnet_b3", "EfficientNet_B3_Weights"),
    "efficientnet_b4": ("efficientnet_b4", "EfficientNet_B4_Weights"),
    "vgg11": ("vgg11", "VGG11_Weights"),
    "vgg16": ("vgg16", "VGG16_Weights"),
    "vgg19": ("vgg19", "VGG19_Weights"),
    "densenet121": ("densenet121", "DenseNet121_Weights"),
    "densenet169": ("densenet169", "DenseNet169_Weights"),
    "densenet201": ("densenet201", "DenseNet201_Weights"),
    "squeezenet1_0": ("squeezenet1_0", "SqueezeNet1_0_Weights"),
    "squeezenet1_1": ("squeezenet1_1", "SqueezeNet1_1_Weights"),
    "shufflenet_v2_x0_5": ("shufflenet_v2_x0_5", "ShuffleNet_V2_X0_5_Weights"),
    "shufflenet_v2_x1_0": ("shufflenet_v2_x1_0", "ShuffleNet_V2_X1_0_Weights"),
    "regnet_y_400mf": ("regnet_y_400mf", "RegNet_Y_400MF_Weights"),
    "regnet_y_800mf": ("regnet_y_800mf", "RegNet_Y_800MF_Weights"),
    "regnet_y_1_6gf": ("regnet_y_1_6gf", "RegNet_Y_1_6GF_Weights"),
    "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights"),
    "convnext_small": ("convnext_small", "ConvNeXt_Small_Weights"),
    "vit_b_16": ("vit_b_16", "ViT_B_16_Weights"),
    "vit_b_32": ("vit_b_32", "ViT_B_32_Weights"),
    "vit_l_16": ("vit_l_16", "ViT_L_16_Weights"),
    "swin_t": ("swin_t", "Swin_T_Weights"),
    "swin_s": ("swin_s", "Swin_S_Weights"),
    "swin_b": ("swin_b", "Swin_B_Weights"),
}


class TorchVisionProvider(ModelProvider):
    """Provider for TorchVision pretrained models.

    Downloads and exports models from torchvision.models with
    pretrained ImageNet weights.
    """

    @property
    def name(self) -> str:
        return "torchvision"

    @property
    def supported_formats(self) -> list[ModelFormat]:
        return [
            ModelFormat.PYTORCH,
            ModelFormat.TORCHSCRIPT,
            ModelFormat.ONNX,
        ]

    def list_models(self, query: Optional[ModelQuery] = None) -> list[dict[str, Any]]:
        """List available TorchVision models."""
        models = []
        for model_id, info in TORCHVISION_MODELS.items():
            model_info = {
                "id": model_id,
                "provider": self.name,
                "benchmarked": True,
                "accuracy": info.get("top1"),
                **info,
            }

            if query is None or query.matches(model_info):
                models.append(model_info)

        return sorted(models, key=lambda m: m.get("parameters", 0))

    def download(
        self,
        model_id: str,
        format: ModelFormat,
        cache_dir: Path,
    ) -> ModelArtifact:
        """Download and export a TorchVision model.

        Args:
            model_id: Model identifier (e.g., 'resnet18', 'efficientnet_b0')
            format: Target export format
            cache_dir: Directory to store the model

        Returns:
            ModelArtifact with path and metadata
        """
        if not self.supports_format(format):
            raise ValueError(
                f"Format {format} not supported. "
                f"Supported: {[f.value for f in self.supported_formats]}"
            )

        # Get model info
        info = TORCHVISION_MODELS.get(model_id)
        if info is None:
            raise ValueError(f"Model '{model_id}' not found in TorchVision catalog")

        # Determine file extension
        ext_map = {
            ModelFormat.PYTORCH: ".pt",
            ModelFormat.TORCHSCRIPT: ".torchscript",
            ModelFormat.ONNX: ".onnx",
        }

        ext = ext_map[format]
        model_filename = f"{model_id}{ext}"
        model_path = cache_dir / model_filename

        # Check if already cached
        if model_path.exists():
            return self._create_artifact(model_id, format, model_path, info)

        # Ensure cache directory exists
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Load model from torchvision
        try:
            import torch
            import torchvision.models as models
        except ImportError:
            raise ImportError(
                "torch and torchvision not installed. "
                "Install with: pip install torch torchvision"
            )

        # Get model loader info
        loader_info = MODEL_LOADERS.get(model_id)
        if loader_info is None:
            raise ValueError(f"No loader configured for '{model_id}'")

        model_fn_name, weights_name = loader_info

        # Load model with pretrained weights
        print(f"[TorchVision] Loading {model_id} with pretrained weights...")
        model_fn = getattr(models, model_fn_name)
        weights_class = getattr(models, weights_name)
        weights = weights_class.DEFAULT

        model = model_fn(weights=weights)
        model.eval()

        # Export to target format
        input_shape = info.get("input_shape", (1, 3, 224, 224))
        dummy_input = torch.randn(*input_shape)

        if format == ModelFormat.PYTORCH:
            torch.save(model.state_dict(), model_path)

        elif format == ModelFormat.TORCHSCRIPT:
            print(f"[TorchVision] Tracing model to TorchScript...")
            traced = torch.jit.trace(model, dummy_input)
            traced.save(str(model_path))

        elif format == ModelFormat.ONNX:
            print(f"[TorchVision] Exporting to ONNX...")
            torch.onnx.export(
                model,
                dummy_input,
                str(model_path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
                opset_version=17,
            )

        if not model_path.exists():
            raise RuntimeError(f"Failed to export model to {model_path}")

        print(f"[TorchVision] Saved to {model_path}")
        return self._create_artifact(model_id, format, model_path, info)

    def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Get detailed model information."""
        info = TORCHVISION_MODELS.get(model_id)
        if info is None:
            raise ValueError(f"Model '{model_id}' not found in catalog")

        return {
            "id": model_id,
            "provider": self.name,
            "benchmarked": True,
            "accuracy": info.get("top1"),
            **info,
        }

    def _create_artifact(
        self,
        model_id: str,
        format: ModelFormat,
        path: Path,
        info: dict[str, Any],
    ) -> ModelArtifact:
        """Create a ModelArtifact from download result."""
        size = path.stat().st_size

        return ModelArtifact(
            model_id=model_id,
            provider=self.name,
            format=format,
            path=path,
            size_bytes=size,
            name=info.get("name", model_id),
            version=info.get("weights"),
            task=info.get("task"),
            parameters=info.get("parameters"),
            input_shape=info.get("input_shape"),
            accuracy=info.get("top1"),
            metadata=info,
        )
