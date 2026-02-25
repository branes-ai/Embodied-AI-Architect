"""Timm provider for PyTorch Image Models.

Provides access to 700+ vision models from Ross Wightman's timm library.
This catalog includes curated models optimized for edge deployment and
embodied AI applications.

Full model list: https://huggingface.co/timm
"""

from pathlib import Path
from typing import Any, Optional

from .base import ModelProvider, ModelFormat, ModelQuery, ModelArtifact

# Curated Timm model catalog focused on edge-efficient and popular models
# Full library has 700+ models - this is a curated selection
TIMM_MODELS = {
    # ==========================================================================
    # EfficientNet Family (Google) - Excellent accuracy/efficiency tradeoff
    # ==========================================================================
    "efficientnet_b0.ra_in1k": {
        "name": "EfficientNet B0 (RA)",
        "task": "classification",
        "parameters": 5_288_548,
        "flops": 390_000_000,
        "top1": 0.7914,
        "input_shape": (1, 3, 224, 224),
        "family": "efficientnet",
        "description": "EfficientNet B0 with RandAugment training",
    },
    "efficientnet_b1.ra_in1k": {
        "name": "EfficientNet B1 (RA)",
        "task": "classification",
        "parameters": 7_794_184,
        "flops": 700_000_000,
        "top1": 0.8072,
        "input_shape": (1, 3, 240, 240),
        "family": "efficientnet",
        "description": "EfficientNet B1 with RandAugment training",
    },
    "efficientnet_b2.ra_in1k": {
        "name": "EfficientNet B2 (RA)",
        "task": "classification",
        "parameters": 9_109_994,
        "flops": 1_000_000_000,
        "top1": 0.8170,
        "input_shape": (1, 3, 260, 260),
        "family": "efficientnet",
        "description": "EfficientNet B2 with RandAugment training",
    },
    "efficientnet_b3.ra2_in1k": {
        "name": "EfficientNet B3 (RA2)",
        "task": "classification",
        "parameters": 12_233_232,
        "flops": 1_800_000_000,
        "top1": 0.8388,
        "input_shape": (1, 3, 288, 288),
        "family": "efficientnet",
        "description": "EfficientNet B3 with RandAugment v2",
    },
    # EfficientNet-Lite (TFLite optimized)
    "efficientnet_lite0.ra_in1k": {
        "name": "EfficientNet Lite0",
        "task": "classification",
        "parameters": 4_652_008,
        "flops": 407_000_000,
        "top1": 0.7532,
        "input_shape": (1, 3, 224, 224),
        "family": "efficientnet",
        "description": "EfficientNet Lite0 - optimized for mobile/edge",
    },
    "tf_efficientnetv2_s.in1k": {
        "name": "EfficientNetV2-S",
        "task": "classification",
        "parameters": 21_458_488,
        "flops": 8_400_000_000,
        "top1": 0.8386,
        "input_shape": (1, 3, 300, 300),
        "family": "efficientnet",
        "description": "EfficientNetV2 Small - faster training, better accuracy",
    },
    "tf_efficientnetv2_m.in1k": {
        "name": "EfficientNetV2-M",
        "task": "classification",
        "parameters": 54_139_356,
        "flops": 24_700_000_000,
        "top1": 0.8530,
        "input_shape": (1, 3, 384, 384),
        "family": "efficientnet",
        "description": "EfficientNetV2 Medium",
    },
    # ==========================================================================
    # MobileNet Family - Designed for mobile/edge
    # ==========================================================================
    "mobilenetv3_large_100.ra_in1k": {
        "name": "MobileNetV3 Large",
        "task": "classification",
        "parameters": 5_483_032,
        "flops": 217_000_000,
        "top1": 0.7560,
        "input_shape": (1, 3, 224, 224),
        "family": "mobilenet",
        "description": "MobileNetV3 Large with RandAugment",
    },
    "mobilenetv3_small_100.lamb_in1k": {
        "name": "MobileNetV3 Small",
        "task": "classification",
        "parameters": 2_542_856,
        "flops": 56_000_000,
        "top1": 0.6780,
        "input_shape": (1, 3, 224, 224),
        "family": "mobilenet",
        "description": "MobileNetV3 Small - ultra lightweight",
    },
    "mobilenetv4_conv_small.e2400_r224_in1k": {
        "name": "MobileNetV4 Conv Small",
        "task": "classification",
        "parameters": 3_770_000,
        "flops": 200_000_000,
        "top1": 0.7380,
        "input_shape": (1, 3, 224, 224),
        "family": "mobilenet",
        "description": "MobileNetV4 Small - latest Google mobile architecture",
    },
    "mobilenetv4_conv_medium.e500_r256_in1k": {
        "name": "MobileNetV4 Conv Medium",
        "task": "classification",
        "parameters": 9_720_000,
        "flops": 970_000_000,
        "top1": 0.7930,
        "input_shape": (1, 3, 256, 256),
        "family": "mobilenet",
        "description": "MobileNetV4 Medium - balanced speed/accuracy",
    },
    # ==========================================================================
    # FastViT (Apple) - State-of-the-art mobile vision transformer
    # ==========================================================================
    "fastvit_t8.apple_in1k": {
        "name": "FastViT T8",
        "task": "classification",
        "parameters": 4_030_000,
        "flops": 700_000_000,
        "top1": 0.7640,
        "input_shape": (1, 3, 256, 256),
        "family": "fastvit",
        "description": "FastViT Tiny - Apple's efficient hybrid architecture",
    },
    "fastvit_t12.apple_in1k": {
        "name": "FastViT T12",
        "task": "classification",
        "parameters": 7_550_000,
        "flops": 1_400_000_000,
        "top1": 0.7940,
        "input_shape": (1, 3, 256, 256),
        "family": "fastvit",
        "description": "FastViT T12 - larger tiny variant",
    },
    "fastvit_s12.apple_in1k": {
        "name": "FastViT S12",
        "task": "classification",
        "parameters": 9_470_000,
        "flops": 1_800_000_000,
        "top1": 0.7970,
        "input_shape": (1, 3, 256, 256),
        "family": "fastvit",
        "description": "FastViT Small 12",
    },
    "fastvit_sa12.apple_in1k": {
        "name": "FastViT SA12",
        "task": "classification",
        "parameters": 11_580_000,
        "flops": 1_900_000_000,
        "top1": 0.8050,
        "input_shape": (1, 3, 256, 256),
        "family": "fastvit",
        "description": "FastViT SA12 with self-attention",
    },
    "fastvit_sa24.apple_in1k": {
        "name": "FastViT SA24",
        "task": "classification",
        "parameters": 21_550_000,
        "flops": 3_800_000_000,
        "top1": 0.8270,
        "input_shape": (1, 3, 256, 256),
        "family": "fastvit",
        "description": "FastViT SA24 - high accuracy variant",
    },
    # ==========================================================================
    # EfficientFormer (Snap) - Efficient vision transformer
    # ==========================================================================
    "efficientformer_l1.snap_dist_in1k": {
        "name": "EfficientFormer L1",
        "task": "classification",
        "parameters": 12_290_000,
        "flops": 1_300_000_000,
        "top1": 0.7994,
        "input_shape": (1, 3, 224, 224),
        "family": "efficientformer",
        "description": "EfficientFormer L1 - Snap's efficient ViT",
    },
    "efficientformer_l3.snap_dist_in1k": {
        "name": "EfficientFormer L3",
        "task": "classification",
        "parameters": 31_410_000,
        "flops": 3_900_000_000,
        "top1": 0.8262,
        "input_shape": (1, 3, 224, 224),
        "family": "efficientformer",
        "description": "EfficientFormer L3",
    },
    "efficientformerv2_s0.snap_dist_in1k": {
        "name": "EfficientFormerV2 S0",
        "task": "classification",
        "parameters": 3_600_000,
        "flops": 400_000_000,
        "top1": 0.7594,
        "input_shape": (1, 3, 224, 224),
        "family": "efficientformer",
        "description": "EfficientFormerV2 S0 - improved efficiency",
    },
    "efficientformerv2_s1.snap_dist_in1k": {
        "name": "EfficientFormerV2 S1",
        "task": "classification",
        "parameters": 6_190_000,
        "flops": 650_000_000,
        "top1": 0.7944,
        "input_shape": (1, 3, 224, 224),
        "family": "efficientformer",
        "description": "EfficientFormerV2 S1",
    },
    "efficientformerv2_s2.snap_dist_in1k": {
        "name": "EfficientFormerV2 S2",
        "task": "classification",
        "parameters": 12_710_000,
        "flops": 1_300_000_000,
        "top1": 0.8198,
        "input_shape": (1, 3, 224, 224),
        "family": "efficientformer",
        "description": "EfficientFormerV2 S2",
    },
    # ==========================================================================
    # MobileViT v2 (Apple) - Improved mobile vision transformer
    # ==========================================================================
    "mobilevitv2_050.cvnets_in1k": {
        "name": "MobileViTV2 0.5",
        "task": "classification",
        "parameters": 1_370_000,
        "flops": 479_000_000,
        "top1": 0.7030,
        "input_shape": (1, 3, 256, 256),
        "family": "mobilevit",
        "description": "MobileViTV2 0.5x - ultra compact",
    },
    "mobilevitv2_075.cvnets_in1k": {
        "name": "MobileViTV2 0.75",
        "task": "classification",
        "parameters": 2_870_000,
        "flops": 1_040_000_000,
        "top1": 0.7510,
        "input_shape": (1, 3, 256, 256),
        "family": "mobilevit",
        "description": "MobileViTV2 0.75x",
    },
    "mobilevitv2_100.cvnets_in1k": {
        "name": "MobileViTV2 1.0",
        "task": "classification",
        "parameters": 4_900_000,
        "flops": 1_840_000_000,
        "top1": 0.7820,
        "input_shape": (1, 3, 256, 256),
        "family": "mobilevit",
        "description": "MobileViTV2 1.0x",
    },
    "mobilevitv2_150.cvnets_in22k_ft_in1k_384": {
        "name": "MobileViTV2 1.5 384",
        "task": "classification",
        "parameters": 10_600_000,
        "flops": 9_200_000_000,
        "top1": 0.8460,
        "input_shape": (1, 3, 384, 384),
        "family": "mobilevit",
        "description": "MobileViTV2 1.5x pretrained on ImageNet-22k",
    },
    # ==========================================================================
    # ConvNeXt V2 (Meta) - Modern pure ConvNet
    # ==========================================================================
    "convnextv2_atto.fcmae_ft_in1k": {
        "name": "ConvNeXtV2 Atto",
        "task": "classification",
        "parameters": 3_710_000,
        "flops": 550_000_000,
        "top1": 0.7648,
        "input_shape": (1, 3, 224, 224),
        "family": "convnext",
        "description": "ConvNeXtV2 Atto - smallest variant",
    },
    "convnextv2_femto.fcmae_ft_in1k": {
        "name": "ConvNeXtV2 Femto",
        "task": "classification",
        "parameters": 5_230_000,
        "flops": 790_000_000,
        "top1": 0.7846,
        "input_shape": (1, 3, 224, 224),
        "family": "convnext",
        "description": "ConvNeXtV2 Femto",
    },
    "convnextv2_pico.fcmae_ft_in1k": {
        "name": "ConvNeXtV2 Pico",
        "task": "classification",
        "parameters": 9_070_000,
        "flops": 1_370_000_000,
        "top1": 0.8026,
        "input_shape": (1, 3, 224, 224),
        "family": "convnext",
        "description": "ConvNeXtV2 Pico",
    },
    "convnextv2_nano.fcmae_ft_in1k": {
        "name": "ConvNeXtV2 Nano",
        "task": "classification",
        "parameters": 15_620_000,
        "flops": 2_450_000_000,
        "top1": 0.8166,
        "input_shape": (1, 3, 224, 224),
        "family": "convnext",
        "description": "ConvNeXtV2 Nano",
    },
    "convnextv2_tiny.fcmae_ft_in1k": {
        "name": "ConvNeXtV2 Tiny",
        "task": "classification",
        "parameters": 28_640_000,
        "flops": 4_470_000_000,
        "top1": 0.8304,
        "input_shape": (1, 3, 224, 224),
        "family": "convnext",
        "description": "ConvNeXtV2 Tiny",
    },
    # ==========================================================================
    # EdgeNeXt (Samsung) - Designed for edge devices
    # ==========================================================================
    "edgenext_xx_small.in1k": {
        "name": "EdgeNeXt XX-Small",
        "task": "classification",
        "parameters": 1_330_000,
        "flops": 261_000_000,
        "top1": 0.7120,
        "input_shape": (1, 3, 256, 256),
        "family": "edgenext",
        "description": "EdgeNeXt XX-Small - Samsung's edge architecture",
    },
    "edgenext_x_small.in1k": {
        "name": "EdgeNeXt X-Small",
        "task": "classification",
        "parameters": 2_340_000,
        "flops": 538_000_000,
        "top1": 0.7480,
        "input_shape": (1, 3, 256, 256),
        "family": "edgenext",
        "description": "EdgeNeXt X-Small",
    },
    "edgenext_small.usi_in1k": {
        "name": "EdgeNeXt Small",
        "task": "classification",
        "parameters": 5_590_000,
        "flops": 1_260_000_000,
        "top1": 0.8120,
        "input_shape": (1, 3, 256, 256),
        "family": "edgenext",
        "description": "EdgeNeXt Small with USI training",
    },
    # ==========================================================================
    # RepVGG - Fast inference through reparameterization
    # ==========================================================================
    "repvgg_a0.rvgg_in1k": {
        "name": "RepVGG A0",
        "task": "classification",
        "parameters": 9_110_000,
        "flops": 1_520_000_000,
        "top1": 0.7240,
        "input_shape": (1, 3, 224, 224),
        "family": "repvgg",
        "description": "RepVGG A0 - simple and fast at inference",
    },
    "repvgg_a1.rvgg_in1k": {
        "name": "RepVGG A1",
        "task": "classification",
        "parameters": 14_090_000,
        "flops": 2_640_000_000,
        "top1": 0.7440,
        "input_shape": (1, 3, 224, 224),
        "family": "repvgg",
        "description": "RepVGG A1",
    },
    "repvgg_a2.rvgg_in1k": {
        "name": "RepVGG A2",
        "task": "classification",
        "parameters": 28_210_000,
        "flops": 5_700_000_000,
        "top1": 0.7660,
        "input_shape": (1, 3, 224, 224),
        "family": "repvgg",
        "description": "RepVGG A2",
    },
    "repvgg_b0.rvgg_in1k": {
        "name": "RepVGG B0",
        "task": "classification",
        "parameters": 15_820_000,
        "flops": 3_420_000_000,
        "top1": 0.7530,
        "input_shape": (1, 3, 224, 224),
        "family": "repvgg",
        "description": "RepVGG B0",
    },
    # ==========================================================================
    # GhostNet (Huawei) - Efficient feature generation
    # ==========================================================================
    "ghostnet_100.in1k": {
        "name": "GhostNet 1.0",
        "task": "classification",
        "parameters": 5_180_000,
        "flops": 141_000_000,
        "top1": 0.7400,
        "input_shape": (1, 3, 224, 224),
        "family": "ghostnet",
        "description": "GhostNet 1.0x - efficient feature reuse",
    },
    "ghostnetv2_100.in1k": {
        "name": "GhostNetV2 1.0",
        "task": "classification",
        "parameters": 6_160_000,
        "flops": 168_000_000,
        "top1": 0.7510,
        "input_shape": (1, 3, 224, 224),
        "family": "ghostnet",
        "description": "GhostNetV2 1.0x - improved ghost modules",
    },
    "ghostnetv2_130.in1k": {
        "name": "GhostNetV2 1.3",
        "task": "classification",
        "parameters": 8_960_000,
        "flops": 271_000_000,
        "top1": 0.7680,
        "input_shape": (1, 3, 224, 224),
        "family": "ghostnet",
        "description": "GhostNetV2 1.3x",
    },
    "ghostnetv2_160.in1k": {
        "name": "GhostNetV2 1.6",
        "task": "classification",
        "parameters": 12_390_000,
        "flops": 400_000_000,
        "top1": 0.7840,
        "input_shape": (1, 3, 224, 224),
        "family": "ghostnet",
        "description": "GhostNetV2 1.6x",
    },
    # ==========================================================================
    # MaxViT - Multi-axis attention
    # ==========================================================================
    "maxvit_tiny_tf_224.in1k": {
        "name": "MaxViT Tiny",
        "task": "classification",
        "parameters": 30_920_000,
        "flops": 5_600_000_000,
        "top1": 0.8350,
        "input_shape": (1, 3, 224, 224),
        "family": "maxvit",
        "description": "MaxViT Tiny - multi-axis vision transformer",
    },
    "maxvit_small_tf_224.in1k": {
        "name": "MaxViT Small",
        "task": "classification",
        "parameters": 68_930_000,
        "flops": 11_700_000_000,
        "top1": 0.8490,
        "input_shape": (1, 3, 224, 224),
        "family": "maxvit",
        "description": "MaxViT Small",
    },
    # ==========================================================================
    # TinyNet (Huawei) - Neural architecture search for tiny models
    # ==========================================================================
    "tinynet_a.in1k": {
        "name": "TinyNet-A",
        "task": "classification",
        "parameters": 6_190_000,
        "flops": 340_000_000,
        "top1": 0.7700,
        "input_shape": (1, 3, 192, 192),
        "family": "tinynet",
        "description": "TinyNet-A - NAS-optimized tiny model",
    },
    "tinynet_b.in1k": {
        "name": "TinyNet-B",
        "task": "classification",
        "parameters": 3_730_000,
        "flops": 206_000_000,
        "top1": 0.7480,
        "input_shape": (1, 3, 188, 188),
        "family": "tinynet",
        "description": "TinyNet-B",
    },
    "tinynet_c.in1k": {
        "name": "TinyNet-C",
        "task": "classification",
        "parameters": 2_460_000,
        "flops": 103_000_000,
        "top1": 0.7100,
        "input_shape": (1, 3, 184, 184),
        "family": "tinynet",
        "description": "TinyNet-C",
    },
    "tinynet_d.in1k": {
        "name": "TinyNet-D",
        "task": "classification",
        "parameters": 1_760_000,
        "flops": 53_000_000,
        "top1": 0.6670,
        "input_shape": (1, 3, 152, 152),
        "family": "tinynet",
        "description": "TinyNet-D - smallest variant",
    },
    "tinynet_e.in1k": {
        "name": "TinyNet-E",
        "task": "classification",
        "parameters": 2_040_000,
        "flops": 25_000_000,
        "top1": 0.5920,
        "input_shape": (1, 3, 106, 106),
        "family": "tinynet",
        "description": "TinyNet-E - extreme compression",
    },
    # ==========================================================================
    # MnasNet - Mobile Neural Architecture Search
    # ==========================================================================
    "mnasnet_050.lamb_in1k": {
        "name": "MnasNet 0.5",
        "task": "classification",
        "parameters": 2_220_000,
        "flops": 105_000_000,
        "top1": 0.6890,
        "input_shape": (1, 3, 224, 224),
        "family": "mnasnet",
        "description": "MnasNet 0.5x - Google NAS mobile model",
    },
    "mnasnet_100.rmsp_in1k": {
        "name": "MnasNet 1.0",
        "task": "classification",
        "parameters": 4_380_000,
        "flops": 312_000_000,
        "top1": 0.7470,
        "input_shape": (1, 3, 224, 224),
        "family": "mnasnet",
        "description": "MnasNet 1.0x",
    },
    # ==========================================================================
    # LCNet (Baidu) - Lightweight CPU network
    # ==========================================================================
    "lcnet_050.ra2_in1k": {
        "name": "LCNet 0.5",
        "task": "classification",
        "parameters": 1_880_000,
        "flops": 47_000_000,
        "top1": 0.6360,
        "input_shape": (1, 3, 224, 224),
        "family": "lcnet",
        "description": "LCNet 0.5x - optimized for CPU inference",
    },
    "lcnet_075.ra2_in1k": {
        "name": "LCNet 0.75",
        "task": "classification",
        "parameters": 2_360_000,
        "flops": 97_000_000,
        "top1": 0.6820,
        "input_shape": (1, 3, 224, 224),
        "family": "lcnet",
        "description": "LCNet 0.75x",
    },
    "lcnet_100.ra2_in1k": {
        "name": "LCNet 1.0",
        "task": "classification",
        "parameters": 2_950_000,
        "flops": 161_000_000,
        "top1": 0.7200,
        "input_shape": (1, 3, 224, 224),
        "family": "lcnet",
        "description": "LCNet 1.0x",
    },
}


class TimmProvider(ModelProvider):
    """Provider for timm (PyTorch Image Models) library.

    Provides access to 700+ vision models. This catalog includes a curated
    selection of models optimized for edge deployment and embodied AI.

    Use `timm.list_models()` to see all available models.
    """

    @property
    def name(self) -> str:
        return "timm"

    @property
    def supported_formats(self) -> list[ModelFormat]:
        return [
            ModelFormat.PYTORCH,
            ModelFormat.TORCHSCRIPT,
            ModelFormat.ONNX,
        ]

    def list_models(self, query: Optional[ModelQuery] = None) -> list[dict[str, Any]]:
        """List available Timm models from curated catalog."""
        models = []
        for model_id, info in TIMM_MODELS.items():
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
        """Download and export a Timm model.

        Args:
            model_id: Timm model identifier (e.g., 'efficientnet_b0.ra_in1k')
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

        # Get model info from catalog
        info = TIMM_MODELS.get(model_id)
        if info is None:
            # Model not in curated catalog - try loading directly from timm
            info = {
                "name": model_id,
                "task": "classification",
                "family": "unknown",
            }

        # Create safe filename
        safe_name = model_id.replace(".", "_").replace("/", "_")

        # Determine file extension
        ext_map = {
            ModelFormat.PYTORCH: ".pt",
            ModelFormat.TORCHSCRIPT: ".torchscript",
            ModelFormat.ONNX: ".onnx",
        }

        ext = ext_map[format]
        model_filename = f"{safe_name}{ext}"
        model_path = cache_dir / model_filename

        # Check if already cached
        if model_path.exists():
            return self._create_artifact(model_id, format, model_path, info)

        # Ensure cache directory exists
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download using timm
        try:
            import torch
            import timm
        except ImportError:
            raise ImportError("timm not installed. Install with: pip install timm")

        print(f"[Timm] Loading {model_id} with pretrained weights...")
        try:
            model = timm.create_model(model_id, pretrained=True)
        except Exception as e:
            raise ValueError(f"Failed to load model '{model_id}': {e}")

        model.eval()

        # Get input shape
        input_shape = info.get("input_shape", (1, 3, 224, 224))

        # Get the model's expected input size from data_config if available
        try:
            data_config = timm.data.resolve_model_data_config(model)
            input_size = data_config.get("input_size", (3, 224, 224))
            input_shape = (1,) + tuple(input_size)
        except Exception:
            pass

        dummy_input = torch.randn(*input_shape)

        # Export to target format
        if format == ModelFormat.PYTORCH:
            print("[Timm] Saving PyTorch weights...")
            torch.save(model.state_dict(), model_path)

        elif format == ModelFormat.TORCHSCRIPT:
            print("[Timm] Tracing to TorchScript...")
            traced = torch.jit.trace(model, dummy_input)
            traced.save(str(model_path))

        elif format == ModelFormat.ONNX:
            print("[Timm] Exporting to ONNX...")
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

        print(f"[Timm] Saved to {model_path}")
        return self._create_artifact(model_id, format, model_path, info)

    def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Get detailed model information."""
        info = TIMM_MODELS.get(model_id)
        if info is None:
            # Try to get info from timm directly
            try:
                import timm

                if model_id in timm.list_models():
                    return {
                        "id": model_id,
                        "name": model_id,
                        "provider": self.name,
                        "task": "classification",
                        "benchmarked": False,
                        "description": "Model available in timm but not in curated catalog",
                    }
            except ImportError:
                pass
            raise ValueError(f"Model '{model_id}' not found in catalog")

        return {
            "id": model_id,
            "provider": self.name,
            "benchmarked": True,
            "accuracy": info.get("top1"),
            **info,
        }

    def list_all_available(self) -> list[str]:
        """List all models available in timm (not just curated catalog).

        Returns:
            List of all model names available in the timm library
        """
        try:
            import timm

            return timm.list_models()
        except ImportError:
            raise ImportError("timm not installed. Install with: pip install timm")

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
            version=None,
            task=info.get("task"),
            parameters=info.get("parameters"),
            input_shape=info.get("input_shape"),
            accuracy=info.get("top1"),
            metadata=info,
        )
