"""HuggingFace Hub provider for transformer models.

Supports vision models from HuggingFace Hub including:
- Image classification (ViT, DeiT, BEiT, ConvNeXt)
- Object detection (DETR, YOLOS, Conditional DETR)
- Semantic segmentation (SegFormer, Mask2Former)
- Depth estimation (DPT, GLPN)
"""

from pathlib import Path
from typing import Any, Optional

from .base import ModelProvider, ModelFormat, ModelQuery, ModelArtifact

# HuggingFace model catalog with known specifications
# Focused on vision models suitable for embodied AI / edge deployment
HUGGINGFACE_MODELS = {
    # ViT - Vision Transformer (Image Classification)
    "google/vit-base-patch16-224": {
        "name": "ViT Base Patch16 224",
        "task": "classification",
        "parameters": 86_000_000,
        "top1": 0.8146,
        "input_shape": (1, 3, 224, 224),
        "family": "vit",
        "description": "Vision Transformer base model pre-trained on ImageNet-21k and fine-tuned on ImageNet-1k",
    },
    "google/vit-base-patch32-224": {
        "name": "ViT Base Patch32 224",
        "task": "classification",
        "parameters": 88_000_000,
        "top1": 0.7892,
        "input_shape": (1, 3, 224, 224),
        "family": "vit",
        "description": "ViT with larger patches (32x32), faster but slightly less accurate",
    },
    "google/vit-large-patch16-224": {
        "name": "ViT Large Patch16 224",
        "task": "classification",
        "parameters": 304_000_000,
        "top1": 0.8500,
        "input_shape": (1, 3, 224, 224),
        "family": "vit",
        "description": "Large Vision Transformer with 16x16 patches",
    },
    # DeiT - Data-efficient Image Transformer
    "facebook/deit-tiny-patch16-224": {
        "name": "DeiT Tiny",
        "task": "classification",
        "parameters": 5_700_000,
        "top1": 0.7220,
        "input_shape": (1, 3, 224, 224),
        "family": "deit",
        "description": "Tiny Data-efficient Image Transformer, good for edge deployment",
    },
    "facebook/deit-small-patch16-224": {
        "name": "DeiT Small",
        "task": "classification",
        "parameters": 22_000_000,
        "top1": 0.7980,
        "input_shape": (1, 3, 224, 224),
        "family": "deit",
        "description": "Small Data-efficient Image Transformer",
    },
    "facebook/deit-base-patch16-224": {
        "name": "DeiT Base",
        "task": "classification",
        "parameters": 86_000_000,
        "top1": 0.8180,
        "input_shape": (1, 3, 224, 224),
        "family": "deit",
        "description": "Base Data-efficient Image Transformer",
    },
    # BEiT - BERT Pre-training of Image Transformers
    "microsoft/beit-base-patch16-224": {
        "name": "BEiT Base",
        "task": "classification",
        "parameters": 86_000_000,
        "top1": 0.8282,
        "input_shape": (1, 3, 224, 224),
        "family": "beit",
        "description": "BEiT pre-trained on ImageNet-21k, fine-tuned on ImageNet-1k",
    },
    # DETR - Detection Transformer
    "facebook/detr-resnet-50": {
        "name": "DETR ResNet-50",
        "task": "detection",
        "parameters": 41_000_000,
        "map50": 0.420,
        "input_shape": (1, 3, 800, 800),
        "family": "detr",
        "description": "DEtection TRansformer with ResNet-50 backbone",
    },
    "facebook/detr-resnet-101": {
        "name": "DETR ResNet-101",
        "task": "detection",
        "parameters": 60_000_000,
        "map50": 0.435,
        "input_shape": (1, 3, 800, 800),
        "family": "detr",
        "description": "DEtection TRansformer with ResNet-101 backbone",
    },
    # Conditional DETR - faster convergence
    "microsoft/conditional-detr-resnet-50": {
        "name": "Conditional DETR ResNet-50",
        "task": "detection",
        "parameters": 44_000_000,
        "map50": 0.430,
        "input_shape": (1, 3, 800, 800),
        "family": "detr",
        "description": "Conditional DETR with faster training convergence",
    },
    # YOLOS - You Only Look at One Sequence
    "hustvl/yolos-tiny": {
        "name": "YOLOS Tiny",
        "task": "detection",
        "parameters": 6_500_000,
        "map50": 0.287,
        "input_shape": (1, 3, 512, 512),
        "family": "yolos",
        "description": "Tiny YOLOS for efficient object detection",
    },
    "hustvl/yolos-small": {
        "name": "YOLOS Small",
        "task": "detection",
        "parameters": 30_000_000,
        "map50": 0.364,
        "input_shape": (1, 3, 512, 512),
        "family": "yolos",
        "description": "Small YOLOS object detector",
    },
    "hustvl/yolos-base": {
        "name": "YOLOS Base",
        "task": "detection",
        "parameters": 127_000_000,
        "map50": 0.420,
        "input_shape": (1, 3, 512, 512),
        "family": "yolos",
        "description": "Base YOLOS object detector",
    },
    # SegFormer - Semantic Segmentation
    "nvidia/segformer-b0-finetuned-ade-512-512": {
        "name": "SegFormer B0 ADE",
        "task": "segmentation",
        "parameters": 3_700_000,
        "miou": 0.374,
        "input_shape": (1, 3, 512, 512),
        "family": "segformer",
        "description": "Lightweight SegFormer for semantic segmentation",
    },
    "nvidia/segformer-b1-finetuned-ade-512-512": {
        "name": "SegFormer B1 ADE",
        "task": "segmentation",
        "parameters": 13_700_000,
        "miou": 0.408,
        "input_shape": (1, 3, 512, 512),
        "family": "segformer",
        "description": "SegFormer B1 for semantic segmentation",
    },
    "nvidia/segformer-b2-finetuned-ade-512-512": {
        "name": "SegFormer B2 ADE",
        "task": "segmentation",
        "parameters": 27_400_000,
        "miou": 0.461,
        "input_shape": (1, 3, 512, 512),
        "family": "segformer",
        "description": "SegFormer B2 for semantic segmentation",
    },
    "nvidia/segformer-b3-finetuned-ade-512-512": {
        "name": "SegFormer B3 ADE",
        "task": "segmentation",
        "parameters": 47_300_000,
        "miou": 0.494,
        "input_shape": (1, 3, 512, 512),
        "family": "segformer",
        "description": "SegFormer B3 for semantic segmentation",
    },
    "nvidia/segformer-b0-finetuned-cityscapes-512-1024": {
        "name": "SegFormer B0 Cityscapes",
        "task": "segmentation",
        "parameters": 3_700_000,
        "miou": 0.714,
        "input_shape": (1, 3, 512, 1024),
        "family": "segformer",
        "description": "SegFormer B0 for urban scene segmentation",
    },
    # DPT - Dense Prediction Transformer (Depth Estimation)
    "Intel/dpt-hybrid-midas": {
        "name": "DPT Hybrid MiDaS",
        "task": "depth_estimation",
        "parameters": 123_000_000,
        "input_shape": (1, 3, 384, 384),
        "family": "dpt",
        "description": "DPT with hybrid backbone for monocular depth estimation",
    },
    "Intel/dpt-large": {
        "name": "DPT Large",
        "task": "depth_estimation",
        "parameters": 343_000_000,
        "input_shape": (1, 3, 384, 384),
        "family": "dpt",
        "description": "Large DPT for high-quality depth estimation",
    },
    # GLPN - Global-Local Path Networks (Depth Estimation)
    "vinvino02/glpn-kitti": {
        "name": "GLPN KITTI",
        "task": "depth_estimation",
        "parameters": 26_000_000,
        "input_shape": (1, 3, 480, 640),
        "family": "glpn",
        "description": "GLPN trained on KITTI for outdoor depth estimation",
    },
    "vinvino02/glpn-nyu": {
        "name": "GLPN NYU",
        "task": "depth_estimation",
        "parameters": 26_000_000,
        "input_shape": (1, 3, 480, 640),
        "family": "glpn",
        "description": "GLPN trained on NYU for indoor depth estimation",
    },
    # MobileViT - Mobile Vision Transformer
    "apple/mobilevit-small": {
        "name": "MobileViT Small",
        "task": "classification",
        "parameters": 5_600_000,
        "top1": 0.7840,
        "input_shape": (1, 3, 256, 256),
        "family": "mobilevit",
        "description": "Lightweight vision transformer for mobile deployment",
    },
    "apple/mobilevit-x-small": {
        "name": "MobileViT X-Small",
        "task": "classification",
        "parameters": 2_300_000,
        "top1": 0.7450,
        "input_shape": (1, 3, 256, 256),
        "family": "mobilevit",
        "description": "Extra small MobileViT for edge deployment",
    },
    "apple/mobilevit-xx-small": {
        "name": "MobileViT XX-Small",
        "task": "classification",
        "parameters": 1_300_000,
        "top1": 0.6900,
        "input_shape": (1, 3, 256, 256),
        "family": "mobilevit",
        "description": "Extremely small MobileViT for constrained devices",
    },
    # Depth Anything
    "LiheYoung/depth-anything-small-hf": {
        "name": "Depth Anything Small",
        "task": "depth_estimation",
        "parameters": 24_800_000,
        "input_shape": (1, 3, 518, 518),
        "family": "depth_anything",
        "description": "Small Depth Anything model for monocular depth estimation",
    },
    "LiheYoung/depth-anything-base-hf": {
        "name": "Depth Anything Base",
        "task": "depth_estimation",
        "parameters": 97_500_000,
        "input_shape": (1, 3, 518, 518),
        "family": "depth_anything",
        "description": "Base Depth Anything model for accurate depth estimation",
    },
}


class HuggingFaceProvider(ModelProvider):
    """Provider for HuggingFace Hub models.

    Downloads and exports transformer models from the HuggingFace Hub.
    Focused on vision models suitable for embodied AI applications.
    """

    @property
    def name(self) -> str:
        return "huggingface"

    @property
    def supported_formats(self) -> list[ModelFormat]:
        return [
            ModelFormat.PYTORCH,
            ModelFormat.ONNX,
        ]

    def list_models(self, query: Optional[ModelQuery] = None) -> list[dict[str, Any]]:
        """List available HuggingFace models."""
        models = []
        for model_id, info in HUGGINGFACE_MODELS.items():
            # Extract short ID for display
            short_id = model_id.split("/")[-1]

            model_info = {
                "id": model_id,
                "short_id": short_id,
                "provider": self.name,
                "benchmarked": True,
                "accuracy": info.get("top1") or info.get("map50") or info.get("miou"),
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
        """Download and export a HuggingFace model.

        Args:
            model_id: HuggingFace model identifier (e.g., 'google/vit-base-patch16-224')
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
        info = HUGGINGFACE_MODELS.get(model_id)
        if info is None:
            raise ValueError(f"Model '{model_id}' not found in HuggingFace catalog")

        # Create safe filename from model ID
        safe_name = model_id.replace("/", "_")

        # Determine file extension
        ext_map = {
            ModelFormat.PYTORCH: ".pt",
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

        # Download using transformers
        try:
            import torch
            from transformers import AutoModel, AutoModelForImageClassification
            from transformers import AutoModelForObjectDetection
            from transformers import AutoModelForSemanticSegmentation
            from transformers import AutoModelForDepthEstimation
        except ImportError:
            raise ImportError(
                "transformers not installed. " "Install with: pip install transformers"
            )

        # Determine model type and load appropriate class
        task = info.get("task", "classification")

        print(f"[HuggingFace] Loading {model_id}...")

        try:
            if task == "classification":
                model = AutoModelForImageClassification.from_pretrained(model_id)
            elif task == "detection":
                model = AutoModelForObjectDetection.from_pretrained(model_id)
            elif task == "segmentation":
                model = AutoModelForSemanticSegmentation.from_pretrained(model_id)
            elif task == "depth_estimation":
                model = AutoModelForDepthEstimation.from_pretrained(model_id)
            else:
                model = AutoModel.from_pretrained(model_id)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_id}': {e}")

        model.eval()

        # Export to target format
        input_shape = info.get("input_shape", (1, 3, 224, 224))
        dummy_input = torch.randn(*input_shape)

        if format == ModelFormat.PYTORCH:
            print(f"[HuggingFace] Saving PyTorch model...")
            torch.save(model.state_dict(), model_path)

        elif format == ModelFormat.ONNX:
            print(f"[HuggingFace] Exporting to ONNX...")
            try:
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(model_path),
                    input_names=["pixel_values"],
                    output_names=["logits"],
                    dynamic_axes={
                        "pixel_values": {0: "batch_size"},
                        "logits": {0: "batch_size"},
                    },
                    opset_version=17,
                )
            except Exception as e:
                # Some models need different export approach
                print(f"[HuggingFace] Standard export failed, trying alternative...")
                try:
                    from transformers.onnx import export as onnx_export
                    from transformers import AutoConfig

                    # Use transformers built-in ONNX export if available
                    raise NotImplementedError("Falling back to torch.onnx")
                except Exception:
                    # Last resort: try with tracing
                    traced = torch.jit.trace(model, dummy_input)
                    torch.onnx.export(
                        traced,
                        dummy_input,
                        str(model_path),
                        input_names=["pixel_values"],
                        output_names=["output"],
                        opset_version=17,
                    )

        if not model_path.exists():
            raise RuntimeError(f"Failed to export model to {model_path}")

        print(f"[HuggingFace] Saved to {model_path}")
        return self._create_artifact(model_id, format, model_path, info)

    def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Get detailed model information."""
        info = HUGGINGFACE_MODELS.get(model_id)
        if info is None:
            raise ValueError(f"Model '{model_id}' not found in catalog")

        return {
            "id": model_id,
            "provider": self.name,
            "benchmarked": True,
            "accuracy": info.get("top1") or info.get("map50") or info.get("miou"),
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
            version=None,
            task=info.get("task"),
            parameters=info.get("parameters"),
            input_shape=info.get("input_shape"),
            accuracy=info.get("top1") or info.get("map50") or info.get("miou"),
            metadata=info,
        )
