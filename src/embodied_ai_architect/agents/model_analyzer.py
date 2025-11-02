"""Model Analysis Agent - Analyzes DNN architectures."""

import torch
import torch.nn as nn
from typing import Any, Dict
from .base import BaseAgent, AgentResult


class ModelAnalyzerAgent(BaseAgent):
    """Analyzes neural network models to extract architecture information.

    This agent can process PyTorch models and extract:
    - Total parameters
    - Trainable parameters
    - Layer types and counts
    - Model structure
    """

    def __init__(self):
        super().__init__(name="ModelAnalyzer")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Analyze a PyTorch model.

        Args:
            input_data: Dictionary with keys:
                - 'model': PyTorch nn.Module or path to model file
                - 'input_shape': Optional tuple for input shape analysis

        Returns:
            AgentResult containing model analysis data
        """
        try:
            model = input_data.get("model")
            input_shape = input_data.get("input_shape")

            if model is None:
                return AgentResult(
                    success=False,
                    data={},
                    error="No model provided in input_data"
                )

            # Load model if it's a path string
            if isinstance(model, str):
                model = torch.load(model)

            if not isinstance(model, nn.Module):
                return AgentResult(
                    success=False,
                    data={},
                    error="Model must be a PyTorch nn.Module or path to model file"
                )

            # Analyze model
            analysis = self._analyze_model(model, input_shape)

            return AgentResult(
                success=True,
                data=analysis,
                metadata={"agent": self.name}
            )

        except Exception as e:
            return AgentResult(
                success=False,
                data={},
                error=f"Error analyzing model: {str(e)}"
            )

    def _analyze_model(self, model: nn.Module, input_shape: tuple | None = None) -> Dict[str, Any]:
        """Internal method to analyze a PyTorch model.

        Args:
            model: PyTorch nn.Module
            input_shape: Optional input shape for inference analysis

        Returns:
            Dictionary containing model analysis results
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Count layer types
        layer_types = {}
        layer_list = []

        for name, module in model.named_modules():
            module_type = type(module).__name__
            if module_type != type(model).__name__:  # Skip the model itself
                layer_types[module_type] = layer_types.get(module_type, 0) + 1
                layer_list.append({
                    "name": name if name else "root",
                    "type": module_type,
                    "params": sum(p.numel() for p in module.parameters())
                })

        analysis = {
            "model_type": type(model).__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "layer_type_counts": layer_types,
            "layer_details": layer_list[:20],  # Limit to first 20 layers for brevity
            "total_layers": len(layer_list)
        }

        # If input shape provided, estimate memory and FLOPs
        if input_shape is not None:
            try:
                model.eval()
                dummy_input = torch.randn(*input_shape)
                with torch.no_grad():
                    output = model(dummy_input)
                analysis["input_shape"] = list(input_shape)
                analysis["output_shape"] = list(output.shape)
            except Exception as e:
                analysis["shape_inference_error"] = str(e)

        return analysis
