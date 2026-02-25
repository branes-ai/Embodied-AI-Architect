"""Hardware Profile Agent - Recommends optimal hardware for models."""

from typing import Any, Dict, List
from ..base import BaseAgent, AgentResult
from .models import HardwareProfile, HardwareRecommendation
from .knowledge_base import get_default_hardware_profiles, get_hardware_by_name


class HardwareProfileAgent(BaseAgent):
    """Agent that recommends hardware based on model characteristics.

    This agent maintains a knowledge base of hardware platforms and provides
    recommendations based on:
    - Model size and memory requirements
    - Operation types (convolutions, matrix multiplies, etc.)
    - Performance constraints (latency, throughput)
    - Power budget
    - Cost considerations
    """

    def __init__(self, custom_profiles: List[HardwareProfile] | None = None):
        """Initialize the hardware profile agent.

        Args:
            custom_profiles: Optional list of custom hardware profiles to add
        """
        super().__init__(name="HardwareProfile")

        # Load knowledge base
        self.hardware_profiles = get_default_hardware_profiles()

        # Add custom profiles if provided
        if custom_profiles:
            self.hardware_profiles.extend(custom_profiles)

        print(f"  Loaded {len(self.hardware_profiles)} hardware profiles")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Analyze model and recommend optimal hardware.

        Args:
            input_data: Dictionary with keys:
                - 'model_analysis': Results from ModelAnalyzerAgent (required)
                - 'constraints': Optional dict with:
                    - 'max_latency_ms': Target latency
                    - 'max_power_watts': Power budget
                    - 'max_cost_usd': Cost budget
                - 'target_use_case': Optional string (edge, cloud, datacenter, etc.)
                - 'top_n': Number of recommendations to return (default: 5)

        Returns:
            AgentResult containing hardware recommendations
        """
        try:
            model_analysis = input_data.get("model_analysis")
            constraints = input_data.get("constraints", {})
            target_use_case = input_data.get("target_use_case")
            top_n = input_data.get("top_n", 5)

            if model_analysis is None:
                return AgentResult(
                    success=False,
                    data={},
                    error="No model_analysis provided. Run ModelAnalyzerAgent first.",
                )

            # Extract model characteristics
            model_params = model_analysis.get("total_parameters", 0)
            model_layers = model_analysis.get("total_layers", 0)
            layer_types = model_analysis.get("layer_type_counts", {})

            # Estimate memory requirements (rough approximation)
            # Parameters in FP32 = 4 bytes per param
            model_memory_mb = (model_params * 4) / (1024 * 1024)

            # Determine operation types from layer types
            operation_types = self._infer_operation_types(layer_types)

            # Extract constraints
            max_latency_ms = constraints.get("max_latency_ms")
            max_power_watts = constraints.get("max_power_watts")
            max_cost_usd = constraints.get("max_cost_usd")

            # Score all hardware
            recommendations = []
            for hw in self.hardware_profiles:
                # Filter by use case if specified
                if target_use_case and target_use_case not in hw.suitable_for:
                    continue

                # Filter by cost if specified
                if max_cost_usd and hw.approximate_cost_usd:
                    if hw.approximate_cost_usd > max_cost_usd:
                        continue

                # Calculate fitness score
                score = hw.get_score_for_model(
                    model_params=model_params,
                    model_memory_mb=model_memory_mb,
                    operation_types=operation_types,
                    target_latency_ms=max_latency_ms,
                    max_power_watts=max_power_watts,
                )

                # Generate reasons and warnings
                reasons, warnings = self._generate_reasoning(
                    hw,
                    model_memory_mb,
                    model_params,
                    operation_types,
                    max_latency_ms,
                    max_power_watts,
                )

                # Estimate performance
                estimated_perf = self._estimate_performance(hw, model_params, model_memory_mb)

                recommendation = HardwareRecommendation(
                    hardware=hw,
                    score=score,
                    reasons=reasons,
                    warnings=warnings,
                    estimated_performance=estimated_perf,
                )
                recommendations.append(recommendation)

            # Sort by score
            recommendations.sort(key=lambda r: r.score, reverse=True)

            # Take top N
            top_recommendations = recommendations[:top_n]

            # Create result data
            result_data = {
                "model_characteristics": {
                    "parameters": model_params,
                    "layers": model_layers,
                    "estimated_memory_mb": model_memory_mb,
                    "operation_types": operation_types,
                },
                "recommendations": [
                    {
                        "rank": i + 1,
                        "name": rec.hardware.name,
                        "vendor": rec.hardware.vendor,
                        "type": rec.hardware.hardware_type.value,
                        "score": round(rec.score, 2),
                        "reasons": rec.reasons,
                        "warnings": rec.warnings,
                        "estimated_performance": rec.estimated_performance,
                        "cost_usd": rec.hardware.approximate_cost_usd,
                        "power_watts": rec.hardware.capabilities.tdp_watts,
                    }
                    for i, rec in enumerate(top_recommendations)
                ],
                "total_evaluated": len(recommendations),
            }

            return AgentResult(
                success=True,
                data=result_data,
                metadata={"agent": self.name, "profiles_evaluated": len(recommendations)},
            )

        except Exception as e:
            return AgentResult(success=False, data={}, error=f"Hardware profiling failed: {str(e)}")

    def _infer_operation_types(self, layer_types: Dict[str, int]) -> List[str]:
        """Infer operation types from layer types.

        Args:
            layer_types: Dictionary of layer type counts

        Returns:
            List of operation type strings
        """
        operations = set()

        # Map layer types to operation types
        if any(name in layer_types for name in ["Conv2d", "Conv1d", "Conv3d"]):
            operations.add("convolution")

        if any(name in layer_types for name in ["Linear", "Dense"]):
            operations.add("matrix_multiply")

        if "Attention" in layer_types or "MultiheadAttention" in layer_types:
            operations.add("attention")

        # Default to general purpose if nothing specific
        if not operations:
            operations.add("general_purpose")

        return list(operations)

    def _generate_reasoning(
        self,
        hw: HardwareProfile,
        model_memory_mb: float,
        model_params: int,
        operation_types: List[str],
        max_latency_ms: float | None,
        max_power_watts: float | None,
    ) -> tuple[List[str], List[str]]:
        """Generate human-readable reasons and warnings for a recommendation.

        Args:
            hw: Hardware profile
            model_memory_mb: Model memory requirement
            model_params: Model parameter count
            operation_types: List of operation types
            max_latency_ms: Latency constraint
            max_power_watts: Power constraint

        Returns:
            Tuple of (reasons, warnings)
        """
        reasons = []
        warnings = []

        # Memory fit
        memory_gb_needed = model_memory_mb / 1024
        if hw.capabilities.memory_gb >= memory_gb_needed * 2:
            reasons.append(
                f"Ample memory: {hw.capabilities.memory_gb}GB (needs ~{memory_gb_needed:.1f}GB)"
            )
        elif hw.capabilities.memory_gb >= memory_gb_needed:
            reasons.append(f"Sufficient memory: {hw.capabilities.memory_gb}GB")
        else:
            warnings.append(
                f"Insufficient memory: {hw.capabilities.memory_gb}GB < {memory_gb_needed:.1f}GB needed"
            )

        # Power budget
        if max_power_watts:
            if hw.capabilities.tdp_watts <= max_power_watts * 0.8:
                reasons.append(
                    f"Low power: {hw.capabilities.tdp_watts}W (budget: {max_power_watts}W)"
                )
            elif hw.capabilities.tdp_watts <= max_power_watts:
                reasons.append(f"Within power budget: {hw.capabilities.tdp_watts}W")
            else:
                warnings.append(
                    f"Exceeds power budget: {hw.capabilities.tdp_watts}W > {max_power_watts}W"
                )

        # Operation matching
        matching_ops = [op for op in operation_types if op in [o.value for o in hw.optimized_for]]
        if matching_ops:
            reasons.append(f"Optimized for: {', '.join(matching_ops)}")

        # Special features
        if hw.capabilities.tensor_cores and "matrix_multiply" in operation_types:
            reasons.append("Has tensor cores for matrix operations")

        if hw.capabilities.sparse_acceleration:
            reasons.append("Supports sparse tensor acceleration")

        # Compute capability
        if hw.capabilities.peak_tflops_fp32:
            reasons.append(f"Peak compute: {hw.capabilities.peak_tflops_fp32} TFLOPS FP32")

        # Use case fit
        if hw.suitable_for:
            reasons.append(f"Suitable for: {', '.join(hw.suitable_for)}")

        return reasons, warnings

    def _estimate_performance(
        self, hw: HardwareProfile, model_params: int, model_memory_mb: float
    ) -> Dict[str, Any]:
        """Estimate performance metrics.

        Args:
            hw: Hardware profile
            model_params: Model parameter count
            model_memory_mb: Model memory requirement

        Returns:
            Dictionary with estimated performance metrics
        """
        estimates = {}

        # Rough estimate of inference time
        # Assumes 2 FLOPs per parameter per inference
        if hw.capabilities.peak_tflops_fp32:
            gflops_needed = (model_params * 2) / 1e9
            tflops_needed = gflops_needed / 1000

            # Assume 50% efficiency
            estimated_latency_ms = (tflops_needed / hw.capabilities.peak_tflops_fp32) * 1000 * 2
            estimates["estimated_latency_ms"] = round(estimated_latency_ms, 2)

        # Memory efficiency
        memory_utilization = (model_memory_mb / 1024) / hw.capabilities.memory_gb * 100
        estimates["memory_utilization_percent"] = round(memory_utilization, 1)

        return estimates

    def list_hardware(self, hardware_type: str | None = None) -> List[str]:
        """List all available hardware in the knowledge base.

        Args:
            hardware_type: Optional filter by hardware type

        Returns:
            List of hardware names
        """
        profiles = self.hardware_profiles
        if hardware_type:
            profiles = [p for p in profiles if p.hardware_type.value == hardware_type]
        return [p.name for p in profiles]

    def get_hardware_info(self, name: str) -> Dict[str, Any] | None:
        """Get detailed information about a specific hardware.

        Args:
            name: Hardware name

        Returns:
            Hardware information dictionary or None
        """
        hw = get_hardware_by_name(name)
        if hw:
            return hw.model_dump()
        return None
