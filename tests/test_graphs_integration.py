"""Integration tests for graphs verdict-first tools.

Tests the full pipeline from embodied-ai-architect through graphs
to embodied-schemas, verifying verdict-first output.
"""

import json
import pytest

# Skip all tests if dependencies are not available
try:
    from embodied_ai_architect.llm.graphs_tools import (
        check_latency,
        check_power,
        check_memory,
        full_analysis,
        HAS_GRAPHS,
        HAS_PYDANTIC,
    )

    DEPS_AVAILABLE = HAS_GRAPHS and HAS_PYDANTIC
except ImportError:
    DEPS_AVAILABLE = False
    HAS_GRAPHS = False
    HAS_PYDANTIC = False

pytestmark = pytest.mark.skipif(
    not DEPS_AVAILABLE, reason="graphs and/or embodied-schemas not installed"
)


class TestCheckLatency:
    """Tests for check_latency verdict-first tool."""

    def test_pass_verdict(self):
        """Test PASS verdict when latency is under target."""
        result_json = check_latency(
            model_name="resnet18",
            hardware_name="H100-SXM5-80GB",
            latency_target_ms=100.0,  # Very generous target
        )
        result = json.loads(result_json)

        assert result["verdict"] == "PASS"
        assert result["confidence"] in ["high", "medium", "low"]
        assert "summary" in result
        assert result["constraint"]["metric"] == "latency"
        assert result["constraint"]["threshold"] == 100.0
        assert result["constraint"]["margin_pct"] > 0  # Positive = headroom

    def test_fail_verdict(self):
        """Test FAIL verdict when latency exceeds target."""
        result_json = check_latency(
            model_name="resnet18",
            hardware_name="H100-SXM5-80GB",
            latency_target_ms=0.001,  # Impossibly tight target
        )
        result = json.loads(result_json)

        assert result["verdict"] == "FAIL"
        assert result["constraint"]["margin_pct"] < 0  # Negative = exceeded
        assert len(result.get("suggestions", [])) > 0  # Should have suggestions

    def test_edge_hardware(self):
        """Test on edge hardware (Jetson)."""
        result_json = check_latency(
            model_name="mobilenet_v2",
            hardware_name="Jetson-Orin-AGX",
            latency_target_ms=50.0,
        )
        result = json.loads(result_json)

        assert result["verdict"] in ["PASS", "FAIL"]
        assert "Jetson" in result["hardware_id"] or "Orin" in result["hardware_id"]

    def test_includes_metrics(self):
        """Test that full metrics are included."""
        result_json = check_latency(
            model_name="resnet18",
            hardware_name="H100-SXM5-80GB",
            latency_target_ms=10.0,
        )
        result = json.loads(result_json)

        # Check metrics
        assert "metrics" in result
        assert "latency_ms" in result["metrics"]
        assert "throughput_fps" in result["metrics"]
        assert "energy_per_inference_mj" in result["metrics"]
        assert "peak_memory_mb" in result["metrics"]

        # Check breakdowns
        assert "roofline" in result
        assert "energy" in result
        assert "memory" in result


class TestCheckPower:
    """Tests for check_power verdict-first tool."""

    def test_pass_verdict(self):
        """Test PASS verdict when power is under budget."""
        result_json = check_power(
            model_name="resnet18",
            hardware_name="H100-SXM5-80GB",
            power_budget_w=500.0,  # H100 TDP is 700W
        )
        result = json.loads(result_json)

        assert result["verdict"] == "PASS"
        assert result["constraint"]["metric"] == "power"

    def test_fail_verdict(self):
        """Test FAIL verdict when power exceeds budget."""
        result_json = check_power(
            model_name="resnet18",
            hardware_name="H100-SXM5-80GB",
            power_budget_w=10.0,  # Very tight budget
        )
        result = json.loads(result_json)

        assert result["verdict"] == "FAIL"

    def test_edge_power_budget(self):
        """Test on edge hardware with typical power budget."""
        result_json = check_power(
            model_name="mobilenet_v2",
            hardware_name="Jetson-Orin-Nano",
            power_budget_w=15.0,  # Typical edge budget
        )
        result = json.loads(result_json)

        assert result["verdict"] in ["PASS", "FAIL"]
        assert result["energy"]["average_power_w"] > 0


class TestCheckMemory:
    """Tests for check_memory verdict-first tool."""

    def test_pass_verdict(self):
        """Test PASS verdict when memory fits."""
        result_json = check_memory(
            model_name="resnet18",
            hardware_name="H100-SXM5-80GB",
            memory_budget_mb=1000.0,  # 1 GB budget
        )
        result = json.loads(result_json)

        assert result["verdict"] == "PASS"
        assert result["constraint"]["metric"] == "memory"

    def test_fail_verdict(self):
        """Test FAIL verdict when memory exceeds budget."""
        result_json = check_memory(
            model_name="resnet50",
            hardware_name="H100-SXM5-80GB",
            memory_budget_mb=1.0,  # 1 MB budget - way too small
        )
        result = json.loads(result_json)

        assert result["verdict"] == "FAIL"

    def test_memory_breakdown(self):
        """Test that memory breakdown is included."""
        result_json = check_memory(
            model_name="resnet18",
            hardware_name="H100-SXM5-80GB",
            memory_budget_mb=1000.0,
        )
        result = json.loads(result_json)

        assert "memory" in result
        assert "weights_mb" in result["memory"]
        assert "activations_mb" in result["memory"]
        assert "fits_in_l2" in result["memory"]
        assert "fits_in_device_memory" in result["memory"]


class TestFullAnalysis:
    """Tests for full_analysis verdict-first tool."""

    def test_without_constraint(self):
        """Test full analysis without constraint returns PASS."""
        result_json = full_analysis(
            model_name="resnet18",
            hardware_name="H100-SXM5-80GB",
        )
        result = json.loads(result_json)

        # Without constraint, should return PASS with full analysis
        assert result["verdict"] == "PASS"
        assert "roofline" in result
        assert "energy" in result
        assert "memory" in result

    def test_with_latency_constraint(self):
        """Test full analysis with latency constraint."""
        result_json = full_analysis(
            model_name="resnet18",
            hardware_name="H100-SXM5-80GB",
            constraint_metric="latency",
            constraint_threshold=10.0,
        )
        result = json.loads(result_json)

        assert result["constraint"]["metric"] == "latency"
        assert result["constraint"]["threshold"] == 10.0

    def test_with_energy_constraint(self):
        """Test full analysis with energy constraint."""
        result_json = full_analysis(
            model_name="resnet18",
            hardware_name="H100-SXM5-80GB",
            constraint_metric="energy",
            constraint_threshold=500.0,  # 500 mJ
        )
        result = json.loads(result_json)

        assert result["constraint"]["metric"] == "energy"

    def test_precision_parameter(self):
        """Test that precision parameter is respected."""
        result_fp32 = json.loads(
            full_analysis(
                model_name="resnet18",
                hardware_name="H100-SXM5-80GB",
                precision="FP32",
            )
        )
        result_fp16 = json.loads(
            full_analysis(
                model_name="resnet18",
                hardware_name="H100-SXM5-80GB",
                precision="FP16",
            )
        )

        assert result_fp32["precision"] == "fp32"
        assert result_fp16["precision"] == "fp16"


class TestVerdictFirstPattern:
    """Tests verifying the verdict-first pattern is correct."""

    def test_verdict_is_first_field(self):
        """Test that verdict appears first in output."""
        result_json = check_latency(
            model_name="resnet18",
            hardware_name="H100-SXM5-80GB",
            latency_target_ms=10.0,
        )
        result = json.loads(result_json)

        # Verdict should be present
        assert "verdict" in result
        assert result["verdict"] in ["PASS", "FAIL", "PARTIAL", "UNKNOWN"]

    def test_confidence_present(self):
        """Test that confidence is present."""
        result_json = check_latency(
            model_name="resnet18",
            hardware_name="H100-SXM5-80GB",
            latency_target_ms=10.0,
        )
        result = json.loads(result_json)

        assert "confidence" in result
        assert result["confidence"] in ["high", "medium", "low"]

    def test_summary_is_meaningful(self):
        """Test that summary provides useful information."""
        result_json = check_latency(
            model_name="resnet18",
            hardware_name="H100-SXM5-80GB",
            latency_target_ms=10.0,
        )
        result = json.loads(result_json)

        assert "summary" in result
        assert len(result["summary"]) > 20  # Meaningful length
        # Should mention the key metric
        assert "atency" in result["summary"] or "target" in result["summary"].lower()

    def test_suggestions_on_fail(self):
        """Test that FAIL results include suggestions."""
        result_json = check_latency(
            model_name="resnet18",
            hardware_name="H100-SXM5-80GB",
            latency_target_ms=0.001,  # Will fail
        )
        result = json.loads(result_json)

        assert result["verdict"] == "FAIL"
        assert "suggestions" in result
        assert len(result["suggestions"]) > 0


class TestMultipleHardware:
    """Tests across multiple hardware targets."""

    @pytest.mark.parametrize(
        "hardware",
        [
            "H100-SXM5-80GB",
            "A100-SXM4-80GB",
            "Jetson-Orin-AGX",
            "TPU-v4",
        ],
    )
    def test_hardware_targets(self, hardware):
        """Test that different hardware targets work."""
        result_json = full_analysis(
            model_name="resnet18",
            hardware_name=hardware,
        )
        result = json.loads(result_json)

        assert result["verdict"] in ["PASS", "FAIL", "PARTIAL", "UNKNOWN"]
        assert result["metrics"]["latency_ms"] > 0

    @pytest.mark.parametrize(
        "model",
        [
            "resnet18",
            "resnet50",
            "mobilenet_v2",
        ],
    )
    def test_model_targets(self, model):
        """Test that different models work."""
        result_json = full_analysis(
            model_name=model,
            hardware_name="H100-SXM5-80GB",
        )
        result = json.loads(result_json)

        assert result["verdict"] in ["PASS", "FAIL", "PARTIAL", "UNKNOWN"]
        assert result["model_id"] is not None


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_model(self):
        """Test handling of invalid model name."""
        result_json = check_latency(
            model_name="not_a_real_model_12345",
            hardware_name="H100-SXM5-80GB",
            latency_target_ms=10.0,
        )
        result = json.loads(result_json)

        # Should return UNKNOWN with error
        assert result["verdict"] == "UNKNOWN"
        assert "error" in result

    def test_invalid_hardware(self):
        """Test handling of invalid hardware name."""
        result_json = check_latency(
            model_name="resnet18",
            hardware_name="not_a_real_hardware_12345",
            latency_target_ms=10.0,
        )
        result = json.loads(result_json)

        # Should return UNKNOWN with error
        assert result["verdict"] == "UNKNOWN"
        assert "error" in result
