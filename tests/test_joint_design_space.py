"""Tests for the joint design space (Phase 3).

Tests cover:
- create_joint_design_space: variable count, dimensions, sampling
- JointEvaluator: full-stack evaluation, accuracy, capability/watt
- Pipeline variables: tracker/state estimator impact on latency
- NAS variables: model variant selection, quantization, pruning
- Compiler variables: runtime utilization, graph fusion speedup
- Integration with MOO engine via explore_joint_design_space tool
"""

from __future__ import annotations

from embodied_ai_architect.graphs.moo.joint_design_space import (
    JointEvaluator,
    create_joint_design_space,
    _resolve_model,
)
from embodied_ai_architect.graphs.moo.design_space import ObjectiveDirection

# ---------------------------------------------------------------------------
# Design space factory tests
# ---------------------------------------------------------------------------


class TestCreateJointDesignSpace:
    """Tests for create_joint_design_space factory."""

    def test_full_joint_space_variable_count(self):
        ds = create_joint_design_space()
        # 7 HW + 3 pipeline + 3 NAS + 4 compiler = 17
        assert ds.n_var == 17

    def test_hw_only_space(self):
        ds = create_joint_design_space(
            include_pipeline=False, include_nas=False, include_compiler=False
        )
        assert ds.n_var == 7

    def test_includes_accuracy_by_default(self):
        ds = create_joint_design_space()
        assert "accuracy_percent" in ds.objectives
        assert "capability_per_watt" in ds.objectives

    def test_objective_directions(self):
        ds = create_joint_design_space()
        dir_map = dict(zip(ds.objectives, ds.directions))
        assert dir_map["power_watts"] == ObjectiveDirection.MINIMIZE
        assert dir_map["latency_ms"] == ObjectiveDirection.MINIMIZE
        assert dir_map["accuracy_percent"] == ObjectiveDirection.MAXIMIZE
        assert dir_map["capability_per_watt"] == ObjectiveDirection.MAXIMIZE

    def test_constraint_bounds(self):
        ds = create_joint_design_space(
            constraints={"max_power_watts": 10.0, "min_accuracy_percent": 30.0}
        )
        assert "power_watts" in ds.constraint_bounds
        assert ds.constraint_bounds["power_watts"] == (0.0, 10.0)
        assert "accuracy_percent" in ds.constraint_bounds
        assert ds.constraint_bounds["accuracy_percent"] == (30.0, 100.0)

    def test_custom_detector_families(self):
        ds = create_joint_design_space(
            constraints={"detector_families": ["yolov8", "efficientdet"]}
        )
        det_var = ds.get_variable("detector_family")
        assert det_var is not None
        assert set(det_var.choices) == {"yolov8", "efficientdet"}

    def test_sample_random(self):
        ds = create_joint_design_space()
        samples = ds.sample_random(10, seed=42)
        assert len(samples) == 10
        # Each sample should have all variables
        for s in samples:
            assert "process_nm" in s
            assert "detector_family" in s
            assert "quantization_dtype" in s
            assert "runtime" in s

    def test_encode_decode_roundtrip(self):
        ds = create_joint_design_space()
        sample = ds.sample_random(1, seed=42)[0]
        encoded = ds.encode(sample)
        decoded = ds.decode(encoded)
        # Categorical values should roundtrip
        assert decoded["detector_family"] == sample["detector_family"]
        assert decoded["runtime"] == sample["runtime"]

    def test_pipeline_variables_present(self):
        ds = create_joint_design_space(include_pipeline=True)
        names = {v.name for v in ds.variables}
        assert "detector_family" in names
        assert "tracker_type" in names
        assert "state_estimator" in names

    def test_nas_variables_present(self):
        ds = create_joint_design_space(include_nas=True)
        names = {v.name for v in ds.variables}
        assert "model_variant_idx" in names
        assert "quantization_dtype" in names
        assert "pruning_ratio" in names

    def test_compiler_variables_present(self):
        ds = create_joint_design_space(include_compiler=True)
        names = {v.name for v in ds.variables}
        assert "runtime" in names
        assert "tile_size" in names
        assert "batch_size" in names
        assert "graph_fusion" in names


# ---------------------------------------------------------------------------
# Model resolution tests
# ---------------------------------------------------------------------------


class TestResolveModel:
    """Tests for variant index → model mapping."""

    def test_yolov8_variants(self):
        assert _resolve_model("yolov8", 0) == ("yolov8", "n")
        assert _resolve_model("yolov8", 1) == ("yolov8", "s")
        assert _resolve_model("yolov8", 4) == ("yolov8", "x")

    def test_clamps_out_of_range(self):
        family, variant = _resolve_model("yolov8", 99)
        assert family == "yolov8"
        assert variant == "x"  # clamped to last

    def test_single_variant_family(self):
        family, variant = _resolve_model("ssd_mobilenet", 0)
        assert variant == "v2"

    def test_unknown_family_fallback(self):
        family, variant = _resolve_model("unknown_model", 0)
        assert family == "unknown_model"
        assert variant == "s"  # default fallback


# ---------------------------------------------------------------------------
# JointEvaluator tests
# ---------------------------------------------------------------------------


class TestJointEvaluator:
    """Tests for full-stack joint evaluation."""

    def _make_evaluator(self, **constraints):
        ds = create_joint_design_space(constraints=constraints)
        return JointEvaluator(
            design_space=ds,
            base_state={"constraints": constraints},
        )

    def test_basic_evaluation(self):
        evaluator = self._make_evaluator()
        ds = create_joint_design_space()
        sample = ds.sample_random(1, seed=42)[0]
        result = evaluator.evaluate(sample)
        assert result.objectives["power_watts"] > 0
        assert result.objectives["latency_ms"] > 0
        assert result.objectives["accuracy_percent"] >= 0

    def test_capability_per_watt_computed(self):
        evaluator = self._make_evaluator()
        ds = create_joint_design_space()
        sample = ds.sample_random(1, seed=42)[0]
        result = evaluator.evaluate(sample)
        assert "capability_per_watt" in result.objectives
        # Verify the math: capability_per_watt = (accuracy/100) / power
        acc = result.objectives["accuracy_percent"]
        power = result.objectives["power_watts"]
        expected = (acc / 100.0) / power if power > 0 else 0.0
        assert abs(result.objectives["capability_per_watt"] - round(expected, 4)) < 0.001

    def test_metadata_has_pipeline_info(self):
        evaluator = self._make_evaluator()
        ds = create_joint_design_space()
        sample = ds.sample_random(1, seed=42)[0]
        result = evaluator.evaluate(sample)
        assert "model_family" in result.metadata
        assert "tracker_type" in result.metadata
        assert "runtime" in result.metadata
        assert "effective_utilization" in result.metadata

    def test_tensorrt_has_higher_utilization(self):
        """TensorRT should give higher utilization than PyTorch."""
        evaluator = self._make_evaluator()
        ds = create_joint_design_space()

        base_params = ds.sample_random(1, seed=42)[0]

        base_params["runtime"] = "tensorrt"
        rt_result = evaluator.evaluate(base_params)

        base_params["runtime"] = "pytorch"
        pt_result = evaluator.evaluate(base_params)

        assert (
            rt_result.metadata["effective_utilization"]
            > pt_result.metadata["effective_utilization"]
        )

    def test_quantization_affects_accuracy(self):
        """INT8 should have lower accuracy than FP32."""
        evaluator = self._make_evaluator()
        ds = create_joint_design_space()

        base_params = ds.sample_random(1, seed=42)[0]
        base_params["detector_family"] = "yolov8"
        base_params["model_variant_idx"] = 1  # yolov8s

        base_params["quantization_dtype"] = "fp32"
        fp32_result = evaluator.evaluate(base_params)

        base_params["quantization_dtype"] = "int8"
        int8_result = evaluator.evaluate(base_params)

        assert (
            fp32_result.objectives["accuracy_percent"] > int8_result.objectives["accuracy_percent"]
        )

    def test_pruning_reduces_accuracy(self):
        """High pruning ratio should reduce accuracy."""
        evaluator = self._make_evaluator()
        ds = create_joint_design_space()

        base_params = ds.sample_random(1, seed=42)[0]
        base_params["detector_family"] = "yolov8"
        base_params["model_variant_idx"] = 1
        base_params["quantization_dtype"] = "fp32"

        base_params["pruning_ratio"] = 0.0
        no_prune = evaluator.evaluate(base_params)

        base_params["pruning_ratio"] = 0.5
        pruned = evaluator.evaluate(base_params)

        assert no_prune.objectives["accuracy_percent"] > pruned.objectives["accuracy_percent"]

    def test_larger_model_more_accurate(self):
        """YOLOv8x should be more accurate than YOLOv8n (same HW)."""
        evaluator = self._make_evaluator()
        ds = create_joint_design_space()

        base_params = ds.sample_random(1, seed=42)[0]
        base_params["detector_family"] = "yolov8"
        base_params["quantization_dtype"] = "fp32"
        base_params["pruning_ratio"] = 0.0

        base_params["model_variant_idx"] = 0  # yolov8n
        small_result = evaluator.evaluate(base_params)

        base_params["model_variant_idx"] = 4  # yolov8x
        large_result = evaluator.evaluate(base_params)

        assert (
            large_result.objectives["accuracy_percent"]
            > small_result.objectives["accuracy_percent"]
        )

    def test_graph_fusion_reduces_latency(self):
        """Graph fusion enabled should reduce latency vs disabled."""
        evaluator = self._make_evaluator()
        ds = create_joint_design_space()

        base_params = ds.sample_random(1, seed=42)[0]

        base_params["graph_fusion"] = True
        fused = evaluator.evaluate(base_params)

        base_params["graph_fusion"] = False
        unfused = evaluator.evaluate(base_params)

        assert fused.objectives["latency_ms"] < unfused.objectives["latency_ms"]

    def test_feasibility_check(self):
        """Designs violating constraints should be marked infeasible."""
        evaluator = self._make_evaluator(max_power_watts=0.001)
        ds = create_joint_design_space(constraints={"max_power_watts": 0.001})
        sample = ds.sample_random(1, seed=42)[0]
        result = evaluator.evaluate(sample)
        # Essentially no design can meet 0.001W
        assert result.feasible is False or result.objectives["power_watts"] <= 0.001

    def test_batch_evaluation(self):
        evaluator = self._make_evaluator()
        ds = create_joint_design_space()
        samples = ds.sample_random(5, seed=42)
        results = evaluator.evaluate_batch(samples)
        assert len(results) == 5
        assert all(r.objectives["power_watts"] > 0 for r in results)

    def test_error_handling(self):
        """Bad params should return a result, not crash."""
        evaluator = self._make_evaluator()
        # Even with bad params, evaluate should not raise
        result = evaluator.evaluate({"process_nm": -999})
        assert result.objectives["power_watts"] > 0


# ---------------------------------------------------------------------------
# Tool registration test
# ---------------------------------------------------------------------------


class TestJointToolRegistration:
    """Verify the joint exploration tool is registered."""

    def test_tool_definition_exists(self):
        from embodied_ai_architect.llm.optimization_tools import (
            get_optimization_tool_definitions,
        )

        defs = get_optimization_tool_definitions()
        names = {d["name"] for d in defs}
        assert "explore_joint_design_space" in names

    def test_executor_exists(self):
        from embodied_ai_architect.llm.optimization_tools import (
            create_optimization_tool_executors,
        )

        executors = create_optimization_tool_executors()
        assert "explore_joint_design_space" in executors
