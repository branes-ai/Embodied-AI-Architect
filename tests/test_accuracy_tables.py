"""Tests for accuracy lookup tables and capability-per-watt integration."""

from embodied_ai_architect.graphs.moo.accuracy_tables import (
    ModelVariantSpec,
    QuantizationImpact,
    estimate_effective_accuracy,
    estimate_effective_gflops,
    get_accuracy_gflops_frontier,
    get_model_accuracy,
    get_model_gflops,
    get_model_spec,
    get_models_for_task,
    get_quantization_impact,
    list_model_families,
    list_variants,
    lookup_accuracy_for_workload,
)


class TestModelLookup:
    def test_list_families(self):
        families = list_model_families()
        assert "yolov8" in families
        assert "resnet" in families
        assert "mobilenet" in families
        assert len(families) >= 8

    def test_list_variants(self):
        variants = list_variants("yolov8")
        assert "n" in variants
        assert "s" in variants
        assert "x" in variants
        assert len(variants) == 5

    def test_get_model_spec(self):
        spec = get_model_spec("yolov8", "s")
        assert spec is not None
        assert isinstance(spec, ModelVariantSpec)
        assert spec.accuracy == 44.9
        assert spec.params_m == 11.2
        assert spec.gflops == 28.6
        assert spec.task == "detection"
        assert spec.primary_metric == "mAP50"

    def test_get_model_spec_not_found(self):
        assert get_model_spec("nonexistent", "v1") is None

    def test_get_model_accuracy(self):
        assert get_model_accuracy("yolov8", "n") == 37.3
        assert get_model_accuracy("resnet", "50") == 76.1
        assert get_model_accuracy("nonexistent", "x") is None

    def test_get_model_gflops(self):
        assert get_model_gflops("yolov8", "s") == 28.6
        assert get_model_gflops("nonexistent", "x") is None

    def test_models_for_task(self):
        detection_models = get_models_for_task("detection")
        assert len(detection_models) > 0
        # Should be sorted by accuracy descending
        for i in range(len(detection_models) - 1):
            assert detection_models[i].accuracy >= detection_models[i + 1].accuracy

        classification_models = get_models_for_task("classification")
        assert len(classification_models) > 0

        segmentation_models = get_models_for_task("segmentation")
        assert len(segmentation_models) > 0

    def test_accuracy_gflops_frontier(self):
        frontier = get_accuracy_gflops_frontier("detection")
        assert len(frontier) >= 2
        # Should be sorted by GFLOPS ascending
        for i in range(len(frontier) - 1):
            assert frontier[i].gflops <= frontier[i + 1].gflops
        # Should have increasing accuracy
        for i in range(len(frontier) - 1):
            assert frontier[i].accuracy < frontier[i + 1].accuracy


class TestQuantizationImpact:
    def test_fp32_baseline(self):
        impact = get_quantization_impact("fp32")
        assert impact is not None
        assert impact.accuracy_drop_pct == 0.0
        assert impact.speedup_factor == 1.0

    def test_int8(self):
        impact = get_quantization_impact("int8")
        assert impact is not None
        assert impact.accuracy_drop_pct > 0.0
        assert impact.speedup_factor > 1.0
        assert impact.memory_reduction > 1.0

    def test_all_dtypes(self):
        for dtype in ["fp32", "fp16", "bf16", "int8", "int4", "mixed_int8_fp16"]:
            impact = get_quantization_impact(dtype)
            assert impact is not None, f"Missing impact for {dtype}"
            assert isinstance(impact, QuantizationImpact)

    def test_unknown_dtype(self):
        assert get_quantization_impact("binary") is None


class TestEffectiveAccuracy:
    def test_fp32_no_pruning(self):
        acc = estimate_effective_accuracy("yolov8", "s", "fp32", 0.0)
        assert acc == 44.9  # No degradation

    def test_int8_reduces_accuracy(self):
        fp32_acc = estimate_effective_accuracy("yolov8", "s", "fp32")
        int8_acc = estimate_effective_accuracy("yolov8", "s", "int8")
        assert int8_acc is not None
        assert int8_acc < fp32_acc

    def test_pruning_reduces_accuracy(self):
        base = estimate_effective_accuracy("yolov8", "s", "fp32", 0.0)
        pruned = estimate_effective_accuracy("yolov8", "s", "fp32", 0.5)
        assert pruned is not None
        assert pruned < base

    def test_heavy_pruning_significant_drop(self):
        base = estimate_effective_accuracy("yolov8", "s", "fp32", 0.0)
        heavy = estimate_effective_accuracy("yolov8", "s", "fp32", 0.8)
        assert heavy is not None
        assert base - heavy > 5.0  # Significant drop at 80% pruning

    def test_combined_quantization_and_pruning(self):
        base = estimate_effective_accuracy("yolov8", "s", "fp32", 0.0)
        combined = estimate_effective_accuracy("yolov8", "s", "int8", 0.3)
        assert combined is not None
        assert combined < base

    def test_nonexistent_model(self):
        assert estimate_effective_accuracy("fake", "model", "fp32") is None

    def test_effective_gflops(self):
        base = estimate_effective_gflops("yolov8", "s", 0.0)
        pruned = estimate_effective_gflops("yolov8", "s", 0.5)
        assert base is not None
        assert pruned is not None
        assert pruned < base

    def test_effective_gflops_nonexistent(self):
        assert estimate_effective_gflops("fake", "model") is None


class TestWorkloadLookup:
    def test_detection_workload(self):
        result = lookup_accuracy_for_workload("detection", 100.0, "fp32")
        assert "accuracy" in result
        assert "model_family" in result
        assert "model_variant" in result
        assert result["accuracy"] > 0

    def test_classification_workload(self):
        result = lookup_accuracy_for_workload("classification", 5.0, "fp32")
        assert result["accuracy"] > 0
        assert result["primary_metric"] == "top1"

    def test_low_gflops_budget(self):
        low = lookup_accuracy_for_workload("detection", 1.0, "fp32")
        high = lookup_accuracy_for_workload("detection", 300.0, "fp32")
        # Higher GFLOPS budget should allow better accuracy
        assert high["accuracy"] >= low["accuracy"]

    def test_quantization_reduces_lookup_accuracy(self):
        fp32 = lookup_accuracy_for_workload("detection", 50.0, "fp32")
        int8 = lookup_accuracy_for_workload("detection", 50.0, "int8")
        assert int8["accuracy"] < fp32["accuracy"]

    def test_unknown_workload_type_defaults_to_detection(self):
        result = lookup_accuracy_for_workload("unknown_type", 50.0)
        assert result["accuracy"] > 0
