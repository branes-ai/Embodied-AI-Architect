"""Tests for the DesignSpace and DesignVariable models."""

import numpy as np
import pytest

from embodied_ai_architect.graphs.moo.design_space import (
    DesignSpace,
    DesignVariable,
    ObjectiveDirection,
    VarType,
    create_soc_design_space,
)


class TestDesignVariable:
    def test_integer_variable(self):
        var = DesignVariable(name="x", var_type=VarType.INTEGER, lower=2, upper=64)
        assert var.n_levels() == 63
        assert var.validate_value(10)
        assert not var.validate_value(100)

    def test_continuous_variable(self):
        var = DesignVariable(name="f", var_type=VarType.CONTINUOUS, lower=100.0, upper=3000.0)
        assert var.n_levels() == 0  # continuous has no discrete levels
        assert var.validate_value(500.0)
        assert not var.validate_value(5000.0)

    def test_categorical_variable(self):
        var = DesignVariable(name="node", var_type=VarType.CATEGORICAL, choices=[7, 14, 28, 65])
        assert var.n_levels() == 4
        assert var.validate_value(28)
        assert not var.validate_value(10)


class TestDesignSpace:
    @pytest.fixture
    def simple_space(self):
        return DesignSpace(
            variables=[
                DesignVariable(name="x", var_type=VarType.CONTINUOUS, lower=0.0, upper=10.0),
                DesignVariable(name="y", var_type=VarType.INTEGER, lower=1, upper=5),
                DesignVariable(name="z", var_type=VarType.CATEGORICAL, choices=["a", "b", "c"]),
            ],
            objectives=["f1", "f2"],
            directions=[ObjectiveDirection.MINIMIZE, ObjectiveDirection.MINIMIZE],
        )

    def test_n_var_n_obj(self, simple_space):
        assert simple_space.n_var == 3
        assert simple_space.n_obj == 2

    def test_sample_random(self, simple_space):
        samples = simple_space.sample_random(50, seed=42)
        assert len(samples) == 50
        for s in samples:
            assert "x" in s
            assert "y" in s
            assert "z" in s
            assert 0.0 <= s["x"] <= 10.0
            assert 1 <= s["y"] <= 5
            assert s["z"] in ["a", "b", "c"]

    def test_encode_decode_roundtrip(self, simple_space):
        params = {"x": 5.5, "y": 3, "z": "b"}
        encoded = simple_space.encode(params)
        assert encoded.shape == (3,)
        decoded = simple_space.decode(encoded)
        assert abs(decoded["x"] - 5.5) < 1e-10
        assert decoded["y"] == 3
        assert decoded["z"] == "b"

    def test_encode_categorical_index(self, simple_space):
        params = {"x": 0.0, "y": 1, "z": "c"}
        encoded = simple_space.encode(params)
        assert encoded[2] == 2.0  # "c" is index 2

    def test_decode_clamps_bounds(self, simple_space):
        x = np.array([15.0, 10.0, 5.0])  # all out of bounds
        decoded = simple_space.decode(x)
        assert decoded["x"] == 10.0  # clamped to upper
        assert decoded["y"] == 5  # clamped to upper
        assert decoded["z"] == "c"  # index 5 -> clamped to last

    def test_get_variable(self, simple_space):
        assert simple_space.get_variable("x") is not None
        assert simple_space.get_variable("nonexistent") is None


class TestDesignSpaceExport:
    @pytest.fixture
    def space(self):
        return create_soc_design_space()

    def test_to_pymoo_problem_kwargs(self, space):
        kwargs = space.to_pymoo_problem_kwargs()
        assert kwargs["n_var"] == space.n_var
        assert kwargs["n_obj"] == space.n_obj
        assert kwargs["xl"].shape == (space.n_var,)
        assert kwargs["xu"].shape == (space.n_var,)

    def test_to_botorch_bounds(self, space):
        bounds = space.to_botorch_bounds()
        assert bounds.shape == (2, space.n_var)
        # Lower bounds <= upper bounds
        assert (bounds[0] <= bounds[1]).all()


class TestCreateSoCDesignSpace:
    def test_default_factory(self):
        ds = create_soc_design_space()
        assert ds.n_var == 7  # 7 default variables
        assert ds.n_obj == 4  # power, latency, area, cost
        assert "process_nm" in [v.name for v in ds.variables]
        assert "clock_mhz" in [v.name for v in ds.variables]

    def test_factory_with_constraints(self):
        ds = create_soc_design_space({"max_power_watts": 5.0, "max_latency_ms": 33.0})
        assert "power_watts" in ds.constraint_bounds
        assert "latency_ms" in ds.constraint_bounds
        assert ds.constraint_bounds["power_watts"] == (0.0, 5.0)

    def test_process_nm_from_technology(self):
        ds = create_soc_design_space()
        proc_var = ds.get_variable("process_nm")
        assert proc_var is not None
        assert proc_var.var_type == VarType.CATEGORICAL
        # Should include at least some standard nodes
        assert 28 in proc_var.choices
        assert 7 in proc_var.choices

    def test_lhs_coverage(self):
        """LHS should spread samples across the design space."""
        ds = create_soc_design_space()
        samples = ds.sample_random(100, seed=42)
        # Check that process_nm has variety
        nodes = set(s["process_nm"] for s in samples)
        assert len(nodes) >= 5  # should hit at least 5 of 17 nodes
