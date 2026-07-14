"""Seam S3 (issue #208): DesignDelta.change is validated against a per-DeltaKind
typed payload model at construction, and _apply_delta consumes the typed fields."""

import pytest
from pydantic import ValidationError

from embodied_ai_architect.graphs.design_state import (
    AddVariablePayload,
    ConstraintRelaxationPayload,
    DesignDelta,
    DeltaKind,
    DesignSpaceEditPayload,
    RemoveVariablePayload,
    SpecialistRetaskPayload,
    VariableBoundChangePayload,
)
from embodied_ai_architect.graphs.loop_agents import Optimizer


def _delta(kind, target, change):
    return DesignDelta(kind=kind, target=target, change=change, rationale="test")


# ---------------------------------------------------------------------------
# Valid payloads construct and parse to the right model
# ---------------------------------------------------------------------------


def test_valid_payloads_construct_and_type():
    cases = [
        (
            DeltaKind.DESIGN_SPACE_EDIT,
            "quantization_dtype",
            {"value": "int8"},
            DesignSpaceEditPayload,
        ),
        (
            DeltaKind.VARIABLE_BOUND_CHANGE,
            "width_scale",
            {"bounds": [0.25, 1.0]},
            VariableBoundChangePayload,
        ),
        (
            DeltaKind.VARIABLE_BOUND_CHANGE,
            "detector",
            {"categories": ["a", "b"]},
            VariableBoundChangePayload,
        ),
        (
            DeltaKind.ADD_VARIABLE,
            "pruning_ratio",
            {"variable": {"type": "continuous"}},
            AddVariablePayload,
        ),
        (
            DeltaKind.REMOVE_VARIABLE,
            "batch_size",
            {},
            RemoveVariablePayload,
        ),
        (
            DeltaKind.CONSTRAINT_RELAXATION,
            "max_power_watts",
            {"to": 6.0, "from": 5.0},
            ConstraintRelaxationPayload,
        ),
        (
            DeltaKind.SPECIALIST_RETASK,
            "bandwidth_validator",
            {"reason": "x"},
            SpecialistRetaskPayload,
        ),
    ]
    for kind, target, change, expected_model in cases:
        d = _delta(kind, target, change)  # must not raise
        assert isinstance(d.typed_change(), expected_model)


def test_typed_change_exposes_fields():
    d = _delta(DeltaKind.DESIGN_SPACE_EDIT, "q", {"value": "int8"})
    assert d.typed_change().value == "int8"
    d = _delta(DeltaKind.CONSTRAINT_RELAXATION, "max_power_watts", {"to": 6.0, "from": 5.0})
    tc = d.typed_change()
    assert tc.to == 6.0 and tc.from_ == 5.0
    d = _delta(DeltaKind.SPECIALIST_RETASK, "bw", {"reason": "re-check", "extra": 1})
    assert d.typed_change().model_dump()["extra"] == 1  # extras pass through


# ---------------------------------------------------------------------------
# Invalid payloads raise at construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,change",
    [
        (DeltaKind.DESIGN_SPACE_EDIT, {}),  # missing value
        (DeltaKind.VARIABLE_BOUND_CHANGE, {}),  # neither bounds nor categories
        (DeltaKind.VARIABLE_BOUND_CHANGE, {"bounds": [1.0], "categories": None}),  # bad arity
        (DeltaKind.VARIABLE_BOUND_CHANGE, {"bounds": [0.0, 1.0], "categories": ["a"]}),  # both
        (DeltaKind.ADD_VARIABLE, {}),  # missing variable
        (DeltaKind.REMOVE_VARIABLE, {"variable": {}}),  # forbidden extra
        (DeltaKind.CONSTRAINT_RELAXATION, {"from": 5.0}),  # missing required 'to'
    ],
)
def test_invalid_payloads_raise_at_construction(kind, change):
    with pytest.raises(ValidationError):
        _delta(kind, "target", change)


# ---------------------------------------------------------------------------
# _apply_delta consumes the typed fields
# ---------------------------------------------------------------------------


def test_apply_delta_consumes_typed_fields():
    opt = Optimizer()
    state = {"design_space_config": {"variables": {"old": {}}}, "iteration": 0}

    opt._apply_delta(state, _delta(DeltaKind.DESIGN_SPACE_EDIT, "q", {"value": "int8"}))
    assert state["design_space_config"]["q"] == "int8"

    opt._apply_delta(
        state, _delta(DeltaKind.VARIABLE_BOUND_CHANGE, "det", {"categories": ["a", "b"]})
    )
    assert state["design_space_config"]["det"]["categories"] == ["a", "b"]

    opt._apply_delta(state, _delta(DeltaKind.VARIABLE_BOUND_CHANGE, "w", {"bounds": [0.25, 1.0]}))
    assert state["design_space_config"]["w"]["bounds"] == [0.25, 1.0]

    opt._apply_delta(state, _delta(DeltaKind.CONSTRAINT_RELAXATION, "max_power_watts", {"to": 6.0}))
    assert state["constraints"]["max_power_watts"] == 6.0

    opt._apply_delta(state, _delta(DeltaKind.SPECIALIST_RETASK, "bw", {"reason": "re-check"}))
    task = state["pending_specialist_tasks"][0]
    assert task["specialist"] == "bw" and task["reason"] == "re-check"


def test_specialist_target_wins_over_payload_extra():
    """A payload extra literally named 'specialist' must not override delta.target."""
    opt = Optimizer()
    state = {"iteration": 0}
    opt._apply_delta(
        state,
        _delta(
            DeltaKind.SPECIALIST_RETASK,
            "bandwidth_validator",
            {"specialist": "evil", "reason": "x"},
        ),
    )
    assert state["pending_specialist_tasks"][0]["specialist"] == "bandwidth_validator"
