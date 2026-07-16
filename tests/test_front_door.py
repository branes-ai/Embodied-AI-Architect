"""Seam S10 (issue #215): the loop front door — an NL mission string seeds a valid
DesignState (constraints + joint design space) with no manual setup."""

from embodied_ai_architect.graphs.design_state import DesignState
from embodied_ai_architect.graphs.loop_convergence_graph import seed_node
from embodied_ai_architect.research.decomposer import MissionDecomposer, plan_to_constraints


def test_nl_mission_seeds_constraints_and_design_space() -> None:
    state: DesignState = {
        "mission_description": "Delivery-drone perception at 5 m/s: detection + "
        "tracking + VIO, 5W power budget, 33ms latency",
    }
    out = seed_node(state)

    # constraints inferred from the mission (no manual setup)
    assert out["constraints"]["max_power_watts"] == 5.0
    assert out["constraints"]["max_latency_ms"] is not None
    # a real joint design space was materialized
    dsc = out["design_space_config"]
    assert dsc["source"] == "mission_decomposer"
    assert dsc["num_variables"] >= 17  # hw + pipeline + NAS + compiler
    assert dsc["num_objectives"] >= 3
    # mission context carried through
    assert out["platform"] == "drone"
    assert out["mission_type"]
    assert "mission_plan" in out


def test_existing_constraints_build_space_without_decompose() -> None:
    """If constraints are already set, the front door materializes a design space
    from them and does not overwrite them."""
    state: DesignState = {"constraints": {"max_power_watts": 8.0, "max_latency_ms": 20.0}}
    out = seed_node(state)
    assert "constraints" not in out  # untouched
    assert out["design_space_config"]["num_variables"] >= 17


def test_no_mission_falls_back_to_stub() -> None:
    out = seed_node({})
    assert out["design_space_config"] == {"source": "default_joint_space"}


def test_already_seeded_is_noop() -> None:
    out = seed_node({"design_space_config": {"num_variables": 17}})
    assert set(out) == {"status"}


def test_plan_to_constraints_maps_named_constraints() -> None:
    plan = MissionDecomposer(llm_client=None).decompose("Drone perception, 5W, 33ms")
    c = plan_to_constraints(plan)
    assert c["max_power_watts"] == 5.0
    assert "max_latency_ms" in c and "workload_type" in c
