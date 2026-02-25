"""Tests for working memory module."""

from embodied_ai_architect.graphs.memory import (
    AgentWorkingMemory,
    WorkingMemoryStore,
)


class TestAgentWorkingMemory:
    def test_defaults(self):
        mem = AgentWorkingMemory(agent_name="optimizer")
        assert mem.agent_name == "optimizer"
        assert mem.decisions_made == []
        assert mem.things_tried == []
        assert mem.constraints_discovered == []
        assert mem.iteration_notes == {}

    def test_serialization_round_trip(self):
        mem = AgentWorkingMemory(
            agent_name="optimizer",
            decisions_made=["chose INT8"],
            things_tried=[{"description": "INT8", "outcome": "power 5.1W", "iteration": 1}],
        )
        data = mem.model_dump()
        restored = AgentWorkingMemory(**data)
        assert restored.agent_name == "optimizer"
        assert restored.decisions_made == ["chose INT8"]
        assert len(restored.things_tried) == 1


class TestWorkingMemoryStore:
    def test_empty_store(self):
        store = WorkingMemoryStore()
        assert store.agents == {}

    def test_get_agent_memory_creates(self):
        store = WorkingMemoryStore()
        mem = store.get_agent_memory("optimizer")
        assert mem.agent_name == "optimizer"
        assert "optimizer" in store.agents

    def test_get_agent_memory_idempotent(self):
        store = WorkingMemoryStore()
        mem1 = store.get_agent_memory("optimizer")
        mem1.decisions_made.append("test")
        mem2 = store.get_agent_memory("optimizer")
        assert mem2.decisions_made == ["test"]

    def test_record_attempt(self):
        store = WorkingMemoryStore()
        store.record_attempt(
            agent_name="design_optimizer",
            description="INT8 quantization",
            outcome="Power reduced from 6.3W to 5.1W",
            iteration=1,
        )
        mem = store.get_agent_memory("design_optimizer")
        assert len(mem.things_tried) == 1
        assert mem.things_tried[0]["description"] == "INT8 quantization"
        assert mem.things_tried[0]["iteration"] == 1

    def test_record_decision(self):
        store = WorkingMemoryStore()
        store.record_decision("optimizer", "Apply INT8 quantization")
        mem = store.get_agent_memory("optimizer")
        assert "Apply INT8 quantization" in mem.decisions_made

    def test_record_constraint(self):
        store = WorkingMemoryStore()
        store.record_constraint("ppa_assessor", "Power exceeds 5W budget")
        mem = store.get_agent_memory("ppa_assessor")
        assert "Power exceeds 5W budget" in mem.constraints_discovered

    def test_get_tried_descriptions(self):
        store = WorkingMemoryStore()
        store.record_attempt("opt", "INT8", "good", 1)
        store.record_attempt("opt", "pruning", "bad", 2)
        tried = store.get_tried_descriptions("opt")
        assert tried == ["INT8", "pruning"]

    def test_get_tried_descriptions_empty(self):
        store = WorkingMemoryStore()
        tried = store.get_tried_descriptions("unknown_agent")
        assert tried == []

    def test_serialization_round_trip(self):
        store = WorkingMemoryStore()
        store.record_attempt("optimizer", "INT8", "power 5.1W", 1)
        store.record_decision("optimizer", "chose INT8")
        store.record_constraint("optimizer", "power budget tight")

        data = store.model_dump()
        restored = WorkingMemoryStore(**data)

        mem = restored.get_agent_memory("optimizer")
        assert len(mem.things_tried) == 1
        assert mem.decisions_made == ["chose INT8"]
        assert mem.constraints_discovered == ["power budget tight"]

    def test_multiple_agents(self):
        store = WorkingMemoryStore()
        store.record_attempt("optimizer", "INT8", "ok", 1)
        store.record_attempt("ppa_assessor", "check power", "fail", 1)

        assert len(store.agents) == 2
        assert store.get_tried_descriptions("optimizer") == ["INT8"]
        assert store.get_tried_descriptions("ppa_assessor") == ["check power"]
