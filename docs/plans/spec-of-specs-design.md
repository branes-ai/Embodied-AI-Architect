# Plan: Spec-of-Specs System (Hybrid Pydantic Registry + Event-Sourced Provenance)

## Context

Embodied AI systems have hundreds of attributes across subsystems (perception, compute, power, sensors, actuators, comms, autonomy, safety). Users need to incrementally define, refine, and version these requirements. Agents need to consume and modify them. Currently, the codebase has flat `PipelineRequirements` models that don't support hierarchy, versioning, or provenance.

This plan implements a **hierarchical spec system** with:
- Pydantic models for validation + structured subsystems
- Append-only event log for full provenance (who changed what, when, why)
- Content-addressed snapshots for versioning
- CLI commands for `branes spec ...`
- LLM tools so the chat agent can read/modify specs

## New Files

```
src/embodied_ai_architect/
  specs/                        # NEW package
    __init__.py                 # Public API exports
    models.py                   # Pydantic model hierarchy (SystemSpec + 8 subsystem specs)
    events.py                   # SpecEvent model, EventLog (append-only JSONL)
    store.py                    # SpecStore: content-addressed blobs + manifest + CRUD
    templates.py                # Predefined archetypes (drone, quadruped, AMR, etc.)
    diff.py                     # Structured diff between two spec versions
    validation.py               # Cross-subsystem consistency checks
    exceptions.py               # SpecNotFoundError, InvalidPathError, etc.
  cli/commands/
    spec.py                     # NEW: Click command group for `branes spec ...`
  llm/
    spec_tools.py               # NEW: LLM tool definitions + executors
```

## Modified Files

- `src/embodied_ai_architect/cli/__init__.py` — add `from ... import spec; cli.add_command(spec.spec)`
- `src/embodied_ai_architect/llm/tools.py` — add try/except import for spec_tools (same pattern as lines 17-30)

## Architecture

### Model Hierarchy (`specs/models.py`)

Follow the pattern from `requirements/models.py`: Pydantic BaseModel, Optional fields with Field descriptors, str Enums.

```
SystemSpec (root)
├── name: str, description: Optional[str], platform_type: Optional[str]
├── perception: Optional[PerceptionSpec]    # cameras, detection, tracking, accuracy, latency, fps
├── compute: Optional[ComputeSpec]          # SoC, CPU, GPU, accelerators, memory, quantization
├── power: Optional[PowerSpec]              # battery, power budget, thermal, cooling, mission duration
├── sensors: Optional[SensorSpec]           # modalities, rates, data rate, environmental rating
├── actuators: Optional[ActuatorSpec]       # DOF, control rate, speed, payload
├── comms: Optional[CommsSpec]              # protocols, bandwidth, latency, range
├── autonomy: Optional[AutonomySpec]        # autonomy level, planning, navigation, decision rate
├── safety: Optional[SafetySpec]            # safety level, redundancy, failsafe, certifications
├── constraints: list[SuccessCriterion]     # cross-cutting (reuse from embodied-schemas)
├── custom: dict[str, Any]                  # extensibility for user-defined subsystems
└── tags: list[str]
```

All subsystems are `Optional[...] = None` — users build up incrementally. Import `SuccessCriterion`/`ConstraintCriticality` from embodied-schemas with try/except fallback.

### Event Log (`specs/events.py`)

Every spec mutation is an event in an append-only JSONL file:

```python
class SpecEvent(BaseModel):
    op: EventOp          # create, set, delete, tag, snapshot, import
    path: Optional[str]  # JSON pointer path, e.g. "/perception/min_fps"
    value: Any           # new value
    author: str          # "user", agent name, "template"
    reason: Optional[str]# why the change was made
    timestamp: str       # ISO format
    spec_name: str
    sequence: int        # monotonic per-spec counter
```

`EventLog` class handles append, replay (with snapshot optimization), and field history queries. Replay uses `set_at_path(obj, path, value)` / `delete_at_path(obj, path)` utility functions. Auto-snapshots every 50 events for bounded replay time.

### Storage Layout (`specs/store.py`)

```
.branes/specs/                     # project-level (default)
  specs.json                       # index: {name -> metadata}
  <spec-name>/
    manifest.json                  # version chain: [{hash, parent, timestamp, author, message, tags}]
    events.jsonl                   # append-only event log
    blobs/
      ab/ab123...json              # content-addressed snapshots (SHA256)
```

`SpecStore` class with:
- `create()`, `get()`, `get_version()` — CRUD
- `set_field()`, `delete_field()` — mutations that record events
- `commit()` — create content-addressed snapshot, update manifest
- `tag()` — name a version
- `history()`, `diff()`, `why()` — queries
- `export()`, `import_spec()` — portability
- `validate()` — run cross-subsystem checks
- `resolve()` — flatten to `Dict[str, Any]` for agent `execute(input_data)`

Index JSON follows the existing registry pattern from `registry/model_registry.py`.

### CLI Commands (`cli/commands/spec.py`)

Click command group with these subcommands:

| Command | Description |
|---------|------------|
| `branes spec new <name> [--template T]` | Create spec (empty or from template) |
| `branes spec list` | List all specs (Rich Table) |
| `branes spec show <name> [--version V]` | Display spec tree (Rich Tree) |
| `branes spec set <name> <path> <value> [-m reason]` | Set a field |
| `branes spec delete <name> <path> [-m reason]` | Remove a field |
| `branes spec commit <name> -m "message"` | Snapshot current state |
| `branes spec history <name>` | Version history (Rich Table) |
| `branes spec diff <name> <v1> <v2>` | Colored structured diff |
| `branes spec tag <name> <tag> [-m msg]` | Tag a version |
| `branes spec export <name> [--format yaml\|json]` | Export spec |
| `branes spec import <name> <file>` | Import from YAML/JSON |
| `branes spec why <name> <path>` | Provenance of a field |
| `branes spec validate <name>` | Consistency check |
| `branes spec resolve <name>` | Flatten for agent consumption |

### LLM Tools (`llm/spec_tools.py`)

6 tools following the pattern from `llm/codebase_tools.py`:
- `read_spec` — get current spec state
- `modify_spec` — set a field (with mandatory `reason`)
- `validate_spec` — check consistency
- `list_specs` — list available specs
- `create_spec` — create from template
- `spec_field_history` — provenance of a field

Agent modifications set `author` to the agent name, enabling user vs agent attribution in provenance.

### Templates (`specs/templates.py`)

Predefined archetypes: `drone-perception`, `quadruped-nav`, `industrial-inspection`, `amr-warehouse`, `edge-camera`, `biped-humanoid`. Each returns a `SystemSpec` with sensible defaults.

### Validation (`specs/validation.py`)

Cross-subsystem checks:
- Power budget vs compute TDP
- Perception latency vs constraint targets
- Platform type implications (drone → low power, battery)
- Safety level vs redundancy requirements

Returns `list[ValidationIssue]` with severity, path, message, suggestion.

## Implementation Order

```
Phase 1: models.py, events.py, exceptions.py, __init__.py
    │     (Pydantic models + event log + path utilities)
    │
    ├──→ Phase 2: store.py, diff.py
    │     (content-addressed storage + manifest + CRUD + diff)
    │
    ├──→ Phase 3: validation.py, templates.py
    │     (cross-subsystem checks + predefined archetypes)
    │
    ├──→ Phase 4: cli/commands/spec.py + modify cli/__init__.py
    │     (all Click subcommands)
    │
    ├──→ Phase 5: llm/spec_tools.py + modify llm/tools.py
    │     (chat agent integration)
    │
    └──→ Phase 6: store.py resolve() + optional orchestrator hook
          (agent consumption)
```

## Verification

1. **Unit tests**: Create `tests/test_spec_models.py` — model serialization, event replay, path utilities
2. **Integration tests**: Create `tests/test_spec_store.py` — full CRUD cycle, commit/tag/diff/history/why
3. **CLI smoke test**: `branes spec new test-drone --template drone-perception && branes spec show test-drone && branes spec set test-drone /perception/min_fps 60 -m "need 60fps" && branes spec commit test-drone -m "initial" && branes spec history test-drone && branes spec why test-drone /perception/min_fps`
4. **Validate**: `branes spec validate test-drone` should report any cross-subsystem issues
5. **Export round-trip**: `branes spec export test-drone --format yaml > /tmp/spec.yaml && branes spec import test-drone-copy /tmp/spec.yaml`
