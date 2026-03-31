# Goal Qualification System — Design Assessment

## Problem Statement

Users give underspecified goals like "drone perception SoC" which are
meaningless without knowing:

- What kind of drone (fixed-wing survey, multirotor delivery, racing FPV, agricultural spraying)
- What perception tasks (obstacle avoidance, object detection, landing zone detection, SLAM, visual odometry, payload inspection)
- Quality requirements (detection range, false positive rate, minimum object size)
- What the perception drives (flight controller, path planner, mission abort, payload release)
- Environmental conditions (indoor warehouse, outdoor urban, GPS-denied, night/IR)
- Regulatory/safety requirements (Part 107, BVLOS, over people)

There is no such thing as "drone perception" in isolation. A delivery drone
obstacle avoidance SoC and a racing FPV SLAM SoC share almost no design
parameters. The system should refuse to proceed with an underspecified goal
and instead help the user refine it through structured Q&A until we have a
"tangible" goal — meaning we can enumerate the actual hw/sw components needed.

This is not sycophantic pushback — it is the fundamental requirement for
producing a meaningful design. Proceeding with assumptions on underspecified
goals wastes compute and produces results the architect cannot evaluate.

## What "Tangible" Means Concretely

A goal is tangible when all of the following can be enumerated from the
stated information:

| Dimension | Vague | Tangible |
|-----------|-------|----------|
| **Platform** | "drone" | "multirotor delivery drone, 2kg payload, 25min endurance" |
| **Perception tasks** | "perception" | "obstacle avoidance: detect >30cm objects at >5m range, >90% recall" |
| **Control output** | (unstated) | "feeds flight controller at 50Hz -> perception budget <20ms" |
| **Power envelope** | "<5W" | "5W compute budget from 100Wh battery shared with motors+avionics" |
| **Environment** | (unstated) | "outdoor urban, GPS available, daytime visible spectrum" |

The concrete test: can we populate at least the `perception`, `power`, and
`autonomy` subsections of a `SystemSpec` with non-None values? If not, the
goal is underspecified.

```python
class GoalQualification(BaseModel):
    platform_identified: bool       # PlatformType resolvable
    perception_tasks_enumerated: bool  # >= 1 task with quality target
    control_output_identified: bool    # What consumes perception output
    power_envelope_bounded: bool       # Watts number exists or derivable
    environment_characterized: bool    # Enough for sensor selection

    @property
    def is_tangible(self) -> bool:
        return all([
            self.platform_identified,
            self.perception_tasks_enumerated,
            self.control_output_identified,
            self.power_envelope_bounded,
        ])
        # environment_characterized is soft -- can be inferred
```

## What Exists Today

The current pipeline accepts a goal string and optional constraints, then
proceeds directly to planning:

```
User goal string -> create_initial_soc_state() -> PlannerNode -> ReviewNode -> Dispatch -> ...
```

There is no validation that the goal is actionable. The `design_soc` tool
requires only `goal` (a string). The `MissionDecomposer` in
`research/decomposer.py` does keyword extraction but never refuses or
pushes back — it always produces output, even from vague input. The
`SystemSpec` in `specs/models.py` has the right subsystem structure
(perception, compute, power, sensors, actuators, comms, autonomy, safety)
but is never consulted before the planner runs.

---

## Design 1: LangGraph Gate Node

### Where It Sits

A new `goal_qualifier` node is inserted as the entry point of the LangGraph
StateGraph, before the planner:

```
goal_qualifier -> [planner -> plan_review -> dispatch -> evaluate -> ...]
```

The node implements a state machine with three internal phases: ASSESS,
QUESTION, QUALIFIED. It uses LangGraph's `interrupt_before` mechanism
(same pattern as `plan_review`) to pause and collect user answers.

### How It Interacts With the User

The node builds a `GoalQualificationSnapshot` (analogous to
`PlanReviewSnapshot`) containing:

- A tangibility scorecard showing which of the 5 required dimensions are
  specified vs. missing
- The next question to ask (chosen from a priority-ordered list of missing
  dimensions)
- Candidate answers derived from the platform/domain (e.g., "For a
  multirotor drone, typical perception tasks include: obstacle avoidance,
  landing zone detection, payload inspection — which apply?")

The user provides a `GoalRefinementInput` with their answer. The node
validates the answer, updates state, re-assesses tangibility, and either
asks the next question or transitions to QUALIFIED.

### Convergence Mechanism

Maximum 5 question rounds (one per dimension). If after 5 rounds the goal
is still not tangible, the system presents what it has, explains what is
missing and what assumptions it will make, and asks for a final
approve/reject.

Questions are asked in dependency order: platform first (it constrains
everything else), then perception tasks, then control output, then power
envelope. Environment is inferred if not stated.

### Connection to Existing Flow

- Adds new fields to `SoCDesignState`: `qualification_snapshot`,
  `qualification_input`, `goal_qualification`
- `build_soc_design_graph()` gets a new `goal_qualification: bool` parameter
  (default True)
- `SoCDesignRunner.start()` returns `"qualifying_goal"` as a new status
- The `design_soc` tool gets a companion `refine_goal` tool

### Pros

- Fits naturally into the existing LangGraph interrupt/resume pattern
- State is persisted in the LangGraph checkpoint, sessions can be
  suspended and resumed
- Clean separation: qualification is its own node, planner code unchanged
- The qualification snapshot is renderable (ASCII, Rich, JSON) like the
  plan review snapshot

### Cons

- Adds complexity to the already 6-node graph topology
- Each qualification round requires a full graph invoke/resume cycle
- The LLM is used both to generate questions and to assess answers —
  expensive for what could be a simple form
- Tight coupling to LangGraph mechanics; harder to test in isolation

---

## Design 2: Structured Questionnaire with Domain Templates

### Where It Sits

A standalone module `src/embodied_ai_architect/qualification/` that runs
**before** any LangGraph graph is constructed. It is invoked by the CLI
command or chat tool, and its output is a populated `SystemSpec` that is
then converted to `SoCDesignState` inputs.

```
User goal string -> GoalQualifier.assess() -> [Q&A loop] -> SystemSpec -> create_initial_soc_state()
```

This is a pure Python class with no LangGraph dependency.

### How It Interacts With the User

The qualifier uses domain-specific question templates organized by platform
type. When the user says "drone perception SoC", the keyword extractor
identifies platform=drone and loads the drone question template:

```python
DRONE_QUESTIONS = [
    Question(
        id="drone_type",
        text="What type of drone?",
        options=["multirotor_delivery", "multirotor_racing", "fixed_wing_survey",
                 "fixed_wing_cargo", "agricultural_sprayer", "fpv_racing"],
        required=True,
        implications={  # each answer pre-fills downstream fields
            "multirotor_delivery": {"max_speed_mps": 15, "payload_kg": 2.0,
                                     "power_budget_watts": 5.0, "mission_duration_min": 25},
            "fixed_wing_survey": {"max_speed_mps": 25, "payload_kg": 0.5,
                                   "power_budget_watts": 3.0, "mission_duration_min": 60},
        },
    ),
    Question(
        id="perception_tasks",
        text="Which perception tasks are needed?",
        options=["obstacle_avoidance", "object_detection", "landing_zone_detection",
                 "visual_odometry", "slam", "payload_inspection", "tracking"],
        multi_select=True,
        required=True,
        min_selections=1,
    ),
    Question(
        id="control_output",
        text="What does the perception drive?",
        options=["flight_controller", "path_planner", "mission_abort",
                 "payload_release", "ground_station_alert"],
        multi_select=True,
        required=True,
    ),
    # ... more questions per platform
]
```

Each answer narrows the design space and pre-fills `SystemSpec` fields via
the `implications` dict. Questions that become answerable from implications
are skipped.

### Tangibility Check

The same `GoalQualification` model, evaluated against the accumulated
`SystemSpec`:

```python
def assess_tangibility(spec: SystemSpec) -> GoalQualification:
    return GoalQualification(
        platform_identified=spec.platform_type is not None,
        perception_tasks_enumerated=(
            spec.perception is not None
            and len(spec.perception.detection_classes) > 0
            and spec.perception.max_latency_ms is not None
        ),
        control_output_identified=(
            spec.actuators is not None
            and spec.actuators.control_rate_hz is not None
        ),
        power_envelope_bounded=(
            spec.power is not None
            and spec.power.compute_power_watts is not None
        ),
        environment_characterized=(
            spec.sensors is not None
            and spec.sensors.environmental_rating is not None
        ),
    )
```

### Convergence Mechanism

The question templates are finite and ordered. Each platform has 5-8
questions. The loop terminates when either (a) `is_tangible` returns True,
or (b) all questions for the platform have been asked. Skipping is allowed —
the user can say "don't know" and the system fills in a domain-appropriate
default with a warning.

### Connection to Existing Flow

A new `convert_spec_to_design_inputs()` function maps `SystemSpec` to the
triple `(goal_string, DesignConstraints, use_case, platform)` that
`create_initial_soc_state()` expects. The existing `design_soc` tool checks
tangibility before calling `SoCDesignRunner.start()` and redirects to the
questionnaire if needed.

### Pros

- No LLM cost for qualification — entirely deterministic and testable
- Domain templates encode real engineering knowledge (a delivery drone
  needs X watts, a racing FPV needs Y watts)
- The `SystemSpec` output is reusable — it can be saved via
  `branes spec create` for future sessions
- Implications propagation means fewer questions: answering "multirotor
  delivery" auto-fills power, weight, speed, duration
- Easy to test: feed in answers, assert output spec
- Fast — no API calls, no graph construction

### Cons

- Template authoring burden: every new platform/domain needs a question set
- Less flexible for novel use cases that do not fit templates
- The "conversational" feel in chat mode depends on the LLM wrapping
  structured questions naturally
- No graceful degradation for truly novel platforms (user says
  "underwater ROV" and there is no template)

---

## Design 3: LLM-Driven Adversarial Qualification Agent

### Where It Sits

A new specialist agent `GoalQualifierAgent` that runs as a **tool** within
the existing `ArchitectAgent` chat loop (same pattern as
`decompose_mission`, `design_soc`, etc.). It does not modify the LangGraph
pipeline at all — it is a pre-processing tool that the LLM agent calls
when it detects an underspecified goal.

```
User: "drone perception SoC"
ArchitectAgent: [calls qualify_goal tool]
qualify_goal tool: returns structured assessment + required follow-ups
ArchitectAgent: asks user the follow-up questions conversationally
ArchitectAgent: [calls qualify_goal tool again with updated info]
... repeats until qualified ...
ArchitectAgent: [calls design_soc with fully qualified goal]
```

### How It Interacts With the User

The `qualify_goal` tool accepts the current goal text plus any accumulated
context, and returns a JSON assessment:

```json
{
  "tangibility_score": 0.3,
  "dimensions": {
    "platform": {
      "status": "partial",
      "value": "drone",
      "missing": "drone subtype (delivery/racing/survey/agricultural)"
    },
    "perception_tasks": {
      "status": "missing",
      "value": null,
      "missing": "specific tasks with quality requirements"
    },
    "control_output": {
      "status": "missing",
      "value": null,
      "missing": "what system consumes perception output"
    },
    "power_envelope": {
      "status": "missing",
      "value": null,
      "missing": "power budget or battery + mission duration"
    },
    "environment": {
      "status": "missing",
      "value": null,
      "missing": "operating environment and conditions"
    }
  },
  "qualified": false,
  "next_questions": [
    "What type of drone is this for? Delivery drones and racing FPV drones have completely different power budgets, latency requirements, and sensor needs.",
    "What specific perception tasks are needed? 'Perception' is too broad -- obstacle avoidance at 15m/s requires <20ms latency, while survey mapping can tolerate 200ms."
  ],
  "pushback": "I cannot design a meaningful SoC from 'drone perception' alone. A delivery drone obstacle avoidance SoC and a racing FPV SLAM SoC share almost no design parameters. I need to know what the drone does, what it needs to see, and what it does with what it sees."
}
```

The critical design element: the `pushback` field is **not sycophantic**.
The system prompt for this tool explicitly instructs it to be direct about
why the goal is insufficient and what will go wrong if it proceeds with
assumptions.

### Tangibility Check

The LLM evaluates tangibility using a scoring rubric embedded in its system
prompt. The rubric mirrors the 5 dimensions but allows the LLM to exercise
judgment about borderline cases. The tool also performs a deterministic
cross-check: it attempts to populate a `SystemSpec` from the stated
information and reports which subsystems remain empty.

The hybrid approach: LLM does the natural-language assessment and question
generation, but the `qualified` boolean is determined by a **deterministic
function** (same `assess_tangibility()` from Design 2) that the LLM cannot
override.

### Convergence Mechanism

The tool tracks a `round_number` in its state. After round 3, it shifts
from asking questions to proposing assumptions: "I will assume this is a
delivery multirotor with 5W compute budget and 30fps obstacle avoidance.
If these assumptions are wrong, correct them now." After round 5, it
proceeds with whatever it has, logging warnings about underspecified
dimensions.

The `ArchitectAgent` system prompt is updated to include:

```
IMPORTANT: Before calling design_soc, you MUST call qualify_goal first.
If qualify_goal returns qualified=false, you MUST NOT call design_soc.
Instead, ask the user the questions from next_questions and call qualify_goal
again with the additional context. Never proceed with an unqualified goal.
```

### Connection to Existing Flow

- New tool definitions in `src/embodied_ai_architect/llm/qualification_tools.py`
  following the `get_*_tool_definitions()` + `create_*_tool_executors()`
  pattern
- The tool is registered in `tools.py` alongside the existing tool families
- The `ArchitectAgent` system prompt in `agent.py` is updated to mandate
  qualification
- For CLI mode, a wrapper in `cli/commands/design.py` calls the qualifier
  directly before invoking the runner
- The qualified goal is converted to `DesignConstraints` + `SystemSpec` for
  downstream consumption

### Pros

- Most natural user experience in chat mode — the LLM asks questions
  conversationally, not as a rigid form
- Can handle novel platforms and edge cases that templates cannot cover
- The adversarial tone (pushback) is natural for an LLM — harder to
  achieve with templates
- Minimal changes to existing architecture — just a new tool, no new
  LangGraph nodes
- The hybrid deterministic/LLM check prevents the LLM from hallucinating
  that a goal is qualified when it is not

### Cons

- Requires LLM API calls for qualification (cost, latency)
- The LLM might generate inconsistent or unhelpful questions across sessions
- The `ArchitectAgent` system prompt instruction to "always qualify first"
  is a soft constraint — the LLM could skip it
- Harder to test deterministically — the LLM's questions vary per invocation
- Does not work in offline/no-API-key mode (would need fallback to Design 2
  templates)

---

## Comparison Matrix

| Criterion | Design 1: LangGraph Gate | Design 2: Templates | Design 3: LLM Agent Tool |
|---|---|---|---|
| LLM cost | Medium (assessment) | **Zero** | High (multi-round) |
| Offline capable | No | **Yes** | No (needs fallback) |
| Novel platform handling | Medium | Poor | **Good** |
| Testability | Medium | **Excellent** | Poor |
| User experience (CLI) | Good (Rich prompts) | Good (Rich prompts) | N/A (CLI needs wrapper) |
| User experience (chat) | Awkward (interrupt/resume) | Decent (structured) | **Excellent** (natural) |
| Implementation complexity | High (state machine in graph) | **Low-medium** | Low-medium |
| Pushback quality | Template-based | Template-based | **Natural, context-aware** |
| Integration with specs | Moderate | **Native** (outputs SystemSpec) | Moderate |
| Convergence guarantee | Yes (5 rounds hard limit) | **Yes** (finite questions) | Soft (LLM-dependent) |

---

## Recommendation

**Design 2 as the foundation with Design 3 as an enhancement layer.**

1. Build the domain question templates and deterministic tangibility
   assessment (Design 2) as the core module in
   `src/embodied_ai_architect/qualification/`
2. Wrap it as an LLM tool (Design 3 pattern) so the chat agent can present
   questions conversationally
3. Use the LLM for the `pushback` text generation only — the questions
   themselves come from templates, keeping them consistent and testable
4. Add a `branes design qualify` CLI subcommand that runs the template
   questionnaire directly with Rich prompts
5. Wire the deterministic `is_tangible` check into `design_soc` as a
   hard gate

This gives deterministic, testable, offline-capable qualification with
natural conversational presentation when an LLM is available.

### Key Implementation Files

- `specs/models.py` — `SystemSpec` defines what "fully specified" means;
  tangibility check maps to which subsystems have non-None values
- `llm/soc_design_tools.py` — `design_soc` must gate on tangibility before
  calling `SoCDesignRunner.start()`
- `research/decomposer.py` — Existing keyword extraction and rule-based
  logic that the qualification templates would extend
- `requirements/wizard.py` — `RequirementsWizard` is the closest existing
  analog to the structured Q&A flow for CLI mode
