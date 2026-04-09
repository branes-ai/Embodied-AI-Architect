# Plan: Registry-Driven Qualification — Replace Hard-Coded Domain Templates

## Context

The `design qualify` command currently only offers 3 platforms (drone, ugv, robot_arm)
because the `GoalQualifier` is tightly coupled to `DomainTemplate` objects that have
hand-crafted question trees. We just built a platform registry with 30 platform
definitions (targeting 285), but when the registry matches a platform like
"surveillance.edge_vision_box", the qualifier has no template for "surveillance" and
falls back to offering only the 3 hard-coded options.

The fundamental problem: **the qualifier treats "no domain template" as "unknown platform"**
when it should treat "registry match" as a rich data source for qualification.

## Approach

**Generate qualification questions dynamically from platform YAML data** instead of
requiring a hand-crafted `DomainTemplate` per category.

Each platform YAML already contains:
- `classification` — locomotion, manipulation, environment, human_proximity (→ platform dimension)
- `attributes.power_watts.typical` — (→ power dimension)
- `attributes.latency_ms.typical` — (→ perception timing)
- `implications.perception` — camera_types, detection_classes, fps (→ perception dimension)
- `implications.actuators` — control_rate_hz (→ control dimension)
- `classification.environment` — (→ environment dimension)
- `qualification.critical_constraints` — what MUST be specified
- `qualification.suggested_questions` — question IDs to ask
- `context` — domain knowledge to show the user

This is enough to satisfy all 5 tangibility dimensions and generate useful questions.

## Implementation — 3 Phases

### Phase 1: Platform Picker + Auto-Qualify from Registry (this PR)

**Goal**: When registry matches, show matched platforms (not drone/ugv/robot_arm),
let user confirm, and auto-fill spec from platform data.

#### Step 1: New `_build_platform_selection_result()` in `qualifier.py`

When the registry matches platforms, instead of showing the 3-domain picker,
show the **top registry matches** as choices:

```
Matched platforms for "edge device running edge detection":

  1. surveillance.edge_vision_box (0.92) — Edge Vision Box
     General-purpose edge AI appliance for running computer vision inference...
  2. surveillance.smart_camera_security (0.45) — Security Smart Camera
  3. edge_vision.industrial_quality (0.38) — Industrial Quality Inspection

  Select (number), or 's' to pick from categories:
```

When the user selects, the platform's `implications` are merged into the spec,
and the platform's `attributes.typical` values pre-fill power/latency/cost.

**Files to modify:**
- `src/embodied_ai_architect/qualification/qualifier.py`
  - Add `_platform_matches` field to `__init__`
  - New `_build_platform_selection_result()` method
  - Update `assess()`: when registry matches and no template exists, call platform selection
  - Update `answer()`: handle `_platform_selection` question_id
  - Update `_assess_tangibility()`: accept platform_type from registry implications

#### Step 2: Generate questions from platform `qualification` data

After user confirms a platform, generate questions from:
1. `qualification.critical_constraints` → one NUMERIC or YES_NO question per constraint
2. `qualification.suggested_questions` → mapped to generic question templates
3. Any missing tangibility dimensions → fill with universal fallback questions

Create a **universal question bank** — generic questions that apply to any platform:
- `power_budget` — "What is the compute power budget?" (NUMERIC, watts)
- `latency_requirement` — "What is the maximum perception latency?" (NUMERIC, ms)
- `deployment_environment` — "Where will this be deployed?" (SINGLE_CHOICE from classification.environment)
- `perception_tasks` — "What perception tasks?" (MULTI_CHOICE, common tasks)
- `control_output` — "What does perception output drive?" (SINGLE_CHOICE)
- `safety_level` — "What safety integrity level?" (SINGLE_CHOICE)

**Files to create/modify:**
- `src/embodied_ai_architect/qualification/generic_questions.py` — universal question bank
- `src/embodied_ai_architect/qualification/qualifier.py`
  - New `_generate_questions_from_platform()` method
  - Build Question objects from platform + generic bank
  - Store generated questions in `self._generated_questions`
  - Update `_find_next_question()` to use generated questions when no template

#### Step 3: Wire into CLI

- `src/embodied_ai_architect/cli/commands/design.py`
  - Update help text for `--domain` to mention registry
  - Handle `_platform_selection` question type in the interactive loop
  - Show platform context after selection

### Phase 2: Coexist with existing templates (future)
Keep drone/ugv/robot_arm templates working as-is for users who type "drone".
Registry-driven path activates only when no hard-coded template matches.

### Phase 3: Migrate existing templates to registry (future)
Convert drone.py, ugv.py, robot_arm.py into platform YAML files with
embedded question definitions.

## Phase 1 Detailed File Changes

### `qualifier.py` — Core changes

1. **`__init__`**: Add `_platform_matches: list` and `_generated_questions: list[Question]`

2. **`assess()`**: Replace current fallback logic:
   ```python
   # Current: falls back to 3-domain picker
   # New: if registry matches, show platform picker
   if not template and self._platform_context:
       self._platform_matches = self._platform_context.get("alternatives", [])
       # Insert top match too
       return self._build_platform_selection_result()
   ```

3. **`answer("_platform_selection", platform_id)`**: 
   - Load full platform from registry
   - Merge platform.implications into spec
   - Apply platform.attributes as constraint defaults
   - Generate questions from platform data
   - Return `_build_result()` (now has generated questions)

4. **`_generate_questions_from_platform(platform)`**:
   - Start with universal questions (power, latency, environment, perception, control)
   - Pre-fill defaults from platform.attributes.typical values
   - Add platform-specific questions from qualification.suggested_questions
   - Store in `self._generated_questions`

5. **`_find_next_question()`**: Check `self._generated_questions` when
   `get_domain_template(self._domain)` returns None

6. **`_assess_tangibility()`**: No changes needed — it already checks spec fields,
   which will be populated by platform implications

### `generic_questions.py` — New file

Universal question bank with ~6 questions that cover the 5 tangibility dimensions:

```python
UNIVERSAL_QUESTIONS = [
    Question(id="power_budget", dimension="power", text="What is the compute power budget?",
             question_type=QuestionType.NUMERIC, numeric_unit="watts", ...),
    Question(id="latency_requirement", dimension="perception", ...),
    Question(id="deployment_environment", dimension="environment", ...),
    Question(id="perception_tasks", dimension="perception", ...),
    Question(id="control_output", dimension="control", ...),
]
```

Each question has implications that map answers to spec fields,
matching the existing tangibility assessment logic.

### `design.py` — CLI changes

- Handle `_platform_selection` question type: show rich platform info + alternatives
- After platform selection, show context (typical_architecture, design_considerations)
- Update `--domain` help text

## Verification

1. `branes design qualify "edge device running edge detection"` →
   shows "surveillance.edge_vision_box" as top match, user confirms,
   questions asked about power/latency/environment, reaches tangibility

2. `branes design qualify "autonomous sprayer for vineyards"` →
   shows agriculture platforms, user picks, gets vineyard-specific questions

3. `branes design qualify "drone perception"` →
   still uses existing drone template (backward compatible)

4. All existing tests pass (no regression)

5. New tests: registry-driven qualification produces tangible result
