# Plan: Enhance `branes codebase scan` with Entry Point Classification

## Context

The `branes codebase scan` command currently enumerates source files with 4 columns
(File, Language, Lines, Role) and stops there. To progress from scan to the LLM-powered
`branes codebase analyze`, the user needs to know *which* entry point is the application
(vs examples, tests, or utility scripts). Currently this requires manual inspection.

The workflow document (`docs/workflow-bottleneck-analysis.md`) also inaccurately claims
scan identifies pipeline stages — it must be corrected to reflect what scan actually does.

**Goal:** Enhance scan to sub-classify entry points as application/example/test,
recommend the main application entry point, and suggest a copy-pasteable next command.
Then fix the workflow document to match.

---

## Files to Modify

| File | Change |
|------|--------|
| `src/embodied_ai_architect/codebase/models.py` | Add `entry_type` field to `SourceFile` |
| `src/embodied_ai_architect/codebase/scanner.py` | Add entry point sub-classification logic |
| `src/embodied_ai_architect/cli/commands/codebase.py` | Add 5th column, recommendation, next-step suggestion |
| `docs/workflow-bottleneck-analysis.md` | Rewrite Step 1 to accurately reflect enhanced scan |

---

## Step 1: Add `entry_type` field to `SourceFile`

**File:** `src/embodied_ai_architect/codebase/models.py` (line 13-19)

Add an optional `entry_type` field to `SourceFile`:

```python
class SourceFile(BaseModel):
    path: str
    language: str
    lines: int = 0
    role: str = "library"
    entry_type: str | None = None  # "application", "example", "test", or None
```

Values:
- `"application"` — main program entry point (the target for `analyze`)
- `"example"` — demonstration/tutorial script
- `"test"` — test runner or test harness with a main guard
- `None` — not an entry point (libraries, configs, build files)

No changes to `ScanResult` or other models needed.

---

## Step 2: Add Entry Point Sub-Classification in Scanner

**File:** `src/embodied_ai_architect/codebase/scanner.py`

### New method: `_classify_entry_type()`

Called after `_classify_role()` returns `"entry_point"`. Uses path-based heuristics
(strongest signal), then filename patterns, then content hints.

**Path heuristics** (checked in order):
| Path pattern | entry_type | Rationale |
|---|---|---|
| `examples/`, `example/`, `demos/`, `demo/`, `samples/` | `"example"` | Standard example directories |
| `tests/`, `test/`, `benchmarks/`, `bench/` | `"test"` | Test directories (rare — most caught by role="test" earlier) |
| `scripts/`, `tools/`, `utils/`, `utilities/` | `"example"` | Utility scripts, not the main app |

**Filename heuristics** (if path didn't match):
| Pattern | entry_type |
|---|---|
| `example_*`, `*_example.*`, `demo_*`, `*_demo.*`, `sample_*` | `"example"` |
| `test_*`, `*_test.*` | `"test"` |
| `run_*`, `main.*`, `app.*`, `__main__.*` | `"application"` |

**Default:** `"application"` — an entry point that doesn't match example/test patterns
is assumed to be an application candidate.

### Integration into `scan()`

In the scan loop (line 152-160), after `_classify_role()` returns, if role is
`"entry_point"`, call `_classify_entry_type()` and set the field:

```python
role = self._classify_role(item, rel_path, lang)
entry_type = None
if role == "entry_point":
    entry_type = self._classify_entry_type(item, rel_path, lang)

source_files.append(
    SourceFile(path=rel_path, language=lang, lines=line_count,
               role=role, entry_type=entry_type)
)
```

### New method: `_recommend_application()`

After scan completes, pick the best application entry point:

1. Filter entry points where `entry_type == "application"`
2. If exactly one → recommend it (high confidence)
3. If multiple → rank by line count descending (larger = more likely main app),
   recommend the largest with a note about alternatives
4. If none → note that no clear application entry point was found;
   suggest the largest entry point as a starting point

Returns a tuple: `(recommended_path: str | None, confidence: str, alternatives: list[str])`

Add a `recommended_entry_point` field to `ScanResult` (optional dict):
```python
class ScanResult(BaseModel):
    ...
    recommended_entry_point: dict | None = None
    # {"path": "...", "confidence": "high"|"medium"|"low", "alternatives": [...]}
```

Call `_recommend_application()` at the end of `scan()`, just before returning.

---

## Step 3: Enhance CLI Display

**File:** `src/embodied_ai_architect/cli/commands/codebase.py`

### Add 5th column to table

In `_display_scan_result()` (line 222-234), add an "Entry Type" column:

```python
table.add_column("Entry Type", style="yellow")
```

For each row, show `entry_type` if present, otherwise empty string:

```python
table.add_row(sf.path, sf.language, str(sf.lines), sf.role,
              sf.entry_type or "")
```

### Add recommendation section

After the file table, if `result.recommended_entry_point` is set, print:

```
Recommended application entry point:
  → examples/full_pipeline.py  (confidence: high)

Next step — run LLM-powered analysis on this entry point:

  branes codebase analyze prototypes/drone_perception
```

If confidence is "medium" (multiple candidates), also list alternatives:

```
Recommended application entry point:
  → examples/full_pipeline.py  (confidence: medium, 3 candidates)
  Alternatives: examples/simple_detection.py, examples/reasoning_pipeline.py

Next step — run LLM-powered analysis on this entry point:

  branes codebase analyze prototypes/drone_perception
```

The `analyze` command takes the project directory (not the entry point file),
so the suggested command uses the original project path argument.

### JSON output

The `entry_type` and `recommended_entry_point` are already part of the Pydantic
model, so `result.model_dump()` will include them in JSON output automatically.

---

## Step 4: Fix Workflow Document

**File:** `docs/workflow-bottleneck-analysis.md`

Rewrite Step 1 to accurately describe what scan does:

- **What it does:** Static file enumeration — lists source files with language,
  line count, role (entry_point/library/test/config), and entry type
  (application/example/test) for entry points.
- **What it produces:** A table of files + a recommendation of which entry point
  is the application, with a suggested next command.
- **What it does NOT do:** It does not parse code semantics, identify pipeline
  stages, extract compute costs, or run any LLM analysis.

Add a clear transition showing that Step 2 (`branes codebase analyze`) is where
pipeline stages, compute kernels, and dataflow get extracted via LLM analysis.

---

## Implementation Order

1. Edit `models.py` — add `entry_type` to `SourceFile`, `recommended_entry_point` to `ScanResult`
2. Edit `scanner.py` — add `_classify_entry_type()`, `_recommend_application()`, integrate into `scan()`
3. Edit `codebase.py` — add 5th column, recommendation display, next-step suggestion
4. Test with drone perception: `.venv/bin/branes codebase scan prototypes/drone_perception`
5. Fix `docs/workflow-bottleneck-analysis.md` Step 1
6. Run lint: `.venv/bin/black --check src/ --line-length 100 && .venv/bin/ruff check src/`

---

## Verification

```bash
# 1. Scan drone perception and verify output has 5 columns + recommendation
.venv/bin/branes codebase scan prototypes/drone_perception

# Expected: entry points classified as "application" or "example"
# Expected: recommendation section with next-step command

# 2. JSON output includes new fields
.venv/bin/branes --json codebase scan prototypes/drone_perception | python -m json.tool | grep entry_type

# 3. Lint passes
.venv/bin/black --check src/ tests/ --line-length 100
.venv/bin/ruff check src/ tests/

# 4. Existing tests still pass
.venv/bin/pytest tests/ -q
```
