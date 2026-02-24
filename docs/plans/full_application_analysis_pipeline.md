# Plan: Full Application Analysis Pipeline

## Context

Users want to bring complete C++, Rust, or Python applications to the Embodied AI Architect
and get hardware assessment across different targets. Today the tool only analyzes individual
PyTorch models. We need to support whole-application analysis: parsing source code, identifying
computational kernels (ML and non-ML), mapping them to hardware operators, and running them
through the existing PPA/hardware assessment pipeline.

**User requirements (confirmed):**
- Support **both** ML-heavy apps (PyTorch/TensorFlow models embedded in C++/Rust/Python) AND
  general embedded apps (signal processing, control loops, sensor fusion)
- Use **LLM code reading + dynamic profiling** — LLM analyzes source structure multi-pass,
  optional profiling captures runtime hotspots
- Expose via **both** CLI subcommands and interactive chat tools

## Deliverables

### 1. NEW: `src/embodied_ai_architect/codebase/models.py`

Data models for codebase analysis results.

```python
class SourceFile(BaseModel):
    path: str
    language: str              # "cpp", "rust", "python"
    lines: int
    role: str                  # "entry_point", "library", "config", "build", "test"

class ComputeKernel(BaseModel):
    name: str
    source_file: str
    line_range: tuple[int, int]
    kernel_type: str           # "ml_inference", "signal_processing", "control_loop",
                               # "image_processing", "sensor_fusion", "io_bound", "general_compute"
    estimated_ops_per_invocation: float
    data_types: list[str]      # ["float32", "int8", "float16"]
    parallelism: str           # "data_parallel", "pipeline", "sequential"
    memory_access_pattern: str # "streaming", "random", "reuse_heavy"
    invocation_frequency_hz: float
    frameworks: list[str]      # ["pytorch", "opencv", "eigen", "custom"]

class DataflowLink(BaseModel):
    source_kernel: str
    sink_kernel: str
    data_size_bytes: int
    transfer_type: str         # "memory", "dma", "network"

class CodebaseAnalysisResult(BaseModel):
    project_name: str
    languages: list[str]
    source_files: list[SourceFile]
    kernels: list[ComputeKernel]
    dataflow: list[DataflowLink]
    ml_models: list[dict]      # detected model files (.onnx, .pt, .tflite)
    build_system: str          # "cmake", "cargo", "pip/poetry", "make"
    dependencies: list[str]
    summary: str               # LLM-generated natural language summary
```

**Reuse:** Aligns with existing `SoftwareArchitecture` schema in embodied-schemas
(`OperatorInstance`, `DataflowEdge`).

### 2. NEW: `src/embodied_ai_architect/codebase/scanner.py`

Static file scanner — no LLM needed, fast pass.

```python
class CodebaseScanner:
    def scan(self, project_path: Path) -> ScanResult
```

- Walks the directory tree, identifies language by extension
- Detects build system (CMakeLists.txt, Cargo.toml, pyproject.toml, Makefile)
- Finds ML model files (.onnx, .pt, .tflite, .pb, .safetensors)
- Extracts dependency lists from build files
- Classifies files by role (entry_point heuristics: main(), fn main(), `if __name__`)
- Returns `ScanResult` with file inventory, prioritized for LLM analysis

### 3. NEW: `src/embodied_ai_architect/codebase/analyzer.py`

LLM-powered multi-pass code analyzer.

```python
class CodeAnalyzer:
    def __init__(self, llm_client: LLMClient): ...
    def analyze(self, scan_result: ScanResult, project_path: Path) -> CodebaseAnalysisResult
```

**Multi-pass strategy** (stays within context limits):
1. **Pass 1 — Build & Config**: Read build files, dependency manifests → understand project structure
2. **Pass 2 — Entry Points**: Read main files, top-level orchestration → identify pipeline structure
3. **Pass 3 — Compute Kernels**: Read implementation files flagged as computationally heavy →
   extract `ComputeKernel` instances with ops estimates
4. **Pass 4 — Synthesis**: Combine passes into `CodebaseAnalysisResult` with dataflow graph

Each pass sends a focused prompt with relevant file contents to the LLM and asks for
structured JSON output. Uses existing `LLMClient` from `src/embodied_ai_architect/llm/client.py`.

### 4. NEW: `src/embodied_ai_architect/codebase/converter.py`

Converts `CodebaseAnalysisResult` into the existing pipeline's `WorkloadProfile` format.

```python
class CodebaseConverter:
    def to_workload_profile(self, analysis: CodebaseAnalysisResult) -> dict
```

- Maps each `ComputeKernel` to operator types understood by `workload_analyzer` and `hw_explorer`
- Kernel type → operator mapping:
  - `ml_inference` → convolution, matrix_multiply, activation (from detected framework)
  - `signal_processing` → fft, filtering, accumulate
  - `image_processing` → convolution, resize, color_convert
  - `control_loop` → matrix_multiply (small), accumulate
  - `sensor_fusion` → matrix_multiply, accumulate
- Aggregates total ops, memory requirements, latency constraints
- Output dict is compatible with existing `architecture_composer` and `ppa_assessor` inputs

**Reuse:** Maps into existing `workload_profile` format already consumed by
`specialists.py:workload_analyzer()` and `hw_explorer()`.

### 5. NEW: `src/embodied_ai_architect/codebase/__init__.py`

Package init exporting public API.

### 6. NEW: `src/embodied_ai_architect/agents/codebase_analyzer.py`

Agent that wraps the scanner + analyzer + converter pipeline.

```python
class CodebaseAnalyzerAgent(BaseAgent):
    agent_name = "CodebaseAnalyzer"
    def execute(self, input_data: dict) -> AgentResult
```

- Accepts `{"project_path": "/path/to/app", "target_hardware": ["jetson_orin", "custom_soc"]}`
- Runs scanner → analyzer → converter → returns workload profile + analysis summary
- Follows existing `BaseAgent` pattern from `agents/base.py`

### 7. NEW: `src/embodied_ai_architect/llm/codebase_tools.py`

Three LLM tools for the interactive chat agent:

```python
def get_codebase_tool_definitions() -> list[dict]
def create_codebase_tool_executors() -> dict[str, Callable]
```

**Tools:**
1. **`scan_project`** — Quick scan, returns file inventory + detected build system + ML models
2. **`analyze_codebase`** — Full multi-pass LLM analysis, returns `CodebaseAnalysisResult`
3. **`assess_codebase_on_hardware`** — End-to-end: scan → analyze → convert → run through
   existing hw_explorer + ppa_assessor pipeline → return hardware recommendations

**Reuse:** Follows exact same pattern as `llm/tools.py`, `llm/architecture_tools.py`,
and `llm/graphs_tools.py` — tool definitions as dicts, executor functions returning JSON strings.

### 8. MODIFY: `src/embodied_ai_architect/llm/tools.py`

- Add `from embodied_ai_architect.llm.codebase_tools import ...` with try/except guard
- Register codebase tools in `get_tool_definitions()` and `create_tool_executors()`

### 9. NEW: `src/embodied_ai_architect/cli/commands/codebase.py`

Click command group with subcommands:

```
embodied-ai codebase scan /path/to/project      # Quick file scan
embodied-ai codebase analyze /path/to/project    # Full LLM analysis
embodied-ai codebase assess /path/to/project     # End-to-end hardware assessment
    --hardware jetson_orin,custom_kpu
    --power-budget 15
    --latency-target 33
```

### 10. MODIFY: `src/embodied_ai_architect/cli/__init__.py`

- Register `codebase` command group in the main CLI.

### 11. NEW: `tests/test_codebase.py`

- `test_scanner_detects_cmake_project` — scan a fixture directory with CMakeLists.txt
- `test_scanner_detects_cargo_project` — Rust project detection
- `test_scanner_detects_python_project` — pyproject.toml detection
- `test_scanner_finds_ml_models` — finds .onnx/.pt files
- `test_converter_ml_kernel_to_workload` — ML inference kernel maps to conv/matmul operators
- `test_converter_signal_processing_kernel` — signal processing maps to fft/filtering
- `test_converter_output_compatible_with_specialists` — converter output works with
  existing `workload_analyzer()` and `hw_explorer()`
- `test_codebase_analyzer_agent` — end-to-end agent test with mock LLM

### 12. Test fixtures: `tests/fixtures/sample_projects/`

Small synthetic projects for testing:
- `cpp_drone/` — CMake project with main.cpp, perception.cpp, control.cpp
- `python_ml/` — pyproject.toml with inference.py, a dummy .onnx file
- `rust_embedded/` — Cargo.toml with sensor driver + control loop

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `src/.../codebase/__init__.py` | CREATE | Package init |
| `src/.../codebase/models.py` | CREATE | Data models (CodebaseAnalysisResult, ComputeKernel, etc.) |
| `src/.../codebase/scanner.py` | CREATE | Static file scanner |
| `src/.../codebase/analyzer.py` | CREATE | LLM multi-pass code analyzer |
| `src/.../codebase/converter.py` | CREATE | Analysis → workload_profile converter |
| `src/.../agents/codebase_analyzer.py` | CREATE | BaseAgent wrapper |
| `src/.../llm/codebase_tools.py` | CREATE | 3 chat tools (scan, analyze, assess) |
| `src/.../llm/tools.py` | MODIFY | Register codebase tools |
| `src/.../cli/commands/codebase.py` | CREATE | CLI subcommands |
| `src/.../cli/__init__.py` | MODIFY | Register codebase command group |
| `tests/test_codebase.py` | CREATE | Unit + integration tests |
| `tests/fixtures/sample_projects/` | CREATE | Synthetic test projects |

## Key Design Decisions

1. **Separate `codebase/` package** — keeps application analysis self-contained, doesn't
   bloat existing modules. Clear boundary: codebase/ handles source code → workload_profile,
   then hands off to existing PPA pipeline.

2. **Multi-pass LLM analysis** — reading entire codebases in one prompt exceeds context limits.
   4-pass strategy (build → entry points → kernels → synthesis) keeps each pass focused and
   within ~50K tokens. Priority-sorted file list ensures most important files are read first.

3. **Converter as bridge** — `CodebaseAnalysisResult` is a rich representation of the app;
   `converter.to_workload_profile()` maps it to the format already consumed by
   `workload_analyzer()`, `hw_explorer()`, and `ppa_assessor()`. No changes needed to the
   existing SoC design pipeline.

4. **No new LLM dependency** — reuses existing `LLMClient` from `llm/client.py`. The analyzer
   sends structured prompts and parses JSON responses, same pattern as `ArchitectAgent`.

5. **Dynamic profiling deferred to Phase 2** — MVP uses static LLM analysis only. Phase 2
   adds optional `profiler.py` that instruments and runs the application to capture runtime
   hotspots, feeding actual measurements back into the workload profile.

## Verification

```bash
# Run new tests
pytest tests/test_codebase.py -v

# Test CLI commands
embodied-ai codebase scan tests/fixtures/sample_projects/cpp_drone/
embodied-ai codebase analyze tests/fixtures/sample_projects/python_ml/

# Test in chat (requires ANTHROPIC_API_KEY)
embodied-ai chat
> scan the project at /path/to/my/app
> analyze this codebase for hardware mapping
> assess this app on jetson orin vs custom kpu with 10W power budget

# Existing tests must still pass
pytest tests/ -v

# Lint
ruff check src/embodied_ai_architect/codebase/
black src/embodied_ai_architect/codebase/ --check --line-length 100
```
