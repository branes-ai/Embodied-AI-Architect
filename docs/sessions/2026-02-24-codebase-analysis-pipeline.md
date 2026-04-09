# Session: Full Application Codebase Analysis Pipeline

**Date:** 2026-02-24
**Commits:** `d6d6799`, `27e725c`, `3823a98`

## Summary

Implemented a complete pipeline for analyzing full C++, Rust, and Python applications and mapping them to hardware — extending the system beyond individual PyTorch models to whole-application workloads including signal processing, control loops, and sensor fusion.

The pipeline has three stages: static scan (no LLM), multi-pass LLM analysis (4 passes), and hardware assessment (reuses existing PPA pipeline). Exposed via CLI subcommands (`branes codebase scan|analyze|assess`) and three interactive chat tools (`scan_project`, `analyze_codebase`, `assess_codebase_on_hardware`).

## Architecture

```
Project Directory
       ↓
┌──────────────────┐
│  CodebaseScanner │  Static file walk — languages, build system,
│  (scanner.py)    │  ML models, dependencies, entry points
└───────┬──────────┘
        ↓ ScanResult
┌──────────────────┐
│  CodeAnalyzer    │  4-pass LLM analysis:
│  (analyzer.py)   │  build → entry points → compute kernels → synthesis
└───────┬──────────┘
        ↓ CodebaseAnalysisResult
┌──────────────────┐
│ CodebaseConverter│  Maps ComputeKernels to operator types
│ (converter.py)   │  → workload_profile dict
└───────┬──────────┘
        ↓ workload_profile
┌──────────────────┐
│ Existing PPA     │  workload_analyzer → hw_explorer → ppa_assessor
│ Pipeline         │
└──────────────────┘
```

## What Was Built

### New Package: `src/embodied_ai_architect/codebase/`

| Module | Purpose |
|--------|---------|
| `models.py` | Pydantic models: `SourceFile`, `ComputeKernel`, `DataflowLink`, `ScanResult`, `CodebaseAnalysisResult` |
| `scanner.py` | `CodebaseScanner.scan()` — walks directory, detects languages (15 extensions), build systems (CMake/Cargo/pip/Make), ML models (.onnx/.pt/.tflite/.pb/.safetensors), entry points, dependencies |
| `analyzer.py` | `CodeAnalyzer.analyze()` — 4-pass LLM strategy (build → entry points → compute kernels → synthesis), 48K char per-pass limit, JSON parsing with markdown fence stripping |
| `converter.py` | `CodebaseConverter.to_workload_profile()` — kernel type → operator mapping (ml_inference → conv/matmul/activation; signal_processing → fft/filtering; etc.), outputs dict compatible with existing specialists |
| `__init__.py` | Public API exports |

### Kernel Type → Operator Mapping

| Kernel Type | Operators |
|-------------|-----------|
| `ml_inference` | convolution, matrix_multiply, activation, batch_norm |
| `signal_processing` | fft, filtering, accumulate |
| `image_processing` | convolution, resize, color_convert |
| `control_loop` | matrix_multiply, accumulate |
| `sensor_fusion` | matrix_multiply, accumulate |
| `io_bound` | memory_copy, dma_transfer |
| `general_compute` | matrix_multiply, accumulate |

### Agent & Tools

| File | Purpose |
|------|---------|
| `agents/codebase_analyzer.py` | `CodebaseAnalyzerAgent(BaseAgent)` — wraps scanner → analyzer → converter pipeline, gracefully handles missing LLM (falls back to scan-only) |
| `llm/codebase_tools.py` | 3 chat tools: `scan_project`, `analyze_codebase`, `assess_codebase_on_hardware` — follows existing tool pattern with try/except import guard |
| `llm/tools.py` | Modified to register codebase tools via `HAS_CODEBASE_TOOLS` flag |

### CLI Commands

| Command | Description |
|---------|-------------|
| `branes codebase scan PATH` | Quick static scan (no API key needed) |
| `branes codebase analyze PATH` | Full LLM-powered 4-pass analysis |
| `branes codebase assess PATH --hardware X --power-budget N --latency-target N` | End-to-end hardware assessment |

### Documentation

| File | Action |
|------|--------|
| `docs/codebase-analysis-guide.md` | **NEW** — Full methodology guide with pipeline stages, kernel reference, CLI/chat/API examples |
| `docs-site/.../features/codebase-analysis.md` | **NEW** — Starlight feature page |
| `docs-site/.../reference/cli.md` | Updated with codebase commands |
| `docs/interactive-chat.md` | Added 3 new tools to table |
| `CLAUDE.md` | Added codebase package and CLI descriptions |
| `docs-site/astro.config.mjs` | Added sidebar entry + `server: { host: '0.0.0.0' }` for headless access |

### Test Fixtures

Three synthetic projects for scanner/agent tests:
- `tests/fixtures/sample_projects/cpp_drone/` — CMake + 3 C++ files (main, perception, control)
- `tests/fixtures/sample_projects/python_ml/` — pyproject.toml + inference.py + model.onnx
- `tests/fixtures/sample_projects/rust_embedded/` — Cargo.toml + 3 Rust files (main, sensor, control)

## Tests

34 tests across 5 test classes, all passing:

| Class | Tests | Coverage |
|-------|-------|----------|
| `TestCodebaseScanner` | 13 | Language detection, build systems, ML model discovery, entry points, dependency extraction, ignore patterns |
| `TestCodebaseConverter` | 7 | ML/signal/control/sensor/io kernel mapping, output format compatibility, empty analysis |
| `TestCodebaseAnalyzerAgent` | 4 | Scan-only mode, full pipeline with mock LLM, missing path error, agent metadata |
| `TestModels` | 5 | Model validation, defaults, serialization round-trips |
| `TestCodebaseTools` | 5 | Tool definitions schema, executor creation, scan tool execution |

## Key Design Decisions

1. **Separate `codebase/` package**: Self-contained module; clear boundary at workload_profile dict, then hands off to existing PPA pipeline with zero changes to specialists.
2. **4-pass LLM analysis**: Entire codebases exceed context limits. Each pass is focused (build → entry → kernels → synthesis) and capped at 48K chars. Priority-sorted file list ensures important files are read first.
3. **Converter as bridge**: Rich `CodebaseAnalysisResult` → flat `workload_profile` dict compatible with `workload_analyzer()`, `hw_explorer()`, and `ppa_assessor()`.
4. **No new dependencies**: Reuses existing `LLMClient` from `llm/client.py`.

## Files Changed

| File | Action | Lines |
|------|--------|-------|
| `src/.../codebase/__init__.py` | CREATE | 25 |
| `src/.../codebase/models.py` | CREATE | 84 |
| `src/.../codebase/scanner.py` | CREATE | 310 |
| `src/.../codebase/analyzer.py` | CREATE | 406 |
| `src/.../codebase/converter.py` | CREATE | 180 |
| `src/.../agents/codebase_analyzer.py` | CREATE | 131 |
| `src/.../llm/codebase_tools.py` | CREATE | 305 |
| `src/.../llm/tools.py` | MODIFY | +20 |
| `src/.../cli/commands/codebase.py` | CREATE | 326 |
| `src/.../cli/__init__.py` | MODIFY | +2 |
| `tests/test_codebase.py` | CREATE | 457 |
| `tests/fixtures/sample_projects/` | CREATE | 248 (across 8 files) |
| `docs/codebase-analysis-guide.md` | CREATE | 272 |
| `docs-site/.../features/codebase-analysis.md` | CREATE | 117 |
| `docs-site/.../reference/cli.md` | MODIFY | +38 |
| `docs/interactive-chat.md` | MODIFY | +3 |
| `CLAUDE.md` | MODIFY | +13 |
| `docs-site/astro.config.mjs` | MODIFY | +2 |
| `docs/plans/full_application_analysis_pipeline.md` | CREATE | 261 |
| **Total** | **29 files** | **+3,200 lines** |
