# Proprietary Package Distribution Strategy

**Date**: 2026-04-12
**Status**: Assessment — pending decision
**Context**: Bugs #162 (embodied-schemas not on PyPI) and #163 (graphs Cython fails)
**Author**: Architecture assessment for Branes.AI executive review

## Problem Statement

Two sibling repositories — `graphs` (roofline models, hardware simulation, calibration)
and `embodied-schemas` (shared Pydantic models, hardware catalog) — are required by
`embodied-ai-architect` for full functionality but are not available on PyPI. The `graphs`
repo contains proprietary IP (performance models, calibration data) that cannot be
published as source code.

### Current State

| Package | Location | Used by | IP Sensitivity |
|---------|----------|---------|----------------|
| `graphs` | `../graphs` (private repo) | `branes mcp analyze/latency/energy/memory/compare`, `llm/graphs_tools.py`, benchmark runner | **HIGH** — proprietary roofline models, hardware calibration data |
| `embodied-schemas` | `../embodied-schemas` (private repo) | 13 files — benchmark, model zoo, specialists, LLM tools, specs | **LOW** — shared Pydantic models, could be open-sourced |

### Impact on Customers

A `pip install embodied-ai-architect` user today:
- **Works**: mission workflow, qualification, sensor/actuator/platform registries, swap analysis, design plan
- **Broken**: `branes mcp analyze` (Cython error), benchmark, model zoo discovery, constraint tiers

## What PyPI Publishing Exposes

| Format | Contents | Protection Level |
|--------|----------|-----------------|
| Source dist (.tar.gz) | ALL Python source, .pyx files, data | None — fully readable |
| Wheel (.whl) | Compiled .so/.pyd for Cython, but pure .py as-is | Low — pure Python readable |

**PyPI is public** — no access control. Anyone can `pip install` and extract source.

## Distribution Options

### Option 1: Private PyPI Server

Run a private package index requiring authentication.

**Providers**:
- AWS CodeArtifact
- Google Artifact Registry
- JFrog Artifactory (self-hosted or cloud)
- devpi (lightweight self-hosted)
- Gemfury ($9/month hosted)

**Customer experience**:
```bash
pip install --extra-index-url https://pypi.branes.ai/simple/ \
    --token $BRANES_TOKEN graphs
pip install embodied-ai-architect[analysis]
```

| Pros | Cons |
|------|------|
| Standard pip workflow | Infrastructure to maintain |
| Versioned releases | Token management |
| Per-customer access control | Build matrix (platform × Python version) |

### Option 2: License-Gated GitHub Releases

Keep `graphs` as a private GitHub repo. Grant customers read access or distribute wheel files.

```toml
# pyproject.toml optional dep
analysis = ["graphs @ git+https://${BRANES_TOKEN}@github.com/branes-ai/graphs.git"]
```

| Pros | Cons |
|------|------|
| No infrastructure beyond GitHub | Exposes source via git clone |
| Simple access control | No compiled-only option |

### Option 3: Cloud API Service (Best IP Protection)

Don't ship the models. Run `graphs` as a service behind a REST API:

```
Customer laptop                          Branes Cloud
┌─────────────────┐                    ┌──────────────────┐
│ branes CLI      │  ── REST/gRPC ──→  │ graphs service   │
│ (open source)   │                    │ (proprietary)    │
│ pip install     │  ←── results ────  │ roofline, calib  │
└─────────────────┘                    └──────────────────┘
```

**Architecture**:
```python
class AnalysisClient:
    """Routes to local (if graphs installed) or cloud API."""

    def analyze(self, model, hardware, **kwargs):
        try:
            from graphs.estimation import roofline_analyze
            return roofline_analyze(model, hardware, **kwargs)
        except ImportError:
            return self._cloud_analyze(model, hardware, **kwargs)

    def _cloud_analyze(self, model, hardware, **kwargs):
        api_key = os.environ.get("BRANES_API_KEY", "")
        resp = httpx.post("https://api.branes.ai/v1/analyze",
                         json={"model": model, "hardware": hardware, **kwargs},
                         headers={"Authorization": f"Bearer {api_key}"})
        return resp.json()
```

| Pros | Cons |
|------|------|
| Zero IP exposure | Requires internet |
| Metered billing | Latency |
| Instant updates | Hosting costs |
| No Cython/NumPy version issues | More complex initial setup |

### Option 4: Compiled-Only Distribution

Build platform-specific wheels with Cython, strip source:

```bash
python setup.py bdist_wheel  # with cythonize
# .whl contains .so files, not .py source
```

Distribute via private PyPI (Option 1).

| Pros | Cons |
|------|------|
| Source not trivially readable | Must build per-platform |
| Standard pip workflow | Determined attacker can decompile |

## Recommendation: Phased Approach

### Phase 1 — Now (unblock customers)

**embodied-schemas**: Publish to PyPI as open source. Low IP sensitivity, shared models only.

**graphs**: Distribute via private GitHub access. Grant paying customers read access
to the private repo. Install via:
```bash
pip install "graphs @ git+https://github.com/branes-ai/graphs.git"
```

Fix the Cython issue (`np.int_t` → `np.intp_t`) in the graphs repo first.

### Phase 2 — 3-6 months (scale)

Set up AWS CodeArtifact or Gemfury as private PyPI. Publish compiled wheels for
`graphs`. Standard `pip install` with token auth.

### Phase 3 — Product-market fit (monetize)

Move graphs analysis behind a cloud API. Three-tier model:

| Tier | Access | Price |
|------|--------|-------|
| **Free** | Open-source CLI: missions, registries, qualification, swap | $0 |
| **API** | Cloud-hosted analysis: roofline, benchmark, optimization | Per-query or subscription |
| **Enterprise** | Local `graphs` package for air-gapped deployments | Annual license |

This is the same model used by:
- Anthropic (Claude client free, inference paid)
- GitHub (client free, Copilot paid)
- Weights & Biases (client free, cloud paid)

## Technical Prerequisites

### For graphs repo
1. Fix `_betweenness_helper.pyx`: change `np.int_t` → `np.intp_t` (~14 occurrences)
2. Add `requires-python = ">=3.11"` to pyproject.toml
3. Test on Python 3.11 and 3.12 with NumPy 2.x
4. Build and test wheel

### For embodied-schemas repo
1. Ensure pyproject.toml has proper PyPI metadata (name, version, description, license)
2. Build: `python -m build`
3. Upload: `twine upload dist/*`
4. Set up Trusted Publisher on PyPI for CI

### For embodied-ai-architect
1. Add `schemas = ["embodied-schemas>=0.1.0"]` to optional deps
2. Add `analysis = ["graphs>=0.2.0"]` to optional deps (once available)
3. Update installation docs with optional dep instructions
4. Implement `AnalysisClient` fallback pattern for cloud API (Phase 3)
