# Documentation & Developer Experience Roadmap

This roadmap defines the user-facing documentation infrastructure for the Embodied AI Architect platform. The goal is to create a comprehensive documentation site that serves both human users and LLM agents.

## Design Principles

1. **LLM-Native Content** - All documentation is structured for both human reading and LLM context injection
2. **Single Source of Truth** - Documentation generates from the same content that feeds the knowledge base
3. **Progressive Disclosure** - Landing page → Features → Tutorials → API Reference → Deep Dives
4. **Searchable & Retrievable** - Full-text search for humans, vector search for LLMs

---

## Technology Stack

### Astro Starlight

**Why Starlight:**
- Built on Astro (fast, modern static site generator)
- Designed for documentation with built-in features:
  - Sidebar navigation
  - Search (Pagefind)
  - Dark/light mode
  - i18n support
  - Markdown + MDX support
- LLM-friendly output (clean HTML, semantic structure)
- Can generate `llms.txt` and structured data for AI consumption

**Directory Structure:**
```
docs-site/
├── astro.config.mjs
├── src/
│   ├── content/
│   │   ├── docs/
│   │   │   ├── index.mdx              # Landing page
│   │   │   ├── getting-started/
│   │   │   │   ├── introduction.md
│   │   │   │   ├── installation.md
│   │   │   │   └── quickstart.md
│   │   │   ├── features/
│   │   │   │   ├── model-analysis.md
│   │   │   │   ├── hardware-selection.md
│   │   │   │   ├── roofline-analysis.md
│   │   │   │   ├── constraint-checking.md
│   │   │   │   └── deployment.md
│   │   │   ├── tutorials/
│   │   │   │   ├── drone-perception.md
│   │   │   │   ├── yolo-on-jetson.md
│   │   │   │   ├── power-optimization.md
│   │   │   │   └── custom-hardware.md
│   │   │   ├── catalog/
│   │   │   │   ├── hardware/
│   │   │   │   ├── models/
│   │   │   │   ├── sensors/
│   │   │   │   └── operators/
│   │   │   ├── reference/
│   │   │   │   ├── cli.md
│   │   │   │   ├── api.md
│   │   │   │   ├── mcp-tools.md
│   │   │   │   └── constraints.md
│   │   │   └── troubleshooting/
│   │   │       ├── common-issues.md
│   │   │       └── faq.md
│   │   └── config.ts
│   └── assets/
│       ├── images/
│       └── diagrams/
├── public/
│   ├── llms.txt                       # LLM-specific sitemap
│   └── llms-full.txt                  # Full content for LLM ingestion
└── package.json
```

---

## Content Architecture

### Site Sections

| Section | Purpose | Audience |
|---------|---------|----------|
| **Landing** | Value proposition, key features, CTA | Prospects, new users |
| **Getting Started** | Installation, first run, quickstart | New users |
| **Features** | Deep dive into each capability | Users learning the product |
| **Tutorials** | Step-by-step problem-solving | Users with specific goals |
| **Catalog** | Hardware, model, sensor specs | Reference lookups |
| **Reference** | CLI, API, MCP tools documentation | Power users, integrators |
| **Troubleshooting** | Common issues, FAQ | Users with problems |

### LLM Integration

**`llms.txt` (lightweight):**
```
# Embodied AI Architect

> Design, analyze, optimize, and deploy embodied AI solutions

## Capabilities
- Model-hardware fit analysis with roofline modeling
- Constraint checking (latency, power, memory)
- Hardware recommendation for edge deployment
- Deployment guide generation

## Quick Links
- Documentation: /docs/
- Hardware Catalog: /docs/catalog/hardware/
- Tutorials: /docs/tutorials/

## API
- MCP Server: branes-mcp serve
- CLI: branes analyze model.pt
```

**`llms-full.txt` (comprehensive):**
- Full catalog content in markdown
- All tutorials concatenated
- Structured for RAG ingestion
- Updated on each build

---

## Phase 1: Foundation (Weeks 1-2)

**Goal**: Minimal viable documentation site with core structure.

| Task | Deliverable |
|------|-------------|
| Initialize Astro Starlight project | `docs-site/` directory |
| Create landing page | Product overview, key features |
| Write Getting Started guide | Installation, quickstart |
| Add 3 feature pages | Model analysis, hardware selection, roofline |
| Deploy to Vercel/Netlify | Live URL |
| Generate `llms.txt` | LLM-friendly sitemap |

**Acceptance Criteria:**
- Site loads at production URL
- Navigation works (sidebar, search)
- `llms.txt` accessible at `/llms.txt`

---

## Phase 2: Catalog Integration (Weeks 3-4)

**Goal**: Auto-generate catalog pages from embodied-schemas.

| Task | Deliverable |
|------|-------------|
| Write catalog generator script | YAML → MDX conversion |
| Generate hardware pages (50+) | One page per hardware entry |
| Generate model pages (10+) | One page per model entry |
| Generate sensor pages | One page per sensor |
| Generate operator pages | One page per operator |
| Add catalog search/filter | Interactive hardware finder |

**Acceptance Criteria:**
- All embodied-schemas entries have documentation pages
- Pages auto-regenerate when YAML changes
- Search finds hardware by name, vendor, specs

---

## Phase 3: Tutorials & Guides (Weeks 5-8)

**Goal**: Comprehensive how-to content for common use cases.

### Priority Tutorials

| # | Tutorial | Use Case |
|---|----------|----------|
| 1 | **Drone Perception Pipeline** | End-to-end drone obstacle avoidance |
| 2 | **YOLOv8 on Jetson Orin** | Deploy detection model to edge GPU |
| 3 | **Coral Edge TPU Deployment** | Ultra-low-power deployment |
| 4 | **INT8 Quantization Guide** | Optimize model for edge |
| 5 | **Power Budget Planning** | Design for battery-powered systems |
| 6 | **Multi-Model Pipelines** | Run detection + tracking + depth |
| 7 | **Custom Hardware Analysis** | Evaluate pre-silicon targets |

### Tutorial Template

```markdown
---
title: "Deploying YOLOv8 on Jetson Orin"
description: "Step-by-step guide to running YOLOv8 on NVIDIA Jetson with TensorRT optimization"
---

## Overview
What you'll learn and achieve.

## Prerequisites
- Hardware: Jetson Orin Nano/NX/AGX
- Software: JetPack 5.1+
- Model: YOLOv8n/s

## Steps

### 1. Analyze Model-Hardware Fit
```bash
branes analyze yolov8n.pt --hardware jetson-orin-nano
```

### 2. Export to ONNX
...

### 3. Convert to TensorRT
...

### 4. Benchmark Performance
...

## Expected Results

| Metric | Value |
|--------|-------|
| Latency | 5.1ms |
| Throughput | 196 FPS |
| Power | 9.8W |

## Troubleshooting

### Issue: "Unsupported operator"
Solution: ...

## Next Steps
- Try INT8 quantization for 2x speedup
- Add tracking with ByteTrack
```

**Acceptance Criteria:**
- 7 tutorials published
- Each tutorial tested and verified
- Code examples are copy-paste ready

---

## Phase 4: Reference Documentation (Weeks 9-10)

**Goal**: Complete API and CLI reference.

| Task | Deliverable |
|------|-------------|
| CLI reference | All commands documented with examples |
| MCP tools reference | All tools with input/output schemas |
| Python API reference | Core classes and functions |
| Constraint tiers reference | Latency, power, memory classes |
| Glossary | Key terms defined |

**Acceptance Criteria:**
- Every CLI command has documentation
- Every MCP tool has input/output examples
- Auto-generated from source where possible

---

## Phase 5: Interactive Features (Weeks 11-12)

**Goal**: Dynamic, interactive documentation elements.

| Feature | Description |
|---------|-------------|
| Hardware Finder | Filter hardware by constraints (TOPS, power, form factor) |
| Model Compatibility Matrix | Which models run on which hardware |
| Constraint Calculator | Input requirements → recommended hardware |
| Code Playground | Interactive examples (Pyodide or similar) |

**Acceptance Criteria:**
- Hardware finder filters work
- Compatibility matrix loads from data
- At least one interactive example

---

## Deployment & CI/CD

### Build Pipeline

```yaml
# .github/workflows/docs.yml
name: Deploy Documentation

on:
  push:
    branches: [main]
    paths:
      - 'docs-site/**'
      - 'embodied-schemas/data/**'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Generate catalog pages
        run: python scripts/generate_catalog_docs.py

      - name: Generate llms.txt
        run: python scripts/generate_llms_txt.py

      - name: Build Astro site
        run: |
          cd docs-site
          npm install
          npm run build

      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
```

### Content Sync

- Catalog pages regenerate when `embodied-schemas` YAML changes
- Tutorials link to knowledge base articles (single source)
- `llms.txt` and `llms-full.txt` regenerate on each build

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Page load time | <2s | Lighthouse |
| Search relevance | >90% | User testing |
| Tutorial completion rate | >70% | Analytics |
| `llms.txt` accuracy | 100% up-to-date | CI validation |
| SEO score | >90 | Lighthouse |
| Mobile usability | 100% | Lighthouse |

---

## Content Ownership

| Section | Owner | Update Frequency |
|---------|-------|------------------|
| Landing, Features | Product | Monthly |
| Getting Started | Engineering | Per release |
| Tutorials | Engineering | Bi-weekly |
| Catalog | Auto-generated | On schema change |
| Reference | Auto-generated | On code change |
| Troubleshooting | Support | As issues arise |

---

## Integration with Knowledge Base

The documentation site and knowledge base share content:

```
                    ┌─────────────────────┐
                    │  embodied-schemas   │
                    │  (YAML source)      │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
    ┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
    │  docs-site/     │ │ branes_mcp/ │ │ llms.txt        │
    │  (Starlight)    │ │ knowledge/  │ │ llms-full.txt   │
    │                 │ │             │ │                 │
    │  Human-readable │ │ LLM context │ │ AI agent        │
    │  documentation  │ │ injection   │ │ discovery       │
    └─────────────────┘ └─────────────┘ └─────────────────┘
```

**Single source, multiple outputs:**
- Same YAML → website pages, knowledge chunks, and LLM metadata
- Tutorials appear in both docs site and knowledge base
- Changes propagate to all outputs automatically

---

## Next Steps

1. **Initialize Starlight project** - `npm create astro@latest -- --template starlight`
2. **Create landing page** - Product positioning and key features
3. **Write installation guide** - First user experience
4. **Build catalog generator** - YAML → MDX automation
5. **Deploy to Vercel** - Get live URL
