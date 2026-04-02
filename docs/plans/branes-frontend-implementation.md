# Branes Frontend — Implementation Plan

## Epic: Visual Design Exploration Dashboard

Build a web-based visualization frontend for the branes embodied AI
architect platform. The frontend reads design session state from the
backend REST API and renders interactive visualizations for Pareto
frontiers, constraint analysis, task graphs, optimization trajectories,
and multi-level metric drill-down.

**Repos:**
- `branes-ai/embodied-ai-architect` — Backend: REST API + session store
- `branes-ai/branes-frontend` — Frontend: React + visualization libraries

**Architecture:** Ollama-style decoupled REST API with Claude Cowork for
prototyping (see `docs/designs/frontend-architecture-assessment.md`).

---

## Phase 1: API Foundation (Backend)

**Goal:** Expose session data via REST endpoints in the branes repo.

### 1.1 FastAPI Server

Add a thin API server that reads from `SessionStore`:

```
src/embodied_ai_architect/api/
├── __init__.py
├── server.py          # FastAPI app, CORS, startup
├── routes/
│   ├── sessions.py    # Session CRUD + derived views
│   └── health.py      # Readiness/liveness
└── schemas.py         # Pydantic response models
```

**Endpoints:**

| Method | Path | Returns |
|--------|------|---------|
| `GET` | `/api/sessions` | List of session summaries |
| `GET` | `/api/sessions/{id}` | Full session state |
| `GET` | `/api/sessions/{id}/pareto` | Pareto points + knee point |
| `GET` | `/api/sessions/{id}/slackness` | Constraint slackness array |
| `GET` | `/api/sessions/{id}/trajectory` | Optimization history |
| `GET` | `/api/sessions/{id}/taskgraph` | Task graph nodes + edges |
| `GET` | `/api/sessions/{id}/workload` | Per-operator breakdown |
| `GET` | `/api/health` | Server status |

**CLI entry point:**
```bash
branes api serve --port 8000
```

### 1.2 OpenAPI Spec

Auto-generated from FastAPI. The frontend uses this to generate TypeScript
types via `openapi-typescript`.

### 1.3 CORS Configuration

Allow `localhost:3000` (dev) and configurable origins (production).

### 1.4 Tests

- Unit tests for each endpoint (mock SessionStore)
- Integration test: create session → GET via API → verify response

**Acceptance criteria:** `curl http://localhost:8000/api/sessions` returns
session list; `curl .../api/sessions/{id}/pareto` returns Pareto data.

---

## Phase 2: Frontend MVP

**Goal:** Session list + session detail with Pareto scatter + slackness bars.

### 2.1 Project Setup

```
branes-frontend/
├── src/
│   ├── api/
│   │   ├── client.ts        # Fetch wrapper, base URL config
│   │   └── types.ts         # Generated from OpenAPI
│   ├── components/
│   │   ├── SessionList.tsx
│   │   ├── SessionDetail.tsx
│   │   ├── ParetoScatter.tsx
│   │   ├── SlacknessBars.tsx
│   │   └── MetricCard.tsx
│   ├── hooks/
│   │   └── useSession.ts    # TanStack Query hooks
│   ├── App.tsx
│   └── main.tsx
├── public/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
└── Dockerfile
```

### 2.2 Session List Page

- Table: session ID, goal, status, iteration, platform, saved at
- Click row → navigate to session detail
- Auto-refresh every 10s (TanStack Query refetchInterval)

### 2.3 Session Detail Page

Layout: sidebar navigation + main content area

**Panels:**
1. **Header**: Goal, platform, status, iteration counter
2. **Pareto Scatter**: Plotly 3D scatter (power × latency × cost)
   - Non-dominated points highlighted
   - Knee point marked
   - Constraint boundary planes
   - Hover: show design details
3. **Slackness Bars**: Horizontal bar chart
   - Green (>20% margin), yellow (5-20%), red (<5% or exceeded)
   - Click bar → drill to that metric
4. **PPA Summary Cards**: Power, latency, area, cost with verdict badges

### 2.4 Development Infrastructure

- Vite dev server with hot reload
- ESLint + Prettier
- GitHub Actions CI (lint, typecheck, build)
- Docker build for deployment

**Acceptance criteria:** Navigate to session list, click a session, see
interactive Pareto scatter plot and constraint slackness dashboard.

---

## Phase 3: Rich Visualizations

**Goal:** Full visualization suite for design exploration.

### 3.1 Task Graph DAG

- Cytoscape.js with dagre layout
- Nodes colored by status (completed/running/failed/pending)
- Click node → show task result summary
- Edges show dependencies

### 3.2 Optimization Trajectory

- Recharts line chart with dual Y-axis
- X: iteration, Y1: power/latency, Y2: cost
- Vertical markers for strategy applications
- Hover: show strategy name and before/after metrics

### 3.3 SWaP-C Radar Chart

- ECharts radar with 6 axes (power, latency, area, cost, weight, volume)
- Overlay multiple design points for comparison
- Budget envelope shown as reference polygon
- Toggle between absolute values and % of budget

### 3.4 Multi-Level Drill-Down Tree

- Treemap or collapsible tree showing metric composition:
  System → Subsystem → Operator → Kernel
- Color intensity by contribution to selected metric
- Click to drill down, breadcrumb to drill up

### 3.5 Design Decision Timeline

- Horizontal timeline of design rationale entries
- Click entry → show decision details (agent, action, rationale)
- Filter by agent type

### 3.6 Cost Breakdown Waterfall

- Stacked waterfall chart: die + package + test + NRE/unit
- Toggle between unit cost and total cost at volume
- Sensitivity slider for production volume

**Acceptance criteria:** All 6 visualization types render with real
session data, interactive navigation works.

---

## Phase 4: Real-Time + Integration

**Goal:** Live updates and Claude integration.

### 4.1 SSE Streaming

- Backend: SSE endpoint `/api/sessions/{id}/stream`
- Sends state updates as optimization runs
- Frontend: `EventSource` consumer, updates charts in real-time

### 4.2 Session Comparison

- Side-by-side comparison of two sessions
- Overlay Pareto frontiers
- Diff constraint slackness

### 4.3 Export

- Export charts as PNG/SVG
- Generate PDF report from session
- PowerPoint export for stakeholder presentations

### 4.4 Claude Cowork Integration

- "Visualize" button sends session data to Claude
- Claude generates custom artifact visualizations
- Proven visualizations get promoted to permanent components

### 4.5 Responsive Design

- Mobile-friendly session list
- Tablet-optimized detail view
- Dark/light theme

---

## Backend Task Breakdown

| Task | Effort | Priority | Dependencies |
|------|--------|----------|-------------|
| FastAPI server skeleton + health | 2h | P0 | None |
| `GET /api/sessions` | 1h | P0 | Server skeleton |
| `GET /api/sessions/{id}` | 1h | P0 | Server skeleton |
| `GET /api/sessions/{id}/pareto` | 2h | P0 | Session endpoint |
| `GET /api/sessions/{id}/slackness` | 2h | P0 | Session endpoint |
| `GET /api/sessions/{id}/trajectory` | 1h | P1 | Session endpoint |
| `GET /api/sessions/{id}/taskgraph` | 1h | P1 | Session endpoint |
| `GET /api/sessions/{id}/workload` | 2h | P1 | Session endpoint |
| `branes api serve` CLI command | 1h | P0 | Server skeleton |
| OpenAPI spec + CORS | 1h | P0 | Server skeleton |
| SSE streaming endpoint | 4h | P2 | Active session tracking |
| Tests for all endpoints | 4h | P0 | All endpoints |

## Frontend Task Breakdown

| Task | Effort | Priority | Dependencies |
|------|--------|----------|-------------|
| Vite + React + TS project setup | 2h | P0 | None |
| TanStack Query + API client | 2h | P0 | Project setup |
| Session list page | 3h | P0 | API client |
| Session detail layout | 3h | P0 | API client |
| Pareto scatter (Plotly 3D) | 4h | P0 | Session detail |
| Slackness bars | 3h | P0 | Session detail |
| PPA summary cards | 2h | P0 | Session detail |
| Task graph DAG (Cytoscape) | 4h | P1 | Session detail |
| Optimization trajectory chart | 3h | P1 | Session detail |
| SWaP-C radar chart | 3h | P1 | Session detail |
| Drill-down tree | 6h | P2 | Session detail |
| Decision timeline | 3h | P2 | Session detail |
| Cost waterfall | 3h | P2 | Session detail |
| SSE consumer | 3h | P2 | Streaming endpoint |
| Docker build | 2h | P1 | Project setup |
| GitHub Actions CI | 2h | P0 | Project setup |
| Dark/light theme | 2h | P3 | All components |
