# Frontend Architecture Assessment: Ollama-Style vs Claude Cowork

## Context

The branes platform needs a visual frontend for design exploration that
can render Pareto frontier plots, constraint dashboards, task graph DAGs,
optimization trajectories, SWaP-C radar charts, and multi-level drill-down
trees. This frontend should be a separate repository with its own
development lifecycle.

Two architectural models are evaluated:
1. **Ollama-style**: Decoupled REST API + independent web frontend
2. **Claude Cowork-style**: AI-generated dynamic UI rendered inline

## What We Need to Visualize

| Visualization | Data Source | Interactivity Required |
|--------------|-------------|----------------------|
| Pareto frontier scatter (2D/3D) | `pareto_points`, `pareto_results` | Hover, select point, filter dominated |
| Constraint slackness dashboard | `optimization_review_snapshot` | Click to drill down |
| Task graph DAG | `task_graph` | Expand/collapse, status coloring |
| Optimization trajectory | `optimization_history` | Scrub iterations, compare |
| SWaP-C radar/spider chart | `ppa_metrics`, `swap_assessment` | Toggle design points |
| Multi-level metric tree | `workload_profile`, `ip_blocks` | Drill down/up levels |
| Design session timeline | `history`, `design_rationale` | Navigate decisions |
| Cost breakdown waterfall | `ppa_metrics.cost_breakdown` | Component contribution |
| Thermal/power density map | `floorplan_estimate` | Zoom, hotspot identification |

All data is already available in `SoCDesignState` (persisted as JSON in
`~/.embodied-ai/sessions/`). The visualization layer needs to read this
data and render it interactively.

---

## Architecture 1: Ollama-Style (Decoupled REST API + Web Frontend)

### How It Works

Ollama exposes a stateless REST API on a fixed port. The UI is a completely
separate project (Open WebUI, Enchanted, Hollama, etc.) that connects to
the API over HTTP. Multiple frontends can connect simultaneously. The
backend manages computation; the frontend manages display.

```
┌──────────────────────┐        ┌──────────────────────┐
│  Frontend (React)    │  HTTP  │  Backend (branes)    │
│  Port 3000           │───────>│  Port 8000           │
│                      │        │                      │
│  Plotly 3D scatter   │  GET   │  /api/sessions       │
│  ECharts radar       │  GET   │  /api/sessions/{id}/ │
│  Cytoscape DAG       │  SSE   │    pareto_front      │
│  Recharts timeline   │        │    trajectory        │
│                      │        │    task_graph        │
└──────────────────────┘        └──────────────────────┘
```

### Applied to Branes

**Backend additions** (in the branes repo):
- FastAPI or Flask REST server exposing session data
- Endpoints read from `SessionStore` (JSON files already on disk)
- SSE streaming for live optimization progress
- No new computation — just data access layer

**Frontend** (separate repo: `branes-frontend`):
- React + TypeScript + Vite
- Plotly.js for Pareto scatter plots (3D, interactive)
- ECharts for radar charts, gauges, heatmaps
- Cytoscape.js or Dagre for task graph DAG rendering
- Recharts for line charts, trajectories, timelines
- TanStack Query for data fetching/caching

### Pros

| Advantage | Why It Matters for Branes |
|-----------|--------------------------|
| **Independent repos and release cycles** | Frontend team iterates on visualizations without touching optimizer code |
| **Multiple frontends** | CLI, web, mobile, embedded dashboard — all from same API |
| **Persistent state** | Sessions survive across browser refreshes, CLI invocations, machine restarts |
| **Real-time streaming** | Watch optimization converge live via SSE/WebSocket |
| **Standard web architecture** | Any web developer can contribute; standard tooling |
| **Shareable URLs** | `https://branes.local/session/soc_abc123` — team reviews a design |
| **Scalable** | Frontend and backend scale independently |
| **Testable** | Mock API → test frontend in isolation; mock frontend → test API |
| **DevOps friendly** | Docker Compose for dev, K8s for production |

### Cons

| Disadvantage | Impact |
|-------------|--------|
| **API design upfront** | Must define endpoints before building visualizations |
| **Two repos to maintain** | More infrastructure, CI/CD pipelines |
| **Version coupling** | API changes require frontend updates |
| **More code for simple things** | A quick scatter plot requires: endpoint + fetch + component |
| **No AI-driven iteration** | "Make bars green" requires code change, not natural language |
| **Cold start** | Need 5-10 endpoints before first useful visualization |

---

## Architecture 2: Claude Cowork-Style (AI-Generated Dynamic UI)

### How It Works

The AI generates complete, self-contained HTML/React/SVG applications
inline during the conversation. The generated code runs in a sandboxed
browser environment alongside the chat. The user refines visualizations
through natural language: "add a red line at the power budget" → code
regenerates.

```
┌──────────────────────────────────────────────────┐
│  Chat Panel            │  Artifact Panel         │
│                        │                         │
│  User: Show me the     │  ┌───────────────────┐  │
│  Pareto frontier       │  │  [Plotly 3D]      │  │
│                        │  │  ● ● ●            │  │
│  Claude: Here's the    │  │    ●  ●           │  │
│  Pareto analysis...    │  │  ●      ●         │  │
│                        │  │     ●             │  │
│  User: Highlight the   │  │  [Interactive]    │  │
│  knee point and add    │  └───────────────────┘  │
│  power budget plane    │                         │
│                        │  Generated React code   │
│  Claude: Updated.      │  rendered in sandbox    │
│                        │                         │
└──────────────────────────────────────────────────┘
```

### Applied to Branes

**Backend**: Minimal — Claude reads session JSON directly (via tool call
to `branes session show --latest --json` or MCP).

**Frontend**: None to build — Claude generates each visualization on demand.
The `/architect-assess` skill could generate an HTML dashboard artifact.
The user says "show me the Pareto frontier" and gets an interactive Plotly
chart right there in the conversation.

### Pros

| Advantage | Why It Matters for Branes |
|-----------|--------------------------|
| **Zero frontend code to maintain** | No separate repo, no CI/CD, no deployment |
| **Natural language iteration** | "Make the failing constraints red" → instant |
| **Custom visualizations per session** | Each design problem gets tailored charts |
| **Rapid prototyping** | Test 10 chart ideas in 10 minutes |
| **Context-aware** | Claude knows the design context, generates relevant views |
| **No API to design** | Data flows through tool calls, not endpoints |

### Cons

| Disadvantage | Impact |
|-------------|--------|
| **No persistence** | Charts disappear when conversation ends |
| **Single user** | Can't share a visualization URL with a teammate |
| **Regeneration latency** | Every change requires full code regeneration (seconds) |
| **No real-time streaming** | Can't watch optimization converge live |
| **Sandbox limitations** | Generated code can't call external APIs directly |
| **Inconsistent UX** | Each generated chart looks slightly different |
| **API cost** | Every visualization costs tokens |
| **No offline mode** | Requires Claude API access |
| **Scalability** | Doesn't scale to dashboards with 10+ panels updating in real-time |
| **No version control** | Can't track visualization evolution in git |

---

## Side-by-Side Comparison

| Dimension | Ollama-Style REST | Claude Cowork | Winner |
|-----------|------------------|---------------|--------|
| **Separate repo lifecycle** | Native (two repos) | N/A (no repo) | Ollama |
| **Persistent visualizations** | Yes (URLs, bookmarks) | No (ephemeral) | Ollama |
| **Multi-user/team** | Yes (shared server) | No (single session) | Ollama |
| **Real-time updates** | SSE/WebSocket | Code regeneration | Ollama |
| **Development velocity** | Weeks to first chart | Minutes | Cowork |
| **Custom one-off analysis** | Requires code | Natural language | Cowork |
| **Consistent UX** | Yes (designed once) | No (varies per generation) | Ollama |
| **Offline operation** | Yes (local server) | No (needs Claude API) | Ollama |
| **Interactive drill-down** | Native (React state) | Limited (regen) | Ollama |
| **Cost to operate** | Server hosting | API tokens per viz | Depends |
| **Integration with CLI** | Same API | Tool calls | Both |
| **10+ panel dashboard** | Native | Impractical | Ollama |
| **Rapid experimentation** | Code→test cycle | Describe→see | Cowork |
| **Version control** | Git (components) | None | Ollama |

---

## Recommendation: Hybrid Architecture

Neither model alone satisfies all requirements. The recommendation is:

### Primary: Ollama-Style REST + React Frontend (production)

This is the **production visualization system** — the dashboard that an
architect opens every morning to check their design sessions.

**What to build first:**
1. REST API server (5 endpoints, reads from SessionStore)
2. Session list/detail page
3. Pareto frontier 3D scatter (Plotly)
4. Constraint slackness bar chart with color-coded headroom
5. Task graph DAG (Cytoscape)

**Repo structure:**
```
branes-frontend/
├── src/
│   ├── api/client.ts         # REST client, types from OpenAPI
│   ├── components/
│   │   ├── ParetoScatter.tsx  # Plotly 3D, interactive
│   │   ├── SlacknessBars.tsx  # Budget utilization
│   │   ├── TaskGraph.tsx      # Cytoscape DAG
│   │   ├── TrajectoryChart.tsx # Recharts line
│   │   ├── SwapRadar.tsx      # ECharts radar
│   │   └── DrillTree.tsx      # Hierarchical metrics
│   ├── pages/
│   │   ├── SessionList.tsx
│   │   └── SessionDetail.tsx
│   └── hooks/
│       ├── useSession.ts
│       └── useStream.ts
├── package.json
├── vite.config.ts
└── Dockerfile
```

### Supplementary: Claude Cowork (exploration)

This is the **design exploration scratchpad** — used when the architect
wants a quick custom visualization to understand a specific trade-off.

**How it works:**
- Architect uses `/architect-assess` or `/architect-drill` skills
- Skill reads session state, can also generate an HTML artifact
- "Show me how power trades off against cost for the top 5 designs" →
  Claude generates a Plotly chart with the specific data points
- When a visualization pattern proves useful, the frontend team
  implements it permanently in the React app

**The funnel:**
```
Idea → Claude generates it → Architect uses it → Valuable? →
  Yes → Frontend team builds permanent React component
  No  → Discard, it was free exploration
```

This means Claude Cowork serves as the **visualization prototyping tool**
that feeds the permanent frontend's feature backlog.

---

## API Design (Minimal Viable Endpoints)

The REST API bridges the existing `SessionStore` (JSON files) with the
frontend. Minimal set to start:

```
GET  /api/sessions
     → [{session_id, goal, status, iteration, platform, saved_at}, ...]

GET  /api/sessions/{id}
     → Full SoCDesignState JSON (same as `branes session show --json`)

GET  /api/sessions/{id}/pareto
     → {points: [{power, latency, cost, dominated, hardware_name}], knee_point_idx}

GET  /api/sessions/{id}/slackness
     → [{name, target, actual, margin_pct, verdict, trend, binding}, ...]

GET  /api/sessions/{id}/trajectory
     → [{iteration, ppa_snapshot, verdicts, strategy_applied}, ...]

GET  /api/sessions/{id}/task_graph
     → {nodes: {id: {name, agent, status, deps}}, execution_order: []}

SSE  /api/sessions/{id}/stream
     → Live state updates during active optimization
```

All of these are simple reads from the JSON files in
`~/.embodied-ai/sessions/`. No new computation required.

---

## Technology Stack

### Backend API (branes repo, minimal addition)

```python
# FastAPI server — thin layer over SessionStore
# ~200 lines of code for all 6 endpoints

from fastapi import FastAPI
from embodied_ai_architect.graphs.session_store import SessionStore

app = FastAPI(title="Branes Architect API")
store = SessionStore()

@app.get("/api/sessions")
async def list_sessions():
    return store.list_sessions()

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    state = store.load(session_id)
    if not state:
        raise HTTPException(404)
    return state
```

### Frontend (separate repo)

| Library | Purpose | Why |
|---------|---------|-----|
| React 18+ | Framework | Component model, hooks, ecosystem |
| TypeScript | Type safety | API response types, catch errors early |
| Vite | Build tool | Fast HMR, simple config |
| Plotly.js | Pareto 3D, scatter | Best 3D interactive charts |
| ECharts | Radar, gauges, heatmaps | Rich chart library, good perf |
| Cytoscape.js | Task graph DAG | Purpose-built for graph visualization |
| Recharts | Line, bar, area | Clean React-native charting |
| TanStack Query | Data fetching | Caching, refetching, loading states |
| Tailwind CSS | Styling | Rapid UI development |

### Deployment

```yaml
# docker-compose.yml (development)
services:
  backend:
    build: ../embodied-ai-architect
    ports: ["8000:8000"]
    volumes: ["~/.embodied-ai:/data"]

  frontend:
    build: .
    ports: ["3000:3000"]
    environment:
      - BRANES_API_URL=http://backend:8000
```

---

## Migration Path

### Phase 1: API Foundation (Week 1-2)
- Add FastAPI server to branes repo (6 endpoints)
- OpenAPI spec generation
- CORS configuration for local development

### Phase 2: MVP Frontend (Week 3-6)
- Create `branes-frontend` repo
- Session list page
- Session detail with Pareto scatter + slackness bars
- Basic task graph rendering

### Phase 3: Rich Visualizations (Week 7-12)
- SWaP-C radar chart
- Optimization trajectory with iteration scrubbing
- Multi-level drill-down tree
- Design comparison (overlay two sessions)

### Phase 4: Real-Time + AI (Week 13+)
- SSE streaming for live optimization progress
- Claude-generated artifact integration
- Export to PDF/PowerPoint for stakeholder presentations
- Embedded mode (iframe into other tools)

---

## Decision Summary

| Question | Answer |
|----------|--------|
| Should the frontend be a separate repo? | **Yes** — different tech stack, team, release cycle |
| Which architecture model? | **Hybrid**: Ollama-style REST as production foundation, Claude Cowork for exploration/prototyping |
| What to build first? | REST API (6 endpoints) + Pareto scatter + slackness dashboard |
| What visualization library? | Plotly.js (3D), ECharts (radar), Cytoscape.js (DAG), Recharts (lines) |
| How does Claude fit? | Generates one-off visualizations from session data; proves concepts before permanent implementation |
