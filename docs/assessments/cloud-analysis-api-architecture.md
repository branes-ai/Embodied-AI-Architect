# Cloud Analysis API — System Architecture

**Date**: 2026-04-13
**Status**: Design — pending approval
**Related**: #182, #163, #181

## Problem

The `branes mcp` commands (`analyze`, `latency`, `energy`, `memory`, `compare`,
`hardware`, `specs`) depend on the proprietary `graphs` package which:
1. Cannot be published to public PyPI (proprietary roofline models, calibration data)
2. Has a Cython build issue on Python 3.12 / NumPy 2.x
3. Requires a source checkout or private distribution channel

A clean `pip install embodied-ai-architect` user today gets zero analysis capability.

## Solution: Cloud Analysis API

Wrap the `graphs.mcp.server` tool functions in a REST API. The branes CLI
transparently routes to local (if graphs installed) or cloud (via API key).

## System Architecture

```
┌──────────────────────────────────────┐
│          Customer Machine            │
│                                      │
│  branes CLI (pip install, open src)  │
│  ┌────────────────────────────────┐  │
│  │ Lifecycle Commands (local)     │  │
│  │ mission, sensor, actuator,     │  │
│  │ design, qualify, plan, swap,   │  │
│  │ optimize, validate, synthesize │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │ AnalysisClient                 │  │
│  │ ┌──────────┐  ┌─────────────┐ │  │
│  │ │ Local    │  │ Cloud       │ │  │
│  │ │ (graphs) │  │ (REST API)  │ │  │
│  │ └──────────┘  └──────┬──────┘ │  │
│  │   try first     fallback      │  │
│  └─────────────────────┼────────┘  │
│                        │            │
└────────────────────────┼────────────┘
                         │ HTTPS
                         ▼
┌────────────────────────────────────────────────────┐
│                 Branes Cloud                        │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │            API Gateway                        │  │
│  │  • Auth (Bearer token)                        │  │
│  │  • Rate limiting (per-key)                    │  │
│  │  • Usage metering (for billing)               │  │
│  │  • CORS, TLS                                  │  │
│  └──────────────┬───────────────────────────────┘  │
│                 │                                    │
│  ┌──────────────▼───────────────────────────────┐  │
│  │         FastAPI Application                   │  │
│  │                                               │  │
│  │  POST /v1/tools/{tool_name}                   │  │
│  │    → graphs.mcp.server.execute_mcp_tool()     │  │
│  │    → JSON response                            │  │
│  │                                               │  │
│  │  Convenience endpoints:                       │  │
│  │  GET  /v1/hardware                            │  │
│  │  GET  /v1/hardware/{id}/specs                 │  │
│  │  POST /v1/analyze                             │  │
│  │  POST /v1/latency                             │  │
│  │  POST /v1/energy                              │  │
│  │  POST /v1/memory                              │  │
│  │  POST /v1/compare                             │  │
│  └──────────────┬───────────────────────────────┘  │
│                 │                                    │
│  ┌──────────────▼───────────────────────────────┐  │
│  │         graphs library (proprietary)          │  │
│  │                                               │  │
│  │  mcp/server.py      → tool dispatch           │  │
│  │  hardware/           → profile registry        │  │
│  │  estimation/         → roofline models         │  │
│  │  calibration/        → measured perf data      │  │
│  │  ir/                 → intermediate repr       │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## API Design

### Authentication

```
Authorization: Bearer brk_live_xxxxxxxxxxxx
```

Keys managed via `branes secrets set BRANES_API_KEY brk_live_xxx` or
environment variable `BRANES_API_KEY`.

### Endpoints

#### Generic tool dispatch

```
POST /v1/tools/{tool_name}
Content-Type: application/json

{
  "model_name": "yolov8n",
  "hardware_name": "Jetson-Orin-Nano-8GB",
  "batch_size": 1,
  "precision": "FP16"
}

→ 200 OK
{
  "model": "yolov8n",
  "hardware": "Jetson-Orin-Nano-8GB",
  "metrics": { ... }
}
```

This mirrors `execute_mcp_tool(tool_name, args)` exactly.

#### Convenience endpoints

| Method | Path | Maps to tool |
|--------|------|-------------|
| GET | `/v1/hardware?type=gpu&query=orin` | `list_hardware` |
| GET | `/v1/hardware/{id}/specs` | `get_hardware_specs` |
| POST | `/v1/analyze` | `analyze_model_detailed` |
| POST | `/v1/latency` | `estimate_latency` |
| POST | `/v1/energy` | `estimate_energy` |
| POST | `/v1/memory` | `estimate_memory` |
| POST | `/v1/compare` | `compare_hardware_targets` |

### Rate Limits

| Tier | Rate | Monthly Quota |
|------|------|---------------|
| Free | 10 req/min | 100 requests |
| Pro | 60 req/min | 10,000 requests |
| Enterprise | unlimited | unlimited |

### Error Responses

```json
{
  "error": "rate_limit_exceeded",
  "message": "10 requests per minute exceeded. Upgrade at https://branes.ai/pricing",
  "retry_after": 45
}
```

## Client Architecture (in embodied-ai-architect)

### New file: `src/embodied_ai_architect/analysis/client.py`

```python
class AnalysisClient:
    """Routes analysis to local graphs library or cloud API.

    Resolution order:
    1. Local import (graphs installed) — zero latency, no API key needed
    2. Cloud API (BRANES_API_KEY set) — ~200ms latency, metered
    3. Raise ImportError with install/signup instructions
    """

    def __init__(self):
        self._local = self._try_local()
        self._api_key = os.environ.get("BRANES_API_KEY", "")
        self._base_url = os.environ.get(
            "BRANES_API_URL", "https://api.branes.ai"
        )

    def _try_local(self):
        try:
            from graphs.mcp.server import execute_mcp_tool
            return execute_mcp_tool
        except ImportError:
            return None

    def call_tool(self, tool_name: str, args: dict) -> dict:
        if self._local:
            result_json = self._local(tool_name, args)
            return json.loads(result_json)
        if self._api_key:
            return self._cloud_call(tool_name, args)
        raise ImportError(
            "Analysis requires either:\n"
            "  1. graphs package installed locally\n"
            "  2. BRANES_API_KEY set for cloud analysis\n"
            "Sign up at https://branes.ai/signup"
        )

    def _cloud_call(self, tool_name: str, args: dict) -> dict:
        resp = httpx.post(
            f"{self._base_url}/v1/tools/{tool_name}",
            json=args,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()
```

### Updated mcp.py

```python
# Replace:
def _call(tool_name, args, json_output=False):
    execute_mcp_tool, _ = _get_server()
    result_json = execute_mcp_tool(tool_name, args)

# With:
def _call(tool_name, args, json_output=False):
    from embodied_ai_architect.analysis.client import AnalysisClient
    client = AnalysisClient()
    data = client.call_tool(tool_name, args)
```

## Cloud Service Architecture

### Deployment

```
                    ┌─────────────┐
                    │  CloudFlare  │ ← TLS termination, DDoS
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  API Gateway │ ← Auth, rate limit (AWS API GW or Kong)
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │  ECS / Cloud Run / k8s  │
              │  ┌────────────────────┐ │
              │  │ FastAPI + graphs   │ │ ← Stateless, horizontally scalable
              │  │ (Docker container) │ │
              │  └────────────────────┘ │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Usage DB (DynamoDB)    │ ← API key → usage counters
              └─────────────────────────┘
```

### Docker image

```dockerfile
FROM python:3.11-slim
COPY graphs/ /app/graphs/
COPY api/ /app/api/
RUN pip install fastapi uvicorn
WORKDIR /app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

The graphs source lives ONLY inside the Docker image — never in the client,
never on PyPI, never in a git repo the customer can access.

## Three-Tier Product Model

```
┌───────────────────────────────────────────────────────┐
│                    Free Tier                           │
│  pip install embodied-ai-architect                     │
│  ✓ Mission workflow (create, qualify, plan, validate)  │
│  ✓ 328 platform registry + 80 sensors + 80 actuators  │
│  ✓ SWaP-C analysis (swap estimate/check/explore)      │
│  ✓ Design optimization (MOO, Pareto)                   │
│  ✓ System synthesis and validation                     │
│  ✗ Hardware analysis (mcp analyze/latency/energy)      │
│  ✗ Competitive benchmarking (mcp compare)              │
├───────────────────────────────────────────────────────┤
│                    API Tier ($)                         │
│  BRANES_API_KEY=brk_live_xxx                           │
│  ✓ Everything in Free                                  │
│  ✓ Cloud-hosted roofline analysis                      │
│  ✓ Hardware catalog with calibrated measurements       │
│  ✓ Model × hardware comparison                         │
│  ✓ Energy and memory estimation                        │
│  Rate limited: 10K requests/month                      │
├───────────────────────────────────────────────────────┤
│                Enterprise Tier ($$)                     │
│  Local graphs package (air-gapped)                     │
│  ✓ Everything in API Tier                              │
│  ✓ Unlimited local analysis                            │
│  ✓ Custom hardware modeling                            │
│  ✓ Custom calibration data                             │
│  ✓ On-premise deployment                               │
│  Annual license + support                              │
└───────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: AnalysisClient abstraction (embodied-ai-architect)
- Create `analysis/client.py` with local/cloud fallback
- Update `cli/commands/mcp.py` to use AnalysisClient
- Add `httpx` to core dependencies
- Graceful error when neither local nor cloud available

### Phase 2: Cloud API service (new repo: branes-cloud)
- FastAPI app wrapping graphs.mcp.server
- Docker containerization
- API key auth middleware
- Deploy to AWS ECS / GCP Cloud Run

### Phase 3: Auth & billing infrastructure
- API key provisioning (signup → key)
- Usage metering (DynamoDB counters)
- Rate limiting (per-key, per-tier)
- Billing integration (Stripe)

### Phase 4: Enterprise distribution
- Private PyPI with compiled wheels
- License key validation in graphs package
- Air-gapped installation playbook
