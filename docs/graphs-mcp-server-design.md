# Graphs MCP Server — Design & Configuration

## Decision: Server Lives in `../graphs`

The MCP server belongs in the `graphs` repo because:

1. **It serves graphs code.** The `UnifiedAnalyzer`, `RooflineAnalyzer`, `EnergyAnalyzer`,
   `HardwareRegistry` — all live in `graphs`. The server is a thin API layer over them.
2. **Proprietary boundary.** Keeping the server co-located means the proprietary estimators
   never leave the `graphs` repo. Clients (this repo, Claude Code, other tools) only see
   MCP tool interfaces.
3. **Versioning.** When graphs adds a new estimator or changes a hardware model, the MCP
   server updates in the same commit. No cross-repo coordination needed.
4. **This repo is already a client.** `embodied-ai-architect` already wraps graphs via
   `llm/graphs_tools.py`. Replacing that with MCP calls is a clean swap.

---

## Current State

| | branes MCP server (this repo) | graphs MCP server (proposed) |
|---|---|---|
| **Location** | `src/embodied_ai_architect/mcp/server.py` | `src/graphs/mcp/server.py` (new) |
| **Framework** | Raw MCP SDK (`mcp>=1.0.0`) | Same |
| **Pattern** | `get_mcp_tool_definitions()` + `execute_mcp_tool()` | Same |
| **Tools** | 5 tools (MOO exploration, Pareto, sensitivity) | 7 tools (see below) |
| **State** | Stateful — `SessionManager` for async optimization | Mostly stateless — analytical queries |
| **Transport** | stdio (local) | SSE or streamable-HTTP (remote, authenticated) |
| **Auth** | None (local process) | Bearer token or mTLS (see Auth section) |

---

## Server Architecture (in `../graphs`)

### File Layout

```
graphs/
  src/graphs/mcp/
    __init__.py
    server.py          # Tool definitions + dispatcher
    auth.py            # Token validation middleware
    transport.py       # SSE/HTTP server with auth layer
  pyproject.toml       # Add: mcp = ["mcp>=1.0.0", "starlette", "uvicorn"]
```

### Tools to Expose

Seven tools, matching the key `graphs` APIs:

#### 1. `analyze_model` — Full unified analysis

```json
{
  "name": "analyze_model",
  "description": "Run unified roofline + energy + memory analysis for a model on target hardware",
  "input_schema": {
    "type": "object",
    "properties": {
      "model_name": { "type": "string", "description": "Model identifier (e.g., 'yolov8n', 'resnet50')" },
      "hardware_name": { "type": "string", "description": "Hardware target (e.g., 'jetson_orin_nano', 'h100_sxm5')" },
      "batch_size": { "type": "integer", "default": 1 },
      "precision": { "type": "string", "enum": ["fp32", "fp16", "int8", "int4"], "default": "fp16" }
    },
    "required": ["model_name", "hardware_name"]
  }
}
```

**Returns:** Latency (ms), energy (mJ), peak memory (MB), bottleneck (compute/memory),
utilization (%), arithmetic intensity, confidence level, recommendations.

#### 2. `estimate_latency` — Roofline-based latency prediction

```json
{
  "name": "estimate_latency",
  "description": "Predict inference latency using roofline model with calibration data",
  "input_schema": {
    "type": "object",
    "properties": {
      "model_name": { "type": "string" },
      "hardware_name": { "type": "string" },
      "batch_size": { "type": "integer", "default": 1 },
      "precision": { "type": "string", "default": "fp16" },
      "thermal_profile": { "type": "string", "enum": ["nominal", "throttled", "boosted"], "default": "nominal" }
    },
    "required": ["model_name", "hardware_name"]
  }
}
```

**Returns:** Compute time, memory time, total latency, bottleneck, per-subgraph breakdown,
confidence (CALIBRATED/INTERPOLATED/THEORETICAL).

#### 3. `estimate_energy` — Power and energy breakdown

```json
{
  "name": "estimate_energy",
  "description": "Estimate energy consumption with component-wise breakdown",
  "input_schema": {
    "type": "object",
    "properties": {
      "model_name": { "type": "string" },
      "hardware_name": { "type": "string" },
      "batch_size": { "type": "integer", "default": 1 },
      "precision": { "type": "string", "default": "fp16" },
      "power_gating_enabled": { "type": "boolean", "default": false }
    },
    "required": ["model_name", "hardware_name"]
  }
}
```

**Returns:** Compute energy, memory energy, static (leakage) energy, total energy,
efficiency ratio, wasted energy.

#### 4. `estimate_memory` — Memory footprint analysis

```json
{
  "name": "estimate_memory",
  "description": "Analyze peak memory usage, activation timeline, and reuse patterns",
  "input_schema": {
    "type": "object",
    "properties": {
      "model_name": { "type": "string" },
      "batch_size": { "type": "integer", "default": 1 },
      "precision": { "type": "string", "default": "fp16" }
    },
    "required": ["model_name"]
  }
}
```

**Returns:** Peak memory, weight memory, activation memory, workspace, timeline,
reuse patterns, fits-in-SRAM verdict.

#### 5. `compare_hardware` — Multi-target ranking

```json
{
  "name": "compare_hardware",
  "description": "Compare a model's performance across multiple hardware targets",
  "input_schema": {
    "type": "object",
    "properties": {
      "model_name": { "type": "string" },
      "hardware_list": { "type": "array", "items": { "type": "string" }, "description": "List of hardware IDs to compare" },
      "batch_size": { "type": "integer", "default": 1 },
      "precision": { "type": "string", "default": "fp16" },
      "sort_by": { "type": "string", "enum": ["latency", "energy", "efficiency"], "default": "latency" }
    },
    "required": ["model_name", "hardware_list"]
  }
}
```

**Returns:** Ranked comparison table with latency, energy, memory, utilization,
confidence per hardware target.

#### 6. `list_hardware` — Hardware catalog discovery

```json
{
  "name": "list_hardware",
  "description": "List available hardware targets, optionally filtered by type",
  "input_schema": {
    "type": "object",
    "properties": {
      "device_type": { "type": "string", "enum": ["cpu", "gpu", "dsp", "tpu", "kpu", "accelerator"], "description": "Filter by device category" },
      "query": { "type": "string", "description": "Fuzzy search query (e.g., 'jetson', 'orin')" }
    }
  }
}
```

**Returns:** List of hardware IDs with summary specs (peak FLOPS, bandwidth, TDP, memory).

#### 7. `get_hardware_specs` — Detailed hardware profile

```json
{
  "name": "get_hardware_specs",
  "description": "Get detailed specifications for a specific hardware target",
  "input_schema": {
    "type": "object",
    "properties": {
      "hardware_id": { "type": "string" }
    },
    "required": ["hardware_id"]
  }
}
```

**Returns:** Full profile — peak FLOPS by precision, memory bandwidth, memory capacity,
power envelope, thermal operating points, supported precisions, calibration status.

---

## Authentication

### Option A: Bearer Token (recommended for dev/staging)

Simple, works with existing infrastructure, easy to rotate.

```python
# graphs/src/graphs/mcp/auth.py
import os
import hmac

def validate_request(authorization_header: str) -> bool:
    """Validate Bearer token against GRAPHS_MCP_TOKEN env var."""
    expected = os.environ.get("GRAPHS_MCP_TOKEN")
    if not expected:
        raise RuntimeError("GRAPHS_MCP_TOKEN not set")
    if not authorization_header.startswith("Bearer "):
        return False
    token = authorization_header[7:]
    return hmac.compare_digest(token, expected)
```

### Option B: mTLS (recommended for production)

Client and server certificates — strongest isolation, no shared secrets in env vars.

```python
# Transport config for mTLS
ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain("server.crt", "server.key")
ssl_context.load_verify_locations("ca.crt")
ssl_context.verify_mode = ssl.CERT_REQUIRED
```

### Recommendation

Start with **Bearer token** for local/dev use. Add mTLS when deploying to
shared infrastructure or CI.

---

## Transport: SSE vs Streamable HTTP

| Transport | Pros | Cons |
|-----------|------|------|
| **stdio** | Zero config, no auth needed | Local only — can't serve remote clients |
| **SSE** | Streaming results, well-supported by MCP SDK | Unidirectional, needs HTTP wrapper for auth |
| **Streamable HTTP** | Full bidirectional, native auth headers | Newer MCP transport, less battle-tested |

**Recommendation:** Use **SSE over HTTP** with a thin Starlette/uvicorn wrapper
that validates the Bearer token before upgrading to SSE. This is the most
compatible option with Claude Code's MCP client.

```python
# graphs/src/graphs/mcp/transport.py
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
from mcp.server.sse import SseServerTransport
from graphs.mcp.server import create_server
from graphs.mcp.auth import validate_request

sse = SseServerTransport("/messages")

async def handle_sse(request):
    auth = request.headers.get("Authorization", "")
    if not validate_request(auth):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await create_server().run(
            streams[0], streams[1], create_server().create_initialization_options()
        )

app = Starlette(routes=[
    Route("/sse", endpoint=handle_sse),
    Route("/messages", endpoint=sse.handle_post_message, methods=["POST"]),
])

# Run: uvicorn graphs.mcp.transport:app --host 0.0.0.0 --port 8100
```

---

## Client Configuration

### In this repo: `.claude/settings.local.json`

Add the graphs MCP server alongside the existing branes server:

```jsonc
{
  "mcpServers": {
    "branes": {
      "command": ".venv/bin/python",
      "args": ["-m", "embodied_ai_architect.mcp.server"]
      // Local stdio — no auth needed
    },
    "graphs": {
      "url": "http://localhost:8100/sse",
      "headers": {
        "Authorization": "Bearer ${GRAPHS_MCP_TOKEN}"
      }
    }
  }
}
```

For remote deployment:

```jsonc
{
  "mcpServers": {
    "graphs": {
      "url": "https://graphs.internal.branes.ai/sse",
      "headers": {
        "Authorization": "Bearer ${GRAPHS_MCP_TOKEN}"
      }
    }
  }
}
```

### Environment Variable

```bash
# .env or shell profile
export GRAPHS_MCP_TOKEN="<generate-with-openssl-rand-hex-32>"
```

---

## Server Implementation Skeleton

```python
# graphs/src/graphs/mcp/server.py
"""MCP server exposing graphs quantitative estimators."""

from mcp.server import Server
from mcp.types import Tool, TextContent

from graphs.estimation.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.hardware.registry.registry import HardwareRegistry

server = Server("graphs")
registry = HardwareRegistry()


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="analyze_model",
            description="Run unified roofline + energy + memory analysis",
            inputSchema={...},  # see tool definitions above
        ),
        Tool(name="estimate_latency", ...),
        Tool(name="estimate_energy", ...),
        Tool(name="estimate_memory", ...),
        Tool(name="compare_hardware", ...),
        Tool(name="list_hardware", ...),
        Tool(name="get_hardware_specs", ...),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    match name:
        case "analyze_model":
            return await _analyze_model(**arguments)
        case "estimate_latency":
            return await _estimate_latency(**arguments)
        case "estimate_energy":
            return await _estimate_energy(**arguments)
        case "estimate_memory":
            return await _estimate_memory(**arguments)
        case "compare_hardware":
            return await _compare_hardware(**arguments)
        case "list_hardware":
            return await _list_hardware(**arguments)
        case "get_hardware_specs":
            return await _get_hardware_specs(**arguments)
        case _:
            raise ValueError(f"Unknown tool: {name}")


async def _analyze_model(
    model_name: str,
    hardware_name: str,
    batch_size: int = 1,
    precision: str = "fp16",
) -> list[TextContent]:
    config = AnalysisConfig()
    analyzer = UnifiedAnalyzer(config)
    result = analyzer.analyze_model(model_name, hardware_name, batch_size, precision)
    return [TextContent(type="text", text=result.to_json())]


async def _list_hardware(
    device_type: str | None = None,
    query: str | None = None,
) -> list[TextContent]:
    if query:
        results = registry.search(query)
    elif device_type:
        results = registry.list_by_type(device_type)
    else:
        results = registry.list_all()
    return [TextContent(type="text", text=json.dumps([r.summary() for r in results]))]


# ... remaining tool handlers follow same pattern


def create_server() -> Server:
    return server
```

---

## What Changes in This Repo

Once the graphs MCP server is running, this repo can:

1. **Add the client config** to `.claude/settings.local.json` (the `"graphs"` entry above)
2. **Optionally deprecate `llm/graphs_tools.py`** — it currently wraps `UnifiedAnalyzer`
   via direct Python imports. With MCP, Claude calls the tools natively, no wrapper needed.
3. **Remove `graphs` from install dependencies** if all access goes through MCP —
   the proprietary code stays in its repo, served only over authenticated MCP.

---

## Deployment Modes

| Mode | Transport | Auth | Use case |
|------|-----------|------|----------|
| **Local dev** | stdio | None (process-local) | Solo developer, graphs repo checked out locally |
| **Team dev** | SSE over HTTP | Bearer token | Shared dev server, team members access same instance |
| **CI/CD** | SSE over HTTP | Bearer token (from secrets) | Automated testing against graphs estimators |
| **Production** | SSE over HTTPS | mTLS + Bearer | Deployed service, customer-facing |

### Local dev shortcut (no auth needed)

If the developer has `../graphs` checked out locally, they can run the server
via stdio with no auth, just like the branes MCP server:

```jsonc
{
  "mcpServers": {
    "graphs": {
      "command": "/home/user/dev/branes/clones/graphs/.venv/bin/python",
      "args": ["-m", "graphs.mcp.transport"],
      "env": {}
    }
  }
}
```

This is the fastest path to getting it working — no tokens, no HTTP.
Add the auth layer when you need remote access.

---

## Implementation Checklist (in `../graphs` repo)

- [x] Create `src/graphs/mcp/__init__.py`
- [x] Create `src/graphs/mcp/server.py` with 7 tool definitions
- [x] Create `src/graphs/mcp/auth.py` with Bearer token validation
- [x] Create `src/graphs/mcp/transport.py` with stdio + SSE + auth middleware
- [x] Add `mcp` optional dependency to `pyproject.toml`
- [x] Add `__main__.py` entry point for `python -m graphs.mcp`
- [x] Write tests: `tests/test_mcp_server.py` (8 tests, all passing)
- [x] Add this repo's client config to `.claude/settings.local.json`
- [ ] Document token generation in `../graphs/README.md`
