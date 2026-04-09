# Session: Interactive Chat Agent Implementation

**Date**: 2025-12-16
**Focus**: Building Claude Code-style interactive chat interface with branes-ai/graphs integration

## Summary

Implemented a conversational AI agent interface for the Embodied AI Architect, enabling users to interact with the system through natural language. The agent can analyze models, recommend hardware, run benchmarks, and provide detailed roofline-based performance analysis.

## What Was Built

### 1. LLM Integration Layer (`src/embodied_ai_architect/llm/`)

Created a complete LLM integration module with:

- **`client.py`**: Claude API wrapper with tool use support
  - Automatic API key detection from environment
  - Tool call parsing into structured objects
  - Streaming support for responsive UI

- **`tools.py`**: Base tool definitions wrapping existing agents
  - `analyze_model`: PyTorch model structure analysis
  - `recommend_hardware`: Hardware recommendations with constraints
  - `run_benchmark`: Performance benchmarking on backends
  - `list_files`, `read_file`: File exploration

- **`agent.py`**: Agentic loop implementation
  - Iterative tool use until task completion
  - Conversation history management
  - Callbacks for UI integration (on_tool_start, on_tool_end, on_thinking)
  - Domain-specific system prompt

### 2. branes-ai/graphs Integration (`graphs_tools.py`)

Integrated 5 powerful tools from the graphs repository:

| Tool | Description |
|------|-------------|
| `analyze_model_detailed` | Roofline analysis with latency, energy, memory, utilization |
| `compare_hardware_targets` | Compare model across 6+ hardware targets, ranked by speed/efficiency |
| `identify_bottleneck` | Compute vs memory bound classification + recommendations |
| `list_available_hardware` | 30+ hardware targets in 7 categories |
| `estimate_power_consumption` | Power/energy breakdown for deployment planning |

Hardware categories:
- Datacenter GPU: H100, A100, V100, L4, T4
- Edge GPU: Jetson Orin AGX, Jetson Orin Nano
- Datacenter CPU: Intel Xeon, AMD EPYC
- TPU: TPU v4, Coral Edge TPU
- Accelerators: KPU, Hailo-8
- Automotive: TDA4VM, TDA4VL

### 3. Interactive CLI (`cli/commands/chat.py`)

Built Rich-based terminal interface:
- Welcome panel with commands
- Tool execution visibility (shows what's being called)
- Success/error indicators for tool results
- Markdown-formatted responses
- Session commands: exit, reset, help

### 4. Documentation (`docs/interactive-chat.md`)

Comprehensive documentation covering:
- Quick start guide
- Architecture diagram
- Component descriptions
- Example conversations
- branes-ai/graphs integration details
- Programmatic usage examples

## Architecture

```
User Input
    ↓
┌─────────────────────────────────────────┐
│           ArchitectAgent                 │
│  ┌─────────────────────────────────┐    │
│  │   Agentic Loop                   │    │
│  │   1. Send to Claude              │    │
│  │   2. If tool_calls → execute     │    │
│  │   3. Add results to context      │    │
│  │   4. Repeat until done           │    │
│  └─────────────────────────────────┘    │
│                 ↓                        │
│  ┌─────────────────────────────────┐    │
│  │   Tools (10 total)               │    │
│  │   Base: analyze, recommend,      │    │
│  │         benchmark, files         │    │
│  │   Graphs: detailed analysis,     │    │
│  │          compare, bottleneck     │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
    ↓
Final Response (Markdown)
```

## Example Usage

```bash
# Install with chat support
pip install -e ".[chat]"

# Set API key
export ANTHROPIC_API_KEY=your-key

# Start interactive session
branes chat
```

Example conversation:
```
You: Compare resnet18 across edge devices

Architect: [calls compare_hardware_targets]

Comparison for ResNet-18:

| Hardware           | Latency | FPS   | Bottleneck    |
|-------------------|---------|-------|---------------|
| Jetson-Orin-AGX   | 0.89 ms | 1124  | memory_bound  |
| Jetson-Orin-Nano  | 2.72 ms | 368   | memory_bound  |
| Coral-Edge-TPU    | 5.1 ms  | 196   | compute_bound |

Fastest: Jetson-Orin-AGX
Most Efficient: Coral-Edge-TPU
```

## Files Created/Modified

### New Files
- `src/embodied_ai_architect/llm/__init__.py`
- `src/embodied_ai_architect/llm/client.py`
- `src/embodied_ai_architect/llm/tools.py`
- `src/embodied_ai_architect/llm/agent.py`
- `src/embodied_ai_architect/llm/graphs_tools.py`
- `src/embodied_ai_architect/cli/commands/chat.py`
- `docs/interactive-chat.md`

### Modified Files
- `src/embodied_ai_architect/cli/__init__.py` - Added chat command
- `pyproject.toml` - Added `[chat]` and `[all]` optional dependencies
- `CLAUDE.md` - Updated with chat feature documentation

## Technical Decisions

1. **Optional Dependency**: anthropic package is optional (`[chat]` extra) to keep base install lightweight

2. **Graceful Degradation**: graphs tools only load if graphs package is installed; base tools always available

3. **Roofline-Based Bottleneck**: Used `num_compute_bound` vs `num_memory_bound` operator counts to determine dominant bottleneck (not a single value)

4. **Tool Result Format**: All tools return JSON strings for consistent LLM parsing

5. **Conversation Persistence**: Messages stored in agent for multi-turn conversations; reset() clears history

## MVP Roadmap Progress

This implements a key Phase 2.2 MVP requirement: Natural Language Interface

- [x] LLM-based orchestrator agent
- [x] Tool use for model analysis
- [x] Hardware recommendation integration
- [x] Bottleneck identification with recommendations
- [ ] Simulation tools (AirSim/Branes CSim) - future
- [ ] Constraint parsing from natural language - future
- [ ] Failure interpretation (log analysis) - future

## Next Steps

1. **Simulation Integration**: Add tools to run models in AirSim/Branes CSim
2. **Constraint Parsing**: "under 10W, faster than 30ms" → structured constraints
3. **Streaming Responses**: Show text as it generates for better UX
4. **Failure Interpretation**: Parse logs and explain what went wrong
5. **Multi-model Workflows**: Analyze entire perception pipelines, not just single models

## Lessons Learned

1. **API Mismatch**: The graphs `RooflineReport` structure differs from documentation - needed to inspect actual attributes at runtime

2. **Attribute Names**: `average_utilization_pct` not `hardware_utilization`, `average_flops_utilization` not `flops_utilization`

3. **Bottleneck Aggregation**: No single `dominant_bottleneck` attribute - must compute from per-operator counts

4. **Tool Testing**: Essential to test tools independently before integrating into agent loop
