# Interactive Chat: Claude Code-Style Agent Interface

The Embodied AI Architect includes an interactive chat interface that allows you to converse with an AI agent capable of analyzing models, recommending hardware, and running benchmarks—similar to how Claude Code works.

## Quick Start

```bash
# Install with chat support
pip install -e ".[chat]"

# Set your Anthropic API key
export ANTHROPIC_API_KEY=your-key-here

# Start the interactive session
branes chat
```

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
│  │   Tools (wrapping agents)        │    │
│  │   • analyze_model                │    │
│  │   • recommend_hardware           │    │
│  │   • run_benchmark                │    │
│  │   • list_files / read_file       │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
    ↓
Final Response (Markdown)
```

## Components

### LLM Client (`llm/client.py`)

Wrapper for the Claude API with tool use support:

```python
from embodied_ai_architect.llm import LLMClient

client = LLMClient(model="claude-sonnet-4-20250514")
response = client.chat(
    messages=[{"role": "user", "content": "Hello"}],
    tools=[...],  # Tool definitions
    system="You are an AI architect...",
)
```

Features:
- Automatic API key detection from `ANTHROPIC_API_KEY`
- Tool call parsing into structured `ToolCall` objects
- Streaming support via `stream_chat()`

### Tool Definitions (`llm/tools.py`)

Tools are defined in Anthropic's tool format and wrap existing agents:

| Tool | Description | Wraps |
|------|-------------|-------|
| `analyze_model` | Analyze PyTorch model structure | `ModelAnalyzerAgent` |
| `recommend_hardware` | Get hardware recommendations | `HardwareProfileAgent` |
| `run_benchmark` | Benchmark on CPU/SSH/K8s | `BenchmarkAgent` |
| `list_files` | Explore directories | Built-in |
| `read_file` | Read text files | Built-in |
| `scan_project` | Scan project directory for files, build system, ML models | `CodebaseScanner` |
| `analyze_codebase` | Multi-pass LLM analysis of application source code | `CodeAnalyzer` |
| `assess_codebase_on_hardware` | End-to-end codebase → hardware assessment | Full pipeline |

Adding a new tool:

```python
# 1. Add definition to get_tool_definitions()
{
    "name": "my_tool",
    "description": "What this tool does",
    "input_schema": {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "..."},
        },
        "required": ["param1"],
    },
}

# 2. Add executor to create_tool_executors()
def my_tool(param1: str) -> str:
    # Execute and return result as string
    return json.dumps(result)

return {
    ...,
    "my_tool": my_tool,
}
```

### Architect Agent (`llm/agent.py`)

The agentic loop that orchestrates tool use:

```python
from embodied_ai_architect.llm import ArchitectAgent

agent = ArchitectAgent(verbose=True)

# Single turn
response = agent.run("Analyze models/yolov8n.pt")

# Conversation continues with context
response = agent.run("What hardware can run it at 30fps?")

# Reset conversation
agent.reset()
```

Callbacks for UI integration:

```python
agent.run(
    user_input,
    on_tool_start=lambda name, args: print(f"Running {name}..."),
    on_tool_end=lambda name, result: print(f"Done: {result[:50]}"),
    on_thinking=lambda text: print(f"Thinking: {text}"),
)
```

### Chat CLI (`cli/commands/chat.py`)

Interactive REPL with Rich formatting:

```
╭───────────────────────╮
│ Embodied AI Architect │
│ Version 0.1.0         │
╰───────────────────────╯

You: Analyze the model in ./models/resnet18.pt
  ▶ analyze_model(model_path=./models/resnet18.pt)
  ✓ {"model_type": "ResNet", "total_parameters": 11689512...

Architect:
The ResNet-18 model has **11.7M parameters** with the following structure:
- 17 convolutional layers
- 1 fully connected layer
- Estimated memory: 44.6 MB (FP32)
...
```

Commands in session:
- `exit` / `quit` - End session
- `reset` - Clear conversation history
- `help` - Show help

## Example Conversations

### Basic Model Analysis

```
You: List the .pt files in ./models

Architect: I found 3 model files:
- yolov8n.pt (6.2 MB)
- resnet18.pt (44.7 MB)
- mobilenet_v2.pt (13.6 MB)

You: Analyze yolov8n.pt

Architect: YOLOv8n has:
- 3.2M parameters
- 225 layers (Conv2d: 65, BatchNorm2d: 65, ...)
- Estimated memory: 12.2 MB
- Primary operations: convolution, batch normalization
```

### Hardware Recommendation

```
You: I need to deploy YOLOv8n on a drone with a 10W power budget.
     What hardware should I use?

Architect: Based on your constraints (10W power, drone deployment),
here are my recommendations:

1. **NVIDIA Jetson Orin Nano** (Score: 89/100)
   - 40 TOPS INT8, 7-15W TDP
   - Est. latency: 8ms (125 fps)
   - Cost: ~$199

2. **Google Coral Edge TPU** (Score: 82/100)
   - 4 TOPS INT8, 2W TDP
   - Est. latency: 15ms (66 fps)
   - Cost: ~$60
   - ⚠️ Requires INT8 quantization

3. **Hailo-8** (Score: 78/100)
   ...
```

### Benchmarking

```
You: Benchmark YOLOv8n on my local CPU

Architect: Running benchmark with 100 iterations...

Results:
- Mean latency: 45.3 ms
- Std deviation: 2.1 ms
- Min/Max: 42.1 / 51.8 ms
- Throughput: 22.1 samples/sec

This is about 5x slower than the estimated Jetson Orin Nano performance.
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | API key for Claude | Required |

### CLI Options

```bash
branes chat --help

Options:
  --model TEXT   Claude model to use [default: claude-sonnet-4-20250514]
  -v, --verbose  Show detailed tool execution info
```

## Extending the Agent

### Adding Domain-Specific Tools

To add tools for your specific use case (e.g., simulation, deployment):

1. Define the tool schema in `llm/tools.py`
2. Implement the executor function
3. The agent will automatically have access to it

### Customizing the System Prompt

Edit `SYSTEM_PROMPT` in `llm/agent.py` to change the agent's behavior:

```python
SYSTEM_PROMPT = """
You are an Embodied AI Architect specializing in drone perception systems.
When recommending hardware, prioritize:
1. Power efficiency (drones have limited battery)
2. Weight (affects flight time)
3. Real-time performance (safety critical)
...
"""
```

### Programmatic Usage

Use the agent in your own scripts:

```python
from embodied_ai_architect.llm import ArchitectAgent

agent = ArchitectAgent()

# Analyze a model
result = agent.run("Analyze models/yolov8n.pt and recommend hardware for a 5W budget")
print(result)

# Continue the conversation
result = agent.run("What if I increase the budget to 15W?")
print(result)
```

## branes-ai/graphs Integration

The chat agent integrates with the [branes-ai/graphs](https://github.com/branes-ai/graphs) repository for detailed model characterization using roofline analysis. This provides more accurate performance predictions than basic analysis.

### Additional Tools (when graphs is installed)

| Tool | Description |
|------|-------------|
| `analyze_model_detailed` | Roofline-based analysis with utilization metrics |
| `compare_hardware_targets` | Compare model across multiple hardware targets |
| `identify_bottleneck` | Determine if model is compute or memory bound |
| `list_available_hardware` | List 30+ hardware targets by category |
| `estimate_power_consumption` | Energy and power estimation |

### Example: Detailed Analysis

```
You: Analyze resnet18 on Jetson Orin Nano with detailed metrics

Architect: [calls analyze_model_detailed]

ResNet-18 on Jetson-Orin-Nano-8GB:

**Performance Metrics:**
- Latency: 2.72 ms (367.9 FPS)
- Energy: 14.73 mJ per inference
- Peak Memory: 6.1 MB

**Bottleneck Analysis:**
- Type: MEMORY-BOUND
- 56 of 69 operators are memory-limited
- Compute utilization: 22.5%
- Memory utilization: 60.0%

This model spends most time waiting for data transfers.
Consider FP16 quantization to reduce bandwidth requirements.
```

### Hardware Categories

```
You: List available edge hardware

Architect: [calls list_available_hardware(category="edge_gpu")]

Edge GPU targets:
- Jetson-Orin-AGX
- Jetson-Orin-Nano
```

## Roadmap

- [ ] **Simulation tools** - Connect to AirSim/Branes CSim for validation
- [ ] **Constraint parsing** - Natural language to structured constraints
- [ ] **Failure interpretation** - Parse logs and explain errors
- [ ] **Streaming responses** - Show text as it generates
- [x] **Graph analysis tools** - Integration with branes-ai/graphs
