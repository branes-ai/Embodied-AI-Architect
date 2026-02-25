# Embodied AI Architect

A design environment for creating and evaluating autonomous agents, with hardware/software codesign space exploration and optimization.

## Features

- **Model Analysis**: Analyze PyTorch model structure and compute requirements
- **Hardware Profiling**: Recommendations for edge/cloud deployment
- **Multi-Hardware Benchmarking**: Local CPU, remote SSH, Kubernetes backends
- **Interactive Chat**: Claude-powered architect for design decisions
- **Codebase Analysis**: Scan and assess application codebases for hardware deployment
- **SoC Optimization**: LangGraph-based RTL optimization loop (experimental)

## Installation

```bash
pip install embodied-ai-architect
```

With optional dependencies:

```bash
# Remote SSH benchmarking
pip install embodied-ai-architect[remote]

# Kubernetes benchmarking
pip install embodied-ai-architect[kubernetes]

# Interactive chat (requires ANTHROPIC_API_KEY)
pip install embodied-ai-architect[chat]

# All optional dependencies
pip install embodied-ai-architect[all]
```

## Usage

```bash
# Show available commands
branes --help

# Analyze a PyTorch model
branes analyze model.pt

# Run full workflow
branes workflow run model.pt

# Benchmark on local CPU
branes benchmark model.pt --backend local

# Scan and assess a codebase for hardware deployment
branes codebase scan /path/to/project
branes codebase assess /path/to/project --hardware jetson_orin

# Interactive chat session
export ANTHROPIC_API_KEY=your-key-here
branes chat
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Required for Claude-powered features (chat, codebase analysis) |

## Documentation

For full documentation, development setup, and contributing guidelines, visit the
[GitHub repository](https://github.com/branes-ai/embodied-ai-architect).

## Related Projects

- [embodied-schemas](https://github.com/branes-ai/embodied-schemas): Shared Pydantic schemas and hardware catalog
- [graphs](https://github.com/branes-ai/graphs): Analysis tools and roofline models

## License

MIT License
