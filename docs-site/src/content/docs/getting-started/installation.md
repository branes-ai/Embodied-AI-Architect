---
title: Installation
description: How to install Embodied AI Architect and its dependencies.
---

## Requirements

- Python 3.11 or higher
- pip or uv package manager

## Basic Installation

Install the core package:

```bash
pip install embodied-ai-architect
```

Or with [uv](https://github.com/astral-sh/uv) for faster installation:

```bash
uv pip install embodied-ai-architect
```

## Installation Options

### Development Mode

For development with all tools:

```bash
git clone https://github.com/branes-ai/embodied-ai-architect.git
cd embodied-ai-architect
pip install -e ".[dev]"
```

### With Remote Backends

For SSH and Kubernetes benchmark backends:

```bash
pip install -e ".[dev,remote,kubernetes]"
```

### With Interactive Chat

For the Claude-powered interactive assistant:

```bash
pip install -e ".[chat]"
```

This requires an Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your-key-here
```

## Related Packages

For full functionality, install the companion packages:

### embodied-schemas

Shared data models and hardware catalog:

```bash
pip install embodied-schemas
# Or from source:
pip install -e ../embodied-schemas
```

### graphs

Roofline analysis and hardware simulation:

```bash
pip install graphs
# Or from source:
pip install -e ../graphs
```

## Verify Installation

Check that everything is installed correctly:

```bash
branes --version
branes --help
```

You should see:

```
Usage: branes [OPTIONS] COMMAND [ARGS]...

  Embodied AI Architect - Design and deploy AI for the physical world

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  analyze    Analyze a model's structure and characteristics
  benchmark  Benchmark model performance on target backends
  chat       Start interactive AI architect session
  workflow   Run the full analysis workflow
  ...
```

## Next Steps

- [Run through the quickstart](/getting-started/quickstart/)
- [Learn about model analysis](/features/model-analysis/)
