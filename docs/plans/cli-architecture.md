# CLI Architecture - Human-Friendly Interface

**Date**: 2025-11-02
**Status**: Design Proposal

## Overview

The CLI provides a human-friendly interface to the Embodied AI Architect system, similar to Claude Code's excellent CLI experience.

## Design Principles

1. **Intuitive**: Commands follow natural language patterns
2. **Discoverable**: Excellent help text and examples
3. **Progressive Disclosure**: Simple by default, powerful when needed
4. **Informative**: Rich output with progress indicators
5. **Safe**: Confirm destructive operations
6. **Scriptable**: Easy to use in CI/CD pipelines

## Command Structure

```
embodied-ai
├── workflow        # Run complete workflows
│   ├── run         # Execute full workflow
│   └── list        # List past workflows
├── analyze         # Model analysis only
├── benchmark       # Benchmarking only
│   ├── run         # Run benchmark
│   ├── list        # List backends
│   └── parallel    # Run parallel benchmarks
├── report          # Report operations
│   ├── view        # View report in browser
│   ├── list        # List all reports
│   ├── compare     # Compare two reports
│   └── export      # Export report (PDF, etc.)
├── config          # Configuration management
│   ├── init        # Initialize configuration
│   ├── show        # Show current config
│   ├── set         # Set configuration value
│   └── validate    # Validate configuration
├── backends        # Backend management
│   ├── list        # List available backends
│   ├── test        # Test backend connection
│   └── add         # Add new backend
├── secrets         # Secrets management
│   ├── list        # List available secrets (keys only!)
│   ├── set         # Set a secret
│   └── validate    # Validate secrets setup
└── init            # Initialize new project
```

## Example Commands

### Quick Start

```bash
# Initialize project
branes init my-project
cd my-project

# Run complete workflow
branes workflow run my_model.pt

# View last report
branes report view --latest
```

### Model Analysis

```bash
# Analyze model
branes analyze my_model.pt

# With custom input shape
branes analyze my_model.pt --input-shape 1,3,224,224

# Output as JSON
branes analyze my_model.pt --json
```

### Benchmarking

```bash
# Benchmark on local CPU
branes benchmark my_model.pt

# Benchmark on specific backend
branes benchmark my_model.pt --backend kubernetes

# Parallel benchmarking
branes benchmark parallel \
  --models model1.pt,model2.pt,model3.pt \
  --backend kubernetes

# GPU comparison
branes benchmark my_model.pt \
  --backends kubernetes-v100,kubernetes-a100,kubernetes-t4
```

### Reports

```bash
# View latest report in browser
branes report view --latest

# View specific report
branes report view abc123

# List all reports
branes report list

# Compare two reports
branes report compare abc123 def456

# Export to PDF
branes report export abc123 --format pdf
```

### Configuration

```bash
# Initialize configuration
branes config init

# Show current configuration
branes config show

# Set value
branes config set backends.kubernetes.namespace embodied-ai-prod

# Validate configuration
branes config validate
```

### Backends

```bash
# List available backends
branes backends list

# Test backend connection
branes backends test kubernetes

# Add new SSH backend
branes backends add ssh \
  --name my-gpu-server \
  --host gpu.example.com \
  --user benchmark
```

### Secrets

```bash
# List available secrets (keys only, not values!)
branes secrets list

# Set secret interactively (secure input)
branes secrets set ssh_key

# Validate secrets setup
branes secrets validate
```

## Interactive Mode

```bash
# Start interactive mode
embodied-ai interactive

# Interactive prompt
embodied-ai> workflow run my_model.pt
Running complete workflow...
✓ Model Analysis complete
✓ Hardware Profiling complete
✓ Benchmarking complete
✓ Report generated: reports/abc123/report.html

embodied-ai> report view abc123
Opening report in browser...

embodied-ai> exit
```

## Output Formatting

### Rich Output (Default)

```
╭─────────────────────────────────────────────────────╮
│  Embodied AI Architect                              │
│  Workflow: abc123                                   │
╰─────────────────────────────────────────────────────╯

📊 Model Analysis
  Model: ResNet50
  Parameters: 25.6M
  Layers: 177
  ✓ Complete (0.5s)

🖥️  Hardware Profiling
  Evaluated: 8 hardware options
  Top recommendation: NVIDIA Jetson AGX Orin (score: 91.4)
  ✓ Complete (1.2s)

⚡ Benchmarking
  Backend: kubernetes
  Mean Latency: 2.34ms
  Throughput: 427 samples/sec
  ✓ Complete (45.3s)

📄 Report Generation
  Report: reports/abc123/report.html
  ✓ Complete (2.1s)

Total time: 49.1s
View report: branes report view abc123
```

### JSON Output (for scripting)

```bash
branes workflow run my_model.pt --json

{
  "workflow_id": "abc123",
  "status": "completed",
  "duration_seconds": 49.1,
  "model_analysis": {...},
  "hardware_recommendations": [...],
  "benchmarks": {...},
  "report_path": "reports/abc123/report.html"
}
```

### Quiet Mode (minimal output)

```bash
branes workflow run my_model.pt --quiet

reports/abc123/report.html
```

## Progress Indicators

```bash
# Spinner for quick operations
⠋ Analyzing model...

# Progress bar for longer operations
Benchmarking ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 100/100 0:00:45

# Multi-step progress
[1/4] Model Analysis    ✓
[2/4] Hardware Profile  ⠋ Running...
[3/4] Benchmarking      ⏸ Waiting
[4/4] Report Generation ⏸ Waiting
```

## Error Handling

```bash
$ branes workflow run nonexistent.pt

❌ Error: Model file not found
   Path: nonexistent.pt

💡 Tip: Check the file path or use --help for examples

$ branes benchmark my_model.pt --backend kubernetes

❌ Error: Backend connection failed
   Backend: kubernetes
   Reason: Kubeconfig not found

💡 Fix:
   1. Configure kubeconfig: branes config set secrets.k8s_kubeconfig /path/to/config
   2. Or set environment variable: export EMBODIED_AI_K8S_KUBECONFIG=/path/to/config
   3. Test connection: branes backends test kubernetes
```

## Configuration File

```yaml
# .embodied-ai/config.yaml
version: "1.0"

# Default backend for benchmarking
default_backend: local_cpu

# Backends configuration
backends:
  kubernetes:
    namespace: embodied-ai
    image: embodied-ai-benchmark:latest
    cpu_request: "2"
    memory_request: "4Gi"

  ssh_remote:
    host: gpu-server.example.com
    port: 22
    user: benchmark

# Report settings
reports:
  auto_open: true  # Open in browser after generation
  format: html

# Workflow settings
workflow:
  default_iterations: 100
  default_warmup: 10
  auto_cleanup: true
```

## Shell Completion

```bash
# Install completion
embodied-ai --install-completion bash

# Now tab completion works
embodied-ai work<TAB>
branes workflow <TAB>
  run   list

branes benchmark --backend <TAB>
  local_cpu  remote_ssh  kubernetes
```

## CI/CD Integration

```yaml
# .github/workflows/benchmark.yml
- name: Run benchmark
  run: |
    branes workflow run model.pt --json > results.json

- name: Check performance
  run: |
    LATENCY=$(jq '.benchmarks.local_cpu.mean_latency_ms' results.json)
    if (( $(echo "$LATENCY > 10" | bc -l) )); then
      echo "Performance regression!"
      exit 1
    fi
```

## Watch Mode

```bash
# Re-run on file changes
branes workflow run my_model.pt --watch

Watching for changes to my_model.pt...
Press Ctrl+C to stop

[12:30:15] File changed, re-running workflow...
[12:30:45] ✓ Complete
[12:30:45] Waiting for changes...
```

## Aliases

```bash
# Common aliases
embodied-ai wf      # workflow
embodied-ai bench   # benchmark
embodied-ai cfg     # config

# Can be configured
branes config set alias.run "workflow run"
embodied-ai run my_model.pt  # Equivalent to: branes workflow run
```

## Plugin System

```bash
# Install plugin
embodied-ai plugin install embodied-ai-plugin-tensorrt

# Now new commands available
embodied-ai tensorrt optimize my_model.pt
```

## Verbose Modes

```bash
# Normal
branes workflow run my_model.pt

# Verbose (show agent details)
branes workflow run my_model.pt -v

# Very verbose (show all debug info)
branes workflow run my_model.pt -vv

# Debug mode (show everything including internal state)
branes workflow run my_model.pt --debug
```

## Comparison with Claude Code CLI

| Feature | Claude Code | Embodied AI |
|---------|-------------|-------------|
| Interactive mode | ✅ | ✅ |
| Rich output | ✅ | ✅ |
| Progress indicators | ✅ | ✅ |
| Shell completion | ✅ | ✅ |
| JSON output | ✅ | ✅ |
| Good error messages | ✅ | ✅ |
| Configuration mgmt | ✅ | ✅ |
| Help text | ✅ | ✅ |

## Implementation

Framework: **Click** (industry standard Python CLI framework)
Enhancements:
- **rich**: Beautiful terminal output
- **click-completion**: Shell completion
- **inquirer**: Interactive prompts
- **tqdm**: Progress bars

## Entry Point

```bash
# After pip install
embodied-ai --help

# Or via python -m
python -m embodied_ai_architect.cli --help
```

## Examples in Help Text

```bash
$ branes workflow --help

Usage: branes workflow [OPTIONS] COMMAND [ARGS]...

  Run complete workflows for model evaluation.

Commands:
  run   Run complete workflow on a model
  list  List past workflow executions

Examples:
  # Run on local CPU
  branes workflow run my_model.pt

  # Use Kubernetes backend
  branes workflow run my_model.pt --backend kubernetes

  # Custom constraints
  branes workflow run my_model.pt \
    --max-latency 50 \
    --max-power 100 \
    --max-cost 3000

  # Output as JSON for scripting
  branes workflow run my_model.pt --json > results.json
```

## Future Enhancements

1. **TUI (Text User Interface)**: Full-screen interactive interface
2. **Remote Control**: Control remote agents
3. **Scheduling**: Schedule periodic benchmarks
4. **Notifications**: Slack/email when workflows complete
5. **Dashboard**: Live dashboard for running workflows
