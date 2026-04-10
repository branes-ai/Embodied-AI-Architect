# Embodied AI Architect

A design environment for creating and evaluating autonomous agents, with hardware/software codesign space exploration and optimization.

## Features

- **Mission-Driven Workflow**: Define a mission, qualify goals, select sensors/actuators, synthesize and analyze a complete system
- **Sensor & Actuator Selection**: Search, compare, budget, and fusion analysis for system components
- **System Synthesis**: Generate system architectures and bills of materials from mission specifications
- **System Analysis**: Power, latency, thermal, SWaP-C, and safety analysis of synthesized systems
- **Model Analysis**: Analyze PyTorch model structure and compute requirements
- **Hardware Profiling**: Recommendations for edge/cloud deployment
- **Multi-Hardware Benchmarking**: Local CPU, remote SSH, Kubernetes backends
- **Interactive Chat**: Claude-powered architect for design decisions
- **SoC Optimization**: LangGraph-based RTL optimization loop (experimental)

## Mission Workflow Quick Start

The mission-driven workflow is the primary way to design an embodied AI system:

```bash
# 1. Create a mission
branes mission new vineyard-sprayer --goal "Autonomous vineyard spraying drone"

# 2. Qualify design goals and constraints
branes design qualify --mission vineyard-sprayer --auto

# 3. Select sensors and actuators
branes sensor search "stereo camera for VIO"
branes sensor select vineyard-sprayer visual.stereo_camera
branes actuator search "pump for spraying"
branes actuator select vineyard-sprayer fluid.sprayer

# 4. Analyze sensor budget and fusion
branes sensor budget vineyard-sprayer
branes sensor fusion vineyard-sprayer

# 5. Generate a design plan and synthesize the system
branes design plan --mission vineyard-sprayer --static
branes synthesize system vineyard-sprayer

# 6. Run system-level analysis
branes analyze-system power vineyard-sprayer
branes analyze-system swap vineyard-sprayer
```

See [docs/quickstart-mission.md](docs/quickstart-mission.md) for a complete walkthrough.

## Installation

### Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/branes-ai/embodied-ai-architect.git
cd embodied-ai-architect

# Run the setup script
./bin/setup-dev-env.sh

# Activate the environment
source .venv/bin/activate

# Verify installation
embodied-ai --help
```

### Setup Options

```bash
# Full installation (Python deps + EDA tools)
./bin/setup-dev-env.sh --all

# Minimal installation (Python deps only)
./bin/setup-dev-env.sh --minimal

# SoC optimizer only (Python deps + EDA tools)
./bin/setup-dev-env.sh --soc
```

### Manual Setup

If you prefer manual installation:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -e .

# Install with all optional dependencies
pip install -e ".[all,dev]"

# For SoC optimizer experiments, also install:
pip install langgraph langchain-anthropic
```

### EDA Tools (for SoC Optimizer)

The SoC optimizer requires open-source EDA tools. The setup script installs [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) which includes:

- **Yosys**: RTL synthesis
- **Verilator**: Fast Verilog linting and simulation
- **Icarus Verilog**: Verilog simulation

Manual installation:
```bash
# Download OSS CAD Suite
wget https://github.com/YosysHQ/oss-cad-suite-build/releases/download/2025-12-12/oss-cad-suite-linux-x64-20251212.tgz
sudo tar -xzf oss-cad-suite-linux-x64-20251212.tgz -C /opt

# Add to PATH
export PATH="/opt/oss-cad-suite/bin:$PATH"
```

## Usage

### CLI Commands

```bash
# Show available commands
embodied-ai --help

# Analyze a PyTorch model
embodied-ai analyze model.pt

# Run full workflow
embodied-ai workflow run model.pt

# Benchmark on local CPU
embodied-ai benchmark model.pt --backend local

# Interactive chat session (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=your-key-here
embodied-ai chat
```

### SoC Optimizer (Experimental)

The LangGraph-based SoC optimizer demonstrates agentic RTL optimization:

```bash
cd experiments/langgraph/soc_optimizer

# Run with mock mode (no LLM)
python workflow.py

# Run with Claude for RTL optimization
export ANTHROPIC_API_KEY=your-key-here
python workflow.py --with-llm

# Run simple loop (no LangGraph dependency)
python workflow.py --simple
```

## Project Structure

```
embodied-ai-architect/
├── bin/                    # Setup scripts
│   └── setup-dev-env.sh    # Development environment setup
├── src/embodied_ai_architect/
│   ├── agents/             # Agent implementations
│   ├── cli/                # Click-based CLI
│   └── llm/                # LLM integration
├── experiments/
│   └── langgraph/          # LangGraph experiments
│       └── soc_optimizer/  # RTL optimization loop
├── prototypes/             # Research prototypes
│   ├── drone_perception/   # Real-time perception pipeline
│   └── multi_rate_framework/  # Zenoh-based multi-rate control
├── docs/                   # Documentation
└── tests/                  # Test suite
```

## Development

```bash
# Run tests
pytest tests/

# Run single test
pytest tests/test_file.py::test_function -v

# Linting and formatting
black src/ tests/ --line-length 100
ruff check src/ tests/
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Required for Claude-powered features |

## Related Repositories

- [embodied-schemas](https://github.com/branes-ai/embodied-schemas): Shared Pydantic schemas
- [graphs](https://github.com/branes-ai/graphs): Analysis tools and roofline models
- [systars](https://github.com/stillwater-sc/systars): Amaranth HDL systolic array generator

## License

MIT License - see [LICENSE](LICENSE) for details.
