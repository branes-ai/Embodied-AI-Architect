# Changelog

All notable changes to the Embodied AI Architect project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure with Python package layout
- Documentation directories: `docs/plans` for architecture plans, `docs/sessions` for session logs
- Architecture plan: Multi-agent system design for Embodied AI evaluation (docs/plans/agent-architecture.md)
- Agent packaging strategy document (docs/plans/agent-packaging-strategy.md)
- Core Orchestrator class for coordinating agent workflows
- BaseAgent abstract class and AgentResult data model
- ModelAnalyzerAgent: Extracts architecture information from PyTorch models
- BenchmarkAgent: Performance profiling with extensible backend architecture
  - Backend plugin system for dispatching benchmarks to different execution environments
  - LocalCPUBackend: Measures inference latency and throughput on CPU
  - Support for future backends (GPU, remote cluster, edge devices, robots)
- HardwareProfileAgent: Hardware recommendation system with comprehensive knowledge base
  - Knowledge base with 8 hardware profiles (CPUs, GPUs, TPUs, edge devices)
  - Profiles include: Intel Xeon, NVIDIA A100, NVIDIA Jetson AGX Orin, Google TPU v4, Google Coral Edge TPU, Intel NCS2, Raspberry Pi 4, AMD MI250X
  - Scoring algorithm for hardware-model fitness based on memory, power, operations, and compute
  - Constraint-aware recommendations (latency, power, cost)
  - Use-case filtering (edge, cloud, datacenter, embedded)
  - Detailed reasoning and warnings for each recommendation
- ReportSynthesisAgent: Comprehensive report generation with visualizations
  - Filesystem-based architecture (Producer pattern) - generates artifacts, doesn't serve HTTP
  - Multiple output formats: HTML (human-readable), JSON (machine-readable)
  - Matplotlib-based visualizations: hardware comparison bar chart, layer distribution pie chart
  - Executive summary with key metrics and constraint satisfaction
  - Automated insights generation from workflow data
  - Actionable recommendations based on results
  - Jinja2-templated HTML reports with embedded CSS
  - Reports saved to `./reports/{workflow_id}/` directory structure
- Optional Report Server: Simple HTTP server for viewing reports (separate from orchestrator)
  - Static file serving from reports directory
  - Auto-generated index page listing all reports
  - Run independently: `python -m embodied_ai_architect.report_server`
  - Clean separation of concerns: generation vs. serving
- Architecture documentation: Reporting architecture patterns and design (docs/plans/reporting-architecture.md)
- Security Architecture: Comprehensive secrets management system
  - SecretsManager: Multi-provider secrets management with audit logging
  - EnvironmentSecretsProvider: Load secrets from environment variables
  - FileSecretsProvider: Load secrets from files with permission validation
  - Automatic secret masking in logs and error messages
  - Configuration reference resolution (${secret:key}, ${env:VAR})
  - Audit trail for all secret accesses
  - Secure defaults: file permission checks, ownership validation
  - .env.example template for secure configuration
  - Updated .gitignore to prevent accidental secret commits
  - Security architecture documentation (docs/plans/security-architecture.md)
- RemoteSSHBackend: Execute benchmarks on remote machines via SSH
  - Secure SSH connection using SecretsManager
  - Model serialization and transfer via SFTP
  - Remote code execution with proper cleanup
  - Result retrieval and parsing
  - Optional dependency (install with [remote] extra)
  - Demonstrates proper secrets handling pattern
- Configuration management examples
  - examples/remote_benchmark_example.py: Secure remote benchmarking demo
  - Security features demonstration (masking, audit, multi-provider)
  - Graceful handling of missing credentials
- KubernetesBackend: Cloud-native benchmarking with horizontal scaling
  - Job-based execution (create, monitor, cleanup)
  - Parallel benchmark execution (horizontal scaling)
  - GPU allocation with node selectors
  - Resource management (CPU/memory requests and limits)
  - ConfigMap for model data storage
  - Automatic cleanup with TTL
  - Secure RBAC configuration
  - Optional dependency (install with [kubernetes] extra)
  - Kubernetes manifests (namespace, service account, RBAC)
  - Configuration examples and README (config/kubernetes/)
  - examples/kubernetes_scaling_example.py: Demonstrates parallel execution
- Architecture documentation: Kubernetes backend design (docs/plans/kubernetes-backend-architecture.md)
- Working example demonstrating full workflow with all four agents (examples/simple_workflow.py)
- Project dependencies via pyproject.toml and requirements.txt (added matplotlib, jinja2, paramiko, kubernetes as optional)
- This CHANGELOG.md file to track project changes
- Command-Line Interface (CLI): Human-friendly CLI inspired by Claude Code
  - Architecture documentation: CLI design patterns and commands (docs/plans/cli-architecture.md)
  - Core CLI framework using Click and Rich for beautiful terminal output
  - Entry point: `embodied-ai` command registered in pyproject.toml
  - Global options: --verbose/-v, --json, --quiet for different output modes
  - Rich output with panels, tables, progress indicators, and colored text
  - Workflow commands: Complete end-to-end model evaluation pipeline
    - `embodied-ai workflow run <model>`: Run full workflow (analyze, profile, benchmark, report)
    - `embodied-ai workflow list`: List past workflow executions
  - Model analysis commands:
    - `embodied-ai analyze <model>`: Analyze model architecture and complexity
    - Layer breakdown table with counts by type
    - Parameter and memory statistics
  - Benchmarking commands:
    - `embodied-ai benchmark run <model>`: Run performance benchmarks
    - `embodied-ai benchmark list`: List available backends
    - Progress indicators for long-running benchmarks
    - Support for --backend, --iterations, --warmup options
  - Report management commands:
    - `embodied-ai report view <workflow_id>`: Open report in browser
    - `embodied-ai report list`: List all available reports
    - `embodied-ai report compare <id1> <id2>`: Compare two reports
  - Configuration commands:
    - `embodied-ai config init`: Initialize configuration file
    - `embodied-ai config show`: Display current configuration with syntax highlighting
    - `embodied-ai config validate`: Validate configuration
  - Backend management commands:
    - `embodied-ai backends list`: List available backends with status
    - `embodied-ai backends test <backend>`: Test backend connection
    - Shows installation requirements for optional backends
  - Secrets management commands:
    - `embodied-ai secrets list`: List secrets (keys only, not values)
    - `embodied-ai secrets validate`: Validate secrets configuration
    - Security-focused: never displays secret values
  - Error handling with helpful messages and tips
  - JSON output mode for scripting and CI/CD integration
  - Dependencies: click>=8.1.0, rich>=13.0.0
- Embodied AI Application Framework: Architecture analysis and implementation planning
  - Architecture analysis document: Evaluated 3 architecture options for application support (docs/plans/embodied-ai-application-architecture.md)
  - Selected hybrid architecture (Option 3): Python Framework + Graph IR + DSL
  - Comprehensive implementation plan: 18-week timeline for core functionality (docs/plans/embodied-ai-application-implementation-plan.md)
  - Planned integration with existing infrastructure:
    - PyTorch FX for graph capture
    - MLIR/IREE for compilation
    - Existing simulators as benchmark backends
    - Existing compiler/runtime modeler for hardware mapping
  - Application framework will support heterogeneous operators:
    - DNNs (PyTorch models)
    - Classical algorithms (Kalman filters, PID controllers)
    - Path planners (RRT*, A*)
    - Custom operators
  - Complete control loop applications (sensors → processing → decision → actuation)
  - End-to-end benchmarking with real datasets (ROS bags, custom formats)
  - Graph IR for analysis and optimization
  - DSL serialization for storage and portability
  - Future: C++/Rust transpilation for production deployment
  - Planned reference applications: drone navigation, robot manipulation, autonomous vehicle, legged robot, humanoid balance
- Multi-Rate Control Framework: Deep research and architecture proposal
  - Comprehensive research document on multi-rate control systems (docs/research/multi-rate-control-architecture.md)
  - Analyzed requirements for humanoid/quadruped robots:
    - Joint control: 100-1000 Hz (tight sensor-control-actuator loops)
    - Perception: 30-60 Hz (vision, state estimation)
    - Planning: 1-10 Hz (path planning, high-level decisions)
  - Selected Zenoh as communication backbone:
    - Zero-copy, real-time pub/sub
    - Automotive-grade certified
    - Open source (Apache 2.0)
    - Multi-transport (shared memory, UDP, DDS bridge)
    - Superior to ROS2/DDS for real-time requirements
  - Evaluated 3 multi-rate scheduling patterns:
    - Time-triggered (deterministic, simple)
    - Data-triggered (event-driven, low latency)
    - Hybrid (selected): Time-triggered with async data caching
  - Designed component model with declarative I/O:
    - `@control_loop` decorator for rate specification
    - `Input`/`Output` descriptors for automatic Zenoh wiring
    - Explicit rate, priority, deadline, CPU affinity
  - Hybrid Python/Rust execution strategy:
    - Python for slow loops (≤ 100 Hz): perception, planning
    - Rust for fast loops (> 500 Hz): joint control, balance
    - All communication via Zenoh (language-agnostic)
  - Phase 1 implementation proposal (docs/plans/phase1-framework-architecture.md):
    - Core framework classes (Component, Application, Scheduler)
    - Multi-rate scheduler (one thread per component)
    - Automatic Zenoh pub/sub creation
    - Deadline monitoring and diagnostics
    - 8-week implementation timeline
  - Example quadruped robot architecture with 4 control rates
  - Real-time considerations (RT-PREEMPT, CPU isolation, zero-copy)
- Multi-Rate Framework Prototype: Working validation of architecture
  - Created minimal working prototype (prototypes/multi_rate_framework/)
  - Implemented core framework classes (~400 lines):
    - Component base class with lifecycle hooks
    - @control_loop decorator for rate specification
    - Input/Output descriptors for automatic Zenoh wiring
    - MultiRateScheduler with thread-per-component
    - Application container
  - Successfully integrated Zenoh (eclipse-zenoh 1.6.2)
  - Created two working examples:
    - Simple: 100 Hz producer, 1 Hz consumer
    - Multi-rate: 100 Hz sensor, 10 Hz control, 1 Hz planning
  - Validation tests passed (test_framework.py):
    - Decorators work correctly
    - Input/Output descriptors functional
    - Component lifecycle correct
    - Timing accuracy < 1% (100.1ms vs 100ms expected)
  - Architecture validated for full implementation
  - Demonstrates hybrid time/data triggered pattern
  - Multi-threading with independent rates working
  - Clean declarative API validated

### Changed

### Deprecated

### Removed

### Fixed

### Security

---

## Guidelines for Changelog Entries

### Categories
- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes

### Format
Each release should include:
- Version number following semantic versioning (MAJOR.MINOR.PATCH)
- Release date in YYYY-MM-DD format
- Organized list of changes by category
