# Embodied AI Architect — System Architecture

## High-Level Architecture

```mermaid
graph TB
    subgraph Entry["Entry Points"]
        CLI["CLI<br/><code>branes</code><br/>22 command groups"]
        API["REST API<br/>FastAPI + SSE<br/><code>api/server.py</code>"]
        CHAT["Interactive Chat<br/>LLM Agent Loop<br/><code>llm/agent.py</code>"]
    end

    subgraph Qualify["Goal Qualification"]
        GQ["GoalQualifier<br/><code>qualifier.py</code>"]
        PR["PlatformRegistry<br/>266 platforms, TF-IDF matching<br/><code>platforms/registry.py</code>"]
        GT["Domain Templates<br/>drone / ugv / robot_arm<br/><code>templates/*.py</code>"]
        UQ["Universal Questions<br/><code>generic_questions.py</code>"]
        YAML["Platform YAML Files<br/>36 categories<br/><code>data/platforms/**/*.yaml</code>"]
    end

    subgraph Design["SoC Design Pipeline"]
        STATE["SoCDesignState<br/>40+ fields<br/><code>soc_state.py</code>"]
        RUNNER["SoCDesignRunner<br/>batch or interactive<br/><code>soc_runner.py</code>"]
        GRAPH["LangGraph StateGraph<br/>plan → dispatch → evaluate → optimize<br/><code>soc_graph.py</code>"]
    end

    subgraph TaskExec["Task Execution"]
        PLAN["PlannerNode<br/>LLM-based goal decomposition<br/><code>planner.py</code>"]
        TG["TaskGraph<br/>DAG of specialist tasks<br/><code>task_graph.py</code>"]
        DISP["Dispatcher<br/>parallel batch execution<br/><code>dispatcher.py</code>"]
    end

    subgraph Specialists["Specialist Agents"]
        WA["workload_analyzer"]
        HW["hw_explorer"]
        AC["architecture_composer"]
        PPA["ppa_assessor"]
        OPT["design_optimizer"]
        CRITIC["critic"]
        RPT["report_generator"]
        MOO_S["moo_explorer"]
        KPU["kpu_configurator"]
    end

    subgraph MOO["Multi-Objective Optimization"]
        ENGINE["OptimizationEngine<br/>3-layer pipeline<br/><code>moo/engine.py</code>"]
        ME["Layer 1: MAP-Elites<br/>quality-diversity search<br/>5K-10K evals"]
        BO["Layer 2a: Bayesian BO<br/>qNEHVI (≤4 objectives)"]
        NSGA["Layer 2b: NSGA-III<br/>many-objective (>4 obj)"]
        DS["DesignSpace<br/>20+ variables<br/><code>design_space.py</code>"]
        EVAL["DesignEvaluator<br/>thread-safe PPA evaluation"]
    end

    subgraph Physical["Physical Estimation"]
        IP["IP Blocks<br/>CPU, GPU, ISP, KPU, NoC<br/><code>ip_blocks.py</code>"]
        TECH["Technology Models<br/>7nm → 65nm<br/><code>technology.py</code>"]
        MFG["Manufacturing Cost<br/>yield, packaging, NRE<br/><code>manufacturing.py</code>"]
        BOM["Bill of Materials<br/>hierarchical system BOM<br/><code>bom.py</code>"]
        SWAP["SWaP-C Analysis<br/>sensitivity, ranking<br/><code>swap_analysis.py</code>"]
    end

    subgraph HumanLoop["Human-in-the-Loop"]
        REV["Plan Review<br/><code>review.py</code>"]
        OPTREV["Optimization Review<br/>Pareto, slackness, trajectory<br/><code>optimization_review.py</code>"]
        STEER["Steering Input<br/>continue / accept / redirect / stop"]
    end

    subgraph LLM["LLM Integration"]
        CLIENT["LLMClient<br/>Anthropic API<br/><code>llm/client.py</code>"]
        TOOLS["Tool Definitions<br/>50+ tools<br/><code>llm/tools.py</code>"]
        ACTX["Architecture Tools"]
        GTOOL["Graphs Tools"]
        CTOOL["Codebase Tools"]
    end

    subgraph Agents["Agent System (Workflow)"]
        ORCH["Orchestrator<br/>sequential agent chain<br/><code>orchestrator.py</code>"]
        MA["ModelAnalyzerAgent"]
        HA["HardwareProfileAgent"]
        BA["BenchmarkAgent"]
        DA["DeploymentAgent"]
    end

    subgraph Persist["Persistence"]
        SESS["SessionStore<br/>JSON per session<br/><code>~/.embodied-ai/sessions/</code>"]
        SPEC["SpecStore<br/>versioned specs<br/><code>.branes/specs/</code>"]
        GOV["Governance<br/>audit trail<br/><code>governance.py</code>"]
        MEM["Working Memory<br/>per-agent state<br/><code>memory.py</code>"]
    end

    subgraph External["External"]
        SCHEMAS["embodied-schemas<br/>shared Pydantic models"]
        GRAPHS["graphs library<br/>roofline, hardware models"]
    end

    %% Entry → Qualification
    CLI --> GQ
    CLI --> RUNNER
    CLI --> API
    CHAT --> CLIENT

    %% Qualification flow
    GQ --> PR
    GQ --> GT
    GQ --> UQ
    PR --> YAML
    GQ --> STATE

    %% Design pipeline
    STATE --> RUNNER
    RUNNER --> GRAPH
    GRAPH --> PLAN
    GRAPH --> DISP
    GRAPH --> OPTREV

    %% Task execution
    PLAN --> TG
    PLAN --> CLIENT
    TG --> DISP
    DISP --> WA
    DISP --> HW
    DISP --> AC
    DISP --> PPA
    DISP --> OPT
    DISP --> CRITIC
    DISP --> RPT
    DISP --> MOO_S
    DISP --> KPU

    %% MOO
    MOO_S --> ENGINE
    ENGINE --> ME
    ME --> BO
    ME --> NSGA
    ENGINE --> DS
    ENGINE --> EVAL

    %% Physical estimation
    EVAL --> IP
    EVAL --> TECH
    EVAL --> MFG
    IP --> BOM
    BOM --> SWAP

    %% Human loop
    GRAPH --> REV
    OPTREV --> STEER
    STEER --> GRAPH

    %% LLM tools
    CLIENT --> TOOLS
    TOOLS --> ACTX
    TOOLS --> GTOOL
    TOOLS --> CTOOL

    %% Agent system
    CLI --> ORCH
    ORCH --> MA
    ORCH --> HA
    HA --> BA
    BA --> DA

    %% Persistence
    RUNNER --> SESS
    STATE --> MEM
    GRAPH --> GOV

    %% External
    HW --> SCHEMAS
    EVAL --> GRAPHS
    GTOOL --> GRAPHS

    %% API reads sessions
    API --> SESS

    %% Context injection
    PR -.->|"context injection"| PLAN
    PR -.->|"context injection"| CHAT
```

## Data Flow: Design Session Lifecycle

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as branes CLI
    participant Q as GoalQualifier
    participant PR as PlatformRegistry
    participant P as PlannerNode
    participant D as Dispatcher
    participant S as Specialists
    participant E as Evaluate
    participant O as Optimizer
    participant R as Report
    participant SS as SessionStore

    U->>CLI: branes design qualify "edge device..."
    CLI->>Q: assess(goal)
    Q->>PR: search(goal)
    PR-->>Q: MatchResult[edge_vision_box]
    Q-->>CLI: platform picker + questions
    U->>CLI: answers (power, latency, env...)
    CLI->>Q: to_design_inputs()

    U->>CLI: branes design plan "goal" --power 15
    CLI->>PR: get_platform_context_for_goal()
    PR-->>CLI: context (architecture, pitfalls, refs)
    CLI->>P: plan(state + context)
    P-->>CLI: TaskGraph (6 tasks)
    CLI-->>U: show plan for review

    U->>CLI: branes chat → "execute this design"
    CLI->>D: run(state)

    loop Dispatch Loop
        D->>S: execute ready tasks in parallel
        S-->>D: results + _state_updates
        D->>E: evaluate constraints
        alt All PASS
            E->>R: generate report
            R-->>SS: save session
        else FAIL
            E->>O: apply optimization strategy
            O-->>D: modified state
        end
    end

    SS-->>U: session saved, report generated
```

## Component Inventory

### By Layer

| Layer | Components | Files |
|-------|-----------|-------|
| **Entry** | CLI (22 commands), REST API, Chat Agent | `cli/`, `api/server.py`, `llm/agent.py` |
| **Qualification** | GoalQualifier, PlatformRegistry, Templates | `qualification/`, `platforms/` |
| **Orchestration** | SoCDesignRunner, LangGraph, Dispatcher | `graphs/soc_runner.py`, `soc_graph.py`, `dispatcher.py` |
| **Planning** | PlannerNode, TaskGraph | `graphs/planner.py`, `task_graph.py` |
| **Specialists** | 14+ specialist agents | `graphs/specialists.py`, `moo/specialist.py` |
| **MOO Engine** | MAP-Elites → BO/NSGA-III pipeline | `graphs/moo/` (8 files) |
| **Physical** | IP blocks, technology, manufacturing, BOM | `graphs/ip_blocks.py`, `technology.py`, etc. |
| **Human Loop** | Plan review, optimization steering | `graphs/review.py`, `optimization_review.py` |
| **LLM** | Claude API client, 50+ tool definitions | `llm/` (10 files) |
| **Agents** | Orchestrator, Model/HW/Benchmark/Deploy | `agents/`, `orchestrator.py` |
| **Persistence** | Sessions, specs, working memory, governance | `graphs/session_store.py`, `specs/` |
| **Data** | 266 platform YAMLs, taxonomy, schema | `data/platforms/` |

### Key Metrics

| Metric | Value |
|--------|-------|
| Python source files | ~80 |
| CLI command groups | 22 |
| Platform definitions | 266 |
| Platform categories | 36 |
| LLM tools | 50+ |
| Specialist agents | 14+ |
| MOO optimization layers | 3 |
| API endpoints | 15+ |
| Total lines of Python | ~25,000 |
