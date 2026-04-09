---
title: MCP Tools Reference
description: Reference for Model Context Protocol tools.
---

Embodied AI Architect exposes tools via MCP for use with Claude and other LLM clients.

## Analysis Tools

### analyze_model_detailed

Perform detailed analysis using roofline modeling.

**Input:**
```json
{
  "model_name": "resnet18",
  "hardware_name": "H100-SXM5-80GB",
  "batch_size": 1,
  "precision": "FP16"
}
```

**Output:**
```json
{
  "model": "resnet18",
  "hardware": "H100-SXM5-80GB",
  "metrics": {
    "latency_ms": 0.82,
    "throughput_fps": 1219.5,
    "energy_mj": 0.45,
    "peak_memory_mb": 89.2
  },
  "bottleneck": {
    "type": "memory_bound",
    "compute_utilization": 12.3,
    "memory_utilization": 89.1
  }
}
```

### compare_hardware_targets

Compare model performance across hardware.

**Input:**
```json
{
  "model_name": "yolov8n",
  "hardware_targets": ["H100-SXM5-80GB", "Jetson-Orin-AGX", "Coral-Edge-TPU"]
}
```

### identify_bottleneck

Identify compute vs memory bottleneck.

**Input:**
```json
{
  "model_name": "resnet50",
  "hardware_name": "A100-SXM4-80GB"
}
```

### list_available_hardware

List supported hardware targets.

**Input:**
```json
{
  "category": "edge_gpu"
}
```

## Constraint Checking Tools

### check_latency

Check if model meets latency target.

**Input:**
```json
{
  "model_name": "yolov8n",
  "hardware_name": "Jetson-Orin-Nano",
  "latency_target_ms": 33
}
```

**Output:**
```json
{
  "verdict": "PASS",
  "confidence": "HIGH",
  "metrics": {
    "latency_ms": 28.5
  },
  "constraint": {
    "metric": "latency",
    "threshold": 33,
    "actual": 28.5,
    "margin_pct": 13.6
  }
}
```

### check_power

Check if model meets power budget.

**Input:**
```json
{
  "model_name": "yolov8n",
  "hardware_name": "Jetson-Orin-Nano",
  "power_budget_w": 15
}
```

### check_memory

Check if model fits in memory budget.

**Input:**
```json
{
  "model_name": "resnet50",
  "hardware_name": "Jetson-Orin-Nano",
  "memory_budget_mb": 4096
}
```

### full_analysis

Complete analysis with optional constraint.

**Input:**
```json
{
  "model_name": "yolov8n",
  "hardware_name": "Jetson-Orin-Nano",
  "constraint_metric": "latency",
  "constraint_threshold": 33
}
```

## Architecture Tools

### analyze_architecture

Analyze a complete pipeline on hardware.

**Input:**
```json
{
  "architecture_id": "drone_perception_v1",
  "hardware_id": "Jetson-Orin-Nano"
}
```

### check_scheduling

Check if all operators meet rate requirements.

**Input:**
```json
{
  "architecture_id": "drone_perception_v1",
  "hardware_id": "Jetson-Orin-Nano"
}
```

## Optimization Tools

### start_exploration

Start a multi-objective design space exploration. Runs the 3-layer MOO pipeline (MAP-Elites + Bayesian BO) in the background. Returns a `session_id` for tracking progress and querying results.

**Input:**
```json
{
  "goal": "drone perception SoC",
  "constraints": {
    "max_power_watts": 5,
    "max_latency_ms": 33,
    "max_cost_usd": 25
  },
  "config": {
    "layers": "auto",
    "max_workers": 8
  }
}
```

**Output:**
```json
{
  "session_id": "opt-abc123",
  "status": "running",
  "message": "Optimization started. Use get_exploration_status to track progress."
}
```

### get_pareto_front

Get Pareto-optimal design points from a completed exploration. Returns top-N designs for LLM context.

**Input:**
```json
{
  "session_id": "opt-abc123",
  "top_n": 5
}
```

**Output:**
```json
{
  "preview": [
    {
      "design_params": {
        "process_nm": 7,
        "clock_mhz": 1200,
        "array_rows": 16,
        "array_cols": 16,
        "sram_kb": 512,
        "num_compute_tiles": 4,
        "noc_link_width_bits": 256
      },
      "objectives": {
        "power_watts": 3.2,
        "latency_ms": 28.1,
        "area_mm2": 42.0,
        "cost_usd": 18.5
      }
    }
  ],
  "total_points": 12,
  "hypervolume": 12.45,
  "knee_point": { "..." : "..." }
}
```

### get_sensitivity

Get parameter sensitivity analysis from a completed exploration. Shows which design parameters most affect each objective. Requires the Bayesian optimization layer.

**Input:**
```json
{
  "session_id": "opt-abc123"
}
```

**Output:**
```json
{
  "sensitivity": {
    "power_watts": {
      "clock_mhz": { "lengthscale": 0.1234, "importance": 0.8912 },
      "process_nm": { "lengthscale": 0.2341, "importance": 0.7234 }
    },
    "latency_ms": {
      "array_rows": { "lengthscale": 0.1567, "importance": 0.8456 }
    }
  },
  "layers_used": ["map_elites", "bayesian"]
}
```

### explain_tradeoff

Explain the tradeoff between two design points from the Pareto front. Shows what changes in parameters lead to what changes in objectives.

**Input:**
```json
{
  "session_id": "opt-abc123",
  "point_a_index": 0,
  "point_b_index": 3
}
```

**Output:**
```json
{
  "objective_deltas": {
    "power_watts": {
      "point_a": 3.2,
      "point_b": 1.8,
      "delta": -1.4,
      "pct_change": -43.8
    },
    "latency_ms": {
      "point_a": 28.1,
      "point_b": 30.5,
      "delta": 2.4,
      "pct_change": 8.5
    }
  },
  "parameter_changes": {
    "process_nm": { "from": 7, "to": 5 },
    "clock_mhz": { "from": 1200, "to": 600 }
  }
}
```

### get_exploration_status

Get the status and progress of an optimization session.

**Input:**
```json
{
  "session_id": "opt-abc123"
}
```

**Output:**
```json
{
  "status": "completed",
  "current_layer": "bayesian",
  "total_evaluations": 5320,
  "elapsed_seconds": 45.2
}
```

## Using with Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "branes": {
      "command": "branes",
      "args": ["mcp", "serve"]
    }
  }
}
```
