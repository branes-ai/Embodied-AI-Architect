"""Report Synthesis Agent - Generates comprehensive reports with visualizations."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template

from ..base import BaseAgent, AgentResult


class ReportSynthesisAgent(BaseAgent):
    """Agent that synthesizes results from all agents into comprehensive reports.

    This agent follows the Producer pattern - it generates artifacts (HTML, JSON, images)
    and writes them to the filesystem. Consuming/serving these artifacts is handled
    by separate components (CLI viewer, web server, etc.).

    Responsibilities:
        - Aggregate data from all agents
        - Calculate derived metrics (speedup, cost-effectiveness, etc.)
        - Generate visualizations (charts, graphs)
        - Render HTML reports
        - Write artifacts to disk

    NOT Responsibilities:
        - Serving HTTP requests (use separate report server)
        - User authentication
        - Real-time updates
        - Data querying
    """

    def __init__(self, reports_dir: str = "./reports"):
        """Initialize the report synthesis agent.

        Args:
            reports_dir: Directory where reports will be saved
        """
        super().__init__(name="ReportSynthesis")
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Generate comprehensive report from workflow results.

        Args:
            input_data: Dictionary with keys:
                - 'workflow_id': Unique workflow identifier (auto-generated if missing)
                - 'agent_results': Results from all agents
                - 'request': Original user request
                - 'timestamp': Workflow start time

        Returns:
            AgentResult with report path and summary
        """
        try:
            # Extract input data
            workflow_id = input_data.get("workflow_id") or str(uuid.uuid4())[:8]
            agent_results = input_data.get("agent_results", {})
            request = input_data.get("request", {})
            timestamp = input_data.get("timestamp") or datetime.now().isoformat()

            # Create report directory
            report_dir = self.reports_dir / workflow_id
            assets_dir = report_dir / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)

            # Synthesize data
            report_data = self._synthesize_data(workflow_id, agent_results, request, timestamp)

            # Generate visualizations
            charts = self._generate_visualizations(report_data, assets_dir)

            # Generate HTML report
            html_content = self._generate_html(report_data, charts)

            # Write artifacts
            (report_dir / "report.json").write_text(json.dumps(report_data, indent=2))
            (report_dir / "report.html").write_text(html_content)

            # Create metadata
            metadata = {
                "workflow_id": workflow_id,
                "generated_at": datetime.now().isoformat(),
                "report_version": "1.0",
            }
            (report_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

            print(f"  Report generated: {report_dir}/report.html")

            return AgentResult(
                success=True,
                data={
                    "report_path": str(report_dir),
                    "report_html": f"{report_dir}/report.html",
                    "report_json": f"{report_dir}/report.json",
                    "workflow_id": workflow_id,
                    "summary": report_data.get("executive_summary", {}),
                },
                metadata={"agent": self.name, "charts_generated": len(charts)},
            )

        except Exception as e:
            return AgentResult(success=False, data={}, error=f"Report generation failed: {str(e)}")

    def _synthesize_data(
        self,
        workflow_id: str,
        agent_results: Dict[str, Any],
        request: Dict[str, Any],
        timestamp: str,
    ) -> Dict[str, Any]:
        """Synthesize data from all agents into a structured report.

        Args:
            workflow_id: Workflow identifier
            agent_results: Results from all agents
            request: Original request
            timestamp: Workflow timestamp

        Returns:
            Structured report data dictionary
        """
        report = {
            "metadata": {"workflow_id": workflow_id, "timestamp": timestamp, "version": "1.0"}
        }

        # Extract model analysis
        if "ModelAnalyzer" in agent_results:
            model_data = agent_results["ModelAnalyzer"].data
            report["model_analysis"] = {
                "model_type": model_data.get("model_type"),
                "total_parameters": model_data.get("total_parameters"),
                "trainable_parameters": model_data.get("trainable_parameters"),
                "total_layers": model_data.get("total_layers"),
                "layer_types": model_data.get("layer_type_counts", {}),
                "memory_mb": (model_data.get("total_parameters", 0) * 4) / (1024 * 1024),
            }

        # Extract hardware recommendations
        if "HardwareProfile" in agent_results:
            hw_data = agent_results["HardwareProfile"].data
            report["hardware_recommendations"] = hw_data.get("recommendations", [])
            report["model_characteristics"] = hw_data.get("model_characteristics", {})

        # Extract benchmarks
        if "Benchmark" in agent_results:
            bench_data = agent_results["Benchmark"].data
            report["benchmarks"] = bench_data.get("benchmarks", {})

        # Generate insights and recommendations
        report["executive_summary"] = self._generate_executive_summary(report, request)
        report["insights"] = self._generate_insights(report)
        report["recommendations"] = self._generate_recommendations(report, request)

        return report

    def _generate_executive_summary(
        self, report_data: Dict[str, Any], request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate executive summary.

        Args:
            report_data: Synthesized report data
            request: Original request

        Returns:
            Executive summary dictionary
        """
        summary = {}

        # Model summary
        if "model_analysis" in report_data:
            model = report_data["model_analysis"]
            summary["model"] = {
                "type": model.get("model_type"),
                "size": f"{model.get('total_parameters', 0) / 1e6:.1f}M parameters",
                "memory": f"{model.get('memory_mb', 0):.1f} MB",
            }

        # Best hardware
        if "hardware_recommendations" in report_data:
            recs = report_data["hardware_recommendations"]
            if recs:
                best = recs[0]
                summary["recommended_hardware"] = {
                    "name": best["name"],
                    "vendor": best["vendor"],
                    "score": best["score"],
                    "cost_usd": best.get("cost_usd"),
                    "power_watts": best["power_watts"],
                }

        # Benchmark summary
        if "benchmarks" in report_data:
            benchmarks = report_data["benchmarks"]
            if benchmarks:
                first_backend = list(benchmarks.keys())[0]
                bench = benchmarks[first_backend]
                summary["performance"] = {
                    "backend": first_backend,
                    "latency_ms": bench.get("mean_latency_ms"),
                    "throughput": bench.get("throughput_samples_per_sec"),
                }

        # Constraints met
        constraints = request.get("constraints", {})
        summary["constraints_met"] = self._check_constraints(report_data, constraints)

        return summary

    def _check_constraints(
        self, report_data: Dict[str, Any], constraints: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Check if constraints are met.

        Args:
            report_data: Report data
            constraints: User constraints

        Returns:
            Dictionary of constraint satisfaction
        """
        met = {}

        if "hardware_recommendations" in report_data and report_data["hardware_recommendations"]:
            best_hw = report_data["hardware_recommendations"][0]

            if "max_power_watts" in constraints:
                met["power"] = best_hw["power_watts"] <= constraints["max_power_watts"]

            if "max_cost_usd" in constraints and best_hw.get("cost_usd"):
                met["cost"] = best_hw["cost_usd"] <= constraints["max_cost_usd"]

        return met

    def _generate_insights(self, report_data: Dict[str, Any]) -> List[str]:
        """Generate insights from the data.

        Args:
            report_data: Report data

        Returns:
            List of insight strings
        """
        insights = []

        # Model size insights
        if "model_analysis" in report_data:
            params = report_data["model_analysis"].get("total_parameters", 0)
            if params < 1e6:
                insights.append(
                    f"Lightweight model ({params/1e3:.0f}K params) suitable for edge deployment"
                )
            elif params > 1e9:
                insights.append(
                    f"Large model ({params/1e9:.1f}B params) requires datacenter-class hardware"
                )

        # Hardware insights
        if "hardware_recommendations" in report_data:
            recs = report_data["hardware_recommendations"]
            if len(recs) > 1:
                top_score = recs[0]["score"]
                second_score = recs[1]["score"]
                if top_score - second_score < 5:
                    insights.append(
                        "Multiple hardware options have similar scores - consider cost and availability"
                    )

        # Performance insights
        if "benchmarks" in report_data:
            for backend, bench in report_data["benchmarks"].items():
                latency = bench.get("mean_latency_ms", 0)
                if latency < 1:
                    insights.append(
                        f"Excellent latency on {backend} ({latency:.3f}ms) - suitable for real-time applications"
                    )
                elif latency > 100:
                    insights.append(
                        f"High latency on {backend} ({latency:.1f}ms) - consider hardware acceleration"
                    )

        return insights

    def _generate_recommendations(
        self, report_data: Dict[str, Any], request: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations.

        Args:
            report_data: Report data
            request: Original request

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Hardware recommendations
        if "hardware_recommendations" in report_data:
            recs = report_data["hardware_recommendations"]
            if recs:
                best = recs[0]
                recommendations.append(
                    f"Deploy on {best['name']} for optimal performance (score: {best['score']:.1f}/100)"
                )

        # Optimization recommendations
        if "model_analysis" in report_data:
            model = report_data["model_analysis"]
            params = model.get("total_parameters", 0)
            if params > 1e6:
                recommendations.append(
                    "Consider model quantization (INT8/FP16) to reduce memory and improve performance"
                )
            if "Conv" in str(model.get("layer_types", {})):
                recommendations.append(
                    "Model uses convolutions - hardware with tensor cores will provide significant speedup"
                )

        return recommendations

    def _generate_visualizations(
        self, report_data: Dict[str, Any], assets_dir: Path
    ) -> Dict[str, str]:
        """Generate visualization charts.

        Args:
            report_data: Report data
            assets_dir: Directory to save images

        Returns:
            Dictionary mapping chart names to file paths
        """
        charts = {}

        # Hardware comparison chart
        if "hardware_recommendations" in report_data:
            chart_path = self._create_hardware_comparison_chart(
                report_data["hardware_recommendations"], assets_dir / "hardware_comparison.png"
            )
            if chart_path:
                charts["hardware_comparison"] = "assets/hardware_comparison.png"

        # Layer distribution pie chart
        if "model_analysis" in report_data:
            layer_types = report_data["model_analysis"].get("layer_types", {})
            if layer_types:
                chart_path = self._create_layer_distribution_chart(
                    layer_types, assets_dir / "layer_distribution.png"
                )
                if chart_path:
                    charts["layer_distribution"] = "assets/layer_distribution.png"

        # Benchmark comparison (if multiple backends)
        if "benchmarks" in report_data:
            benchmarks = report_data["benchmarks"]
            if len(benchmarks) > 1:
                chart_path = self._create_benchmark_comparison_chart(
                    benchmarks, assets_dir / "benchmark_comparison.png"
                )
                if chart_path:
                    charts["benchmark_comparison"] = "assets/benchmark_comparison.png"

        return charts

    def _create_hardware_comparison_chart(
        self, recommendations: List[Dict], output_path: Path
    ) -> Path | None:
        """Create hardware comparison bar chart.

        Args:
            recommendations: Hardware recommendations
            output_path: Output file path

        Returns:
            Path to generated chart or None
        """
        if not recommendations:
            return None

        try:
            # Take top 5
            top_recs = recommendations[:5]

            names = [r["name"] for r in top_recs]
            scores = [r["score"] for r in top_recs]
            colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(names))]

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(names, scores, color=colors)

            ax.set_xlabel("Fitness Score", fontsize=12)
            ax.set_title("Hardware Recommendations Comparison", fontsize=14, fontweight="bold")
            ax.set_xlim(0, 100)
            ax.grid(axis="x", alpha=0.3)

            # Add value labels
            for bar, score in zip(bars, scores):
                width = bar.get_width()
                ax.text(
                    width + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.1f}",
                    va="center",
                    fontsize=10,
                )

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            return output_path

        except Exception as e:
            print(f"  Warning: Could not generate hardware comparison chart: {e}")
            return None

    def _create_layer_distribution_chart(
        self, layer_types: Dict[str, int], output_path: Path
    ) -> Path | None:
        """Create layer type distribution pie chart.

        Args:
            layer_types: Dictionary of layer types and counts
            output_path: Output file path

        Returns:
            Path to generated chart or None
        """
        if not layer_types:
            return None

        try:
            # Filter out very small counts
            filtered = {k: v for k, v in layer_types.items() if v > 0}
            if not filtered:
                return None

            labels = list(filtered.keys())
            sizes = list(filtered.values())

            fig, ax = plt.subplots(figsize=(10, 8))
            colors = plt.cm.Set3(range(len(labels)))

            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
            )

            # Make percentage text more readable
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")

            ax.set_title("Model Layer Type Distribution", fontsize=14, fontweight="bold")

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            return output_path

        except Exception as e:
            print(f"  Warning: Could not generate layer distribution chart: {e}")
            return None

    def _create_benchmark_comparison_chart(
        self, benchmarks: Dict[str, Dict], output_path: Path
    ) -> Path | None:
        """Create benchmark comparison chart.

        Args:
            benchmarks: Benchmark results
            output_path: Output file path

        Returns:
            Path to generated chart or None
        """
        if not benchmarks:
            return None

        try:
            backends = list(benchmarks.keys())
            latencies = [benchmarks[b].get("mean_latency_ms", 0) for b in backends]

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(backends, latencies, color="#3498db")

            ax.set_ylabel("Mean Latency (ms)", fontsize=12)
            ax.set_title("Benchmark Results Comparison", fontsize=14, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

            # Add value labels
            for bar, latency in zip(bars, latencies):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{latency:.3f}ms",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            return output_path

        except Exception as e:
            print(f"  Warning: Could not generate benchmark comparison chart: {e}")
            return None

    def _generate_html(self, report_data: Dict[str, Any], charts: Dict[str, str]) -> str:
        """Generate HTML report.

        Args:
            report_data: Report data
            charts: Chart file paths

        Returns:
            HTML content string
        """
        template_str = self._get_html_template()
        template = Template(template_str)

        html = template.render(
            report=report_data,
            charts=charts,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        return html

    def _get_html_template(self) -> str:
        """Get HTML template string.

        Returns:
            HTML template string
        """
        # Inline template for simplicity - could be loaded from file
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embodied AI Report - {{ report.metadata.workflow_id }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f7fa;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        header { border-bottom: 3px solid #3498db; padding-bottom: 20px; margin-bottom: 30px; }
        h1 { color: #2c3e50; font-size: 2.5em; margin-bottom: 10px; }
        h2 { color: #34495e; font-size: 1.8em; margin-top: 30px; margin-bottom: 15px; border-left: 4px solid #3498db; padding-left: 15px; }
        h3 { color: #7f8c8d; font-size: 1.2em; margin-top: 20px; margin-bottom: 10px; }
        .metadata { color: #7f8c8d; font-size: 0.9em; }
        .key-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; }
        .metric-card.green { background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); }
        .metric-card.blue { background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); }
        .metric-card.orange { background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); }
        .metric-value { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
        .metric-label { font-size: 0.9em; opacity: 0.9; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ecf0f1; }
        th { background: #34495e; color: white; font-weight: 600; }
        tr:hover { background: #f8f9fa; }
        .chart { margin: 30px 0; text-align: center; }
        .chart img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .insight-box { background: #e8f5e9; border-left: 4px solid #4caf50; padding: 15px; margin: 15px 0; border-radius: 4px; }
        .warning-box { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; border-radius: 4px; }
        .recommendation-box { background: #e3f2fd; border-left: 4px solid #2196f3; padding: 15px; margin: 15px 0; border-radius: 4px; }
        ul { margin: 10px 0 10px 30px; }
        li { margin: 8px 0; }
        .badge { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; }
        .badge.success { background: #d4edda; color: #155724; }
        .badge.warning { background: #fff3cd; color: #856404; }
        .badge.info { background: #d1ecf1; color: #0c5460; }
        footer { margin-top: 50px; padding-top: 20px; border-top: 1px solid #ecf0f1; color: #7f8c8d; font-size: 0.9em; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 Embodied AI Architect Report</h1>
            <div class="metadata">
                Workflow ID: <strong>{{ report.metadata.workflow_id }}</strong> |
                Generated: {{ generated_at }}
            </div>
        </header>

        <section id="executive-summary">
            <h2>📊 Executive Summary</h2>
            {% if report.executive_summary %}
            <div class="key-metrics">
                {% if report.executive_summary.model %}
                <div class="metric-card blue">
                    <div class="metric-label">Model</div>
                    <div class="metric-value">{{ report.executive_summary.model.size }}</div>
                    <div class="metric-label">{{ report.executive_summary.model.type }}</div>
                </div>
                {% endif %}

                {% if report.executive_summary.recommended_hardware %}
                <div class="metric-card green">
                    <div class="metric-label">Recommended Hardware</div>
                    <div class="metric-value">{{ report.executive_summary.recommended_hardware.score | round(1) }}</div>
                    <div class="metric-label">{{ report.executive_summary.recommended_hardware.name }}</div>
                </div>
                {% endif %}

                {% if report.executive_summary.performance %}
                <div class="metric-card orange">
                    <div class="metric-label">Performance</div>
                    <div class="metric-value">{{ report.executive_summary.performance.latency_ms | round(3) }}ms</div>
                    <div class="metric-label">Inference Latency</div>
                </div>
                {% endif %}
            </div>

            {% if report.executive_summary.constraints_met %}
            <div class="insight-box">
                <strong>Constraints:</strong>
                {% for constraint, met in report.executive_summary.constraints_met.items() %}
                    <span class="badge {{ 'success' if met else 'warning' }}">
                        {{ constraint }}: {{ 'met' if met else 'not met' }}
                    </span>
                {% endfor %}
            </div>
            {% endif %}
            {% endif %}
        </section>

        {% if report.model_analysis %}
        <section id="model-analysis">
            <h2>🔬 Model Analysis</h2>
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Model Type</td><td>{{ report.model_analysis.model_type }}</td></tr>
                <tr><td>Total Parameters</td><td>{{ "{:,}".format(report.model_analysis.total_parameters) }}</td></tr>
                <tr><td>Trainable Parameters</td><td>{{ "{:,}".format(report.model_analysis.trainable_parameters) }}</td></tr>
                <tr><td>Total Layers</td><td>{{ report.model_analysis.total_layers }}</td></tr>
                <tr><td>Estimated Memory</td><td>{{ report.model_analysis.memory_mb | round(2) }} MB</td></tr>
            </table>

            {% if charts.layer_distribution %}
            <div class="chart">
                <img src="{{ charts.layer_distribution }}" alt="Layer Distribution">
            </div>
            {% endif %}
        </section>
        {% endif %}

        {% if report.hardware_recommendations %}
        <section id="hardware-recommendations">
            <h2>🖥️ Hardware Recommendations</h2>

            {% if charts.hardware_comparison %}
            <div class="chart">
                <img src="{{ charts.hardware_comparison }}" alt="Hardware Comparison">
            </div>
            {% endif %}

            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Hardware</th>
                        <th>Vendor</th>
                        <th>Score</th>
                        <th>Type</th>
                        <th>Power (W)</th>
                        <th>Cost (USD)</th>
                    </tr>
                </thead>
                <tbody>
                    {% for rec in report.hardware_recommendations[:5] %}
                    <tr>
                        <td><strong>{{ rec.rank }}</strong></td>
                        <td>{{ rec.name }}</td>
                        <td>{{ rec.vendor }}</td>
                        <td><span class="badge info">{{ rec.score | round(1) }}</span></td>
                        <td>{{ rec.type.upper() }}</td>
                        <td>{{ rec.power_watts }}</td>
                        <td>{{ "${:,}".format(rec.cost_usd) if rec.cost_usd else "N/A" }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>

            <h3>Top Recommendation Details</h3>
            {% if report.hardware_recommendations %}
            {% set top = report.hardware_recommendations[0] %}
            <div class="recommendation-box">
                <strong>{{ top.name }}</strong> ({{ top.vendor }})<br>
                <strong>Reasons:</strong>
                <ul>
                    {% for reason in top.reasons[:5] %}
                    <li>{{ reason }}</li>
                    {% endfor %}
                </ul>
                {% if top.warnings %}
                <div class="warning-box" style="margin-top: 10px;">
                    <strong>⚠️ Warnings:</strong>
                    <ul>
                        {% for warning in top.warnings %}
                        <li>{{ warning }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}
            </div>
            {% endif %}
        </section>
        {% endif %}

        {% if report.benchmarks %}
        <section id="benchmarks">
            <h2>⚡ Benchmark Results</h2>

            {% if charts.benchmark_comparison %}
            <div class="chart">
                <img src="{{ charts.benchmark_comparison }}" alt="Benchmark Comparison">
            </div>
            {% endif %}

            <table>
                <thead>
                    <tr>
                        <th>Backend</th>
                        <th>Mean Latency (ms)</th>
                        <th>Std Dev (ms)</th>
                        <th>Min/Max (ms)</th>
                        <th>Throughput (samples/sec)</th>
                    </tr>
                </thead>
                <tbody>
                    {% for backend, result in report.benchmarks.items() %}
                    <tr>
                        <td><strong>{{ backend.upper() }}</strong></td>
                        <td>{{ result.mean_latency_ms | round(3) }}</td>
                        <td>{{ result.std_latency_ms | round(3) }}</td>
                        <td>{{ result.min_latency_ms | round(3) }} / {{ result.max_latency_ms | round(3) }}</td>
                        <td>{{ result.throughput_samples_per_sec | round(2) if result.throughput_samples_per_sec else "N/A" }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </section>
        {% endif %}

        {% if report.insights %}
        <section id="insights">
            <h2>💡 Insights</h2>
            {% for insight in report.insights %}
            <div class="insight-box">
                {{ insight }}
            </div>
            {% endfor %}
        </section>
        {% endif %}

        {% if report.recommendations %}
        <section id="recommendations">
            <h2>🎯 Recommendations</h2>
            {% for recommendation in report.recommendations %}
            <div class="recommendation-box">
                {{ recommendation }}
            </div>
            {% endfor %}
        </section>
        {% endif %}

        <footer>
            <p>Generated by <strong>Embodied AI Architect</strong> | Version 1.0</p>
            <p>For more information, visit the project repository</p>
        </footer>
    </div>
</body>
</html>"""
