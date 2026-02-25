"""Simple HTTP server for viewing generated reports.

This server is OPTIONAL and completely separate from the orchestrator.
It simply serves static files from the reports directory.

Usage:
    python -m embodied_ai_architect.report_server
    # or
    python src/embodied_ai_architect/report_server.py

Then open http://localhost:8000 in your browser.
"""

import http.server
import socketserver
import os
import json
from pathlib import Path
from typing import List, Dict, Any


class ReportHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler to serve reports and provide index page."""

    def __init__(self, *args, **kwargs):
        # Set the directory to serve
        super().__init__(*args, directory="reports", **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        # Root path - show index of all reports
        if self.path == "/" or self.path == "/index.html":
            self.serve_index()
        else:
            # Serve static files from reports directory
            super().do_GET()

    def serve_index(self):
        """Serve an index page listing all available reports."""
        reports = self.list_reports()

        html = self.generate_index_html(reports)

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html.encode())

    def list_reports(self) -> List[Dict[str, Any]]:
        """List all available reports.

        Returns:
            List of report metadata dictionaries
        """
        reports = []
        reports_dir = Path("reports")

        if not reports_dir.exists():
            return reports

        for report_dir in sorted(reports_dir.iterdir(), reverse=True):
            if report_dir.is_dir():
                metadata_file = report_dir / "metadata.json"
                report_file = report_dir / "report.html"

                if report_file.exists():
                    metadata = {}
                    if metadata_file.exists():
                        try:
                            metadata = json.loads(metadata_file.read_text())
                        except Exception:
                            pass

                    reports.append(
                        {
                            "workflow_id": report_dir.name,
                            "path": f"/{report_dir.name}/report.html",
                            "generated_at": metadata.get("generated_at", "Unknown"),
                            "has_json": (report_dir / "report.json").exists(),
                        }
                    )

        return reports

    def generate_index_html(self, reports: List[Dict[str, Any]]) -> str:
        """Generate HTML index page.

        Args:
            reports: List of report metadata

        Returns:
            HTML string
        """
        reports_html = ""
        if not reports:
            reports_html = """
            <div style="text-align: center; padding: 40px; color: #7f8c8d;">
                <p>No reports available yet.</p>
                <p>Run a workflow to generate reports.</p>
            </div>
            """
        else:
            for report in reports:
                reports_html += f"""
                <div class="report-card">
                    <h3>
                        <a href="{report['path']}">
                            Workflow {report['workflow_id']}
                        </a>
                    </h3>
                    <div class="report-meta">
                        <span>📅 {report['generated_at']}</span>
                    </div>
                    <div class="report-actions">
                        <a href="{report['path']}" class="btn btn-primary">View HTML</a>
                        {f'<a href="/{report["workflow_id"]}/report.json" class="btn btn-secondary">View JSON</a>' if report['has_json'] else ''}
                    </div>
                </div>
                """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embodied AI Architect - Reports</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            padding: 20px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        header {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        h1 {{
            color: #2c3e50;
            font-size: 2em;
        }}
        .subtitle {{
            color: #7f8c8d;
            margin-top: 10px;
        }}
        .report-card {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .report-card h3 {{
            color: #34495e;
            margin-bottom: 10px;
        }}
        .report-card h3 a {{
            color: #3498db;
            text-decoration: none;
        }}
        .report-card h3 a:hover {{
            text-decoration: underline;
        }}
        .report-meta {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 15px;
        }}
        .report-actions {{
            display: flex;
            gap: 10px;
        }}
        .btn {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 4px;
            text-decoration: none;
            font-size: 0.9em;
            font-weight: 600;
        }}
        .btn-primary {{
            background: #3498db;
            color: white;
        }}
        .btn-primary:hover {{
            background: #2980b9;
        }}
        .btn-secondary {{
            background: #95a5a6;
            color: white;
        }}
        .btn-secondary:hover {{
            background: #7f8c8d;
        }}
        footer {{
            text-align: center;
            color: #7f8c8d;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 Embodied AI Architect</h1>
            <div class="subtitle">Report Viewer</div>
        </header>

        <div class="reports-list">
            {reports_html}
        </div>

        <footer>
            <p>Report Server v1.0 | Total Reports: {len(reports)}</p>
            <p>This server is optional and separate from the orchestrator</p>
        </footer>
    </div>
</body>
</html>"""
        return html


def start_server(port: int = 8000, host: str = "localhost"):
    """Start the report server.

    Args:
        port: Port to listen on
        host: Host to bind to
    """
    # Change to project root if we're in src/
    if Path.cwd().name == "src":
        os.chdir("..")

    # Create reports directory if it doesn't exist
    Path("reports").mkdir(exist_ok=True)

    handler = ReportHandler
    with socketserver.TCPServer((host, port), handler) as httpd:
        print(f"""
╔════════════════════════════════════════════════════════════╗
║  Embodied AI Architect - Report Server                    ║
╚════════════════════════════════════════════════════════════╝

  Server running at: http://{host}:{port}

  • View all reports: http://{host}:{port}/
  • Press Ctrl+C to stop

  This server simply serves static files from ./reports/
  It is completely separate from the orchestrator.
        """)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n✓ Server stopped")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embodied AI Architect Report Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", default="localhost", help="Host to bind to")

    args = parser.parse_args()
    start_server(port=args.port, host=args.host)
