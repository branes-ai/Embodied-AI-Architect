"""REST API server for the Branes Embodied AI Architect platform.

Thin read-only layer over SessionStore, exposing design session data
for the branes-frontend and other consumers.

Usage:
    # CLI
    branes api serve --port 8000

    # Programmatic
    from embodied_ai_architect.api import create_app
    app = create_app()
"""

from .server import create_app

__all__ = ["create_app"]
