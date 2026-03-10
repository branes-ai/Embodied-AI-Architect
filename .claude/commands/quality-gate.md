Run the mandatory pre-push quality gate:
1. `.venv/bin/black --check src/ tests/ --line-length 100`
2. `.venv/bin/ruff check src/ tests/`
3. `.venv/bin/pytest tests/ -q`
Report results concisely. If anything fails, fix it.
