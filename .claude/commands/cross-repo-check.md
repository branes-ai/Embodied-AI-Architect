Check cross-repo consistency:
1. Verify embodied-schemas is importable: `.venv/bin/python -c "from embodied_schemas import Registry; print('schemas OK')"`
2. Verify graphs is importable: `.venv/bin/python -c "from graphs.analysis.unified_analyzer import UnifiedAnalyzer; print('graphs OK')"`
3. Run this repo's full test suite: `.venv/bin/pytest tests/ -q`
4. Check for import errors in any test file.
Report which repos are healthy and flag any cross-boundary breakage.
