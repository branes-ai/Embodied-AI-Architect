# Contributing to Embodied AI Architect

## Development Setup

```bash
# Clone and install in development mode
git clone <repo-url>
cd embodied-ai-architect
pip install -e ".[dev]"
```

## Workflow

We use a **feature branch workflow** with squash merges to `main`.

### 1. Create a Feature Branch

```bash
git checkout -b feat/my-feature main
```

### 2. Make Changes and Test Locally

```bash
# Run linting
black --check src/ tests/ --line-length 100
ruff check src/ tests/

# Run tests
pytest tests/ -v --tb=short --ignore=tests/agents/

# Auto-format
black src/ tests/ --line-length 100
```

### 3. Commit with Conventional Commits

We use [Conventional Commits](https://www.conventionalcommits.org/) for automatic versioning and releases.

| Prefix | Example | Release |
|--------|---------|---------|
| `feat` | `feat(codebase): add ONNX export` | Minor (0.X.0) |
| `fix` | `fix(benchmark): correct latency calc` | Patch (0.0.X) |
| `perf` | `perf(pipeline): batch inference` | Patch (0.0.X) |
| `refactor` | `refactor(agents): simplify base class` | Patch (0.0.X) |
| `docs` | `docs: update README` | No release |
| `test` | `test: add hardware profile tests` | No release |
| `ci` | `ci: add Python 3.12 to matrix` | No release |
| `chore` | `chore: update dependencies` | No release |

**Breaking changes:** Add `BREAKING CHANGE:` in the commit footer or `!` after the type (e.g., `feat!: ...`) for a major version bump.

### 4. Push and Open a PR

```bash
git push -u origin feat/my-feature
```

Open a PR targeting `main`. CI will run automatically:
- **Lint** — Black + Ruff formatting and style checks
- **Test** — Full test suite on Python 3.11 and 3.12
- **Test (optional deps)** — Optimization tests (non-blocking)

### 5. Squash Merge

PRs are squash-merged. **The PR title becomes the commit message on `main`**, so use conventional commit format for your PR title:

```
feat(codebase): add ONNX export support
```

After merge, `python-semantic-release` automatically:
1. Determines the version bump from the commit message
2. Updates `pyproject.toml` version
3. Creates a git tag (e.g., `v0.6.0`)
4. Publishes to PyPI

## Testing Optional Dependencies

Some tests require optional packages (botorch, pymoo, langgraph). These tests auto-skip if the dependency isn't installed.

```bash
# Install optimization deps and run those tests
pip install -e ".[dev,optimization]"
pytest tests/test_moo_*.py tests/test_bayesian_*.py -v

# Install chat deps
pip install -e ".[dev,chat]"
```

## Code Style

- **Line length:** 100 characters
- **Python:** 3.11+
- **Formatter:** Black
- **Linter:** Ruff
- **Type hints:** Required for public APIs
