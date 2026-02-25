# Conventional Commits and Semantic Versioning

## Quick Reference: What Bumps What?

Given a version `MAJOR.MINOR.PATCH` (e.g., `0.5.3`):

```
PATCH bump (0.5.3 -> 0.5.4)     fix:  perf:  refactor:
MINOR bump (0.5.3 -> 0.6.0)     feat:
MAJOR bump (0.5.3 -> 1.0.0)     feat!:  fix!:  or BREAKING CHANGE in footer
NO bump                          chore:  ci:  docs:  style:  test:
```

> **Note:** This project sets `major_on_zero = false` in `pyproject.toml`,
> so breaking changes while on `0.x` produce a minor bump instead of a major bump.
> A `1.0.0` release must be done intentionally.

## Commit Message Format

```
<type>[optional scope][!]: <description>

[optional body]

[optional footer(s)]
```

### Examples

```bash
# Patch bump
git commit -m "fix: correct timeout handling in SSH backend"
git commit -m "perf: reduce model loading time by 40%"
git commit -m "refactor: simplify agent pipeline execution"

# Minor bump
git commit -m "feat: add Kubernetes benchmark backend"
git commit -m "feat(cli): add codebase assessment command"

# Major bump (breaking change)
git commit -m "feat!: redesign CLI command structure"
git commit -m "fix!: change benchmark output format"

# Major bump via footer
git commit -m "feat: redesign CLI command structure

BREAKING CHANGE: all subcommands have been renamed"

# No version bump
git commit -m "chore: update dependencies"
git commit -m "ci: fix release workflow"
git commit -m "docs: add deployment guide"
git commit -m "style: format with Black"
git commit -m "test: add benchmark backend tests"
```

## Allowed Commit Types

| Type | Purpose | Version Bump |
|------|---------|--------------|
| `feat` | New feature or capability | Minor |
| `fix` | Bug fix | Patch |
| `perf` | Performance improvement | Patch |
| `refactor` | Code restructuring (no behavior change) | Patch |
| `docs` | Documentation only | None |
| `style` | Formatting, whitespace, linting | None |
| `test` | Adding or updating tests | None |
| `chore` | Maintenance, dependencies, tooling | None |
| `ci` | CI/CD pipeline changes | None |

## How It Works in This Project

1. **You push to `main`** (directly or via PR merge)
2. **GitHub Actions** runs the release workflow (`.github/workflows/release.yml`)
3. **`python-semantic-release`** scans all commits since the last version tag
4. If any commit warrants a bump, it:
   - Updates the version in `pyproject.toml`
   - Creates a git tag (e.g., `v0.6.0`)
   - Builds and publishes to PyPI
5. If no commit warrants a bump, the release step is skipped

### Multiple Commits — Highest Wins

When multiple commits are included (e.g., in a merge), the **highest bump wins**:

```
fix: correct edge case        -> would be patch
feat: add new command         -> would be minor
                              => result: minor bump
```

### Squash Merges vs Merge Commits

- **Squash merge**: Only the squash commit message matters. Use a conventional commit
  prefix in the PR title (GitHub uses the PR title as the squash commit message).
- **Merge commit**: All individual commits in the PR are scanned.

## Configuration

The semantic-release configuration lives in `pyproject.toml`:

```toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
major_on_zero = false        # prevents accidental 1.0.0
branch = "main"
tag_format = "v{version}"

[tool.semantic_release.commit_parser_options]
allowed_tags = ["feat", "fix", "docs", "chore", "refactor", "test", "ci", "perf", "style"]
minor_tags = ["feat"]
patch_tags = ["fix", "perf", "refactor"]
```

## References

- [Conventional Commits Specification](https://www.conventionalcommits.org)
- [Semantic Versioning (SemVer)](https://semver.org)
- [python-semantic-release docs](https://python-semantic-release.readthedocs.io)
