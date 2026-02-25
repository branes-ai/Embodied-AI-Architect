## Summary

<!-- Brief description of the changes -->

## Type of Change

- [ ] `feat`: New feature (minor version bump)
- [ ] `fix`: Bug fix (patch version bump)
- [ ] `perf`: Performance improvement (patch version bump)
- [ ] `refactor`: Code refactoring (patch version bump)
- [ ] `docs`: Documentation only (no release)
- [ ] `test`: Tests only (no release)
- [ ] `ci`: CI/CD changes (no release)
- [ ] `chore`: Maintenance (no release)

## Testing

- [ ] Tests pass locally (`pytest tests/ -v --tb=short --ignore=tests/agents/`)
- [ ] Linting passes (`black --check src/ tests/ --line-length 100 && ruff check src/ tests/`)
- [ ] New tests added for new functionality (if applicable)

## Conventional Commits

> **Reminder:** The PR title becomes the commit message on `main` after squash merge.
> Use the format: `type(scope): description`
>
> Examples: `feat(codebase): add ONNX export support`, `fix(benchmark): correct latency calculation`

## Notes for Reviewers

<!-- Optional: anything reviewers should know -->
