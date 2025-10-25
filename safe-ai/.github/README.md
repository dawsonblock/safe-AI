# .github Directory

This directory contains GitHub-specific files:

- `workflows/ci.yml` - Continuous Integration pipeline

The CI workflow:
1. Runs linting (ruff)
2. Runs type checking (mypy)
3. Runs tests with coverage
4. Validates Docker build
5. Checks for security issues

Status badges are shown in the main README.
