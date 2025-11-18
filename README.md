# MCGrad

Production-ready multicalibration for machine learning.

## Development Setup

### Installation

Install the package in development mode with dev dependencies:

```bash
python3 -m pip install --user -e ".[dev]"
```

### Pre-commit Hooks

This project uses pre-commit hooks to maintain code quality. The hooks will automatically format your code before commits and run tests before pushes.

**Setup:**

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# Install pre-push hooks (for pytest)
pre-commit install --hook-type pre-push
```

**What runs:**

- **On every commit:** `isort` and `black` format your code automatically
- **On git push:** `pytest` runs the test suite

**Configuration:**

Code formatting settings are defined in `pyproject.toml` under `[tool.black]` and `[tool.isort]`.

**Manual execution:**

```bash
# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
```

## Continuous integration status
[![CI](https://github.com/facebookincubator/MCGrad/actions/workflows/main.yaml/badge.svg)](https://github.com/facebookincubator/MCGrad/actions/workflows/main.yaml)
