---
sidebar_position: 7
description: How to contribute to MCGrad. Development setup, coding standards, and pull request guidelines.
---

# Contributing

We welcome contributions to MCGrad! This guide will help you get started.

## Development Setup

1. Fork the repository on GitHub

2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/MCGrad.git
   cd MCGrad
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Set up pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   pre-commit install --hook-type pre-push
   ```

## Code Quality

We use several tools to maintain code quality:

- **flake8** - Code linting
- **pytest** - Unit tests

Run all checks:
```bash
pre-commit run --all-files
```

## Running Tests

```bash
pytest src/mcgrad/tests/ -v
```

With coverage:
```bash
pytest src/mcgrad/tests/ --cov=mcgrad --cov-report=html
```

## Submitting Changes

1. Create a new branch:
   ```bash
   git checkout -b my-feature
   ```

2. Make changes and commit:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

3. Push to your fork:
   ```bash
   git push origin my-feature
   ```

4. Open a Pull Request on GitHub

## Pull Request Guidelines

- Include tests for new functionality
- Update documentation as needed
- Ensure all tests pass
- Follow the existing code style
- Write clear commit messages

## Documentation

To build the documentation locally:

```bash
cd website
npm install
npm start
```

This will start a local dev server at `http://localhost:3000`.

## License

By contributing, you agree that your contributions will be licensed under the MIT license.
