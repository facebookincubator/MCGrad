# MCGrad

Production-ready multicalibration for machine learning.

## Installation

### Basic Installation

```bash
pip install -e .
```

### Development Installation

For development with testing and type checking:

```bash
# Clone with submodules (includes typeshed for Pyre)
git clone --recurse-submodules <repository-url>
cd MCGrad

# OR if already cloned, initialize submodules:
git submodule update --init --recursive

# Install development dependencies
pip install -e ".[dev]"
```

## Type Checking with Pyre

This project uses [Pyre](https://pyre-check.org/) for static type checking, Meta's preferred type checker.

### Running Pyre

After installing dev dependencies:

```bash
pyre check
```

### Configuration

- **`.pyre_configuration`** - Pyre configuration (committed to repo)
- **`third_party/typeshed/`** - Git submodule with Python standard library type stubs
- **`pyproject.toml`** - Contains third-party type stub packages (e.g., `pandas-stubs`, `types-requests`)

The configuration uses:
- **Git submodule** for typeshed (standard library stubs) - recommended approach per Pyre documentation
- **PyPI packages** (`types-*`) for third-party library stubs - installed via `pip install -e ".[dev]"`
- **PEP 561 search strategy** - automatically finds installed stub packages in site-packages

This ensures consistent typeshed versions across all contributors and follows Meta's open-source best practices.

## Testing

Run tests with:

```bash
pytest
```
