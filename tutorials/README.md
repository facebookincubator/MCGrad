# MCGrad Tutorials

Interactive tutorials demonstrating how to use MCGrad for multicalibration.

## ⚠️ Important: Do Not Edit `.ipynb` Files Directly

The `.ipynb` notebook files are **auto-generated** from the `.py` source files.

- ✅ **Edit:** `01_mcgrad_core.py` (source of truth)
- ❌ **Don't edit:** `01_mcgrad_core.ipynb` (auto-generated)

The sync happens automatically via pre-commit hooks. If you're contributing:

```bash
# Install pre-commit hooks (one-time setup)
pip install jupytext pre-commit
pre-commit install

# Generate initial .ipynb files (first time only)
./scripts/sync_notebooks.sh
# Or manually: jupytext --to notebook tutorials/01_mcgrad_core.py

# Now when you commit changes to .py files, .ipynb files are auto-regenerated
git add tutorials/01_mcgrad_core.py
git commit -m "Update tutorial"  # .ipynb is regenerated and added automatically
```

## Available Tutorials

| Tutorial | Description | Colab |
|----------|-------------|-------|
| [01. MCGrad Core Algorithm](01_mcgrad_core.py) | Complete introduction to multicalibration with MCGrad | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookincubator/MCGrad/blob/main/tutorials/01_mcgrad_core.ipynb) |

## Running Tutorials

### Option 1: Google Colab (Recommended for Quick Start)

Click the "Open in Colab" badge next to any tutorial above. No local setup required!

### Option 2: Local Jupyter

1. Install MCGrad and tutorial dependencies:

```bash
pip install "MCGrad[tutorials] @ git+https://github.com/facebookincubator/MCGrad.git"
```

2. Install Jupytext to work with percent-format Python files:

```bash
pip install jupytext
```

3. Convert to notebook and open:

```bash
jupytext --to notebook 01_mcgrad_core.py
jupyter notebook 01_mcgrad_core.ipynb
```

Or open directly in JupyterLab (with Jupytext extension installed):

```bash
jupyter lab 01_mcgrad_core.py
```

### Option 3: VS Code

Install the [Jupytext extension](https://marketplace.visualstudio.com/items?itemName=congyiwu.vscode-jupytext) and open the `.py` files directly as notebooks.

## Tutorial Format

Tutorials are written as [Jupytext](https://jupytext.readthedocs.io/) percent-format Python files. This format:

- **Version control friendly**: Clean diffs, no JSON noise
- **Lintable and testable**: Regular Python files
- **Flexible**: Open as notebooks or run as scripts

## Contributing

See the [Contributing Guide](../CONTRIBUTING.md) for information on adding new tutorials.
