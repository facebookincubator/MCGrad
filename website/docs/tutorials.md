---
sidebar_position: 5
---

# Tutorials

Interactive Jupyter notebooks demonstrating how to use MCGrad for multicalibration.

## Available Tutorials

| Tutorial | Description | Launch |
|----------|-------------|--------|
| **MCGrad Core Algorithm** | Complete introduction to multicalibration with MCGrad | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookincubator/MCGrad/blob/main/tutorials/01_mcgrad_core.ipynb) |

## What You'll Learn

### 01. MCGrad Core Algorithm

This comprehensive tutorial covers:

1. **Why Multicalibration Matters** - Understand the limitations of global calibration and why segment-level calibration is important
2. **MCGrad Basics** - Learn how to use the `fit()` and `predict()` API
3. **Measuring Multicalibration** - Use the Multicalibration Error (MCE) metric to evaluate calibration quality
4. **Visualization** - Plot global and segment-level calibration curves
5. **Advanced Features** - Explore feature importance, model serialization, numerical features, and custom hyperparameters

## Running Tutorials

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badge above to run tutorials directly in your browser. No local setup required!

### Option 2: Local Jupyter

1. Install MCGrad with tutorial dependencies:

```bash
pip install "MCGrad[tutorials] @ git+https://github.com/facebookincubator/MCGrad.git"
```

2. Clone the repository and navigate to tutorials:

```bash
git clone https://github.com/facebookincubator/MCGrad.git
cd MCGrad/tutorials
```

3. Convert to notebook format and open:

```bash
jupytext --to notebook 01_mcgrad_core.py
jupyter notebook 01_mcgrad_core.ipynb
```

### Option 3: VS Code

Install the [Jupytext extension](https://marketplace.visualstudio.com/items?itemName=congyiwu.vscode-jupytext) and open the `.py` files directly as notebooks.

## Tutorial Format

Our tutorials are written as [Jupytext](https://jupytext.readthedocs.io/) percent-format Python files (`.py` with `# %%` cell markers). This format provides several benefits:

- **Version control friendly** - Clean diffs without JSON noise
- **Lintable and testable** - Can be checked with standard Python tools
- **Flexible** - Open as notebooks or run as Python scripts
- **CI integration** - Tutorials are tested on every push to ensure they stay up-to-date

## Contributing Tutorials

We welcome contributions! If you'd like to add a new tutorial:

1. Create a new `.py` file in the `tutorials/` directory following the Jupytext percent format
2. Follow the naming convention: `XX_descriptive_name.py`
3. Include a Colab badge at the top
4. Add the tutorial to this documentation page
5. Submit a pull request

See the [Contributing Guide](contributing.md) for more details.
