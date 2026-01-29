# MCGrad Tutorials

Interactive Jupyter notebooks demonstrating how to use MCGrad for multicalibration.

## Available Tutorials

| Tutorial | Description | Colab |
|----------|-------------|-------|
| [01. MCGrad Core Algorithm](01_mcgrad_core.ipynb) | Complete introduction to multicalibration with MCGrad | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookincubator/MCGrad/blob/main/tutorials/01_mcgrad_core.ipynb) |

## Running Tutorials

### Option 1: Google Colab (Recommended for Quick Start)

Click the "Open in Colab" badge next to any tutorial above. No local setup required!

### Option 2: Local Jupyter

1. Install MCGrad and tutorial dependencies:

```bash
pip install "MCGrad[tutorials] @ git+https://github.com/facebookincubator/MCGrad.git"
```

2. Clone and open notebooks:

```bash
git clone https://github.com/facebookincubator/MCGrad.git
cd MCGrad/tutorials
jupyter notebook 01_mcgrad_core.ipynb
```

### Option 3: VS Code

Open the `.ipynb` files directly in VS Code with the built-in Jupyter extension.

## Contributing

To add a new tutorial:

1. Create a new `.ipynb` file following the naming convention `XX_descriptive_name.ipynb`
2. Include a Colab setup cell at the top (see existing tutorials for the pattern)
3. Add your tutorial to the table above and to the website docs
4. Submit a pull request

See the [Contributing Guide](../CONTRIBUTING.md) for more details.
