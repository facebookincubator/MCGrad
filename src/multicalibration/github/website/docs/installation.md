---
sidebar_position: 3
---

# Installation

## Requirements

MCGrad requires Python 3.10 or later.

## From Source

To install from source:

```bash
git clone https://github.com/facebookincubator/MCGrad.git
cd MCGrad
pip install .
```

## Development Installation

For development, install with development dependencies:

```bash
pip install -e ".[dev]"
```

This will install the package in editable mode along with:
- pytest for running tests
- black for code formatting
- isort for import sorting

## Dependencies

MCGrad depends on several core libraries:
- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **scipy** - Scientific computing
- **scikit-learn** - Machine learning utilities
- **torch** - PyTorch for deep learning components
- **lightgbm** - Gradient boosting framework (core for MCBoost)
- **plotly** - Interactive visualizations
- **matplotlib** - Static visualizations

All dependencies will be automatically installed when you install MCGrad.

## Verification

Verify your installation:

```python
import multicalibration
print("MCGrad installed successfully!")
```

## Next Steps

- [Quick Start](quickstart.md) - Start using MCBoost
- [API Reference](api/methods.md) - Explore the API
