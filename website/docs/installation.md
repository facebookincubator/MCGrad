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
- **lightgbm** - Gradient boosting framework (core for MCGrad)
- **plotly** - Interactive visualizations
- **matplotlib** - Static visualizations

All dependencies will be automatically installed when you install MCGrad.

## Verification

Verify your installation by running a minimal example:

```python
from multicalibration import methods
import numpy as np
import pandas as pd

# Create minimal test data
df = pd.DataFrame({
    'prediction': np.array([0.2, 0.8]),
    'label': np.array([0, 1]),
    'segment': ['a', 'b']
})

# Verify MCGrad can be instantiated and fit
mcboost = methods.MCBoost()
mcboost.fit(
    df_train=df,
    prediction_column_name='prediction',
    label_column_name='label',
    categorical_feature_column_names=['segment']
)
```

## Next Steps

- [Quick Start](quickstart.md) - Start using MCGrad
- [API Reference](api/methods.md) - Explore the API
