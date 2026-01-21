---
sidebar_position: 3
description: How to install MCGrad for Python. Requirements, pip installation, and development setup instructions.
---

# Installation

### Requirements

MCGrad requires Python 3.10 or later.

### From Source

To install from source:

```bash
git clone https://github.com/facebookincubator/MCGrad.git
cd MCGrad
pip install .
```

### Development Installation

For development, install with development dependencies:

```bash
pip install -e ".[dev]"
```

This will install the package in editable mode along with:
- pytest for running tests;
- flake8 for code linting.

### Verification

Verify your installation by running a minimal example:

```python
from multicalibration import methods
import numpy as np
import pandas as pd

# Create sample dataframe with 100 rows
rng = np.random.default_rng(42)
n_samples = 100
df = pd.DataFrame(
        {
            "device_type": rng.choice(["mobile", "desktop", "tablet"], size=n_samples),
            "market": rng.choice(["US", "UK"], size=n_samples),
            "prediction": rng.uniform(0, 1, size=n_samples),
            "label": rng.integers(0, 2, size=n_samples),
        }
    )

# Verify MCGrad can be instantiated and fit
mcgrad = methods.MCGrad()
mcgrad.fit(
    df_train=df,
    prediction_column_name='prediction',
    label_column_name='label',
    categorical_feature_column_names=['device_type', 'market']
)
```

---

### Next Steps

- [Quick Start](quickstart.md) - Start using MCGrad.
- [API Reference](api/methods.md) - Explore the API.
