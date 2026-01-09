---
sidebar_position: 3
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

# Create minimal dataframe
df = pd.DataFrame(
        {
            "device_type": np.array(["mobile", "desktop", "tablet", "tablet", "desktop"]),
            "market": np.array(["US", "UK", "US", "UK", "US"]),
            "prediction": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            "label": np.array([0, 1, 0, 1, 0]),
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
