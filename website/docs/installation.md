---
sidebar_position: 3
description: How to install MCGrad for Python. Requirements, pip installation, and development setup instructions.
---

# Installation

### Requirements

MCGrad requires Python 3.10 or later.

### Pip

Install the latest stable release from PyPI:
```bash
pip install mcgrad
```

To install the latest development version from GitHub:
```bash
pip install git+https://github.com/facebookincubator/MCGrad.git
```

See [Verification](#verification) below to confirm the install worked.

### Conda / Mamba

MCGrad is also available via conda:
```bash
conda install mcgrad
```

Or, using [mamba](https://mamba.readthedocs.io/):
```bash
mamba install mcgrad
```

### From Source (advanced/contributors)

Most users should install via pip above. Installing from source is intended for advanced users who need to build locally or contributors working on the codebase.

To install from source:

```bash
git clone https://github.com/facebookincubator/MCGrad.git
cd MCGrad
pip install .
```

### Development Installation (contributors only)

This is only needed if you plan to contribute to MCGrad or develop locally. It installs MCGrad in editable mode and adds development dependencies.

```bash
pip install -e ".[dev]"
```

This includes development tooling:
- pytest for running tests;
- flake8 for code linting.

### Verification

You can verify that MCGrad is correctly installed (regardless of the installation method used) by running a minimal example:

```python
from mcgrad import methods
import numpy as np
import pandas as pd

# Create sample DataFrame with 100 rows
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

- [Quick Start](quickstart.md) — Start using MCGrad.
- [API Reference](https://mcgrad.readthedocs.io/) — Explore the API.
