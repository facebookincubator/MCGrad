---
sidebar_position: 1
---

# Methods

Multicalibration methods for improving model calibration.

:::info Python API Documentation
For detailed Python API documentation with docstrings, please refer to the source code in `src/multicalibration/methods.py` or use Python's built-in help:

```python
from multicalibration import methods
help(methods.MCBoost)
```
:::

## MCGrad

The main multicalibration method using gradient boosting.

### Overview

MCGrad takes base model predictions and features, then builds a lightweight calibration layer using LightGBM. The resulting model is calibrated both globally and across virtually any segment defined by the features.

### Basic Usage

```python
from multicalibration.methods import MCBoost
import pandas as pd
import numpy as np

# Initialize MCGrad
mcboost = MCBoost(
    num_rounds=100,
    learning_rate=0.1,
    max_depth=3
)

# Prepare your data
df_train = pd.DataFrame({
    'prediction': np.array([...]),  # Your model's predictions
    'label': np.array([...]),        # Ground truth labels
    'country': [...],                 # Categorical features
    'content_type': [...],            # defining segments
    'numeric_feature': [...],         # Numerical features
})

# Fit on training data
mcboost.fit(
    df_train=df_train,
    prediction_column_name='prediction',
    label_column_name='label',
    categorical_feature_column_names=['country', 'content_type'],
    numerical_feature_column_names=['numeric_feature']
)

# Get calibrated predictions
calibrated_preds = mcboost.predict(
    df=df_train,
    prediction_column_name='prediction',
    categorical_feature_column_names=['country', 'content_type'],
    numerical_feature_column_names=['numeric_feature']
)
```

### Key Parameters

- `n_estimators`: Number of boosting rounds
- `learning_rate`: Step size shrinkage
- `max_depth`: Maximum tree depth
- `min_child_samples`: Minimum samples per leaf

### Methods

#### `fit(predictions, features, labels)`
Train the MCGrad calibration model.

**Parameters:**
- `predictions`: Array of base model predictions
- `features`: DataFrame of segment-defining features
- `labels`: Array of ground truth labels

#### `predict(predictions, features)`
Get calibrated predictions.

**Parameters:**
- `predictions`: Array of base model predictions to calibrate
- `features`: DataFrame of segment-defining features

**Returns:**
- Calibrated predictions as numpy array

## Other Calibration Methods

Additional calibration methods are available in the `methods` module for comparison:

- **Isotonic Regression** - Traditional univariate calibration
- **Platt Scaling** - Logistic regression-based calibration
- **Temperature Scaling** - Single parameter scaling method

See the [source code](https://github.com/facebookincubator/MCGrad/blob/main/src/multicalibration/methods.py) for full implementation details.
