---
sidebar_position: 1
---

# Methods

Multi-calibration methods for improving model calibration.

:::info Python API Documentation
For detailed Python API documentation with docstrings, please refer to the source code in `src/multicalibration/methods.py` or use Python's built-in help:

```python
from multicalibration import methods
help(methods.MCBoost)
```
:::

## MCBoost

The main multi-calibration method using gradient boosting.

### Overview

MCBoost takes base model predictions and features, then builds a lightweight calibration layer using LightGBM. The resulting model is calibrated both globally and across virtually any segment defined by the features.

### Basic Usage

```python
from multicalibration.methods import MCBoost

# Initialize
mcboost = MCBoost(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)

# Fit on training data
mcboost.fit(predictions, features, labels)

# Get calibrated predictions
calibrated_preds = mcboost.predict(predictions, features)
```

### Key Parameters

- `n_estimators`: Number of boosting rounds
- `learning_rate`: Step size shrinkage
- `max_depth`: Maximum tree depth
- `min_child_samples`: Minimum samples per leaf

### Methods

#### `fit(predictions, features, labels)`
Train the MCBoost calibration model.

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
