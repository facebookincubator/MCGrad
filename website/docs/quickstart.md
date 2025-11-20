---
sidebar_position: 4
---

# Quick Start

This guide will help you get started with MCBoost for multi-calibration.

## Basic Workflow

Here's a simple example of using MCBoost:

```python
from multicalibration import methods
import numpy as np
import pandas as pd

# Your model's predictions
predictions = np.array([0.1, 0.3, 0.7, 0.9, 0.5, 0.2])

# Features that define segments
features = pd.DataFrame({
    'country': ['US', 'UK', 'US', 'UK', 'US', 'UK'],
    'content_type': ['photo', 'video', 'photo', 'video', 'photo', 'video'],
    'surface': ['feed', 'feed', 'stories', 'stories', 'feed', 'stories'],
})

# Ground truth labels
labels = np.array([0, 0, 1, 1, 1, 0])

# Apply MCBoost
mcboost = methods.MCBoost()
mcboost.fit(predictions, features, labels)

# Get calibrated predictions
calibrated_predictions = mcboost.predict(predictions, features)
```

## Understanding the Output

The calibrated predictions will be:
- **Globally calibrated** - well-calibrated across all data
- **Multi-calibrated** - well-calibrated for any segment defined by the features
  - For `country=US`
  - For `content_type=photo`
  - For `country=US AND content_type=photo`
  - And many more combinations!

## Working with Different Methods

MCGrad provides several calibration methods. Explore the [methods API](api/methods.md) for available options.

## Evaluation Metrics

Use the metrics module to evaluate calibration quality:

```python
from multicalibration import metrics

# Evaluate calibration error
calibration_error = metrics.expected_calibration_error(
    predictions=calibrated_predictions,
    labels=labels
)

print(f"Expected Calibration Error: {calibration_error:.4f}")
```

See the [metrics API](api/metrics.md) for more evaluation options.

## Visualization

The plotting module provides tools for visualizing calibration:

```python
from multicalibration import plotting

# Create calibration plot
fig = plotting.plot_calibration_curve(
    predictions=calibrated_predictions,
    labels=labels
)
fig.show()
```

See the [plotting API](api/plotting.md) for more visualization options.

## Next Steps

- [Methodology](methodology.md) - Understand how MCBoost works
- [API Reference](api/methods.md) - Explore all available methods
- [Contributing](contributing.md) - Contribute to the project
