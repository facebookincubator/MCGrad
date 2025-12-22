---
sidebar_position: 4
---

# Quick Start

This guide will help you get started with MCGrad for multicalibration.

## Basic Workflow

Here's a simple example of using MCGrad:

```python
from multicalibration import methods
import numpy as np
import pandas as pd

# Prepare your data in a DataFrame
df = pd.DataFrame({
    'prediction': np.array([0.1, 0.3, 0.7, 0.9, 0.5, 0.2]),
    'label': np.array([0, 0, 1, 1, 1, 0]),
    'country': ['US', 'UK', 'US', 'UK', 'US', 'UK'],
    'content_type': ['photo', 'video', 'photo', 'video', 'photo', 'video'],
    'surface': ['feed', 'feed', 'stories', 'stories', 'feed', 'stories'],
})

# Apply MCGrad
mcboost = methods.MCBoost()
mcboost.fit(
    df_train=df,
    prediction_column_name='prediction',
    label_column_name='label',
    categorical_feature_column_names=['country', 'content_type', 'surface']
)

# Get calibrated predictions
calibrated_predictions = mcboost.predict(
    df=df,
    prediction_column_name='prediction',
    categorical_feature_column_names=['country', 'content_type', 'surface']
)
```

## Understanding the Output

The calibrated predictions will be:
- **Globally calibrated** - Well-calibrated across all data
- **Multi-calibrated** - Well-calibrated for any segment defined by the features
  - For `country=US`
  - For `content_type=photo`
  - For intersections like `country=US AND content_type=photo`
  - And many more combinations!

## Working with Different Methods

MCGrad provides several calibration methods. Explore the [methods API](api/methods.md) for available options.

## Evaluation Metrics

Use the metrics module to evaluate calibration quality:

```python
from multicalibration.metrics import expected_calibration_error
import numpy as np

# Extract labels from the DataFrame
labels = df['label'].values

# Evaluate calibration error
ece = expected_calibration_error(
    labels=labels,
    predicted_scores=calibrated_predictions,
    num_bins=10
)

print(f"Expected Calibration Error: {ece:.4f}")
```

See the [metrics API](api/metrics.md) for more evaluation options.

## Visualization

The plotting module provides tools for visualizing calibration:

```python
from multicalibration import plotting
import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax = plt.subplots()

# Create calibration plot
plotting.plot_calibration_curve(
    scores=calibrated_predictions,
    y=df['label'].values,
    df=df,
    segment_cols=['country', 'content_type', 'surface'],
    ax=ax,
    num_bins=10
)

plt.show()
```

See the [plotting API](api/plotting.md) for more visualization options.

## Next Steps

- [Methodology](methodology.md) - Understand how MCGrad works
- [API Reference](api/methods.md) - Explore all available methods
- [Contributing](contributing.md) - Contribute to the project
