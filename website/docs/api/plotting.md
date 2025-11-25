---
sidebar_position: 3
---

# Plotting

Visualization tools for calibration analysis.

:::info Python API Documentation
For detailed documentation, refer to the source code or use Python's help:

```python
from multicalibration import plotting
help(plotting)
```
:::

## Calibration Curves

```python
from multicalibration import plotting
import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax = plt.subplots()

# Plot calibration curve
plotting.plot_calibration_curve(
    scores=calibrated_predictions,
    y=labels,
    df=df,
    segment_cols=['country', 'content_type'],
    ax=ax,
    num_bins=10
)

plt.show()
```

## Multi-Calibration Analysis

Visualize calibration across segments:

```python
from multicalibration import plotting
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

plotting.plot_calibration_curve_by_segment(
    scores=calibrated_predictions,
    y=labels,
    df=df,
    segment_cols=['country', 'content_type'],
    ax=ax
)

plt.show()
```

See the [source code](https://github.com/facebookincubator/MCGrad/blob/main/src/multicalibration/plotting.py) for more visualization options.
