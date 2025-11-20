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
from multicalibration.plotting import plot_calibration_curve

fig = plot_calibration_curve(
    predictions=preds,
    labels=labels,
    n_bins=10
)
fig.show()
```

## Multi-Calibration Analysis

Visualize calibration across segments:

```python
from multicalibration.plotting import plot_segment_calibration

fig = plot_segment_calibration(
    predictions=preds,
    labels=labels,
    segments=features
)
fig.show()
```

See the [source code](https://github.com/facebookincubator/MCGrad/blob/main/src/multicalibration/plotting.py) for more visualization options.
