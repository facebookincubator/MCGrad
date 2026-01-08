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

## Global Calibration Curves

```python
from multicalibration import plotting

# Plot global calibration curve
fig = plotting.plot_global_calibration_curve(
    data=df,
    score_col='prediction',
    label_col='label',
    sample_weight_col='weights',  # optional
)

fig.show()
```

## Multicalibration Analysis

Visualize calibration across segments:

```python
from multicalibration import plotting

# Plot calibration curves for each segment
fig = plotting.plot_calibration_curve_by_segment(
    data=df,
    group_var='country',
    score_col='prediction',
    label_col='label',
)

fig.show()
```

## Segment Calibration Errors

Visualize calibration errors across multiple segments:

```python
from multicalibration import metrics, plotting

# Create a MulticalibrationError object
mce = metrics.MulticalibrationError(
    df=df,
    label_column='label',
    score_column='prediction',
    categorical_segment_columns=['country', 'content_type'],
)

# Plot segment calibration errors
fig = plotting.plot_segment_calibration_errors(
    mce=mce,
    quantity='segment_ecces_sigma_scale',
)

fig.show()
```

See the [source code](https://github.com/facebookincubator/MCGrad/blob/main/src/multicalibration/plotting.py) for more visualization options.
