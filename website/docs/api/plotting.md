---
sidebar_position: 3
description: Reference for MCGrad plotting functions. Visualize global and local calibration curves for your ML models.
---

# Plotting

Visualization tools for calibration analysis.

:::info Full API Reference
For complete API documentation with all parameters, return types, and detailed docstrings, see the [Sphinx API Reference](https://mcgrad.readthedocs.io/en/latest/api/plotting.html).
:::

## Global Calibration Curves

```python
from mcgrad import plotting

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
from mcgrad import plotting

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
from mcgrad import metrics, plotting

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
    quantity='segments_ecce_sigma',
)

fig.show()
```

See the [source code](https://github.com/facebookincubator/MCGrad/blob/main/src/mcgrad/plotting.py) for more visualization options.
