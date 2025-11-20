---
sidebar_position: 2
---

# Metrics

Metrics for evaluating calibration quality.

:::info Python API Documentation
For detailed Python API documentation with docstrings, refer to the source code or use Python's help:

```python
from multicalibration import metrics
help(metrics)
```
:::

## Calibration Metrics

### Expected Calibration Error (ECE)

Measures the average difference between predicted probabilities and observed frequencies across bins.

```python
from multicalibration.metrics import expected_calibration_error

ece = expected_calibration_error(
    predictions=preds,
    labels=labels,
    n_bins=10
)
```

###Maximum Calibration Error (MCE)

Measures the maximum difference between predicted probabilities and observed frequencies.

```python
from multicalibration.metrics import maximum_calibration_error

mce = maximum_calibration_error(
    predictions=preds,
    labels=labels,
    n_bins=10
)
```

## Multi-Calibration Metrics

### Segment Calibration Error

Evaluates calibration within specific segments.

```python
from multicalibration.metrics import segment_calibration_error

segment_errors = segment_calibration_error(
    predictions=preds,
    labels=labels,
    segments=segment_ids
)
```

## Predictive Performance Metrics

Standard ML metrics for comparing model performance:

- **Log Loss** - Negative log likelihood
- **Brier Score** - Mean squared difference between predictions and labels
- **PRAUC** - Precision-Recall Area Under Curve

See the [source code](https://github.com/facebookincubator/MCGrad/blob/main/src/multicalibration/metrics.py) for full implementation details.
