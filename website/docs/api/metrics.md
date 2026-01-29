---
sidebar_position: 2
description: Reference for MCGrad metrics including MulticalibrationError (MCE) for measuring segment-level calibration quality.
---

# Metrics

Metrics for evaluating calibration quality.

:::info Full API Reference
For complete API documentation with all parameters, return types, and detailed docstrings, see the [Sphinx API Reference](https://mcgrad.readthedocs.io/en/latest/api/metrics.html).
:::

## Multicalibration Metrics

### Multicalibration Error (MCE)

Measures calibration quality across multiple segments of the data, not just globally.

```python
from mcgrad.metrics import MulticalibrationError

mce = MulticalibrationError(
    df=df,
    label_column='label',
    score_column='prediction',
    categorical_segment_columns=['country', 'content_type'],
    numerical_segment_columns=['numeric_feature']
)

print(f"MCE: {mce.mce}%")
print(f"P-value: {mce.mce_pvalue}")
print(f"Sigma scale: {mce.mce_sigma}")
```

The MulticalibrationError metric is based on the ECCE and provides a principled way to measure multi-calibration quality. For details on the methodology, see:

**Guy, I., Haimovich, D., Linder, F., Okati, N., Perini, L., Tax, N., & Tygert, M. (2025).** [Measuring multi-calibration](https://arxiv.org/abs/2506.11251). arXiv:2506.11251.

### Expected Calibration Error (ECE)

Measures the average difference between predicted probabilities and observed frequencies across bins.

```python
from mcgrad.metrics import expected_calibration_error

ece = expected_calibration_error(
    labels=labels,
    predicted_scores=predictions,
    num_bins=10
)
```

### Maximum Calibration Error

Measures the maximum difference between predicted probabilities and observed frequencies within bins.

This is available as a derived metric from the calibration error calculation.

## Predictive Performance Metrics

Standard ML metrics for comparing model performance:

- **Log Loss** - Negative log likelihood
- **Brier Score** - Mean squared difference between predictions and labels
- **PRAUC** - Precision-Recall Area Under Curve

See the [source code](https://github.com/facebookincubator/MCGrad/blob/main/src/mcgrad/metrics.py) for full implementation details.
