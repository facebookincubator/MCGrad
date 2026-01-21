---
sidebar_position: 4
description: Get started with MCGrad for multicalibration. Learn how to install, train, and evaluate multicalibrated ML models.
---

# Quick Start

This guide will help you get started with MCGrad for multicalibration.

## A Basic Multicalibration Workflow

### 1. Prepare your data

You need a DataFrame with the following columns:
- `label` - the true binary label of the data point;
- `prediction` - the uncalibrated prediction score;
- `categorical_feature_1`, `categorical_feature_2`, ... - *optional* categorical features to identify the segments;
- `numerical_feature_1`, `numerical_feature_2`, ... - *optional* numerical features to identify the segments.

Note that you don't need to use all the features, but at least one feature is desirable as it is necessary to identify the segments.

Also, MCGrad requires that predictions:
- are in the range [0, 1];
- cannot be NaNs.

```python
from multicalibration import methods, metrics, plotting
import numpy as np
import pandas as pd

# Generate synthetic data with segment-specific miscalibration
rng = np.random.default_rng(42)
n_samples = 10000

df = pd.DataFrame({
    'country': rng.choice(['US', 'UK'], size=n_samples),
    'content_type': rng.choice(['photo', 'video'], size=n_samples),
    'surface': rng.choice(['feed', 'stories'], size=n_samples),
})

# True probability depends on segments (US and video have higher rates)
df['true_prob'] = 0.5 + 0.15 * (df['country'] == 'US') + 0.1 * (df['content_type'] == 'video')
df['label'] = (rng.uniform(size=n_samples) < df['true_prob']).astype(int)

# Predictions ignore segment effects (miscalibrated)
df['prediction'] = np.clip(rng.uniform(0.3, 0.7, size=n_samples), 0.01, 0.99)
```

### 2. Apply MCGrad to Uncalibrated Predictions with Features

MCGrad requires consistency between the categorical and numerical features passed to the `fit` and to the `predict` methods.

Despite being the same for this trivial example, the dataframes passed to the `fit` and to the `predict` methods can be (and usually are) different.

```python
# Apply MCGrad
mcgrad = methods.MCGrad()
mcgrad.fit(
    df_train=df,
    prediction_column_name='prediction',
    label_column_name='label',
    categorical_feature_column_names=['country', 'content_type', 'surface'],
    numerical_feature_column_names=[] # optional, can be empty
)

# Get multicalibrated predictions and add to the dataframe
df['prediction_mcgrad'] = mcgrad.predict(
    df=df,
    prediction_column_name='prediction',
    categorical_feature_column_names=['country', 'content_type', 'surface'],
    numerical_feature_column_names=[] # optional, can be empty
)
```

The multicalibrated predictions will be both **globally calibrated** and **multicalibrated**:
- **Globally calibrated** - Well-calibrated across all data.
- **Multicalibrated** - Well-calibrated for any segment defined by the features, i.e.
  - For `country=US`, `country=UK`, `content_type=photo`, ...
  - For intersections like `country=US AND content_type=photo`;
  - ... And many more combinations!

### 3. Apply Global Calibration Models for Comparison

The MCGrad package provides a few global calibration methods for comparison. For example, you can apply Isotonic Regression on the previous dataframe:

```python
# Apply Isotonic Regression
isotonic_regression = methods.IsotonicRegression()
isotonic_regression.fit(
    df_train=df,
    prediction_column_name='prediction',
    label_column_name='label')

# Get globally calibrated predictions and add to the dataframe
df['prediction_isotonic'] = isotonic_regression.predict(
    df=df,
    prediction_column_name='prediction'
)
```

Explore the [methods API](api/methods.md) for other available global calibration methods.


### 4. Model Evaluation: the Multicalibration Error Metric

To rigorously evaluate whether your model is multicalibrated, you can use the `MulticalibrationError` class, which provides several important attributes:
- **Multicalibration Error (MCE)**: The plain Multicalibration Error metric that is expressed in a percent scale. Briefly, it measures the largest deviation from perfect calibration over all segments. This is computed by `.mce` attribute.
- **MCE Sigma Scale**: The Multicalibration Error normalized by its standard deviation under the null hypothesis of perfect calibration. It represents the largest segment error in the number of standard deviations of the metric. Conceptually equivalent to a p-value, it allows to assess the amount of statistical evidence of miscalibration. This is computed by `.mce_sigma_scale` attribute.
- **p-value**: Statistical p-value measured under the null hypothesis of perfect calibration. It can help us determine whether the miscalibration we are seeing is statistically significant. This is computed by `.p_value` attribute.
- **Minimum Detectable Error (MDE)**: This tells us roughly what values of MCE (percent scale) can be detected using the dataset. It is computed by `.mde` attributed.

```python
# Initialize the MulticalibrationError metric
mce = metrics.MulticalibrationError(
    df=df,
    label_column='label',
    score_column='prediction',
    categorical_segment_columns=['country', 'content_type', 'surface'],
    numerical_segment_columns=[],
)

# Print key calibration metrics
print(f"Multicalibration Error (MCE): {mce.mce:.3f}%")
print(f"MCE Sigma Scale: {mce.mce_sigma_scale:.3f}")
print(f"MCE p-value: {mce.p_value:.4f}")
print(f"Minimum Detectable Error (MDE): {mce.mde:.3f}%")
```

See the [metrics API](api/metrics.md) for more evaluation options.

### 5. Compare Calibration Methods
Compare the multicalibration error metrics across the three methods: uncalibrated, isotonic regression, and MCGrad. Given that the data presents segment-specific miscalibration, we expect MCGrad to outperform the other methods.

```python
# Define methods to compare
score_columns = {
    'Uncalibrated': 'prediction',
    'Isotonic': 'prediction_isotonic',
    'MCGrad': 'prediction_mcgrad',
}

# Compute MCE metrics for each method
results = []
for method_name, score_col in score_columns.items():
    mce = metrics.MulticalibrationError(
        df=df,
        label_column='label',
        score_column=score_col,
        categorical_segment_columns=['country', 'content_type', 'surface'],
    )
    results.append({
        'Method': method_name,
        'MCE': round(mce.mce, 2),
        'MCE σ': round(mce.mce_sigma_scale, 2),
        'p-value': round(mce.p_value, 4),
    })

# Display comparison table
pd.DataFrame(results).set_index('Method')
```

The expected output is:

| Method       | MCE   | MCE σ | p-value |
|--------------|-------|-------|---------|
| Uncalibrated | 36.33 | 28.34 | 0.0000  |
| Isotonic     | 17.30 | 13.54 | 0.0000  |
| MCGrad       | 2.99  | 2.38  | 0.0691  |

As expected, MCGrad significantly reduces the Multicalibration Error compared to both uncalibrated predictions and Isotonic Regression. The p-value for MCGrad (0.0691) indicates that there is no statistically significant evidence of miscalibration remaining.

### 6. Visualization of (Multi)calibration Error

The plotting module provides tools for visualizing (multi)calibration.

**Global Calibration Curves**

The global calibration curve shows the average label per score bin. Perfect calibration corresponds to all bin means overlapping with the diagonal line (i.e. the average label is equal to the average score in the bin). 95% confidence intervals for the estimate of the average label are shown. The histogram in the background shows the distribution of the score.

You can plot the global calibration curve for any other model by changing the `score_col` argument.

```python
# Plot Global Calibration Curve
fig = plotting.plot_global_calibration_curve(
    data=df,
    score_col="prediction_mcgrad",  # Change this to any model's score column
    label_col="label",
    num_bins=40,
).update_layout(title="Global Calibration Curve for MCGrad", width=700)

fig.show()
```

<div style={{textAlign: 'center'}}>
<img src={require('../static/img/MCGrad_global_calibration_curve.png').default} alt="global calibration curve mcgrad" width="50%" />
</div>

**Local Calibration Curves**

Here you can plot the calibration curves for single feature segment. I.e. one curve per feature value. This allows you to inspect if the model is calibrated for specific segments (e.g. `country=US`, `content_type=video`), even if it appears calibrated globally.

```python
features_to_plot = ['country', 'content_type']

for feature in features_to_plot:
    n_cats = df[feature].nunique()

    # Plot Local Calibration Curves
    fig = plotting.plot_calibration_curve_by_segment(
        data=df,
        group_var=feature,
        score_col="prediction_mcgrad", # Change this to any model's score column
        label_col="label",
        n_cols=3,
    ).update_layout(
        title=f"MCGrad Local Calibration for {feature}",
        width=2000,
        height=max(10.0, 500 * (np.ceil(n_cats / 3))),
    )
    fig.show()
```

<div style={{textAlign: 'center'}}>
<img src={require('../static/img/mcgrad_country.png').default} alt="local calibration country mcgrad" width="80%" />
</div>

<div style={{textAlign: 'center'}}>
<img src={require('../static/img/mcgrad_content.png').default} alt="local calibration content mcgrad" width="80%" />
</div>

See the [plotting API](api/plotting.md) for more visualization options.

## Next Steps

- [Methodology](methodology.md) - Understand how MCGrad works formally.
- [Measuring Multicalibration](measuring-multicalibration.md) - Undertand how the Multicalibration Error works formally.
- [API Reference](api/methods.md) - Explore all available methods.
- [Contributing](contributing.md) - Contribute to the project.
