---
sidebar_position: 1
---

# Introduction

**MCGrad** is a scalable algorithm for **multicalibration** — ensuring your model's predictions are calibrated not just globally, but across all meaningful subgroups of your data.

## The Problem

A model can be well-calibrated overall but poorly calibrated for specific segments. For example, your fraud detection model might be accurate on average, but systematically overconfident for mobile transactions in certain countries. These local calibration errors harm decision quality.

## The Solution

MCGrad takes your base model's predictions and automatically finds and fixes miscalibrated regions — without requiring you to specify which groups to protect.

```python
from multicalibration import methods

mcgrad = methods.MCBoost()
mcgrad.fit(
    df_train=df,
    prediction_column_name='prediction',
    label_column_name='label',
    categorical_feature_column_names=['country', 'device_type', 'content_type']
)

calibrated = mcgrad.predict(df, 'prediction', ['country', 'device_type', 'content_type'])
```

The result: predictions calibrated globally AND for any segment defined by your features.

## Key Benefits

- **No group specification** — Just provide features, MCGrad finds miscalibrated regions automatically
- **Improves performance** — Often improves log loss and PRAUC, not just calibration
- **Production-ready** — Built on LightGBM, scales to billions of samples

## Next Steps

- [Why MCGrad?](why-mcgrad.md) — Understand the challenges MCGrad solves and see results
- [Installation](installation.md) — Get started
- [Quick Start](quickstart.md) — Detailed usage examples
