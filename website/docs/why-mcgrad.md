---
sidebar_position: 2
---

# Why MCGrad?

MCGrad offers significant advantages over traditional calibration methods like Isotonic Regression.

## The Problem with Traditional Calibration

While methods like Isotonic Regression work well globally, they fail to maintain calibration when looking at specific segments of the data.

For example, when examining different segments (e.g., reactively reported Videos vs. proactively reported Posts), Isotonic Regression results in major calibration deviations, while MCGrad maintains excellent calibration across all segments.

## Key Benefits

### 1. Powerful Multicalibration

MCGrad can take a virtually unlimited number of features and optimize for multicalibration with respect to all of them, ensuring good calibration across segments.

**Unlike traditional methods**, MCGrad improves calibration not just for a handful of segments, but for a huge number of segments that don't need to be pre-specified.

### 2. Data Efficiency

MCGrad borrows information from similar samples just like any other modern ML model. As a result, it can calibrate far more small segments than alternatives which calibrate each segment separately.

### 3. Lightweight Training and Inference

The method is implemented using LightGBM, a highly optimized gradient boosting framework. It is typically **orders of magnitude faster** than training heavier NN-based models.

### 4. Improved Predictive Performance

MCGrad is a **likelihood-improving procedure**, meaning it can only improve model performance on training data (the same isn't true for e.g. Isotonic Regression).

In many cases, we observe:
- Significant likelihood/PRAUC improvements
- Better model performance metrics
- Improved business outcomes

## Learn More

- [Methodology](methodology.md) - Deep dive into how MCGrad works
- [Quick Start](quickstart.md) - Start using MCGrad
- [Research Paper](https://arxiv.org/abs/2509.19884) - MCGrad: Multicalibration at Web Scale (KDD 2026)
