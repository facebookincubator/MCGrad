---
sidebar_position: 6
description: How to measure multicalibration using the Multicalibration Error (MCE) metric. Understand segment-level calibration quality.
---

# Measuring Multicalibration

The Multicalibration Error (MCE) metric quantifies how well a model is multicalibrated across relevant segments (also known as protected groups) of your data. It is used to assess the success of applying a multicalibration algorithm, like MCGrad, by measuring MCE both before and after the multicalibration step. For additional details, see [1].

## What is MCE and How Can I Use It?

The Multicalibration Error measures the worst-case noise-weighted calibration error across all segments that can be formed from your specified features. It answers the question: "What is the maximum miscalibration in any segment of my data?"

#### Key Properties
- $\text{MCE} = 0$ indicates perfect multicalibration;
- Lower values are better;
- Parameter-free - no arbitrary binning decisions required;
- Statistically interpretable - includes p-values.


#### Quick Start: Computing the MCE

```python
from mcgrad.metrics import MulticalibrationError

mce = MulticalibrationError(
    df=df,
    label_column='label',
    score_column='prediction',
    categorical_segment_columns=['region', 'age_group'],
    numerical_segment_columns=['income']
)

print(f"MCE: {mce.mce:.2f}%")
print(f"P-value: {mce.mce_pvalue:.4f}")
print(f"Worst segment: {mce.mce_sigma:.2f} standard deviations")
```

The output is interpreted as follows:

- **`mce`**: The multicalibration error.
- **`mce_pvalue`**: Statistical significance of the worst miscalibration.
- **`mce_sigma`**: MCE in units of standard deviations (for statistical significance).

## How Does the MCE Work?

The Multicalibration Error metric is computed in three steps:

<div style={{textAlign: 'center'}}>
<img src={require('../static/img/mce_metric_design.png').default} alt="mce metric design" width="100%" />
</div>

### Step 1: Define a Set of Segments

First, specify the collection of segments over which calibration will be assessed. In some applications, segments may be predefined based on domain knowledge (e.g., demographic groups, device types). More commonly, a systematic approach is used:

- Each feature is partitioned into $k$ categories (such as quantiles for numerical features or grouped levels for categorical features).
- All intersections of up to $n$ features are considered, generating a rich set of segments that capture meaningful subpopulations in the data.

**Example:** In the [ad click prediction scenario](intro.md#a-motivating-example), to assess multicalibration using the features `{market, device_type}`, the set of segments would include:

- **Segment 0:** All samples (global population);
- **Segment 1:** Users in the US market;
- **Segment 2:** Users in non-US markets;
- **Segment 3:** Users on mobile devices;
- **Segment 4:** Users on desktop devices;
- **Segment 5:** Users in the US market AND using mobile;
- **Segment 6:** Users in the US market AND using desktop;
- **Segment 7:** Users in non-US markets AND using mobile;
- **Segment 8:** Users in non-US markets AND using desktop.

The number of segments grows exponentially with the number of features. Therefore, the number of features and categories per feature used to identify segments are limited to $n$ and $k$, respectively.

### Step 2: Measure Calibration Error Within Each Segment

After defining segments, assess how well the model's predicted probabilities match the observed outcomes within each segment.

For each segment, the **[Estimated Cumulative Calibration Error (ECCE)][2]** is computed, which captures the sum of deviations between predicted scores and actual labels over any interval of scores in that segment.

Why ECCE?

- **ECCE is parameter-free:** It does not require arbitrary choices like bin sizes.
- **ECCE is statistically interpretable:** Its value can be directly related to statistical significance, enabling distinction between genuine miscalibration and random fluctuations.

**Example:**
Continuing with the ad click prediction scenario, for each segment (e.g., "US market AND mobile", "non-US market AND desktop"):

- Samples in the segment are sorted by predicted probability.
- For every possible interval of scores, the cumulative difference between actual clicks and predicted probabilities is calculated.
- The ECCE for the segment is the range of these cumulative differences, normalized by segment size.

Global miscalibrations are also detected, as the first segment represents the entire dataset.

### Step 3: Combine Segment Errors Into an Interpretable Single Score

After computing the calibration error (ECCE) for each segment, results are aggregated to produce a single summary metric: the **Multicalibration Error (MCE) on the sigma-scale**.

- For each segment, the ECCE is normalized by its estimated standard deviation (the "sigma-scale"). This accounts for the statistical reliability of each segment and ensures that segments with few samples do not dominate the metric due to noise.
- The **maximum** normalized error across all segments is taken. This approach highlights the segment with the worst signal-to-noise-weighted miscalibration, making the metric easy to interpret and actionable.

**Example:**
In the ad click prediction scenario, suppose the segment "US market AND mobile" has the highest normalized ECCE. The MCE reflects this value, indicating that this segment is the most miscalibrated and should be the priority for further model improvement.

#### Alternative Scale: Rescaling MCE to the Percent-Scale

The raw MCE value represents how many standard deviations the worst segment deviates from perfect calibration. Higher values indicate stronger statistical evidence for miscalibration. This scale depends on dataset properties, such as minority class proportion, score distribution, and sample size.

For applications requiring magnitude of miscalibration, the **Multicalibration Error (MCE) on the percent-scale** is available: by multiplying the MCE (sigma-scale) by the expected standard deviation for the global segment and dividing by the minority class prevalence, the result is a relative effect size in [0, 1]. This represents the magnitude of miscalibration and is useful for comparing models or tracking improvements over time, as it is independent of sample size.

The maximum is used because it aligns with theoretical definitions of multicalibration.


## How Does the MCE Relate to the Theoretical Definition of Multicalibration?

The MCE (sigma scale) metric has a precise mathematical relationship to the theoretical definition of multicalibration, providing a finite-sample estimator of the minimal error parameter in approximate multicalibration.

### The Definition of $\alpha$-Multicalibration

The classic definition of multicalibration is straightforward: for every segment $h$ and every predicted probability $p$, the average outcome should match the prediction:

$$
\mathbb{E}[Y - f(X) \mid h(X)=1, f(X) = p] = 0.
$$

In practice, this condition rarely holds exactly. Therefore, it is relaxed to allow for small error: A predictor $f$ is **$\alpha$-multicalibrated** with respect to $\mathcal{H}$ if

$$
\left| \mathbb{E}\left[h(X) \cdot \mathbf{1}_{f(X) \in [a,b]} \cdot (Y-f(X))\right] \right| \leq \alpha \cdot \tau_h(f,[a,b])
$$

for all segments $h \in \mathcal{H}$ and all score intervals $[a,b]$, where:

- $\mathbf{1}_{f(X) \in [a,b]}$ is the interval indicator function;
- $\tau_h(f,[a,b]) = \sqrt{\mathbb{E}[h(X) \cdot \mathbf{1}_{f(X) \in [a,b]} \cdot f(X)(1-f(X))]}$ is the theoretical scale parameter.

This definition captures the intuition that miscalibration should be bounded relative to the statistical uncertainty inherent in each segment and score range.

### The Bridge: MCE Directly Estimates $\alpha$

The **key insight** is that MCE provides a finite-sample estimate of the minimal $\alpha$ for which a model satisfies $\alpha$-multicalibration.

**MCE Computation:**

$$
\text{MCE}(f) = \max_{h \in \mathcal{H}} \frac{\text{ECCE}_h(f)}{\sigma_h(f)}
$$

where:
- $\text{ECCE}_h(f)$ measures the range of cumulative calibration differences in segment $h$;
- $\sigma_h(f)$ is the empirical standard deviation of ECCE under perfect calibration.

**Scale Parameter Relationship:**

The connection between theory and practice lies in the relationship between scale parameters:
- Theoretical scale: $\tau_h(f,[a,b])$ measures population-level uncertainty;
- Empirical scale: $\sigma_h(f)$ estimates finite-sample uncertainty.

Under standard conditions, these are asymptotically equivalent: $\sigma_h(f) \approx \tau_h(f) / \sqrt{n_h}$ where $n_h$ is the segment size.

:::tip Mathematical Connection
A predictor $f$ is $\alpha$-multicalibrated with respect to $\mathcal{H}$ if and only if:

$$
\text{MCE}(f) \le \alpha \sqrt{n}
$$

where $n$ is the sample size.

**Proof:** Available in Appendix A.1 of [MCGrad: Multicalibration at Web Scale](https://arxiv.org/abs/2509.19884).

:::

**Why this matters:** MCE directly estimates the minimal α for which your model is theoretically guaranteed to be α-multicalibrated across all considered segments.


---


## References

[1] **Guy, I., Haimovich, D., Linder, F., Okati, N., Perini, L., Tax, N., & Tygert, M. (2025).** [Measuring multi-calibration](https://arxiv.org/abs/2506.11251). *arXiv:2506.11251*.



[2] **Arrieta-Ibarra, I., Gujral, P., Tannen, J., Tygert, M., & Xu, C. (2022).** [Metrics of calibration for probabilistic predictions](https://www.jmlr.org/papers/volume23/22-0658/22-0658.pdf). *Journal of Machine Learning Research, 23(351), 1-54.*

If you use MCE in academic work, please cite our paper:
```bibtex
@article{guy2025measuring,
  title={Measuring multi-calibration},
  author={Guy, Ido and Haimovich, Daniel and Linder, Fridolin and Okati, Nastaran and Perini, Lorenzo and Tax, Niek and Tygert, Mark},
  journal={arXiv preprint arXiv:2506.11251},
  year={2025}
}
```
