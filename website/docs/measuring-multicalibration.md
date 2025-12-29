---
sidebar_position: 6
---

# Measuring Multicalibration

How do you know if your model is multicalibrated? This page introduces the **Multicalibration Error (MCE)** metric—a principled way to measure calibration quality across subpopulations.

## The Challenge

Standard calibration metrics like Expected Calibration Error (ECE) measure global calibration, but a model can appear well-calibrated overall while being poorly calibrated for specific subgroups. Measuring multicalibration requires:

1. Evaluating calibration across many subpopulations
2. Combining these measurements into a single metric
3. Accounting for statistical noise (smaller subgroups have noisier estimates)

## The MCE Metric

The MCE metric addresses these challenges using the **Kuiper statistic**—a classical statistical test that avoids the binning problems of ECE.

### Key Design Choices

| Design Choice | Rationale |
|--------------|-----------|
| **Kuiper statistic** | Avoids arbitrary binning; based on cumulative differences |
| **Signal-to-noise weighting** | Weights subpopulations by statistical reliability, not just size |
| **Maximum over subpopulations** | Focuses on the worst-calibrated subgroup |

The signal-to-noise weighting is critical: without it, estimates from finite samples become dominated by noise, especially for smaller subpopulations. This insight comes from the theoretical work of [Haghtalab, Jordan, and Zhao (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e55edcdb01ac45c839a602f96e09fbcb-Abstract-Conference.html).

## Usage

```python
from multicalibration.metrics import MulticalibrationError

mce = MulticalibrationError(
    df=df,
    label_column='label',
    score_column='prediction',
    categorical_segment_columns=['region', 'age_group'],
    numerical_segment_columns=['income']
)

print(f"MCE: {mce.mce:.2f}%")
print(f"P-value: {mce.p_value:.4f}")
print(f"Worst segment: {mce.mce_sigma_scale:.2f} standard deviations")
```

### Key Outputs

- **`mce`**: The multicalibration error as a percentage (relative to prevalence)
- **`mce_sigma_scale`**: MCE in units of standard deviations (for statistical significance)
- **`p_value`**: Statistical significance of the worst miscalibration
- **`segment_ecces`**: Calibration errors for each individual segment

## Interpreting Results

- **`p_value`**: Use standard significance thresholds (e.g., 0.05, 0.01) to determine whether the observed miscalibration is statistically significant
- **`mce_sigma_scale`**: Higher values indicate stronger evidence of miscalibration; useful for comparing across models or tracking improvements over time
- **`mce`**: The magnitude of miscalibration as a percentage, interpretable in terms of prediction bias

## Reference

For full details on the methodology, see:

**Guy, I., Haimovich, D., Linder, F., Okati, N., Perini, L., Tax, N., & Tygert, M. (2025).** [Measuring multi-calibration](https://arxiv.org/abs/2506.11251). *arXiv:2506.11251*.

```bibtex
@article{guy2025measuring,
  title={Measuring multi-calibration},
  author={Guy, Ido and Haimovich, Daniel and Linder, Fridolin and Okati, Nastaran and Perini, Lorenzo and Tax, Niek and Tygert, Mark},
  journal={arXiv preprint arXiv:2506.11251},
  year={2025}
}
```
