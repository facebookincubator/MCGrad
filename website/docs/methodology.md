---
sidebar_position: 5
---

# Methodology

This page provides an overview of how MCGrad works under the hood.

## The Multicalibration Problem

Traditional calibration ensures that predictions match outcomes globally:
```
E[Y | f(X) = p] = p
```

Multicalibration extends this to segments:
```
E[Y | f(X) = p, g(X) = v] = p  for all functions g
```

Where `g(X)` represents any segment defined by the features.

## MCGrad Algorithm

MCGrad uses gradient boosting (LightGBM) to iteratively improve calibration across segments:

1. **Initialize** - Start with the base model predictions
2. **Iterate** over boosting rounds:
   - Identify segments with the largest calibration errors
   - Train a weak learner to correct those errors
   - Update predictions by adding the corrections
3. **Regularize** - Apply regularization techniques to prevent overfitting

## Why Gradient Boosting?

LightGBM provides several advantages for multicalibration:

- **Automatic feature selection** - finds relevant segments
- **Efficient** - handles large datasets and many features
- **Regularization** - built-in controls against overfitting
- **Interpretable** - tree-based models show which segments matter

## Theoretical Guarantees

MCGrad is a likelihood-improving procedure:
- **On training data**: guaranteed to improve or maintain likelihood
- **On test data**: improved with high probability given proper regularization
- **Multicalibration**: achieves near-optimal calibration across exponentially many segments

## Implementation Details

The implementation uses:
- **LightGBM** as the core gradient boosting framework
- **Custom loss functions** designed for calibration
- **Cross-validation** for hyperparameter tuning
- **Early stopping** to prevent overfitting

## Comparison with Alternatives

| Method | Global Calibration | Multicalibration | Data Efficiency | Speed |
|--------|-------------------|-------------------|-----------------|-------|
| **MCGrad** | ✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| Isotonic Regression | ✓ | ✗ | ✓✓ | ✓✓✓ |
| Platt Scaling | ✓ | ✗ | ✓✓✓ | ✓✓✓ |
| Temperature Scaling | ✓ | ✗ | ✓✓✓ | ✓✓✓ |
| Beta Calibration | ✓ | ✗ | ✓✓ | ✓✓✓ |
| Separate Calibration per Segment | ✓ | ✓ | ✗ | ✓ |

## Research Paper

For a detailed theoretical analysis and experimental results, see our paper:

**Perini, L., Haimovich, D., Linder, F., Tax, N., Karamshuk, D., Vojnovic, M., Okati, N., & Apostolopoulos, P. A. (2025).** [MCGrad: Multicalibration at Web Scale](https://arxiv.org/abs/2509.19884). arXiv:2509.19884. To appear in KDD 2026.

### Related Work

For more on multicalibration theory and applications:

- **Measuring Multi-Calibration:** Guy, I., Haimovich, D., Linder, F., Okati, N., Perini, L., Tax, N., & Tygert, M. (2025). [Measuring multi-calibration](https://arxiv.org/abs/2506.11251). arXiv:2506.11251.

- **Multicalibration Applications:** Baldeschi, R. C., Di Gregorio, S., Fioravanti, S., Fusco, F., Guy, I., Haimovich, D., Leonardi, S., et al. (2025). [Multicalibration yields better matchings](https://arxiv.org/abs/2511.11413). arXiv:2511.11413.

## Next Steps

- [Quick Start](quickstart.md) - Start using MCGrad
- [API Reference](api/methods.md) - Explore the implementation
