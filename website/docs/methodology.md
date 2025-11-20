---
sidebar_position: 5
---

# Methodology

This page provides an overview of how MCBoost works under the hood.

## The Multi-Calibration Problem

Traditional calibration ensures that predictions match outcomes globally:
```
E[Y | f(X) = p] = p
```

Multi-calibration extends this to segments:
```
E[Y | f(X) = p, g(X) = v] = p  for all functions g
```

Where `g(X)` represents any segment defined by the features.

## MCBoost Algorithm

MCBoost uses gradient boosting (LightGBM) to iteratively improve calibration across segments:

1. **Initialize** with base model predictions
2. **Iterate** over boosting rounds:
   - Identify segments with largest calibration errors
   - Train weak learner to correct those errors
   - Update predictions
3. **Regularize** to prevent overfitting

## Why Gradient Boosting?

LightGBM provides several advantages for multi-calibration:

- **Automatic feature selection** - finds relevant segments
- **Efficient** - handles large datasets and many features
- **Regularization** - built-in controls against overfitting
- **Interpretable** - tree-based models show which segments matter

## Theoretical Guarantees

MCBoost is a likelihood-improving procedure:
- **On training data**: guaranteed to improve or maintain likelihood
- **On test data**: improved with high probability given proper regularization
- **Multi-calibration**: achieves near-optimal calibration across exponentially many segments

## Implementation Details

The implementation uses:
- **LightGBM** as the core gradient boosting framework
- **Custom loss functions** designed for calibration
- **Cross-validation** for hyperparameter tuning
- **Early stopping** to prevent overfitting

## Comparison with Alternatives

| Method | Global Calibration | Multi-Calibration | Data Efficiency | Speed |
|--------|-------------------|-------------------|-----------------|-------|
| **MCBoost** | ✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| Isotonic Regression | ✓ | ✗ | ✓✓ | ✓✓✓ |
| Platt Scaling | ✓ | ✗ | ✓✓✓ | ✓✓✓ |
| Temperature Scaling | ✓ | ✗ | ✓✓✓ | ✓✓✓ |
| Beta Calibration | ✓ | ✗ | ✓✓ | ✓✓✓ |
| Separate Calibration per Segment | ✓ | ✓ | ✗ | ✓ |

## Research Paper

For a detailed theoretical analysis and experimental results, see our paper:

**MCBoost: A Tool for Multi-Calibration**
ArXiv: [2509.19884](https://arxiv.org/pdf/2509.19884)

## Next Steps

- [Quick Start](quickstart.md) - Start using MCBoost
- [API Reference](api/methods.md) - Explore the implementation
