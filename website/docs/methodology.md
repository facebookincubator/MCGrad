---
sidebar_position: 5
---

# Methodology

This page provides an overview of how MCGrad works under the hood. For full theoretical details, see the [research paper](https://arxiv.org/abs/2509.19884).

## The Multicalibration Problem

Traditional calibration ensures that predictions match outcomes globally:
```
E[Y | f(X) = p] = p
```

Multicalibration extends this to all segments defined by group membership functions:
```
E[Y | f(X) ∈ I, h(X) = 1] = E[f(X) | f(X) ∈ I, h(X) = 1]
```

Where `I` is any score interval and `h(X)` is any group membership function (e.g., "user is in country X and content type is Y").

## The Key Insight

MCGrad's core insight is that **gradient boosted decision trees (GBDT) naturally achieve multicalibration** when the feature space is augmented with the base model's predictions.

When we train a GBDT with features `(X, f₀(X))` — the original features plus the base model's predictions — the decision trees can split on both:
- **Feature values** (e.g., `country = US`)
- **Score intervals** (e.g., `f₀(X) ∈ [0.7, 0.8]`)

This means the GBDT automatically identifies regions defined by the intersection of groups and score intervals — exactly what multicalibration requires.

## The Algorithm

MCGrad runs multiple rounds to achieve convergence:

```
1. Start with base predictor f₀
2. For each round t:
   a. Train GBDT on features (X, f_{t-1}(X)) to predict Y
   b. Rescale logits to compensate for shrinkage
   c. Update: f_t = σ(θ_t · (F_{t-1} + h_t))
3. Stop when validation loss stops improving
4. Retrain final model on full data with optimal T rounds
```

**Why multiple rounds?** Correcting predictions in some regions may introduce miscalibration in others. Each round refines the previous round's output until convergence.

## Design Choices for Scale and Safety

### 1. Efficient Gradient Boosting

MCGrad delegates compute-intensive operations to LightGBM, a highly optimized GBDT implementation. This makes it orders of magnitude faster than alternatives that iterate over groups explicitly.

### 2. Logit Rescaling

GBDTs use shrinkage (step size < 1) for regularization, which can require many trees to achieve the optimal scale. MCGrad applies a simple rescaling after each round:

```
θ_t = argmin_θ E[L(θ · (F_{t-1} + h_t), Y)]
```

This reduces the number of trees needed while preserving regularization benefits.

### 3. Early Stopping on Rounds

While multiple rounds are needed for convergence, they also increase model capacity. MCGrad uses early stopping on the number of rounds to prevent overfitting:

```
T = last round before validation loss increases
```

**Safe by design**: If the first round hurts performance, MCGrad returns T=0 (the original predictions unchanged).

### 4. Min-Hessian Regularization

Augmenting features with predictions creates regions prone to overfitting (e.g., the tail where `f(x) < 0.01` contains only negative labels). MCGrad uses LightGBM's min-sum-Hessian constraint to prevent splits in these regions.

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
