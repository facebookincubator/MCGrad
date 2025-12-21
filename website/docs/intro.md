---
sidebar_position: 1
---

# Introduction to MCGrad

**Multicalibration** means calibration across segments of the data. It is the form of calibration with highest relevance to the business metrics of ML-based systems.

**MCGrad** is a scalable and easy-to-use tool for multicalibration, providing production-ready implementations developed by Meta's Central Applied Science team.

## Why Improve Multicalibration?

1. **Metrics aren't enough** - Predictive performance metrics (e.g., NE, PRAUC) are not sufficient on their own. Multicalibration is also important for the performance of ML-based systems. Any system that assumes predictions are probabilities would benefit from better calibration: threshold-based actions, value models, ranking, auctions, etc.

2. **Improves predictive performance** - Improving multicalibration often results in significant gains in predictive metrics like NE (Normalized Entropy) and PRAUC (Precision-Recall Area Under Curve)!

## How MCGrad Helps

MCGrad takes a base model and training data, then builds a lightweight calibration layer on top.
- **Globally** - across all data
- **Per-segment** - across virtually any segment that can be defined using the input features

## Example Use Case

ReviewBot is an LLM developed by Integrity which predicts violation probabilities across ~50 violation types. MCGrad is trained with:
- Base model scores
- Features: country, content_type, surface, etc.

The resulting predictions are multi-calibrated, meaning they're calibrated:
- Globally
- For specific segments like `country=US`
- For `content_type=photo`
- For `surface=feed`
- For intersections like `country=US AND content_type=photo AND surface=feed`

## Next Steps

- [Installation](installation.md) - Get started with MCGrad
- [Why MCGrad?](why-mcgrad.md) - Learn more about the benefits
- [Quick Start](quickstart.md) - Start using MCGrad

## Research

MCGrad is based on research in multicalibration:

- **Perini, L., Haimovich, D., Linder, F., Tax, N., Karamshuk, D., Vojnovic, M., Okati, N., & Apostolopoulos, P. A. (2025).** [MCGrad: Multicalibration at Web Scale](https://arxiv.org/abs/2509.19884). arXiv:2509.19884. To appear in KDD 2026.
