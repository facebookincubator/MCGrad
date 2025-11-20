---
sidebar_position: 1
---

# Introduction to MCGrad

**Multi-calibration** means calibration across segments of the data. It is the form of calibration with highest relevance to the business metrics of ML-based systems.

**MCBoost** is a scalable and easy-to-use tool for multi-calibration, providing production-ready implementations developed by Meta's Central Applied Science team.

## Why Improve Multi-Calibration?

1. **Metrics aren't enough** - Predictive performance metrics (e.g. NE, PRAUC) aren't sufficient. Multi-calibration is also important for the performance of ML-based systems. Any system which assumes that predictions are probabilities would benefit: threshold-based actions, value models, ranking, auctions, etc.

2. **Improves predictive performance** - Improving multi-calibration often results in significant NE/PRAUC gains!

## How MCBoost Helps

MCBoost takes a base model and training data, then builds a lightweight calibration layer on top. The resulting model is calibrated:
- **Globally** - across all data
- **Per-segment** - across virtually any segment that can be defined using the input features

## Example Use Case

ReviewBot is an LLM developed by Integrity which predicts violation probabilities across ~50 violation types. MCBoost is trained with:
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
- [Why MCBoost?](why-mcboost.md) - Learn more about the benefits
- [Quick Start](quickstart.md) - Start using MCBoost
