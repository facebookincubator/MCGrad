---
sidebar_position: 2
description: Why choose MCGrad for multicalibration? Learn about the challenges of segment-level calibration and how MCGrad addresses them.
---

# Why MCGrad?

**Multicalibration**—calibration across all relevant segments (also known as protected groups)—is essential for ML systems that make decisions based on predicted probabilities. Whether in content ranking, recommender systems, digital advertising, or content moderation, predictions must be calibrated not just globally, but for every user segment, content type, and other relevant dimensions.

Despite its importance, multicalibration has seen limited industry adoption due to three core challenges:

**1. Manual Segment Specification.** Existing multicalibration methods require manual definition of segments—not just which features matter, but explicit segment indicators (e.g., "is an adult in the US?"). This approach is impractical because:

- Practitioners may not know all relevant segments in advance.
- Segment definitions change over time (legal, policy, ethical considerations).
- Many teams simply want better calibration and performance, not segment engineering.

**2. Lack of Scalability.** Existing methods do not scale to large datasets or many segments. Some algorithms scale linearly (or worse) in time and memory with the number of segments, making them infeasible to productionize at web scale, where thousands of features and billions of samples define a vast number of segments.

**3. Risk of Harming Performance.** Existing methods lack guardrails for safe deployment. As a post-processing method, the risk that multicalibration might *harm* model performance (e.g., through overfitting) prevents adoption.


### MCGrad: Designed for Real-World Impact

MCGrad is a scalable multicalibration algorithm that overcomes these barriers:

- **Pass features, not segments**: MCGrad requires only a set of *segment-defining features* (rather than explicit segments). It then automatically identifies miscalibrated regions within this feature space, calibrating predictions across all segments defined by these features.
- **Scalable to billions of samples**: By reducing multicalibration to gradient boosting, MCGrad inherits the scalability of optimized GBDT libraries like LightGBM. This enables deployment at web scale with minimal computational overhead at training or inference time.
- **Safe by design**: MCGrad is a *likelihood-improving procedure*—it can only improve model performance on training data. Combined with early stopping, this ensures model performance is not harmed.
- **Proven in production**: Deployed at Meta on hundreds of models, MCGrad delivers robust improvements in calibration, log-loss, and PRAUC.


### When Should You Use MCGrad?

Traditional calibration methods like *Isotonic Regression*, *Platt Scaling*, or *Temperature Scaling* work well for global calibration—but they fail to maintain calibration across specific segments of your data.

Use MCGrad when:

- Your system makes *segment-specific decisions* (e.g., different thresholds per market).
- You need *fair predictions* across demographic or interest segments.
- You observe *poor calibration in some segments* despite good global calibration.
- You want to *improve overall performance*—multicalibration often improves log-loss and PRAUC.
- Predictions feed into *downstream optimization* (e.g., matching, ranking, auctions)—even unbiased predictors can lead to [poor decisions without multicalibration](https://arxiv.org/abs/2511.11413).
- You need *robustness to distribution shifts*—multicalibrated models generalize better ([Kim et al., 2022](https://www.pnas.org/doi/10.1073/pnas.2108097119); [Wu et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/859b6564b04959833fdf52ae6f726f84-Abstract-Conference.html)).


### Results at Scale

MCGrad has been deployed at **Meta** on hundreds of production models, generating over a million multicalibrated predictions per second:

- **A/B tests**: 24 of 27 models significantly outperformed Platt scaling.
- **Offline evaluation** (120+ models): Improved log-loss for 88.7%, ECE for 86.0%, PRAUC for 76.7% of models.
- **Inference**: Constant ~20μs latency vs 1,000+ μs for alternatives at scale.

On public benchmarks, MCGrad achieves 56% average [Multicalibration Error](https://arxiv.org/pdf/2506.11251) reduction while never harming base model performance. See the [MCGrad research paper](https://arxiv.org/abs/2509.19884) for full experimental details.

---

### Next Steps

- [Methodology](methodology.md) — Deep dive into how MCGrad works.
- [Installation](installation.md) — Install MCGrad and dependencies.
