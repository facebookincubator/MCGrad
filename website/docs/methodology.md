---
sidebar_position: 5
description: Technical details of MCGrad. Mathematical foundations and algorithm design.
---

# Methodology

This page explains how MCGrad works, from the mathematical foundations of multicalibration to the specific algorithm design that makes it scalable and safe for production.
For additional details, see the full paper [1].

## 1. Introduction: Beyond Global Calibration

A machine learning model is **calibrated** if its predicted probabilities match the true frequency of outcomes. When a model predicts 0.8 for a set of events, 80% of those events should actually occur.

However, global calibration is often insufficient. A model can be calibrated on average while being systematically miscalibrated for specific segments, also known as protected groups (e.g., overestimating risk for one demographic while underestimating it for another). These local miscalibrations cancel each other out in the global average, masking the problem.

**Multicalibration** addresses this by requiring the model to be calibrated not just globally, but simultaneously across all meaningful segments (e.g., defined by `age`, `region`, `device`, etc.). Multicalibration has several application areas, including matchings [2].

## 2. Formal Definitions

Let $X$ be the input features and $Y \in \{0,1\}$ be the binary target. A predictor $f_0(x)$ estimates $P(Y=1|X=x)$. We use capital $F_0(x)$ to denote the **logit** of $f_0(x)$, i.e., $F_0(x) = \log(f_0(x) / (1-f_0(x)))$.

### Calibration
A predictor $f_0$ is perfectly calibrated if, for all $p$:

$$
\mathbb{P}(Y=1 \mid f_0(X)=p) = p
$$

In practice, miscalibration is measured using metrics like **ECCE (Estimated Cumulative Calibration Error)** [3], which quantifies the deviation between predictions and outcomes without relying on arbitrary binning.

### Multicalibration
We define a collection of segments $\mathcal{H}$ (e.g., "users in US", "users on Android"). A predictor $f_0$ is **multicalibrated** with respect to $\mathcal{H}$ if it is calibrated on every segment $h \in \mathcal{H}$.

Formally, for all segments $h \in \mathcal{H}$ and all prediction values $p$, the conditional probability should match the prediction:

$$
\mathbb{P}(Y=1 \mid h(X)=1, f_0(X) = p) = p
$$

This is equivalent to requiring the expected residual to be zero:

$$
\mathbb{E}[Y - f_0(X) \mid h(X)=1, f_0(X) = p] = 0
$$

This residual formulation connects directly to the gradient-based approach in MCGrad (see Insight 3 below).

Existing algorithms often require manual specification of $\mathcal{H}$, i.e., the segments to calibrate. MCGrad relaxes this requirement: you need only provide the features, and MCGrad implicitly calibrates across all segments definable by decision trees on those features.

## 3. The MCGrad Algorithm

MCGrad achieves multicalibration by iteratively training a **Gradient Boosted Decision Tree (GBDT)** model. The core innovation follows three key insights.

#### Insight 1: Segments as Decision Trees
A segment in data is typically defined by intersections of attributes (e.g., "People aged 35–50 in the UK").
*   **Categorical attributes** are simple equivalences: `country == 'UK'`.
*   **Numerical attributes** are intervals, i.e., intersections of inequalities: `35 <= age <= 50`, equivalent to `age >= 35 AND age <= 50`.

This structure corresponds exactly to a **leaf in a Decision Tree**. Any meaningful segment of the data can be represented (or approximated) by a leaf in a sufficiently deep tree.

**Implication:** The abstract space of all possible segments $\mathcal{H}$ can be replaced with the space of **Decision Trees**. Instead of iterating over infinite segments, we search for decision trees.

#### Insight 2: Augmenting the Feature Space
Multicalibration requires checking calibration for every segment *and* every prediction (e.g., "Prediction equals 0.8"). Without loss of generality, this definition can be relaxed to intervals: "Prediction is in the range [0.7, 0.9]". Instead of treating "prediction intervals" as a separate constraint, the **prediction itself becomes a feature**.

**Implication:** By training trees on the augmented features $(X, f_0(X))$, the tree learning algorithm naturally finds splits like:
`if country='UK' AND prediction >= 0.7 AND prediction <= 0.9:`
This split automatically isolates a specific *segment* in a specific *prediction range*. No special logic for "intervals" is needed—the decision tree handles it natively.

#### Insight 3: Transformation to Regression
With the first two insights, we have a target (the residual $Y - f_0(X)$) and a feature space $(X, f_0(X))$. The goal is to train a model $\phi$ that finds regions in this space where the residual is non-zero.

Gradient Boosted Decision Trees (GBDTs) work by iteratively training trees to predict the **negative gradient** of a loss function.
For binary classification using **Log-Loss** $\mathcal{L}$, the negative gradient is mathematically identical to the residual:

$$
- \nabla \mathcal{L} = Y - f_0(X)
$$

**Implication:** No specialized "multicalibration algorithm" is needed. A standard GBDT trained to minimize Log-Loss on augmented features suffices:
*   The GBDT automatically searches for trees that predict the residual $Y - f_0(X)$.
*   If it finds a tree with a non-zero prediction, it has found a miscalibrated segment.
*   The boosting update then corrects the model in that exact direction.
*   When the boosting algorithm can no longer find trees that improve the loss, no efficiently discoverable segments remain miscalibrated.

### Iterative Process
Correcting calibration in one region can create miscalibration in other regions. This occurs because the post-processed model has different predictions, and conditioning on intervals over its predictions differs from conditioning on intervals over $f_0(x)$. Therefore, MCGrad proceeds in rounds:

**Algorithm:**
1.  **Initialize**: Start with the base model's logits $F_0(x)$.
2.  **Iterate**: For round $t = 1, \dots, T$:
    *   Train a GBDT $\phi_t$ to predict the residual $Y - f_{t-1}(X)$ using features $(X, f_{t-1}(X))$.
    *   Update the model: $F_t(x) = F_{t-1}(x) + \eta \cdot \phi_t(x, f_{t-1}(x))$.
    *   Update predictions: $f_t(x) = \text{sigmoid}(F_t(x))$.
3.  **Converge**: Stop when performance on a validation set no longer improves.

This recursive structure ensures progressive elimination of calibration errors until the model is multicalibrated.

## 4. Scalability and Safety Design

MCGrad is designed for production systems serving millions of predictions. It includes several optimizations:

### Efficient Implementation
Rather than custom boosting logic, MCGrad delegates to **LightGBM**, a highly optimized GBDT library. This ensures:
*   **Speed**: Training and inference are extremely fast.
*   **Scalability**: Large datasets and high-dimensional feature spaces are handled efficiently.

### Logit Rescaling
GBDTs use a small learning rate to prevent overfitting, but this can result in under-confident updates that require many trees to correct.
MCGrad adds a **rescaling step** after each round: it learns a single scalar multiplier $\theta_t$ to scale the update optimally.

$$
F_t(x) = \theta_t \cdot (F_{t-1}(x) + \phi_t(\dots))
$$

This approach drastically reduces the number of rounds needed for convergence.

### Safety Guardrails
Post-processing methods risk overfitting, potentially harming the base model. MCGrad prevents this through:
1.  **Early Stopping**: Validation loss is tracked after every round. Training stops immediately when loss ceases to improve. If the first round harms performance, MCGrad returns the original model ($T=0$).
2.  **Min-Hessian Regularization**: Augmenting the data with the previous round’s model necessarily gives rise to regions of the augmented feature space that are particularly prone to overfitting. As the predicted probabilities become close to 0 or 1, the Hessian in these regions becomes smaller. Enforcing a minimum Hessian ensures that these regions are no longer considered.

---

### References

[1] **Tax, N., Perini, L., Linder, F., Haimovich, D., Karamshuk, D., Okati, N., Vojnovic, M., & Apostolopoulos, P. A.**
[MCGrad: Multicalibration at Web Scale](https://arxiv.org/abs/2509.19884).
*Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 (KDD 2026).* DOI: 10.1145/3770854.3783954

[2] **Baldeschi, R. C., Di Gregorio, S., Fioravanti, S., Fusco, F., Guy, I., Haimovich, D., Leonardi, S., Linder, F., Perini, L., Russo, M., & Tax, N.** [Multicalibration yields better matchings](https://arxiv.org/abs/2511.11413).
*ArXiv preprint, 2025.*

[3] **Arrieta-Ibarra, I., Gujral, P., Tannen, J., Tygert, M., & Xu, C.** [Metrics of calibration for probabilistic predictions](https://www.jmlr.org/papers/volume23/22-0658/22-0658.pdf).
*Journal of Machine Learning Research, 23(351), 1-54, 2022.*
