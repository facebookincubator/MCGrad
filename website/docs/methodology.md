---
sidebar_position: 5
---

# Methodology

This page explains how MCGrad works, from the mathematical foundations of multicalibration to the specific algorithm design that makes it scalable and safe for production.
For additional information, please check the full paper [1].

## 1. Introduction: Beyond Global Calibration

A machine learning model is **calibrated** if its predicted probabilities match the true frequency of outcomes. In other words, if we take all the events for which the model predicts 0.8, then 80% of them should actually occur.

However, global calibration is often insufficient. A model can be calibrated on average while being systematically miscalibrated for specific segments (e.g., overestimating risk for one demographic while underestimating it for another). These local miscalibrations cancel each other out in the global average, hiding the problem.

**Multicalibration** solves this by requiring the model to be calibrated not just globally, but simultaneously across all meaningful segments (e.g., defined by `age`, `region`, `device`, etc.). Multicalibration has several application areas, including matchings [2].

## 2. Formal Definitions

Let $X$ be the input features and $Y \in \{0,1\}$ be the binary target. A predictor $f_0(x)$ estimates $P(Y=1|X=x)$. We use the capital $F_0(x)$ to denote the **logit** of $f_0(x)$, i.e. $F_0(x) = \log(f_0(x) / (1-f_0(x)))$.

### Calibration
A predictor $f_0$ is perfectly calibrated if, for all $p$:

$$ \mathbb{P}(Y=1 \mid f_0(X)=p) = p$$

Practically, we measure miscalibration using metrics like **ECCE (Estimated Cumulative Calibration Error)**, which quantifies the deviation between predictions and outcomes without relying on arbitrary binning.

### Multicalibration
We define a collection of segments $\mathcal{H}$ (e.g., "users in US", "users on Android"). A predictor $f_0$ is **multicalibrated** with respect to $\mathcal{H}$ if it is calibrated on every segment $h \in \mathcal{H}$.

Formally, for all segments $h \in \mathcal{H}$ and all prediction values $p$, the conditional probability should match the prediction:

$$ \mathbb{P}(Y=1 \mid h(X)=1, f_0(X) = p) = p$$

This is equivalent to saying that the expected residual should be zero:

$$ \mathbb{E}[Y - f_0(X) \mid h(X)=1, f_0(X) = p] = 0 $$

This residual formulation directly connects to the gradient-based approach in MCGrad (see Insight 3 below).

Existing algorithms often require you to manually specify $\mathcal{H}$ (the "protected groups"). MCGrad relaxes this: you only need to provide the features, and MCGrad implicitly calibrates across all segments definable by decision trees on those features.

## 3. The MCGrad Algorithm

MCGrad achieves multicalibration by iteratively training a **Gradient Boosted Decision Tree (GBDT)** model. The core innovation follows three simple key insights.

#### Insight 1: Segments as Decision Trees
A "segment" (or a group) in data is typically defined by intersections of attributes (e.g., "People aged 35-50 in the UK").
*   **Categorical attributes** are simple equivalences: `country == 'UK'`.
*   **Numerical attributes** are intervals, i.e. intersections of inequalities: `35 <= age <= 50`, equivalent to `age >= 35 AND age <= 50`.

This structure corresponds exactly to a **leaf in a Decision Tree**. Any meaningful segment of the data can be represented (or approximated) by a leaf in a sufficiently deep tree.

**Implication:** We can replace the abstract space of all possible segments $\mathcal{H}$ with the space of **Decision Trees**. Instead of iterating over infinite segments, we just need to search for decision trees.

#### Insight 2: Augmenting the Feature Space
Multicalibration requires checking calibration for every segment *and* every prediction (e.g., "Prediction equals 0.8"). Without losing on generality, we can relax the definition to intervals: "Prediction is in the range [0.7, 0.9]". In this way, instead of treating "prediction intervals" as a separate constraint, we can treat the **prediction itself as a feature**.

**Implication:** By training trees on the augmented features $(X, f_0(X))$, the tree learning algorithm naturally finds splits like:
`if country='UK' AND prediction >= 0.7 AND prediction <= 0.9:`
This split automatically isolates a specific *segment* in a specific *prediction range*. We don't need special logic for "intervals"; the decision tree handles it natively.

#### Insight 3: Transformation to Regression
With the first two insights, we have a target (the residual $Y - f_0(X)$) and a feature space $(X, f_0(X))$. The goal is to train a model $\phi$ that finds regions in this space where the residual is non-zero.

How do we do this efficiently? This is where Gradient Boosting comes in.
Gradient Boosted Decision Trees (GBDTs) work by iteratively training trees to predict the **negative gradient** of a loss function.
For binary classification using **Log-Loss** $\mathcal{L}$, the negative gradient is mathematically identical to the residual:

$$ - \nabla \mathcal{L} = Y - f_0(X) $$

**Implication:** This means we don't need a specialized "multicalibration algorithm." We can simply train a standard GBDT to minimize Log-Loss on our augmented features.
*   The GBDT automatically searches for trees that predict the residual $Y - f_0(X)$.
*   If it finds a tree with a non-zero prediction, it has found a miscalibrated segment.
*   The boosting update then corrects the model in that exact direction.
*   When the boosting algorithm can no longer find trees that improve the loss, it means no efficiently discoverable segments are miscalibrated.

### Iterative Process
Correcting calibration in one region can create other regions with miscalibration. This happens because the post-processed model has different predictions and conditioning on intervals over its predictions is not the same as conditioning on intervals over $f_0(x)$. Therefore, MCGrad proceeds in rounds:

**Algorithm:**
1.  **Initialize**: Start with the base model's logits $F_0(x)$.
2.  **Iterate**: For round $t = 1, \dots, T$:
    *   Train a GBDT $\phi_t$ to predict the residual $Y - f_{t-1}(X)$ using features $(X, f_{t-1}(X))$.
    *   Update the model: $F_t(x) = F_{t-1}(x) + \eta \cdot \phi_t(x, f_{t-1}(x))$.
    *   Update predictions: $f_t(x) = \text{sigmoid}(F_t(x))$.
3.  **Converge**: Stop when performance on a validation set no longer improves.

This recursive structure ensures that we progressively eliminate calibration errors until the model is multicalibrated.

## 4. Scalability & Safety Design

MCGrad is designed for production systems serving millions of predictions. It includes several optimizations:

### Efficient Implementation
Instead of custom boosting logic, MCGrad delegates the heavy lifting to **LightGBM**, a highly optimized GBDT library. This ensures:
*   **Speed**: Training and inference are extremely fast.
*   **Scalability**: Handles large datasets and high-dimensional feature spaces efficiently.

### Logit Rescaling
GBDTs use "shrinkage" (small learning rate) to prevent overfitting, but this can result in under-confident updates that require many trees to fix.
MCGrad adds a **rescaling step** after each round: it learns a single scalar multiplier $\theta_t$ to perfectly scale the update.

$$ F_t(x) = \theta_t \cdot (F_{t-1}(x) + \phi_t(\dots)) $$

This drastically reduces the number of rounds needed for convergence.

### Safety Guardrails
Post-processing methods risk overfitting, potentially harming the base model. MCGrad prevents this via:
1.  **Early Stopping**: We track validation loss after every round. We stop immediately when loss stops improving. If the first round hurts performance, MCGrad returns the original model ($T=0$).
2.  **Min-Hessian Regularization**: In the augmented feature space $(X, f_0(X))$, some regions (e.g., $f_0(x) \approx 0$) contain almost no signal. Standard split rules can overfit here. MCGrad enforces a minimum "Hessian" (second derivative sum) in leaf nodes, which naturally prevents splits in regions where the model is already confident and correct.

---

### References

[1] **Tax, N., Perini, L., Linder, F., Haimovich, D., Karamshuk, D., Okati, N., Vojnovic, M., & Apostolopoulos, P. A.**
[MCGrad: Multicalibration at Web Scale](https://arxiv.org/abs/2509.19884).
*Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 (KDD 2026).* DOI: 10.1145/3770854.3783954

[2] **Baldeschi, R. C., Di Gregorio, S., Fioravanti, S., Fusco, F., Guy, I., Haimovich, D., Leonardi, S., Linder, F., Perini, L., Russo, M., & Tax, N.** [Multicalibration yields better matchings](https://arxiv.org/abs/2511.11413).
*ArXiv preprint, 2025.*
