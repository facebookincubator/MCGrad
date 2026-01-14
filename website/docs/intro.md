---
sidebar_position: 1
---

# Why Multicalibration?

Machine learning models are increasingly used to make decisions that affect people, content, and business outcomes. For these systems, *calibration*—ensuring that predicted probabilities match real-world outcomes—is essential for trust, fairness, and optimal decision-making.
However, global calibration alone is *not enough*. Even if a model is well-calibrated on average, it can still systematically overestimate or underestimate for specific groups (segments)—such as users in a particular country, transactions on a certain device, or content of a certain type. These hidden calibration gaps can lead to unfair, unreliable, or suboptimal decisions for those groups.

## A Motivating Example

Imagine you have trained a model to predict the likelihood of a user clicking on an ad. You check its reliability plot, and the model is clearly miscalibrated. So, you apply a global calibration method, like Isotonic Regression, and the reliability plot looks almost perfect.


<div style={{textAlign: 'center'}}>
<img src={require('../static/img/global_calibration.png').default} alt="global calibration" width="90%" />
</div>

But let's dig deeper. You decide to look at two specific groups: the US and the non US markets. To your surprise, the reliability plots for these groups tell a different story. For the US market, the model underestimates—its predictions are lower than the actual click rates. For non US markets, the model overestimates, consistently predicting higher probabilities than reality.

<div style={{textAlign: 'center'}}>
<img src={require('../static/img/local_miscalibration.png').default} alt="local miscalibration" width="50%" />
</div>

This is a common pitfall: global calibration can mask significant miscalibration in groups. These hidden errors can have real consequences and lead to poor business outcomes.

Now, you apply **MCGrad** instead of Isotonic Regression, and revisit the reliability plots for those same groups. This time, the curves are much closer to the diagonal. The predictions for US and non US markets now accurately reflect the true click rates. MCGrad hasn't just fixed the model globally — it makes sure every meaningful group gets calibrated predictions.

<div style={{textAlign: 'center'}}>
<img src={require('../static/img/mcgrad.png').default} alt="mcgrad fixes local miscalibration" width="50%" />
</div>

## How MCGrad Works

MCGrad automatically discovers and corrects miscalibrated regions defined by these features, including intersections like `market=US AND device_type=mobile`, no need to manually specify which groups to protect. Just provide your features, and MCGrad finds and fixes calibration issues across all relevant groups.

```python
from multicalibration import methods

# df is a DataFrame with columns:
#   - 'prediction': the uncalibrated model predictions (floats in [0, 1])
#   - 'label': the true binary labels (0 or 1)
#   - categorical features such as 'market', 'device_type', or 'content_type'
#   - Optionally, numerical features like 'age', etc.

mcgrad = methods.MCGrad()
mcgrad.fit(
    df_train=df,
    prediction_column_name='prediction',
    label_column_name='label',
    categorical_feature_column_names=['market', 'device_type', 'content_type'],
    numerical_feature_column_names=['age']  # MCGrad also supports continuous features
)

mc_probs = mcgrad.predict(
    df,
    prediction_column_name='prediction',
    categorical_feature_column_names=['market', 'device_type', 'content_type'],
    numerical_feature_column_names=['age']
)
```

The result: predictions calibrated globally AND for virtually any group defined by your features.

## Get Started

- [Why MCGrad?](why-mcgrad.md) — Learn about the challenges MCGrad solves and see results.
- [Installation](installation.md) — Step-by-step instructions for installing MCGrad.
- [Quick Start](quickstart.md) — Example workflows.

## Citation

If you use MCGrad in academic work, please cite [our paper](https://arxiv.org/abs/2509.19884):

```bibtex
@inproceedings{tax2026mcgrad,
  title={{MCGrad: Multicalibration at Web Scale}},
  author={Tax, Niek and Perini, Lorenzo and Linder, Fridolin and Haimovich, Daniel and Karamshuk, Dima and Okati, Nastaran and Vojnovic, Milan and Apostolopoulos, Pavlos Athanasios},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 (KDD 2026)},
  year={2026},
  doi={10.1145/3770854.3783954}
}
```
