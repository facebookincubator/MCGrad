<p align="center">
  <img src="website/static/img/logo.png" alt="MCGrad: Production-ready multicalibration" width="240" />
</p>

<p align="center">
  <strong>Production-ready multicalibration</strong>
</p>

<p align="center">
  <a href="https://github.com/facebookincubator/MCGrad/actions/workflows/main.yaml"><img src="https://github.com/facebookincubator/MCGrad/actions/workflows/main.yaml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/facebookincubator/MCGrad"><img src="https://codecov.io/gh/facebookincubator/MCGrad/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://pypi.org/project/mcgrad/"><img src="https://img.shields.io/pypi/v/mcgrad.svg" alt="PyPI"></a>
  <a href="https://doi.org/10.1145/3770854.3783954"><img src="https://img.shields.io/badge/DOI-10.1145%2F3770854.3783954-blue" alt="DOI"></a>
  <a href="https://mcgrad.dev/"><img src="https://img.shields.io/badge/docs-mcgrad.dev-blue.svg" alt="Website"></a>
  <a href="https://mcgrad.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/mcgrad/badge/?version=latest" alt="API Reference"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
</p>

---

## What is MCGrad?

**MCGrad** is a scalable and easy-to-use tool for **multicalibration**. It ensures your ML model predictions are well-calibrated not just globally (across all data), but also across virtually any segment defined by your features (e.g., by country, content type, or any combination).

Traditional calibration methods, like Isotonic Regression or Platt Scaling, only ensure global calibration—meaning predicted probabilities match observed outcomes *on average* across all data—but your model can still be systematically overconfident or underconfident for specific groups. MCGrad automatically identifies and corrects these hidden calibration gaps without requiring you to manually specify protected groups.

<p align="center">
  <img src="website/static/img/global_calibration.png" alt="Global calibration curve showing well-calibrated predictions on average" width="90%" />
</p>
<p align="center">
  <em>A globally well-calibrated model: predictions match observed outcomes on average.</em>
</p>

<p align="center">
  <img src="website/static/img/local_miscalibration.png" alt="Segment-level calibration curves revealing hidden miscalibration in specific groups" width="90%" />
</p>
<p align="center">
  <em>The same model showing hidden miscalibration when broken down by segment. MCGrad fixes this.</em>
</p>

## 🌟 Why MCGrad?

- **State-of-the-art multicalibration** — Best-in-class calibration quality across a vast number of segments.
- **Easy to use** — Familiar interface. Pass features, not segments.
- **Highly scalable** — Fast to train, low inference overhead, even on web-scale data.
- **Safe by design** — Likelihood-improving updates with validation-based early stopping.

## 🏭 Production Proven

MCGrad has been deployed at **Meta** on hundreds of production models. See the [research paper](https://arxiv.org/abs/2509.19884) for detailed experimental results.

## 📦 Installation

**Requirements:** Python 3.10+

Stable release:
```bash
pip install mcgrad
```

Latest development version:
```bash
pip install git+https://github.com/facebookincubator/MCGrad.git
```

## 🚀 Quick Start

```python
from mcgrad import methods
import numpy as np
import pandas as pd

# Prepare your data in a DataFrame
df = pd.DataFrame({
    'prediction': np.array([0.1, 0.3, 0.7, 0.9, 0.5, 0.2]),  # Your model's predictions
    'label': np.array([0, 0, 1, 1, 1, 0]),  # Ground truth labels
    'country': ['US', 'UK', 'US', 'UK', 'US', 'UK'],  # Categorical feature
    'content_type': ['photo', 'video', 'photo', 'video', 'photo', 'video'],  # Categorical feature
})

# Apply MCGrad
mcgrad = methods.MCGrad()
mcgrad.fit(
    df_train=df,
    prediction_column_name='prediction',
    label_column_name='label',
    categorical_feature_column_names=['country', 'content_type']
)

# Get calibrated predictions
calibrated_predictions = mcgrad.predict(
    df=df,
    prediction_column_name='prediction',
    categorical_feature_column_names=['country', 'content_type']
)
# Returns: numpy array of calibrated probabilities, e.g., [0.12, 0.28, 0.72, ...]
```

## 📚 Documentation

- **Website & Guides:** [mcgrad.dev](https://mcgrad.dev/)
  - [Why MCGrad?](https://mcgrad.dev/docs/why-mcgrad) — Learn about the challenges MCGrad solves
  - [Quick Start](https://mcgrad.dev/docs/quickstart) — Get started quickly
  - [Methodology](https://mcgrad.dev/docs/methodology) — Deep dive into how MCGrad works
  - [API Reference](https://mcgrad.readthedocs.io/en/latest/) — Full API documentation

## 💬 Community & Support

- **Questions & Bugs:** Open an issue on [GitHub Issues](https://github.com/facebookincubator/MCGrad/issues)
- **Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to MCGrad

## 📖 Citation

If you use MCGrad in your research, please cite [our paper](https://arxiv.org/abs/2509.19884).

```bibtex
@inproceedings{tax2026mcgrad,
  title={{MCGrad: Multicalibration at Web Scale}},
  author={Tax, Niek and Perini, Lorenzo and Linder, Fridolin and Haimovich, Daniel and Karamshuk, Dima and Okati, Nastaran and Vojnovic, Milan and Apostolopoulos, Pavlos Athanasios},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 (KDD 2026)},
  year={2026},
  doi={10.1145/3770854.3783954}
}
```

### Related Publications

Some of our team's other work on multicalibration:

- **A New Metric to Measure Multicalibration:** Guy, I., Haimovich, D., Linder, F., Okati, N., Perini, L., Tax, N., & Tygert, M. (2025). [Measuring multi-calibration](https://arxiv.org/abs/2506.11251). arXiv:2506.11251.

- **Theoretical Results on Value of Multicalibration:** Baldeschi, R. C., Di Gregorio, S., Fioravanti, S., Fusco, F., Guy, I., Haimovich, D., Leonardi, S., et al. (2025). [Multicalibration yields better matchings](https://arxiv.org/abs/2511.11413). arXiv:2511.11413.
