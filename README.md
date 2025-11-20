# MCGrad

Production-ready multicalibration for machine learning.

**MCBoost** is a scalable and easy-to-use tool for multi-calibration, ensuring your ML model predictions are calibrated not just globally, but across virtually any segment defined by your features.

## 🌟 Key Features

- **Powerful Multi-Calibration** - Calibrates across unlimited segments without pre-specification
- **Data Efficient** - Borrows information like modern ML models
- **Lightweight & Fast** - Orders of magnitude faster than NN-based calibration
- **Improved Performance** - Likelihood-improving with significant PRAUC gains

## 📚 Documentation

Full documentation is available at: https://facebookincubator.github.io/MCGrad/

- [Why MCBoost?](https://facebookincubator.github.io/MCGrad/docs/why-mcboost) - Learn about the benefits
- [Quick Start](https://facebookincubator.github.io/MCGrad/docs/quickstart) - Get started quickly
- [API Reference](https://facebookincubator.github.io/MCGrad/docs/api/methods) - Detailed API docs

## 🚀 Quick Start

```python
from multicalibration import methods
import numpy as np
import pandas as pd

# Your model predictions and segment features
predictions = np.array([0.1, 0.3, 0.7, 0.9, 0.5, 0.2])
features = pd.DataFrame({
    'country': ['US', 'UK', 'US', 'UK', 'US', 'UK'],
    'content_type': ['photo', 'video', 'photo', 'video', 'photo', 'video'],
})
labels = np.array([0, 0, 1, 1, 1, 0])

# Apply MCBoost
mcboost = methods.MCBoost()
mcboost.fit(predictions, features, labels)
calibrated_predictions = mcboost.predict(predictions, features)
```

## 📦 Installation

```bash
pip install git+https://github.com/facebookincubator/MCGrad.git
```

For development:

```bash
git clone https://github.com/facebookincubator/MCGrad.git
cd MCGrad
pip install -e ".[dev]"
```

## 🎯 Success Stories

- **Integrity ReviewBot**: 6x reduction in multi-calibration error, +12pp PRAUC, +10pp automation rate
- **Instagram Age Prediction**: 1.84% reduction in false negative rate while holding precision constant
- **Confidence Aware NSMIEs**: Unblocked label-free precision readouts for integrity experiments

## 🔧 Development

### Pre-commit Hooks

This project uses pre-commit hooks for code quality:

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

**What runs:**
- **On commit:** `isort` and `black` format your code
- **On push:** `pytest` runs the test suite

### Building Documentation

```bash
cd website
npm install
npm start
```

Open http://localhost:3000 to view the docs locally.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

We welcome contributions! See [Contributing Guide](https://facebookincubator.github.io/MCGrad/docs/contributing) for details.

## 📖 Citation

```bibtex
@article{mcboost2024,
  title = {{MCBoost: A Tool for Multi-Calibration}},
  author = {Meta Central Applied Science Team},
  journal = {arXiv preprint arXiv:2509.19884},
  year = {2024},
  url = {https://arxiv.org/pdf/2509.19884}
}
```

## 📊 CI Status

[![CI](https://github.com/facebookincubator/MCGrad/actions/workflows/main.yaml/badge.svg)](https://github.com/facebookincubator/MCGrad/actions/workflows/main.yaml)
