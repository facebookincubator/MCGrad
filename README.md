# MCGrad

Production-ready multicalibration for machine learning.

**MCGrad** is a scalable and easy-to-use tool for multicalibration. It ensures your ML model predictions are well-calibrated not just globally (across all data), but also across virtually any segment defined by your features (e.g., by country, content type, or any combination).

## 🌟 Key Features

- **Powerful Multicalibration** - Calibrates across unlimited segments without pre-specification
- **Data Efficient** - Borrows information like modern ML models
- **Lightweight & Fast** - Orders of magnitude faster than NN-based calibration
- **Improved Performance** - Likelihood-improving with significant PRAUC gains

## 📚 Documentation

Full documentation is available at: https://facebookincubator.github.io/MCGrad/

- [Why MCGrad?](https://facebookincubator.github.io/MCGrad/docs/why-mcgrad) - Learn about the benefits
- [Quick Start](https://facebookincubator.github.io/MCGrad/docs/quickstart) - Get started quickly
- [API Reference](https://mcgrad.readthedocs.io/) - Auto-generated API documentation from Python docstrings

### Two Documentation Systems

This project uses a dual documentation approach:

1. **User Guide (Docusaurus)** - Available at https://facebookincubator.github.io/MCGrad/
   - Getting started guides, tutorials, and conceptual documentation
   - Built from the `website/` directory

2. **API Reference (Sphinx)** - Available at https://multicalibration.readthedocs.io/
   - Auto-generated from Python docstrings
   - Detailed API documentation for all classes and functions
   - Built from the `sphinx/` directory

## 🚀 Quick Start

```python
from multicalibration import methods
import numpy as np
import pandas as pd

# Prepare your data in a DataFrame
df = pd.DataFrame({
    'prediction': np.array([0.1, 0.3, 0.7, 0.9, 0.5, 0.2]),  # Your model's predictions
    'label': np.array([0, 0, 1, 1, 1, 0]),  # Ground truth labels
    'country': ['US', 'UK', 'US', 'UK', 'US', 'UK'],  # Categorical features
    'content_type': ['photo', 'video', 'photo', 'video', 'photo', 'video'],  # defining segments
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

## 🔧 Development

### Pre-commit Hooks

This project uses pre-commit hooks for code quality:

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

**What runs:**
- **On commit:** `flake8` checks your code
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

If you use MCGrad in your research, please cite:

```bibtex
@inproceedings{tax2026mcgrad,
  title={{MCGrad: Multicalibration at Web Scale}},
  author={Tax, Niek and Perini, Lorenzo and Linder, Fridolin and Haimovich, Daniel and Karamshuk, Dima and Okati, Nastaran and Vojnovic, Milan and Apostolopoulos, Pavlos Athanasios},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 (KDD 2026)},
  year={2026},
  doi={10.1145/3770854.3783954}
}
```

**Paper:** [MCGrad: Multicalibration at Web Scale](https://arxiv.org/abs/2509.19884) (KDD 2026)

### Related Publications

For more on multicalibration theory and applications:

- **Measuring Multi-Calibration:** Guy, I., Haimovich, D., Linder, F., Okati, N., Perini, L., Tax, N., & Tygert, M. (2025). [Measuring multi-calibration](https://arxiv.org/abs/2506.11251). arXiv:2506.11251.

- **Multicalibration Applications:** Baldeschi, R. C., Di Gregorio, S., Fioravanti, S., Fusco, F., Guy, I., Haimovich, D., Leonardi, S., et al. (2025). [Multicalibration yields better matchings](https://arxiv.org/abs/2511.11413). arXiv:2511.11413.

## 📊 CI Status

[![CI](https://github.com/facebookincubator/MCGrad/actions/workflows/main.yaml/badge.svg)](https://github.com/facebookincubator/MCGrad/actions/workflows/main.yaml)
