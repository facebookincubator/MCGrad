---
sidebar_position: 5
---

# Tutorials

Interactive Jupyter notebooks demonstrating how to use MCGrad for multicalibration.

## Available Tutorials

| Tutorial | Description | Launch |
|----------|-------------|--------|
| **MCGrad Core Algorithm** | Complete introduction to multicalibration with MCGrad | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookincubator/MCGrad/blob/main/tutorials/01_mcgrad_core.ipynb) |
| **Trustworthy LLM Confidence for Downstream Decisions with MCGrad** | Apply MCGrad to a Claude Opus classifier so confidence-based decisions (auto-action thresholds, human-routing, prevalence estimation) hold up across segments of your data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookincubator/MCGrad/blob/main/tutorials/02_calibrating_llm_outputs.ipynb) |

## What You'll Learn

### 01. MCGrad Core Algorithm

This comprehensive tutorial covers:

1. **Why Multicalibration Matters** - Understand the limitations of global calibration and why segment-level calibration is important
2. **MCGrad Basics** - Learn how to use the `fit()` and `predict()` API
3. **Measuring Multicalibration** - Use the Multicalibration Error (MCE) metric to evaluate calibration quality
4. **Visualization** - Plot global and segment-level calibration curves
5. **Advanced Features** - Explore feature importance, model serialization, numerical features, and custom hyperparameters

### 02. Trustworthy LLM Confidence for Downstream Decisions with MCGrad

This tutorial applies the MCGrad workflow to an LLM classifier (Claude Opus 4.6 on Comparative Agendas Project documents) and covers:

1. **Why Raw LLM Confidence Isn't Trustworthy** - Diagnose how an LLM's elicited probabilities over-estimate prevalence per country, by different amounts each time
2. **Global Calibration Isn't Enough** - See why isotonic regression closes the global gap but leaves significant per-segment miscalibration
3. **Fitting MCGrad on LLM Outputs** - Train MCGrad with document metadata as segment features
4. **Risk-Calibrated Thresholds** - Set a confidence threshold once and get the same per-item risk in every country, content type, or cohort
5. **Per-Segment Prevalence Estimation** - Recover unbiased positive rates within any sub-population, ready for stakeholder breakdowns

## Running Tutorials

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badge above to run tutorials directly in your browser. No local setup required!

### Option 2: Local Jupyter

1. Install MCGrad with tutorial dependencies:

```bash
pip install "MCGrad[tutorials] @ git+https://github.com/facebookincubator/MCGrad.git"
```

2. Clone the repository and navigate to tutorials:

```bash
git clone https://github.com/facebookincubator/MCGrad.git
cd MCGrad/tutorials
jupyter notebook 01_mcgrad_core.ipynb
```

### Option 3: VS Code

Open the `.ipynb` files directly in VS Code with the built-in Jupyter extension.

## Contributing Tutorials

We welcome contributions! If you'd like to add a new tutorial:

1. Create a new `.ipynb` file in the `tutorials/` directory
2. Follow the naming convention: `XX_descriptive_name.ipynb`
3. Include a Colab setup cell at the top (see existing tutorials for the pattern)
4. Add the tutorial to this documentation page
5. Submit a pull request

See the [Contributing Guide](contributing.md) for more details.
