# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MCGrad Core Algorithm Tutorial
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookincubator/MCGrad/blob/main/tutorials/01_mcgrad_core.ipynb)
#
# This tutorial provides a comprehensive introduction to **MCGrad** (Multicalibration Gradient Boosting),
# a powerful algorithm for calibrating machine learning predictions across arbitrary population segments.
#
# ## What You'll Learn
#
# 1. **Why Multicalibration Matters** - The limitations of global calibration
# 2. **MCGrad Basics** - How to use MCGrad to multicalibrate predictions
# 3. **Measuring Multicalibration** - Using the Multicalibration Error metric
# 4. **Visualization** - Plotting calibration curves globally and by segment
# 5. **Advanced Features** - Early stopping, feature importance, and serialization
#
# ## Prerequisites
#
# - Basic Python and pandas knowledge
# - Understanding of ML model calibration (helpful but not required)

# %% [markdown]
# ## Setup
#
# First, let's install MCGrad and import the required libraries.

# %%
# Install MCGrad (uncomment if running in Colab or if not already installed)
# !pip install "MCGrad @ git+https://github.com/facebookincubator/MCGrad.git"

# %%

# TODO: This is just AI generated for now. Replace with real content.

import numpy as np

# For reproducibility
np.random.seed(42)

# %% [markdown]
# ## Part 1: Why Multicalibration Matters
#
# ### The Problem with Global Calibration
#
# Traditional calibration methods (like Platt Scaling or Isotonic Regression) ensure that
# predictions are well-calibrated **on average** across all data. However, they can still
# be severely miscalibrated for specific subgroups.
#
# Let's create a synthetic dataset that demonstrates this problem.
