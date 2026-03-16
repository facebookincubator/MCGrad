# Changelog

All notable changes to MCGrad will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Fixed documentation inaccuracies in the MCE metrics page: corrected `mce` formatting from percent to absolute scale, added documentation for MDE and all four available scales, and clarified statistical interpretation of sigma threshold and p-value.

### Changed
- Improved early stopping performance by caching per-fold metric DataFrames and fold splits, avoiding redundant DataFrame reconstruction on every round × fold × metric evaluation (#229)
- Speed up fitting with early stopping and save training performance enabled: Remove redundant `_predict` call on the training set in `_determine_best_num_rounds`, reusing predictions already returned by `_fit_single_round` instead (#236)
- Unified `logistic()` and `logistic_vectorized()` into a single `logistic()` function that handles both scalar and array inputs. The `logistic_vectorized` alias has been removed. Performance improvement of ~16x on large arrays by using native NumPy operations instead of `np.vectorize`.

### Added
- Conda / Mamba installation instructions to the documentation (#234)
- `youdens_j` metric (Youden's J statistic / informedness) in the `metrics` module. Computes the continuous net detection rate as `E[scores | positive] - E[scores | negative]`, ranging from -1 to 1, with support for sample weights (#233)
- Support for float calibration targets (soft labels) in `MCGrad`. Labels can now be continuous values in [0, 1] instead of only binary 0/1 or True/False. This enables calibrating to confidence-weighted labels that account for label uncertainty and quality.
- `soft_label_log_loss` function in the `metrics` module for soft-label cross-entropy, used as the default early stopping metric in `MCGrad`.
- Feature consistency check in `predict()` that validates feature names and order match those used during `fit()`. Previously, mismatched features produced confusing encoder errors or silently wrong predictions when feature order was swapped.
- Feature names are now included in the serialized model JSON for consistency validation after deserialization
- Regression support in `MulticalibrationError` — new `outcome_type="regression"` parameter uses successive-differences variance estimation (equation 6, appendix C from the [MCE paper](https://arxiv.org/abs/2506.11251)) instead of Bernoulli variance (#218)
- Regression support in `tune_mcgrad_params` — now accepts `RegressionMCGrad` models, uses the model's `early_stopping_score_func` instead of hardcoded `normalized_entropy` (#213)

## [0.1.3] - 2026-02-11

### Changed
- Lowered psutil dependency to 5.9.0 to fix compatibility with Google Colab (#210)

## [0.1.2] - 2026-02-10

### Fixed
- Fixed optimization direction in `tune_mcgrad_params` — was incorrectly maximizing normalized entropy instead of minimizing it (#202)

### Added
- `use_model_predictions` parameter to `tune_mcgrad_params` to control whether the surrogate model's predicted best or the actual best observed trial is
  returned (#196)
- `mcgrad.tutorials` submodule with shared tutorial helpers (data loading,
  plotting, logging utilities)

### Changed
- Moved `folktables` from optional `[tutorials]` dependency to core dependencies


## [0.1.1] - 2026-01-30

Note: Early pre-release versions (i.e., 0.0.0 and 0.1.0) were yanked due to packaging/versioning cleanup; 0.1.1 is the first stable public release.
### Added
- Initial release of this library. To cite this library, please cite our KDD 2026 paper ["MCGrad: Multicalibration at Web Scale"](https://doi.org/10.1145/3770854.3783954)
- Core multicalibration algorithm (`MCGrad`, `RegressionMCGrad`)
- Traditional calibration methods (`IsotonicRegression`, `PlattScaling`, `TemperatureScaling`)
- Segment-aware calibrators for grouped calibration
- Comprehensive metrics module with calibration error metrics, Kuiper statistics, and ranking metrics
- `MulticalibrationError` class for measuring multicalibration quality
- Plotting utilities for calibration curves and diagnostics
- Hyperparameter tuning integration with Ax platform
- Full documentation website at mcgrad.dev
- Sphinx API documentation at mcgrad.readthedocs.io


<!-- Link will be updated after first release tag is created -->
[Unreleased]: https://github.com/facebookincubator/MCGrad/commits/main
