# Changelog

All notable changes to MCGrad will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Improved early stopping performance by caching per-fold metric DataFrames and fold splits, avoiding redundant DataFrame reconstruction on every round × fold × metric evaluation (#229)
- Refactored `_compute_unshrink_factor`

### Added
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
