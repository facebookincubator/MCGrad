# Changelog

All notable changes to MCGrad will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
