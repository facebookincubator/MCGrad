# Changelog

All notable changes to MCGrad will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial open-source release of MCGrad
- Core multicalibration algorithm (`MCGrad`, `RegressionMCGrad`)
- Traditional calibration methods (`IsotonicRegression`, `PlattScaling`, `TemperatureScaling`)
- Segment-aware calibrators for grouped calibration
- Comprehensive metrics module with calibration error metrics, Kuiper statistics, and ranking metrics
- `MulticalibrationError` class for measuring multicalibration quality
- Plotting utilities for calibration curves and diagnostics
- Hyperparameter tuning integration with Ax platform
- Full documentation website at mcgrad.dev
- Sphinx API documentation at mcgrad.readthedocs.io

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

<!-- Link will be updated after first release tag is created -->
[Unreleased]: https://github.com/facebookincubator/MCGrad/commits/main
