# Changelog

All notable changes to MCGrad will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `num_rounds_trained` property on `MCGrad` / `RegressionMCGrad`, returning the number of boosting rounds actually trained — useful after early stopping, where this can be strictly less than the configured `num_rounds`.
- Schema versioning for serialized models via a `schema_version` field in the JSON payload. Deserialize rejects unknown versions with a clear error; pre-existing serialized models without this field continue to load unchanged (with a deprecation warning).

### Changed
- Bumped serialization `schema_version` from `1` to `2`. The new version uses an identical wire format; the bump signals that downstream consumers should enforce explicit version checks. `deserialize()` accepts both `1` and `2`.
- `MCGrad.serialize()` / `deserialize()` now round-trip the full set of JSON-serializable constructor arguments (e.g. `monotone_t`, `early_stopping`, `patience`, `n_folds`, `lightgbm_params`, timeouts). Previously only a small subset survived a round-trip. Non-JSON-serializable state — custom `early_stopping_score_func`, `early_stopping_minimize_score`, `monitored_metrics_during_training`, and `random_state` — is still reset to defaults; re-fit flows that depend on those must set them explicitly after `deserialize`.
- Behavior change on `num_rounds` after `deserialize`: in schema v1, `self.num_rounds` is restored to the configured upper bound rather than being overwritten with the trained-booster count. Callers that relied on `num_rounds == len(mr)` should read `num_rounds_trained` instead.

### Fixed
- Fixed `_BaseMCGrad.deserialize()` not restoring `allow_missing_segment_feature_values`, causing models serialized with `allow_missing_segment_feature_values=False` to silently revert to the default (`True`) after deserialization.

## [0.1.4] - 2026-03-26

### Fixed
- Fixed documentation inaccuracies in the MCE metrics page: corrected `mce` formatting from percent to absolute scale, added documentation for MDE and all four available scales, and clarified statistical interpretation of sigma threshold and p-value.
- Fixed `calibration_free_normalized_entropy` not passing `sample_weight` to the final `normalized_entropy` call, causing weighted calibration-free NE to silently ignore sample weights.
- Fixed `predictions_to_labels` ignoring the `threshold_column` parameter — the threshold comparison hardcoded `.threshold` via pandas attribute access instead of using `[threshold_column]`, causing an `AttributeError` when a non-default column name was passed.
- Fixed `fpr_at_precision` not using `sample_weight` when computing the false positive rate, causing weighted FPR to silently use unweighted counts.

### Changed
- Improved early stopping performance by caching per-fold metric DataFrames and fold splits, avoiding redundant DataFrame reconstruction on every round × fold × metric evaluation (#229)
- Speed up fitting with early stopping and save training performance enabled: Remove redundant `_predict` call on the training set in `_determine_best_num_rounds`, reusing predictions already returned by `_fit_single_round` instead (#236)
- Skip redundant `_inverse_transform_predictions` calls in `_predict` when `return_all_rounds=False`. Previously called the logistic function on every boosting round but only used the last result; now calls it once. Reduces total inverse-transform calls during early stopping from O(R²) to O(R) (#241)
- Unified `logistic()` and `logistic_vectorized()` into a single `logistic()` function that handles both scalar and array inputs. The `logistic_vectorized` alias has been removed. Performance improvement of ~16x on large arrays by using native NumPy operations instead of `np.vectorize`.
- Vectorized `segments_ecce_pvalue` computation in `MulticalibrationError` using NumPy broadcasting instead of `np.vectorize`. Unified `_ecce_cdf` and `ecce_pvalue_from_sigma` to accept both scalar and array inputs, removing the separate `_ecce_pvalue_from_sigma_vectorized` helper.
- Eliminated redundant `segments * sample_weight` broadcast in `_calculate_cumulative_differences` (~6% faster on the ECCE computation hot path).
- Eliminated redundant `segments * sample_weight` broadcast in `_ecce_standard_deviation_per_segment` (~30% faster on the std dev computation hot path).
- Replaced per-segment `pd.DataFrame` creation with batch list accumulation in `get_segment_masks`, yielding 2.2-3.1x speedup on the segment generation hot path (~53% of total pipeline time).

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


[Unreleased]: https://github.com/facebookincubator/MCGrad/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/facebookincubator/MCGrad/compare/v0.1.3...v0.1.4
