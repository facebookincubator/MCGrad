# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

import math
import os
import pickle
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import scipy

from multicalibration import methods
from multicalibration.tuning import (
    DEFAULT_CV_METRICS,
    DEFAULT_PARAMETER_SPACE,
    ORIGINAL_LIGHTGBM_PARAMS,
    ParameterConfig,
    _aggregate_metrics_and_add_parameters,
    _cache_results,
    _check_cache,
    _compute_metrics,
    _confidence_interval,
    _kuiper_p_value_metric,
    _kuiper_statistic_sd_normalized,
    _make_segments_df,
    _mce_metric,
    _multi_ace_metric,
    _prauc_metric,
    _set_up_cv,
    calibration_ratio,
    default_parameter_configurations,
    mean_score,
    num_negatives,
    num_positives,
    num_rows,
    numerical_features_max_abs_rank_correlation_with_residuals,
    numerical_features_min_rank_correlation_pval_with_residuals,
    prevalence,
    tune_mcboost,
    tune_mcboost_params,
    var_score,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def sample_data(rng):
    n_samples = 100

    df = pd.DataFrame(
        {
            "prediction": rng.uniform(0.1, 0.9, n_samples),
            "label": rng.binomial(1, 0.3, n_samples),
            "weight": rng.uniform(0.5, 2.0, n_samples),
            "cat_feature": rng.choice(["A", "B", "C"], n_samples),
            "num_feature": rng.normal(0, 1, n_samples),
        }
    )

    return df


@pytest.fixture
def sample_val_data(rng):
    n_samples = 30
    return pd.DataFrame(
        {
            "prediction": rng.uniform(0.1, 0.9, n_samples),
            "label": rng.binomial(1, 0.3, n_samples),
            "weight": rng.uniform(0.5, 2.0, n_samples),
            "cat_feature": rng.choice(["A", "B", "C"], n_samples),
            "num_feature": rng.normal(0, 1, n_samples),
        }
    )


@pytest.fixture
def mock_mcboost_model(rng):
    model = Mock(spec=methods.MCBoost)
    model.predict = Mock(return_value=rng.uniform(0.1, 0.9, 80))
    return model


@pytest.fixture
def hyperparams_for_tuning():
    default_hyperparams = methods.MCBoost().DEFAULT_HYPERPARAMS
    lightgbm_params = default_hyperparams["lightgbm_params"]
    return default_hyperparams, lightgbm_params


@pytest.mark.arm64_incompatible
@patch("multicalibration.tuning.normalized_entropy")
def test_tune_mcboost_params_with_weights(
    mock_normalized_entropy,
    sample_data,
    mock_mcboost_model,
):
    # Setup mocks
    mock_normalized_entropy.return_value = 0.5

    result_model, trial_results = tune_mcboost_params(
        model=mock_mcboost_model,
        df_train=sample_data,
        prediction_column_name="prediction",
        label_column_name="label",
        weight_column_name="weight",
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        n_trials=3,
    )

    assert result_model is not None
    assert isinstance(trial_results, pd.DataFrame)

    # Verify that fit was called with weight_column_name
    assert mock_mcboost_model.fit.call_count >= 1
    fit_calls = mock_mcboost_model.fit.call_args_list
    for call in fit_calls:
        assert call[1]["weight_column_name"] == "weight"

    # Verify that normalized_entropy was called with sample_weight
    assert mock_normalized_entropy.call_count >= 1
    entropy_calls = mock_normalized_entropy.call_args_list
    for call in entropy_calls:
        assert "sample_weight" in call[1]
        assert call[1]["sample_weight"] is not None


@pytest.mark.arm64_incompatible
@patch("multicalibration.tuning.normalized_entropy")
def test_tune_mcboost_params_without_weights(
    mock_normalized_entropy,
    sample_data,
    mock_mcboost_model,
):
    mock_normalized_entropy.return_value = 0.5

    result_model, trial_results = tune_mcboost_params(
        model=mock_mcboost_model,
        df_train=sample_data,
        prediction_column_name="prediction",
        label_column_name="label",
        weight_column_name=None,
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        n_trials=3,
    )

    assert result_model is not None
    assert isinstance(trial_results, pd.DataFrame)

    # Verify that fit was called with weight_column_name=None
    assert mock_mcboost_model.fit.call_count >= 1
    fit_calls = mock_mcboost_model.fit.call_args_list
    for call in fit_calls:
        assert call[1]["weight_column_name"] is None

    # Verify that normalized_entropy was called with sample_weight=None
    assert mock_normalized_entropy.call_count >= 1
    entropy_calls = mock_normalized_entropy.call_args_list
    for call in entropy_calls:
        assert "sample_weight" in call[1]
        assert call[1]["sample_weight"] is None


@pytest.mark.arm64_incompatible
@patch("multicalibration.tuning.normalized_entropy")
def test_tune_mcboost_params_default_parameters(
    mock_normalized_entropy,
    sample_data,
    mock_mcboost_model,
):
    mock_normalized_entropy.return_value = 0.5

    result_model, trial_results = tune_mcboost_params(
        model=mock_mcboost_model,
        df_train=sample_data,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["cat_feature"],
        n_trials=2,
    )

    assert result_model is not None
    assert isinstance(trial_results, pd.DataFrame)

    # Verify that fit was called with correct values
    assert mock_mcboost_model.fit.call_count >= 1
    fit_calls = mock_mcboost_model.fit.call_args_list
    for call in fit_calls:
        assert call[1]["weight_column_name"] is None
        assert call[1]["categorical_feature_column_names"] == ["cat_feature"]
        assert call[1]["numerical_feature_column_names"] is None


@pytest.mark.arm64_incompatible
@patch("multicalibration.tuning.normalized_entropy")
def test_tune_mcboost_params_ax_client_setup(
    mock_normalized_entropy,
    sample_data,
    mock_mcboost_model,
):
    mock_normalized_entropy.return_value = 0.5

    result_model, trial_results = tune_mcboost_params(
        model=mock_mcboost_model,
        df_train=sample_data,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["cat_feature"],
        n_trials=2,
    )

    assert result_model is not None
    assert isinstance(trial_results, pd.DataFrame)
    assert len(trial_results) == 2

    # Verify that fit was called multiple times (once per trial + final fit)
    assert mock_mcboost_model.fit.call_count >= 2


@pytest.mark.arm64_incompatible
@patch("multicalibration.tuning.normalized_entropy")
@patch("multicalibration.tuning.train_test_split")
def test_tune_mcboost_params_data_splitting(
    mock_train_test_split,
    mock_normalized_entropy,
    sample_data,
    mock_mcboost_model,
):
    # Setup mock to return specific train/val splits
    train_data = sample_data.iloc[:80]
    val_data = sample_data.iloc[80:]
    mock_train_test_split.return_value = (train_data, val_data)

    mock_normalized_entropy.return_value = 0.5

    tune_mcboost_params(
        model=mock_mcboost_model,
        df_train=sample_data,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["cat_feature"],
        n_trials=1,
    )

    # Verify train_test_split was called with correct parameters
    mock_train_test_split.assert_called_once()
    call_args = mock_train_test_split.call_args
    assert (
        call_args[0][0] is sample_data
    )  # First argument should be the input dataframe
    assert call_args[1]["test_size"] == 0.2
    assert call_args[1]["random_state"] == 42


@pytest.mark.arm64_incompatible
@patch("multicalibration.tuning.normalized_entropy")
def test_tune_mcboost_params_with_subset_of_parameters(
    mock_normalized_entropy,
    sample_data,
):
    mock_normalized_entropy.return_value = 0.5
    model = methods.MCBoost()

    subset_params = ["learning_rate", "max_depth"]
    subset_configs = [
        config
        for config in default_parameter_configurations
        if config.name in subset_params
    ]

    _, trial_results = tune_mcboost_params(
        model=model,
        df_train=sample_data,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        n_trials=3,
        parameter_configurations=subset_configs,
    )

    # Verify that only the specified parameters were tuned
    for param in subset_params:
        assert param in trial_results.columns

    # Check that parameters not in our subset are not in the results (except defaults)
    excluded_params = [
        config.name
        for config in default_parameter_configurations
        if config.name not in subset_params
    ]

    for param in excluded_params:
        assert param not in trial_results.columns


def test_mcboost_and_lightgbm_default_hyperparams_are_within_bounds_for_tuning(
    hyperparams_for_tuning,
):
    _, lightgbm_params = hyperparams_for_tuning

    for config in default_parameter_configurations:
        param_name = config.name
        bounds = config.bounds
        if param_name in lightgbm_params:
            default_value = lightgbm_params[param_name]
            original_value = ORIGINAL_LIGHTGBM_PARAMS[param_name]
            assert (
                bounds[0] <= default_value <= bounds[1]
            ), f"Default {param_name} ({default_value}) is outside of bounds ({bounds})"
            assert (
                bounds[0] <= original_value <= bounds[1]
            ), f"Original {param_name} ({original_value}) is outside of bounds ({bounds})"
            # Check value type
            if config.value_type == "int":
                assert isinstance(
                    default_value, int
                ), f"Default {param_name} ({default_value}) should be an integer"
            elif config.value_type == "float":
                assert isinstance(
                    default_value, float
                ), f"Default {param_name} ({default_value}) should be a float"


@pytest.mark.arm64_incompatible
def test_warm_starting_trials_produces_the_right_number_of_sobol_and_bayesian_trials(
    rng,
):
    df_train = pd.DataFrame(
        {
            "feature1": rng.randint(0, 3, 20),
            "prediction": rng.rand(20),
            "label": rng.randint(0, 2, 20),
            "weight": 1,
        }
    )

    n_warmup_random_trials = 1
    total_trials = 3

    _, trial_results = tune_mcboost_params(
        model=methods.MCBoost(num_rounds=0, early_stopping=False),
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        weight_column_name="weight",
        categorical_feature_column_names=["feature1"],
        numerical_feature_column_names=[],
        n_trials=total_trials,
        n_warmup_random_trials=n_warmup_random_trials,
    )

    value_counter = trial_results["generation_node"].value_counts().to_dict()
    sobol_count = value_counter["GenerationStep_0"]
    BoTorch_count = value_counter["GenerationStep_1"]

    assert len(trial_results) == total_trials, "Expected {} trials, got {}.".format(
        total_trials, len(trial_results)
    )
    assert (
        sobol_count == n_warmup_random_trials
    ), "Expected {} Sobol trials, got {}.".format(n_warmup_random_trials, sobol_count)
    assert (
        BoTorch_count == total_trials - n_warmup_random_trials - 1
    ), "Expected {} BoTorch trials, got {}.".format(
        total_trials - n_warmup_random_trials - 1, BoTorch_count
    )


# Tests for ParameterConfig class
def test_parameter_config_to_dict_works_properly():
    config = ParameterConfig(
        name="learning_rate",
        bounds=[0.01, 0.3],
        value_type="float",
        log_scale=True,
        config_type="range",
    )

    expected_dict = {
        "name": "learning_rate",
        "bounds": [0.01, 0.3],
        "value_type": "float",
        "log_scale": True,
        "type": "range",
    }

    assert config.to_dict() == expected_dict


def test_default_parameter_configurations_have_valid_types():
    for config in default_parameter_configurations:
        assert isinstance(config, ParameterConfig)
        assert isinstance(config.name, str)
        assert len(config.name) > 0
        assert isinstance(config.bounds, list)
        assert len(config.bounds) == 2
        assert config.bounds[0] < config.bounds[1]
        assert config.value_type in ["int", "float"]
        assert isinstance(config.log_scale, bool)
        assert config.config_type == "range"


# Tests for metric functions
def test_kuiper_statistic_sd_normalized_return_values_not_NaNs(sample_data):
    result = _kuiper_statistic_sd_normalized(
        df=sample_data,
        label_column="label",
        score_column="prediction",
        weight_column="weight",
    )

    assert isinstance(result, float)
    assert not np.isnan(result)


def test_kuiper_p_value_metric_returns_float_not_NaN(sample_data):
    result = _kuiper_p_value_metric(
        df=sample_data,
        label_column="label",
        score_column="prediction",
        weight_column="weight",
    )

    assert isinstance(result, float)
    assert 0 <= result <= 1
    assert not np.isnan(result)


def test_prauc_metric_returns_float_not_NaN(sample_data):
    result = _prauc_metric(
        df=sample_data,
        label_column="label",
        score_column="prediction",
        weight_column="weight",
    )

    assert isinstance(result, float)
    assert 0 <= result <= 1
    assert not np.isnan(result)


def test_make_segments_df_with_categorical_only_works_as_expected(sample_data):
    result = _make_segments_df(
        df=sample_data,
        categorical_segment_columns=["cat_feature"],
        numerical_segment_columns=None,
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["cat_feature"]
    assert len(result) == len(sample_data)


def test_make_segments_df_with_numerical_only_works_as_expected(sample_data):
    result = _make_segments_df(
        df=sample_data,
        categorical_segment_columns=None,
        numerical_segment_columns=["num_feature"],
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["num_feature"]
    assert len(result) == len(sample_data)


def test_make_segments_df_with_both_types_works_as_expected(sample_data):
    result = _make_segments_df(
        df=sample_data,
        categorical_segment_columns=["cat_feature"],
        numerical_segment_columns=["num_feature"],
    )

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"cat_feature", "num_feature"}
    assert len(result) == len(sample_data)


def test_make_segments_df_raises_error_without_columns():
    df = pd.DataFrame({"col1": [1, 2, 3]})

    with pytest.raises(AssertionError):
        _make_segments_df(
            df=df,
            categorical_segment_columns=None,
            numerical_segment_columns=None,
        )


def test_multi_ace_metric_returns_float_not_NaN(sample_data):
    result = _multi_ace_metric(
        df=sample_data,
        label_column="label",
        score_column="prediction",
        weight_column="weight",
        categorical_segment_columns=["cat_feature"],
        numerical_segment_columns=["num_feature"],
    )

    assert isinstance(result, float)
    assert result >= 0
    assert not np.isnan(result)


def test_mce_metric_returns_float_not_NaN(sample_data):
    result = _mce_metric(
        df=sample_data,
        label_column="label",
        score_column="prediction",
        weight_column="weight",
        categorical_segment_columns=["cat_feature"],
        numerical_segment_columns=["num_feature"],
    )

    assert result.dtype in [np.float64, np.float32, np.float16]
    assert result >= 0
    assert not np.isnan(result)


def test_mce_metric_with_kwargs_returns_float_not_NaN(sample_data):
    result = _mce_metric(
        df=sample_data,
        label_column="label",
        score_column="prediction",
        weight_column="weight",
        categorical_segment_columns=["cat_feature"],
        numerical_segment_columns=["num_feature"],
        max_depth=2,
        max_values_per_segment_feature=2,
        min_samples_per_segment=5,
    )

    assert result.dtype in [np.float64, np.float32, np.float16]
    assert result >= 0
    assert not np.isnan(result)


# Tests for aggregation and context feature functions
def test_prevalence_in_tuning_file_is_float_btw0and1_and_is_the_mean(sample_data):
    result = prevalence(
        df=sample_data,
        scores_column="prediction",
        label_column="label",
        categorical_segment_columns=["cat_feature"],
        numerical_segment_columns=["num_feature"],
    )

    assert isinstance(result, float)
    assert 0 <= result <= 1
    expected = sample_data["label"].mean()
    assert abs(result - expected) < 1e-10


def test_num_positives_returns_int_not_NaN(sample_data):
    result = num_positives(
        df=sample_data,
        scores_column="prediction",
        label_column="label",
        categorical_segment_columns=["cat_feature"],
        numerical_segment_columns=["num_feature"],
    )

    assert isinstance(result, np.int64 | np.int32)
    assert result >= 0
    expected = sample_data["label"].sum()
    assert abs(result - expected) < 1e-10


def test_num_negatives_returns_int_not_NaN(sample_data):
    result = num_negatives(
        df=sample_data,
        scores_column="prediction",
        label_column="label",
        categorical_segment_columns=["cat_feature"],
        numerical_segment_columns=["num_feature"],
    )

    assert isinstance(result, np.int64 | np.int32)
    assert result >= 0
    expected = len(sample_data) - sample_data["label"].sum()
    assert abs(result - expected) < 1e-10


def test_num_rows_works_properly(sample_data):
    result = num_rows(
        df=sample_data,
        scores_column="prediction",
        label_column="label",
        categorical_segment_columns=["cat_feature"],
        numerical_segment_columns=["num_feature"],
    )

    assert isinstance(result, int)
    assert result == len(sample_data)


def test_mean_score_metric_works_properly(sample_data):
    result = mean_score(
        df=sample_data,
        scores_column="prediction",
        label_column="label",
        categorical_segment_columns=["cat_feature"],
        numerical_segment_columns=["num_feature"],
    )

    assert isinstance(result, float)
    expected = sample_data["prediction"].mean()
    assert abs(result - expected) < 1e-10


def test_var_score_returns_float_not_NaN(sample_data):
    """Test var_score function."""
    result = var_score(
        df=sample_data,
        scores_column="prediction",
        label_column="label",
        categorical_segment_columns=["cat_feature"],
        numerical_segment_columns=["num_feature"],
    )

    assert isinstance(result, float)
    assert result >= 0
    expected = sample_data["prediction"].var()
    assert abs(result - expected) < 1e-10


def test_calibration_ratio_returns_float_not_NaN(sample_data):
    """Test calibration_ratio function."""
    result = calibration_ratio(
        df=sample_data,
        scores_column="prediction",
        label_column="label",
        categorical_segment_columns=["cat_feature"],
        numerical_segment_columns=["num_feature"],
    )

    assert isinstance(result, float)
    assert result > 0
    expected = sample_data["prediction"].mean() / sample_data["label"].mean()
    assert abs(result - expected) < 1e-10


def test_calibration_ratio_with_zero_prevalence_returns_inf():
    df = pd.DataFrame(
        {
            "prediction": [0.1, 0.2, 0.3],
            "label": [0, 0, 0],
            "cat_feature": ["A", "B", "C"],
            "num_feature": [1, 2, 3],
        }
    )

    result = calibration_ratio(
        df=df,
        scores_column="prediction",
        label_column="label",
        categorical_segment_columns=["cat_feature"],
        numerical_segment_columns=["num_feature"],
    )

    # Should return inf when dividing by zero
    assert np.isinf(result)


def test_numerical_features_max_abs_rank_correlation_with_residuals_returns_float(
    sample_data,
):
    result = numerical_features_max_abs_rank_correlation_with_residuals(
        df=sample_data,
        scores_column="prediction",
        label_column="label",
        categorical_segment_columns=["cat_feature"],
        numerical_segment_columns=["num_feature"],
    )

    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_numerical_features_min_rank_correlation_pval_with_residuals_returns_float(
    sample_data,
):
    result = numerical_features_min_rank_correlation_pval_with_residuals(
        df=sample_data,
        scores_column="prediction",
        label_column="label",
        categorical_segment_columns=["cat_feature"],
        numerical_segment_columns=["num_feature"],
    )

    assert isinstance(result, float)
    assert 0 <= result <= 1


# Tests for cache management functions
def test_check_cache_no_cache_file_works_properly():
    fold_results, fold_params, last_iteration = _check_cache(
        cache_file=None,
        continue_from_cache=False,
    )

    assert fold_results == []
    assert fold_params == []
    assert last_iteration == 0


def test_check_cache_with_cache_file_works_properly():
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        cache_file = tmp_file.name
        test_fold_results = [{"param_iteration": 0}, {"param_iteration": 1}]
        test_fold_params = [{"param1": 1}, {"param2": 2}]
        pickle.dump((test_fold_results, test_fold_params), tmp_file)

    try:
        fold_results, fold_params, last_iteration = _check_cache(
            cache_file=cache_file,
            continue_from_cache=True,
        )

        assert fold_results == test_fold_results
        assert fold_params == test_fold_params
        assert last_iteration == 1
    finally:
        os.unlink(cache_file)


def test_cache_results_works_properly():
    test_fold_results = [{"metric": 0.5}]
    test_fold_params = [{"param": 1}]

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        cache_file = tmp_file.name

    try:
        _cache_results(cache_file, test_fold_results, test_fold_params)

        # Verify the file was created and contains correct data
        with open(cache_file, "rb") as f:
            loaded_results, loaded_params = pickle.load(f)

        assert loaded_results == test_fold_results
        assert loaded_params == test_fold_params
    finally:
        os.unlink(cache_file)


def test_cache_results_no_file_does_not_raise_error():
    _cache_results(None, [], [])


# Tests for cross-validation setup
def test_set_up_cv_default_parameter_space():
    """Test _set_up_cv with default parameter space."""
    splitter, sampler = _set_up_cv(
        parameter_space=None,
        n_parameter_samples=5,
        n_folds=3,
        random_state=42,
    )

    assert splitter.n_splits == 3
    assert splitter.shuffle is True
    assert splitter.random_state == 42

    # Check that sampler uses default parameter space
    sample_params = list(sampler)
    assert len(sample_params) == 5

    # Verify that sampled parameters contain expected keys
    for params in sample_params:
        assert "learning_rate" in params
        assert "num_leaves" in params


def test_compute_metrics_works_properly(sample_data):
    metrics = {
        "test_metric": lambda df, label_column, score_column, **kwargs: 0.5,
    }

    result = _compute_metrics(
        metrics=metrics,
        df=sample_data,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=["cat_feature"],
        numerical_segment_columns=["num_feature"],
        weight_column="weight",
    )

    assert isinstance(result, dict)
    assert "test_metric" in result
    assert result["test_metric"] == 0.5


def test_compute_metrics_with_failing_metric_returns_NaNs(sample_data):
    def failing_metric(**kwargs):
        raise ValueError("Test error")

    metrics = {
        "failing_metric": failing_metric,
        "working_metric": lambda **kwargs: 0.8,
    }

    result = _compute_metrics(
        metrics=metrics,
        df=sample_data,
        label_column="label",
        score_column="prediction",
    )

    assert isinstance(result, dict)
    assert np.isnan(result["failing_metric"])
    assert result["working_metric"] == 0.8


# Tests for confidence interval function
def test_confidence_interval_returns_float_not_NaN():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    lower, upper = _confidence_interval(data)

    assert isinstance(lower, float)
    assert isinstance(upper, float)
    assert lower < upper
    assert lower <= np.mean(data) <= upper


def test_confidence_interval_single_value_returns_float_not_NaN():
    data = np.array([5.0])

    lower, upper = _confidence_interval(data)

    # With single value, confidence interval should be very wide
    assert isinstance(lower, float)
    assert isinstance(upper, float)


# Tests for aggregate metrics function
def test_aggregate_metrics_and_add_parameters_works_properly():
    fold_results = [
        {
            "param_iteration": 0,
            "num_rounds": 1,
            "method": "MCBoost",
            "metric1": 0.5,
            "context": {"feature": 1},
        },
        {
            "param_iteration": 0,
            "num_rounds": 1,
            "method": "MCBoost",
            "metric1": 0.6,
            "context": {"feature": 1},
        },
        {
            "param_iteration": 1,
            "num_rounds": 1,
            "method": "MCBoost",
            "metric1": 0.7,
            "context": {"feature": 2},
        },
    ]

    fold_params = [
        {"param_iteration": 0, "num_rounds": 1, "param1": 0.1},
        {"param_iteration": 0, "num_rounds": 1, "param1": 0.1},
        {"param_iteration": 1, "num_rounds": 1, "param1": 0.2},
    ]

    result = _aggregate_metrics_and_add_parameters(fold_results, fold_params)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Two unique param_iteration values
    assert "metric1_mean" in result.columns
    assert "metric1_min" in result.columns
    assert "metric1_max" in result.columns
    assert "param_dict" in result.columns
    assert "context" in result.columns


@pytest.mark.arm64_incompatible
def test_non_default_parameters_preserved_when_not_in_tuning_configurations(
    sample_data,
):
    non_default_params = {
        "learning_rate": 0.05,
        "max_depth": 8,
        "lambda_l2": 5.0,
    }
    model = methods.MCBoost(lightgbm_params=non_default_params)

    for param, value in non_default_params.items():
        assert model.lightgbm_params[param] == value

    with patch("multicalibration.tuning.normalized_entropy", return_value=0.5):
        # Create parameter configurations that don't include our non-default parameters
        # This will tune only num_leaves and min_child_samples
        tune_params = ["num_leaves", "min_child_samples"]
        parameter_configs = [
            config
            for config in default_parameter_configurations
            if config.name in tune_params
        ]

        # Run tuning with these limited configurations
        result_model, _ = tune_mcboost_params(
            model=model,
            df_train=sample_data,
            prediction_column_name="prediction",
            label_column_name="label",
            categorical_feature_column_names=["cat_feature"],
            numerical_feature_column_names=["num_feature"],
            n_trials=2,
            parameter_configurations=parameter_configs,
        )

    assert result_model.lightgbm_params["learning_rate"] == 0.05
    assert result_model.lightgbm_params["max_depth"] == 8
    assert result_model.lightgbm_params["lambda_l2"] == 5.0


def test_default_cv_metrics_are_callable():
    for metric_name, metric_func in DEFAULT_CV_METRICS.items():
        assert callable(metric_func), f"{metric_name} is not callable"


def test_default_parameter_space_structure():
    assert isinstance(DEFAULT_PARAMETER_SPACE, dict)
    assert len(DEFAULT_PARAMETER_SPACE) > 0

    for param_name, param_dist in DEFAULT_PARAMETER_SPACE.items():
        assert isinstance(param_name, str)
        assert isinstance(param_dist, list) or hasattr(
            param_dist, "rvs"
        ), f"{param_name} has invalid distribution type"


def test_prevalence_returns_NaN_with_empty_dataframe():
    empty_df = pd.DataFrame(columns=["label", "prediction", "cat_feature"])

    assert math.isnan(
        prevalence(
            df=empty_df,
            scores_column="prediction",
            label_column="label",
            categorical_segment_columns=["cat_feature"],
            numerical_segment_columns=[],
        )
    )


def test_metric_functions_with_single_row():
    single_row_df = pd.DataFrame(
        {
            "label": [1],
            "prediction": [0.8],
            "cat_feature": ["A"],
            "num_feature": [1.0],
        }
    )

    assert (
        prevalence(
            df=single_row_df,
            scores_column="prediction",
            label_column="label",
            categorical_segment_columns=["cat_feature"],
            numerical_segment_columns=["num_feature"],
        )
        == 1.0
    )

    assert (
        num_rows(
            df=single_row_df,
            scores_column="prediction",
            label_column="label",
            categorical_segment_columns=["cat_feature"],
            numerical_segment_columns=["num_feature"],
        )
        == 1.0
    )


def test_prevalence_variance_functions_with_all_same_values_work_as_expected():
    same_values_df = pd.DataFrame(
        {
            "label": [1, 1, 1, 1],
            "prediction": [0.5, 0.5, 0.5, 0.5],
            "cat_feature": ["A", "A", "A", "A"],
            "num_feature": [1.0, 1.0, 1.0, 1.0],
        }
    )

    assert (
        prevalence(
            df=same_values_df,
            scores_column="prediction",
            label_column="label",
            categorical_segment_columns=["cat_feature"],
            numerical_segment_columns=["num_feature"],
        )
        == 1.0
    )

    assert (
        var_score(
            df=same_values_df,
            scores_column="prediction",
            label_column="label",
            categorical_segment_columns=["cat_feature"],
            numerical_segment_columns=["num_feature"],
        )
        == 0.0
    )


def test_set_up_cv_custom_parameter_space_doesnt_raise_errors():
    custom_space = {
        "param1": scipy.stats.uniform(0, 1),
        "param2": scipy.stats.randint(1, 10),
    }

    splitter, sampler = _set_up_cv(
        parameter_space=custom_space,
        n_parameter_samples=3,
        n_folds=5,
        random_state=123,
    )

    assert splitter.n_splits == 5
    assert splitter.random_state == 123

    sample_params = list(sampler)
    assert len(sample_params) == 3

    for params in sample_params:
        assert "param1" in params
        assert "param2" in params
        assert 0 <= params["param1"] <= 1
        assert 1 <= params["param2"] < 10


@pytest.mark.arm64_incompatible
@patch("multicalibration.tuning.normalized_entropy")
@patch("multicalibration.tuning.train_test_split")
def test_tune_mcboost_params_with_explicit_validation_set(
    mock_train_test_split,
    mock_normalized_entropy,
    sample_data,
    sample_val_data,
    mock_mcboost_model,
):
    mock_normalized_entropy.return_value = 0.5

    tune_mcboost_params(
        model=mock_mcboost_model,
        df_train=sample_data,
        prediction_column_name="prediction",
        label_column_name="label",
        df_val=sample_val_data,
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        n_trials=2,
    )

    # Verify train_test_split was not called
    mock_train_test_split.assert_not_called()

    # Verify that the model was fit with the training data
    assert mock_mcboost_model.fit.call_count >= 1
    fit_calls = mock_mcboost_model.fit.call_args_list
    for call in fit_calls:
        assert call[1]["df_train"] is sample_data

    # Verify that the model was evaluated on the validation data
    assert mock_mcboost_model.predict.call_count >= 1
    predict_calls = mock_mcboost_model.predict.call_args_list
    for call in predict_calls:
        assert call.kwargs["df"] is sample_val_data


@pytest.mark.arm64_incompatible
@patch("multicalibration.tuning.normalized_entropy")
@patch("multicalibration.tuning.train_test_split")
def test_tune_mcboost_params_fallback_to_train_test_split(
    mock_train_test_split,
    mock_normalized_entropy,
    sample_data,
    mock_mcboost_model,
):
    """Test that when df_val is None, train_test_split is used."""
    # Setup mock to return specific train/val splits
    train_data = sample_data.iloc[:80]
    val_data = sample_data.iloc[80:]
    mock_train_test_split.return_value = (train_data, val_data)
    mock_normalized_entropy.return_value = 0.5

    tune_mcboost_params(
        model=mock_mcboost_model,
        df_train=sample_data,
        prediction_column_name="prediction",
        label_column_name="label",
        df_val=None,  # Explicitly set to None
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        n_trials=2,
    )

    # Verify train_test_split was called
    mock_train_test_split.assert_called_once()
    call_args = mock_train_test_split.call_args
    assert call_args[0][0] is sample_data
    assert call_args[1]["test_size"] == 0.2
    assert call_args[1]["random_state"] == 42
    assert call_args[1]["stratify"] is sample_data["label"]

    # Verify that the model was fit with the training portion
    assert mock_mcboost_model.fit.call_count >= 1
    fit_calls = mock_mcboost_model.fit.call_args_list
    for call in fit_calls:
        assert call[1]["df_train"] is train_data

    # Verify that the model was evaluated on the validation portion
    assert mock_mcboost_model.predict.call_count >= 1
    predict_calls = mock_mcboost_model.predict.call_args_list
    for call in predict_calls:
        assert call.kwargs["df"] is val_data


def test_parameter_raises_when_no_segment_columns_specified(rng):
    n = 10
    predictions = rng.uniform(low=0.0, high=1.0, size=n)
    labels = scipy.stats.binom.rvs(1, predictions, size=n, random_state=rng)
    df = pd.DataFrame(
        {
            "prediction": predictions,
            "label": labels,
            "group": rng.choice(["A", "B"], size=n),
        }
    )

    max_num_rounds = 2
    n_parameter_samples = 2
    n_estimators = scipy.stats.randint(1, 3)
    n_estimators.random_state = rng
    max_depth = scipy.stats.randint(1, 3)
    max_depth.random_state = rng

    with pytest.raises(ValueError):
        tune_mcboost(
            train_df=df,
            scores_column="prediction",
            label_column="label",
            categorical_segment_columns=["group"],
            numerical_segment_columns=[],
            n_parameter_samples=n_parameter_samples,
            n_folds=2,
            max_num_rounds=max_num_rounds,
            parameter_space={
                "monotone_t": [True, False],
                "n_estimators": n_estimators,
                "max_depth": max_depth,
            },
        )


@pytest.mark.arm64_incompatible
def test_that_parameter_tuning_return_shape_correct(rng):
    n = 100
    predictions = rng.uniform(low=0.0, high=1.0, size=n)
    labels = scipy.stats.binom.rvs(1, predictions, size=n, random_state=rng)
    df = pd.DataFrame(
        {
            "prediction": predictions,
            "label": labels,
            "group": rng.choice(["A", "B"], size=n),
        }
    )

    max_num_rounds = 2
    n_parameter_samples = 2
    n_estimators = scipy.stats.randint(1, 3)
    n_estimators.random_state = rng
    max_depth = scipy.stats.randint(1, 3)
    max_depth.random_state = rng
    cv_results = tune_mcboost(
        train_df=df,
        scores_column="prediction",
        label_column="label",
        categorical_segment_columns=["group"],
        numerical_segment_columns=[],
        categorical_evaluation_segment_columns=["group"],
        n_parameter_samples=n_parameter_samples,
        n_folds=2,
        max_num_rounds=max_num_rounds,
        parameter_space={
            "early_stopping": [False],
            "monotone_t": [True, False],
            "n_estimators": n_estimators,
            "max_depth": max_depth,
        },
        random_state=rng,
    )

    # We're getting free additional "iterations" by taking all intermediate results
    # of round < max_num_rounds. Early stopping may cause len(cv_results) < max_num_rounds * n_parameter_samples
    assert len(cv_results) <= max_num_rounds * n_parameter_samples
    # At least there should be one round per parameter sample
    assert len(cv_results) >= n_parameter_samples


@pytest.mark.arm64_incompatible
def test_that_parameter_tuning_can_continue_from_previous_run(tmp_path, rng):
    n = 100
    predictions = rng.uniform(low=0.0, high=1.0, size=n)
    labels = scipy.stats.binom.rvs(1, predictions, size=n, random_state=rng)
    df = pd.DataFrame(
        {
            "prediction": predictions,
            "label": labels,
            "group": rng.choice(["A", "B"], size=n),
        }
    )

    n_parameter_samples = 2
    max_num_rounds = 2
    n_estimators = scipy.stats.randint(1, 3)
    n_estimators.random_state = rng
    max_depth = scipy.stats.randint(1, 3)
    max_depth.random_state = rng

    temp_file = tmp_path / "temp_file.pkl"
    cv_results_initial = tune_mcboost(
        train_df=df,
        scores_column="prediction",
        label_column="label",
        categorical_segment_columns=["group"],
        numerical_segment_columns=[],
        categorical_evaluation_segment_columns=["group"],
        n_parameter_samples=n_parameter_samples,
        continue_from_cache=False,
        cache_file=temp_file,
        n_folds=2,
        max_num_rounds=max_num_rounds,
        parameter_space={
            "early_stopping": [False],
            "monotone_t": [True, False],
            "n_estimators": n_estimators,
            "max_depth": max_depth,
        },
        random_state=rng,
    )

    # Note, this doesn't test if the first iteration was actually cached.
    cv_results_continued = tune_mcboost(
        train_df=df,
        scores_column="prediction",
        label_column="label",
        categorical_segment_columns=["group"],
        numerical_segment_columns=[],
        categorical_evaluation_segment_columns=["group"],
        n_parameter_samples=n_parameter_samples + 1,
        continue_from_cache=True,
        cache_file=temp_file,
        n_folds=2,
        max_num_rounds=max_num_rounds,
        parameter_space={
            "early_stopping": [False],
            "monotone_t": [True, False],
            "n_estimators": n_estimators,
            "max_depth": max_depth,
        },
        random_state=rng,
    )

    assert len(cv_results_initial) <= len(cv_results_continued)
