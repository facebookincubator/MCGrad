# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

import warnings
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from ax.api.configs import RangeParameterConfig
from sklearn.model_selection import train_test_split

from .. import methods, tuning as tuning_module
from ..tuning import (
    default_parameter_configurations,
    ORIGINAL_LIGHTGBM_PARAMS,
    tune_mcgrad_params,
)

TUNING_MODULE = tuning_module.__name__


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def sample_data(rng):
    n_samples = 50

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
def mock_mcgrad_model(rng):
    model = Mock(spec=methods.MCGrad)
    model.predict = Mock(return_value=rng.uniform(0.1, 0.9, 80))
    model.early_stopping_estimation_method = methods._EstimationMethod.HOLDOUT
    return model


@pytest.fixture
def hyperparams_for_tuning():
    default_hyperparams = methods.MCGrad().DEFAULT_HYPERPARAMS
    lightgbm_params = default_hyperparams["lightgbm_params"]
    return default_hyperparams, lightgbm_params


@pytest.mark.arm64_incompatible
@patch(f"{TUNING_MODULE}.normalized_entropy")
def test_tune_mcgrad_params_with_weights(
    mock_normalized_entropy,
    sample_data,
    mock_mcgrad_model,
):
    # Setup mocks
    mock_normalized_entropy.return_value = 0.5

    result_model, trial_results = tune_mcgrad_params(
        model=mock_mcgrad_model,
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
    assert mock_mcgrad_model.fit.call_count >= 1
    fit_calls = mock_mcgrad_model.fit.call_args_list
    for call in fit_calls:
        assert call[1]["weight_column_name"] == "weight"

    # Verify that normalized_entropy was called with sample_weight
    assert mock_normalized_entropy.call_count >= 1
    entropy_calls = mock_normalized_entropy.call_args_list
    for call in entropy_calls:
        assert "sample_weight" in call[1]
        assert call[1]["sample_weight"] is not None


@pytest.mark.arm64_incompatible
@patch(f"{TUNING_MODULE}.normalized_entropy")
def test_tune_mcgrad_params_without_weights(
    mock_normalized_entropy,
    sample_data,
    mock_mcgrad_model,
):
    mock_normalized_entropy.return_value = 0.5

    result_model, trial_results = tune_mcgrad_params(
        model=mock_mcgrad_model,
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
    assert mock_mcgrad_model.fit.call_count >= 1
    fit_calls = mock_mcgrad_model.fit.call_args_list
    for call in fit_calls:
        assert call[1]["weight_column_name"] is None

    # Verify that normalized_entropy was called with sample_weight=None
    assert mock_normalized_entropy.call_count >= 1
    entropy_calls = mock_normalized_entropy.call_args_list
    for call in entropy_calls:
        assert "sample_weight" in call[1]
        assert call[1]["sample_weight"] is None


@pytest.mark.arm64_incompatible
@patch(f"{TUNING_MODULE}.normalized_entropy")
def test_tune_mcgrad_params_default_parameters(
    mock_normalized_entropy,
    sample_data,
    mock_mcgrad_model,
):
    mock_normalized_entropy.return_value = 0.5

    result_model, trial_results = tune_mcgrad_params(
        model=mock_mcgrad_model,
        df_train=sample_data,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["cat_feature"],
        n_trials=2,
    )

    assert result_model is not None
    assert isinstance(trial_results, pd.DataFrame)

    # Verify that fit was called with correct values
    assert mock_mcgrad_model.fit.call_count >= 1
    fit_calls = mock_mcgrad_model.fit.call_args_list
    for call in fit_calls:
        assert call[1]["weight_column_name"] is None
        assert call[1]["categorical_feature_column_names"] == ["cat_feature"]
        assert call[1]["numerical_feature_column_names"] is None


@pytest.mark.arm64_incompatible
@patch(f"{TUNING_MODULE}.normalized_entropy")
def test_tune_mcgrad_params_ax_client_setup(
    mock_normalized_entropy,
    sample_data,
    mock_mcgrad_model,
):
    mock_normalized_entropy.return_value = 0.5

    result_model, trial_results = tune_mcgrad_params(
        model=mock_mcgrad_model,
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
    assert mock_mcgrad_model.fit.call_count >= 2


@pytest.mark.arm64_incompatible
@patch(f"{TUNING_MODULE}.normalized_entropy")
@patch(f"{TUNING_MODULE}.train_test_split")
def test_tune_mcgrad_params_data_splitting(
    mock_train_test_split,
    mock_normalized_entropy,
    rng,
    sample_data,
    mock_mcgrad_model,
):
    # Setup mock to return specific train/val splits using sklearn's splitter
    train_data, val_data = train_test_split(
        sample_data, test_size=0.2, random_state=rng
    )
    mock_train_test_split.return_value = (train_data, val_data)

    mock_normalized_entropy.return_value = 0.5

    tune_mcgrad_params(
        model=mock_mcgrad_model,
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
@patch(f"{TUNING_MODULE}.normalized_entropy")
def test_tune_mcgrad_params_with_subset_of_parameters(
    mock_normalized_entropy,
    sample_data,
):
    mock_normalized_entropy.return_value = 0.5
    model = methods.MCGrad()

    subset_params = ["learning_rate", "max_depth"]
    subset_configs = [
        config
        for config in default_parameter_configurations
        if config.name in subset_params
    ]

    _, trial_results = tune_mcgrad_params(
        model=model,
        df_train=sample_data,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        n_trials=2,
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


def test_mcgrad_and_lightgbm_default_hyperparams_are_within_bounds_for_tuning(
    hyperparams_for_tuning,
):
    _, lightgbm_params = hyperparams_for_tuning

    for config in default_parameter_configurations:
        param_name = config.name
        bounds = config.bounds
        if param_name in lightgbm_params:
            default_value = lightgbm_params[param_name]
            original_value = ORIGINAL_LIGHTGBM_PARAMS[param_name]
            assert bounds[0] <= default_value <= bounds[1], (
                f"Default {param_name} ({default_value}) is outside of bounds ({bounds})"
            )
            assert bounds[0] <= original_value <= bounds[1], (
                f"Original {param_name} ({original_value}) is outside of bounds ({bounds})"
            )
            # Check value type
            if config.parameter_type == "int":
                assert isinstance(default_value, int), (
                    f"Default {param_name} ({default_value}) should be an integer"
                )
            elif config.parameter_type == "float":
                assert isinstance(default_value, float), (
                    f"Default {param_name} ({default_value}) should be a float"
                )


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
    total_trials = 4

    # Suppress botorch/ax warnings about constant/non-standardized input data
    # These are expected with minimal synthetic test data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Data.*is not standardized")
        warnings.filterwarnings("ignore", message="An input array is constant")
        _, trial_results = tune_mcgrad_params(
            model=methods.MCGrad(num_rounds=0, early_stopping=False),
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
    # Ax API uses descriptive node names: "CenterOfSearchSpace", "Sobol", "MBM"
    # The generation strategy is: attached trial (defaults) -> CenterOfSearchSpace -> Sobol -> MBM
    # With initialization_budget = n_warmup_random_trials + 1:
    # - 1 attached trial (defaults, counted in initialization)
    # - 1 center of search space trial
    # - n_warmup_random_trials Sobol trials
    # - remaining trials are MBM (Bayesian optimization)
    sobol_count = value_counter.get("Sobol", 0)
    center_count = value_counter.get("CenterOfSearchSpace", 0)
    botorch_count = value_counter.get("MBM", 0)

    # initialization_budget = n_warmup_random_trials + 1 = 2 (for attached + generated init trials)
    # The attached trial is separate from the generated CenterOfSearchSpace + Sobol trials
    # So we expect: 1 attached + 1 center + 1 Sobol = 3 initialization trials, then 1 MBM
    expected_initialization = (
        n_warmup_random_trials + 1
    )  # center + Sobol (attached trial is separate)
    expected_botorch = (
        total_trials - expected_initialization - 1
    )  # -1 for attached trial
    assert len(trial_results) == total_trials, (
        f"Expected {total_trials} trials, got {len(trial_results)}."
    )
    assert sobol_count + center_count == expected_initialization, (
        f"Expected {expected_initialization} initialization trials (Sobol + Center), got {sobol_count + center_count}."
    )
    assert botorch_count == expected_botorch, (
        f"Expected {expected_botorch} BoTorch trials, got {botorch_count}."
    )


# Tests for RangeParameterConfig usage
def test_range_parameter_config_float_with_log_scale():
    config = RangeParameterConfig(
        name="learning_rate",
        bounds=(0.01, 0.3),
        parameter_type="float",
        scaling="log",
    )

    assert config.name == "learning_rate"
    assert config.bounds == (0.01, 0.3)
    assert config.parameter_type == "float"
    assert config.scaling == "log"


def test_range_parameter_config_int_without_log_scale():
    config = RangeParameterConfig(
        name="max_depth",
        bounds=(2.0, 15.0),
        parameter_type="int",
        scaling="linear",
    )

    assert config.name == "max_depth"
    assert config.bounds == (2.0, 15.0)
    assert config.parameter_type == "int"
    assert config.scaling == "linear"


def test_all_default_configs_are_range_parameter_configs():
    for config in default_parameter_configurations:
        assert isinstance(config, RangeParameterConfig)
        assert config.name is not None


@pytest.mark.arm64_incompatible
def test_non_default_parameters_preserved_when_not_in_tuning_configurations(
    sample_data,
):
    non_default_params = {
        "learning_rate": 0.05,
        "max_depth": 8,
        "lambda_l2": 5.0,
    }
    model = methods.MCGrad(lightgbm_params=non_default_params)

    for param, value in non_default_params.items():
        assert model.lightgbm_params[param] == value

    with patch(f"{TUNING_MODULE}.normalized_entropy", return_value=0.5):
        # Create parameter configurations that don't include our non-default parameters
        # This will tune only num_leaves and min_child_samples
        tune_params = ["num_leaves", "min_child_samples"]
        parameter_configs = [
            config
            for config in default_parameter_configurations
            if config.name in tune_params
        ]

        # Run tuning with these limited configurations
        result_model, _ = tune_mcgrad_params(
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


@pytest.mark.arm64_incompatible
@patch(f"{TUNING_MODULE}.normalized_entropy")
@patch(f"{TUNING_MODULE}.train_test_split")
def test_tune_mcgrad_params_with_explicit_validation_set(
    mock_train_test_split,
    mock_normalized_entropy,
    sample_data,
    sample_val_data,
    mock_mcgrad_model,
):
    mock_normalized_entropy.return_value = 0.5

    tune_mcgrad_params(
        model=mock_mcgrad_model,
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
    assert mock_mcgrad_model.fit.call_count >= 1
    fit_calls = mock_mcgrad_model.fit.call_args_list
    for call in fit_calls:
        assert call[1]["df_train"] is sample_data

    # Verify that the model was evaluated on the validation data
    assert mock_mcgrad_model.predict.call_count >= 1
    predict_calls = mock_mcgrad_model.predict.call_args_list
    for call in predict_calls:
        assert call.kwargs["df"] is sample_val_data


@pytest.mark.arm64_incompatible
@patch(f"{TUNING_MODULE}.normalized_entropy")
@patch(f"{TUNING_MODULE}.train_test_split")
def test_tune_mcgrad_params_fallback_to_train_test_split(
    mock_train_test_split,
    mock_normalized_entropy,
    rng,
    sample_data,
    mock_mcgrad_model,
):
    """Test that when df_val is None, train_test_split is used."""
    # Setup mock to return specific train/val splits using sklearn's splitter
    train_data, val_data = train_test_split(
        sample_data, test_size=0.2, random_state=rng
    )
    mock_train_test_split.return_value = (train_data, val_data)
    mock_normalized_entropy.return_value = 0.5

    tune_mcgrad_params(
        model=mock_mcgrad_model,
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
    pd.testing.assert_series_equal(call_args[1]["stratify"], sample_data["label"])

    # Verify that the model was fit with the training portion
    assert mock_mcgrad_model.fit.call_count >= 1
    fit_calls = mock_mcgrad_model.fit.call_args_list
    for call in fit_calls:
        assert call[1]["df_train"] is train_data

    # Verify that the model was evaluated on the validation portion
    assert mock_mcgrad_model.predict.call_count >= 1
    predict_calls = mock_mcgrad_model.predict.call_args_list
    for call in predict_calls:
        assert call.kwargs["df"] is val_data


@pytest.mark.arm64_incompatible
@patch(f"{TUNING_MODULE}.normalized_entropy")
def test_tune_mcgrad_params_pass_df_val_into_tuning_true(
    mock_normalized_entropy,
    sample_data,
    sample_val_data,
    mock_mcgrad_model,
):
    """Test that df_val is passed to model.fit during tuning when pass_df_val_into_tuning=True."""
    mock_normalized_entropy.return_value = 0.5

    tune_mcgrad_params(
        model=mock_mcgrad_model,
        df_train=sample_data,
        prediction_column_name="prediction",
        label_column_name="label",
        df_val=sample_val_data,
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        n_trials=2,
        pass_df_val_into_tuning=True,
        pass_df_val_into_final_fit=False,
    )

    # Get all fit calls
    fit_calls = mock_mcgrad_model.fit.call_args_list

    # All tuning fit calls (all except the last one) should have df_val=sample_val_data
    tuning_fit_calls = fit_calls[:-1]
    for call in tuning_fit_calls:
        assert call[1]["df_val"] is sample_val_data

    # The final fit call should have df_val=None (since pass_df_val_into_final_fit=False)
    final_fit_call = fit_calls[-1]
    assert final_fit_call[1]["df_val"] is None


@pytest.mark.arm64_incompatible
@patch(f"{TUNING_MODULE}.normalized_entropy")
def test_tune_mcgrad_params_pass_df_val_into_tuning_false(
    mock_normalized_entropy,
    sample_data,
    sample_val_data,
    mock_mcgrad_model,
):
    """Test that df_val is not passed to model.fit during tuning when pass_df_val_into_tuning=False."""
    mock_normalized_entropy.return_value = 0.5

    tune_mcgrad_params(
        model=mock_mcgrad_model,
        df_train=sample_data,
        prediction_column_name="prediction",
        label_column_name="label",
        df_val=sample_val_data,
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        n_trials=2,
        pass_df_val_into_tuning=False,
        pass_df_val_into_final_fit=False,
    )

    # All fit calls should have df_val=None
    fit_calls = mock_mcgrad_model.fit.call_args_list
    for call in fit_calls:
        assert call[1]["df_val"] is None


@pytest.mark.arm64_incompatible
@patch(f"{TUNING_MODULE}.normalized_entropy")
def test_tune_mcgrad_params_pass_df_val_into_final_fit_true(
    mock_normalized_entropy,
    sample_data,
    sample_val_data,
    mock_mcgrad_model,
):
    """Test that df_val is passed to model.fit during final fit when pass_df_val_into_final_fit=True."""
    mock_normalized_entropy.return_value = 0.5

    tune_mcgrad_params(
        model=mock_mcgrad_model,
        df_train=sample_data,
        prediction_column_name="prediction",
        label_column_name="label",
        df_val=sample_val_data,
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        n_trials=2,
        pass_df_val_into_tuning=False,
        pass_df_val_into_final_fit=True,
    )

    # Get all fit calls
    fit_calls = mock_mcgrad_model.fit.call_args_list

    # All tuning fit calls (all except the last one) should have df_val=None
    tuning_fit_calls = fit_calls[:-1]
    for call in tuning_fit_calls:
        assert call[1]["df_val"] is None

    # The final fit call should have df_val=sample_val_data
    final_fit_call = fit_calls[-1]
    assert final_fit_call[1]["df_val"] is sample_val_data


@pytest.mark.arm64_incompatible
@patch(f"{TUNING_MODULE}.normalized_entropy")
def test_tune_mcgrad_params_pass_df_val_into_final_fit_false(
    mock_normalized_entropy,
    sample_data,
    sample_val_data,
    mock_mcgrad_model,
):
    """Test that df_val is not passed to model.fit during final fit when pass_df_val_into_final_fit=False."""
    mock_normalized_entropy.return_value = 0.5

    tune_mcgrad_params(
        model=mock_mcgrad_model,
        df_train=sample_data,
        prediction_column_name="prediction",
        label_column_name="label",
        df_val=sample_val_data,
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        n_trials=2,
        pass_df_val_into_tuning=False,
        pass_df_val_into_final_fit=False,
    )

    # The final fit call should have df_val=None
    fit_calls = mock_mcgrad_model.fit.call_args_list
    final_fit_call = fit_calls[-1]
    assert final_fit_call[1]["df_val"] is None


@pytest.mark.arm64_incompatible
@patch(f"{TUNING_MODULE}.normalized_entropy")
def test_tune_mcgrad_params_pass_df_val_into_both_tuning_and_final_fit(
    mock_normalized_entropy,
    sample_data,
    sample_val_data,
    mock_mcgrad_model,
):
    """Test that df_val is passed to both tuning and final fit when both flags are True."""
    mock_normalized_entropy.return_value = 0.5

    tune_mcgrad_params(
        model=mock_mcgrad_model,
        df_train=sample_data,
        prediction_column_name="prediction",
        label_column_name="label",
        df_val=sample_val_data,
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        n_trials=2,
        pass_df_val_into_tuning=True,
        pass_df_val_into_final_fit=True,
    )

    # All fit calls should have df_val=sample_val_data
    fit_calls = mock_mcgrad_model.fit.call_args_list
    for call in fit_calls:
        assert call[1]["df_val"] is sample_val_data


@pytest.mark.arm64_incompatible
@pytest.mark.parametrize(
    "pass_df_val_into_tuning,pass_df_val_into_final_fit",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
@patch(f"{TUNING_MODULE}.normalized_entropy")
def test_tune_mcgrad_params_df_val_passing_defaults_to_false(
    mock_normalized_entropy,
    sample_data,
    sample_val_data,
    mock_mcgrad_model,
    pass_df_val_into_tuning,
    pass_df_val_into_final_fit,
):
    """Test all combinations of pass_df_val_into_tuning and pass_df_val_into_final_fit flags."""
    mock_normalized_entropy.return_value = 0.5

    tune_mcgrad_params(
        model=mock_mcgrad_model,
        df_train=sample_data,
        prediction_column_name="prediction",
        label_column_name="label",
        df_val=sample_val_data,
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        n_trials=2,
        pass_df_val_into_tuning=pass_df_val_into_tuning,
        pass_df_val_into_final_fit=pass_df_val_into_final_fit,
    )

    fit_calls = mock_mcgrad_model.fit.call_args_list

    # Verify tuning fit calls
    tuning_fit_calls = fit_calls[:-1]
    expected_tuning_df_val = sample_val_data if pass_df_val_into_tuning else None
    for call in tuning_fit_calls:
        assert call[1]["df_val"] is expected_tuning_df_val

    # Verify final fit call
    final_fit_call = fit_calls[-1]
    expected_final_df_val = sample_val_data if pass_df_val_into_final_fit else None
    assert final_fit_call[1]["df_val"] is expected_final_df_val


@pytest.mark.arm64_incompatible
@patch(f"{TUNING_MODULE}.normalized_entropy")
def test_tune_mcboost_params_does_not_modify_input_dataframes(
    mock_normalized_entropy,
    sample_data,
    sample_val_data,
    mock_mcgrad_model,
):
    mock_normalized_entropy.return_value = 0.5

    df_train_original = sample_data.copy()
    df_val_original = sample_val_data.copy()

    tune_mcgrad_params(
        model=mock_mcgrad_model,
        df_train=sample_data,
        prediction_column_name="prediction",
        label_column_name="label",
        df_val=sample_val_data,
        weight_column_name="weight",
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        n_trials=2,
    )

    pd.testing.assert_frame_equal(sample_data, df_train_original)
    pd.testing.assert_frame_equal(sample_val_data, df_val_original)


@pytest.mark.arm64_incompatible
@patch(f"{TUNING_MODULE}.normalized_entropy")
@patch(f"{TUNING_MODULE}.train_test_split")
def test_tune_mcboost_params_does_not_modify_input_dataframe_when_no_df_val(
    mock_train_test_split,
    mock_normalized_entropy,
    rng,
    sample_data,
    mock_mcgrad_model,
):
    train_data, val_data = train_test_split(
        sample_data, test_size=0.2, random_state=rng
    )
    mock_train_test_split.return_value = (train_data, val_data)
    mock_normalized_entropy.return_value = 0.5

    df_train_original = sample_data.copy()

    tune_mcgrad_params(
        model=mock_mcgrad_model,
        df_train=sample_data,
        prediction_column_name="prediction",
        label_column_name="label",
        df_val=None,
        weight_column_name="weight",
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        n_trials=2,
    )

    pd.testing.assert_frame_equal(sample_data, df_train_original)
