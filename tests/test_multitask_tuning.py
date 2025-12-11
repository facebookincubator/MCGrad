# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

import numpy as np
import pandas as pd
import pytest

from multicalibration.multitask import methods


@pytest.fixture
def multitask_tuning_data():
    """Create data suitable for multitask tuning tests."""
    np.random.seed(42)
    n_samples_per_task = 50
    tasks = ["TASK_A", "TASK_B"]

    train_data = []
    val_data = []

    for task in tasks:
        for _ in range(n_samples_per_task):
            train_data.append(
                {
                    "prediction": np.random.uniform(0.1, 0.9),
                    "label": np.random.binomial(1, 0.3),
                    "task": task,
                    "cat_feature": np.random.choice(["X", "Y", "Z"]),
                    "num_feature": np.random.normal(0, 1),
                }
            )
            val_data.append(
                {
                    "prediction": np.random.uniform(0.1, 0.9),
                    "label": np.random.binomial(1, 0.3),
                    "task": task,
                    "cat_feature": np.random.choice(["X", "Y", "Z"]),
                    "num_feature": np.random.normal(0, 1),
                }
            )

    return {
        "train": pd.DataFrame(train_data),
        "val": pd.DataFrame(val_data),
    }


@pytest.mark.arm64_incompatible
def test_tune_mcboost_params_with_df_val_and_pass_into_tuning(multitask_tuning_data):
    """Test tuning with pass_df_val_into_tuning=True, df_val provided, early_stopping_use_crossvalidation=False."""
    from multicalibration.multitask.tuning import tune_mcboost_params

    model = methods.MultitaskMCBoost(
        num_rounds=2,
        early_stopping=False,
        lightgbm_params={"max_depth": 2, "n_estimators": 2},
    )

    tuned_model, trial_results = tune_mcboost_params(
        model=model,
        df_train=multitask_tuning_data["train"],
        prediction_column_name="prediction",
        label_column_name="label",
        task_column_name="task",
        df_val=multitask_tuning_data["val"],
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        n_trials=2,
        pass_df_val_into_tuning=True,
        pass_df_val_into_final_fit=False,
    )

    assert tuned_model is not None
    assert isinstance(trial_results, dict)
    assert len(trial_results) == 2
    assert "TASK_A" in trial_results
    assert "TASK_B" in trial_results
    for _task_key, task_trial_df in trial_results.items():
        assert isinstance(task_trial_df, pd.DataFrame)
        assert len(task_trial_df) == 2


@pytest.mark.arm64_incompatible
def test_tune_mcboost_params_with_df_val_and_pass_into_final_fit(multitask_tuning_data):
    """Test tuning with pass_df_val_into_final_fit=True, df_val provided, early_stopping_use_crossvalidation=False."""
    from multicalibration.multitask.tuning import tune_mcboost_params

    model = methods.MultitaskMCBoost(
        num_rounds=2,
        early_stopping=False,
        lightgbm_params={"max_depth": 2, "n_estimators": 2},
    )

    tuned_model, trial_results = tune_mcboost_params(
        model=model,
        df_train=multitask_tuning_data["train"],
        prediction_column_name="prediction",
        label_column_name="label",
        task_column_name="task",
        df_val=multitask_tuning_data["val"],
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        n_trials=2,
        pass_df_val_into_tuning=False,
        pass_df_val_into_final_fit=True,
    )

    assert tuned_model is not None
    assert isinstance(trial_results, dict)
    assert len(trial_results) == 2
    for _task_key, task_trial_df in trial_results.items():
        assert isinstance(task_trial_df, pd.DataFrame)


@pytest.mark.arm64_incompatible
def test_tune_mcboost_params_with_both_pass_df_val_flags_true(multitask_tuning_data):
    """Test tuning with both pass_df_val_into_tuning=True and pass_df_val_into_final_fit=True."""
    from multicalibration.multitask.tuning import tune_mcboost_params

    model = methods.MultitaskMCBoost(
        num_rounds=2,
        early_stopping=False,
        lightgbm_params={"max_depth": 2, "n_estimators": 2},
    )

    tuned_model, trial_results = tune_mcboost_params(
        model=model,
        df_train=multitask_tuning_data["train"],
        prediction_column_name="prediction",
        label_column_name="label",
        task_column_name="task",
        df_val=multitask_tuning_data["val"],
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        n_trials=2,
        pass_df_val_into_tuning=True,
        pass_df_val_into_final_fit=True,
    )

    assert tuned_model is not None
    assert isinstance(trial_results, dict)
    assert len(trial_results) == 2

    assert len(tuned_model.mcboost_models) == 2
    assert "TASK_A" in tuned_model.mcboost_models
    assert "TASK_B" in tuned_model.mcboost_models


@pytest.mark.arm64_incompatible
def test_tune_mcboost_params_tuned_model_can_predict(multitask_tuning_data):
    """Test that the tuned model can make predictions."""
    from multicalibration.multitask.tuning import tune_mcboost_params

    model = methods.MultitaskMCBoost(
        num_rounds=2,
        early_stopping=False,
        lightgbm_params={"max_depth": 2, "n_estimators": 2},
    )

    tuned_model, _ = tune_mcboost_params(
        model=model,
        df_train=multitask_tuning_data["train"],
        prediction_column_name="prediction",
        label_column_name="label",
        task_column_name="task",
        df_val=multitask_tuning_data["val"],
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        n_trials=2,
        pass_df_val_into_tuning=True,
        pass_df_val_into_final_fit=True,
    )

    assert tuned_model is not None

    predictions = tuned_model.predict(
        df=multitask_tuning_data["val"],
        prediction_column_name="prediction",
        task_column_name="task",
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
    )

    assert predictions is not None
    assert len(predictions) == len(multitask_tuning_data["val"])
    assert all(0 <= p <= 1 for p in predictions)


@pytest.mark.arm64_incompatible
def test_tune_mcboost_params_with_early_stopping_no_crossvalidation(
    multitask_tuning_data,
):
    """Test tuning with early_stopping=True, early_stopping_use_crossvalidation=False, and df_val provided.

    This tests the scenario where a customer wants to use early stopping with holdout-based
    validation instead of cross-validation, which is the preferred approach when a separate
    validation set is available.
    """
    from multicalibration.multitask.tuning import tune_mcboost_params

    model = methods.MultitaskMCBoost(
        num_rounds=5,
        early_stopping=True,
        early_stopping_use_crossvalidation=False,
        patience=2,
        lightgbm_params={"max_depth": 2, "n_estimators": 2},
    )

    tuned_model, trial_results = tune_mcboost_params(
        model=model,
        df_train=multitask_tuning_data["train"],
        prediction_column_name="prediction",
        label_column_name="label",
        task_column_name="task",
        df_val=multitask_tuning_data["val"],
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        n_trials=2,
        pass_df_val_into_tuning=True,
        pass_df_val_into_final_fit=True,
    )

    assert tuned_model is not None
    assert isinstance(trial_results, dict)
    assert len(trial_results) == 2
    assert "TASK_A" in trial_results
    assert "TASK_B" in trial_results

    # Verify the tuned models have the correct early stopping settings
    assert len(tuned_model.mcboost_models) == 2
    for task_name, task_model in tuned_model.mcboost_models.items():
        assert (
            task_model.EARLY_STOPPING is True
        ), f"Task {task_name}: early_stopping should be True"
        assert task_model.PATIENCE == 2, f"Task {task_name}: patience should be 2"

    # Verify the tuned model can make predictions
    predictions = tuned_model.predict(
        df=multitask_tuning_data["val"],
        prediction_column_name="prediction",
        task_column_name="task",
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
    )

    assert predictions is not None
    assert len(predictions) == len(multitask_tuning_data["val"])
    assert all(0 <= p <= 1 for p in predictions)
