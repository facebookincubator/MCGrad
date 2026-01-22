# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict

"""
Hyperparameter tuning utilities for MCGrad models.

This module provides utilities for automatic hyperparameter optimization of
MCGrad calibrators using the Ax experimentation platform.
"""

import copy
import logging
import uuid
from contextlib import contextmanager
from typing import Any, Generator

import pandas as pd
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from sklearn.model_selection import train_test_split

from . import methods
from .metrics import normalized_entropy

logger: logging.Logger = logging.getLogger(__name__)


default_parameter_configurations: list[RangeParameterConfig] = [
    RangeParameterConfig(
        name="learning_rate",
        bounds=(0.002, 0.2),
        parameter_type="float",
        scaling="log",
    ),
    RangeParameterConfig(
        name="min_child_samples",
        bounds=(5.0, 201.0),
        parameter_type="int",
        scaling="linear",
    ),
    RangeParameterConfig(
        name="num_leaves",
        bounds=(2.0, 44.0),
        parameter_type="int",
        scaling="linear",
    ),
    RangeParameterConfig(
        name="n_estimators",
        bounds=(10.0, 500.0),
        parameter_type="int",
        scaling="linear",
    ),
    RangeParameterConfig(
        name="lambda_l2",
        bounds=(0.0, 100.0),
        parameter_type="float",
        scaling="linear",
    ),
    RangeParameterConfig(
        name="min_gain_to_split",
        bounds=(0.0, 0.2),
        parameter_type="float",
        scaling="linear",
    ),
    RangeParameterConfig(
        name="max_depth",
        bounds=(2.0, 15.0),
        parameter_type="int",
        scaling="log",
    ),
    RangeParameterConfig(
        name="min_sum_hessian_in_leaf",
        bounds=(1e-3, 1200.0),
        parameter_type="float",
        scaling="log",
    ),
]

# Default hyperparameters from the original LightGBM library.
# Reference: https://lightgbm.readthedocs.io/en/v4.5.0/Parameters.html
ORIGINAL_LIGHTGBM_PARAMS: dict[str, int | float] = {
    "learning_rate": 0.1,
    "min_child_samples": 20,
    "num_leaves": 31,
    "n_estimators": 100,
    "lambda_l2": 0.0,
    "min_gain_to_split": 0.0,
    # Original uses -1 (no limit) but that often leads to overfitting.
    "max_depth": 15,
    "min_sum_hessian_in_leaf": 1e-3,
}


@contextmanager
def _suppress_logger(logger: logging.Logger) -> Generator[None, None, None]:
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


def tune_mcgrad_params(
    model: methods.MCGrad,
    df_train: pd.DataFrame,
    prediction_column_name: str,
    label_column_name: str,
    df_val: pd.DataFrame | None = None,
    weight_column_name: str | None = None,
    categorical_feature_column_names: list[str] | None = None,
    numerical_feature_column_names: list[str] | None = None,
    n_trials: int = 20,
    n_warmup_random_trials: int | None = None,
    parameter_configurations: list[RangeParameterConfig] | None = None,
    pass_df_val_into_tuning: bool = False,
    pass_df_val_into_final_fit: bool = False,
) -> tuple[methods.MCGrad | None, pd.DataFrame]:
    """
    Tune the hyperparameters of an MCGrad model using Ax.

    :param model: The MCGrad model to be tuned. It could be a fitted model or an unfitted model.
    :param df_train: The training data: 80% of the data is used for training the model, and the remaining 20% is used for validation.
    :param prediction_column_name: The name of the prediction column in the data.
    :param label_column_name: The name of the label column in the data.
    :param df_val: The validation data. If None, 20% of the training data is used for validation.
    :param weight_column_name: The name of the weight column in the data. If None, all samples are treated equally.
    :param categorical_feature_column_names: The names of the categorical feature columns in the data.
    :param numerical_feature_column_names: The names of the numerical feature columns in the data.
    :param n_trials: The number of trials to run. Defaults to 20.
    :param n_warmup_random_trials: The number of random trials to run before starting the Ax optimization.
           Defaults to None, which uses calculate_num_initialization_trials to determine the number of warmup trials, which uses the following rules:
           (i) At least 16 (Twice the number of tunable parameters), (ii) At most 1/5th of num_trials.
    :param parameter_configurations: The list of parameter configurations to tune. If None, the default parameter configurations are used.
    :param pass_df_val_into_tuning: Whether to pass the validation data into the tuning process. If True, the validation data is passed into the tuning process.
    :param pass_df_val_into_final_fit: Whether to pass the validation data into the final fit. If True, the validation data is passed into the final fit.

    :returns: A tuple containing:
        - The fitted MCGrad model with the best hyperparameters found during tuning.
        - A DataFrame containing the results of all trials, sorted by normalized entropy.
    """

    if df_val is None:
        df_train, df_val = train_test_split(
            df_train,
            test_size=0.2,
            random_state=42,
            stratify=df_train[label_column_name],
        )
    if df_val is None:
        raise ValueError(
            "df_val must be provided or train_test_split must produce a validation set"
        )

    if (
        model.early_stopping_estimation_method
        == methods._EstimationMethod.CROSS_VALIDATION
        and (pass_df_val_into_tuning or pass_df_val_into_final_fit)
    ):
        raise ValueError(
            "Early stopping with cross validation is not supported when passing validation data into tuning or final fit."
        )

    df_param_val: pd.DataFrame | None = None
    if pass_df_val_into_tuning:
        logger.info(
            f"Passing validation data with {len(df_val)} into fit during tuning process"
        )
        df_param_val = df_val

    if parameter_configurations is None:
        parameter_configurations = default_parameter_configurations

    model = copy.copy(model)

    def _train_evaluate(parameterization: dict[str, Any]) -> float:
        # suppressing logger to avoid expected warning about setting lightgbm params on a (potentially) fitted model
        with _suppress_logger(logger):
            model._set_lightgbm_params(parameterization)
        model.fit(
            df_train=df_train,
            prediction_column_name=prediction_column_name,
            label_column_name=label_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
            weight_column_name=weight_column_name,
            df_val=df_param_val,
        )

        prediction = model.predict(
            # pyre-ignore[6] we assert above that df_val is not None
            df=df_val,
            prediction_column_name=prediction_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
        )
        # pyre-ignore[16] we assert above that df_val is not None
        sample_weight = df_val[weight_column_name] if weight_column_name else None
        return normalized_entropy(
            labels=df_val[label_column_name],
            predicted_scores=prediction,
            sample_weight=sample_weight,
        )

    ax_client = Client()

    ax_client.configure_experiment(
        name=f"lightgbm_autotuning_{uuid.uuid4().hex[:8]}",
        parameters=list(parameter_configurations),
    )

    ax_client.configure_optimization(objective="normalized_entropy")

    # Configure generation strategy with initialization budget
    # -1 is because we add an initial trial with default parameters
    # +1 to account for the manually added trial with default parameters.
    initialization_budget = (
        n_warmup_random_trials + 1 if n_warmup_random_trials is not None else None
    )
    ax_client.configure_generation_strategy(
        initialization_budget=initialization_budget,
    )

    # Construct a set of parameters for the first trial which contains the defaults for every parameter that is tuned.
    # If a default is not available use the LightGBM default
    initial_trial_parameters: dict[str, float | int] = {}
    mcgrad_defaults = methods.MCGrad.DEFAULT_HYPERPARAMS["lightgbm_params"]
    for config in parameter_configurations:
        if config.name in mcgrad_defaults:
            initial_trial_parameters[config.name] = mcgrad_defaults[config.name]
        else:
            initial_trial_parameters[config.name] = ORIGINAL_LIGHTGBM_PARAMS[
                config.name
            ]

    logger.info(
        f"Adding initial configuration from defaults to trials: {initial_trial_parameters}"
    )

    with _suppress_logger(methods.logger):
        # Attach and complete the initial trial with default hyperparameters.
        # Note that we're only using the defaults for the parameters that are being tuned.
        # That is, this configuration does not necessarily correspond to the out-of-the-box defaults.
        initial_trial_index = ax_client.attach_trial(
            parameters=initial_trial_parameters
        )
        initial_score = _train_evaluate(initial_trial_parameters)
        ax_client.complete_trial(
            trial_index=initial_trial_index,
            raw_data={"normalized_entropy": initial_score},
        )
        logger.info(f"Initial trial completed with score: {initial_score}")

        for _ in range(n_trials - 1):
            # get_next_trials returns dict[int, TParameterization]
            trials = ax_client.get_next_trials(max_trials=1)
            for trial_index, parameters in trials.items():
                score = _train_evaluate(dict(parameters))
                ax_client.complete_trial(
                    trial_index=trial_index,
                    raw_data={"normalized_entropy": score},
                )

    # Get trial results using summarize()
    trial_results = ax_client.summarize().sort_values("normalized_entropy")

    # get_best_parameterization returns (params, outcome, trial_idx, arm_name)
    best_params, _, _, _ = ax_client.get_best_parameterization()

    logger.info(f"Best parameters: {best_params}")
    logger.info("Fitting model with best parameters")

    with _suppress_logger(methods.logger):
        model._set_lightgbm_params(dict(best_params))

    df_final_val: pd.DataFrame | None = None
    if pass_df_val_into_final_fit:
        logger.info(f"Passing validation data with {len(df_val)} into final fit")
        df_final_val = df_val

    model.fit(
        df_train=df_train,
        prediction_column_name=prediction_column_name,
        label_column_name=label_column_name,
        categorical_feature_column_names=categorical_feature_column_names,
        numerical_feature_column_names=numerical_feature_column_names,
        weight_column_name=weight_column_name,
        df_val=df_final_val,
    )

    return model, trial_results


# @oss-disable: # Alias for backward compatibility and internal use.
# @oss-disable[end= ]: tune_mcboost_params = tune_mcgrad_params
