# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict

import copy
import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator, List

import pandas as pd

from ax.service.ax_client import AxClient, ObjectiveProperties
from multicalibration import methods
from multicalibration.metrics import normalized_entropy
from sklearn.model_selection import train_test_split

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class ParameterConfig:
    name: str
    bounds: List[float | int]
    value_type: str
    log_scale: bool
    config_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bounds": self.bounds,
            "value_type": self.value_type,
            "log_scale": self.log_scale,
            "type": self.config_type,
        }


default_parameter_configurations: list[ParameterConfig] = [
    ParameterConfig(
        name="learning_rate",
        bounds=[0.002, 0.2],
        value_type="float",
        log_scale=True,
        config_type="range",
    ),
    ParameterConfig(
        name="min_child_samples",
        bounds=[5, 201],
        value_type="int",
        log_scale=False,
        config_type="range",
    ),
    ParameterConfig(
        name="num_leaves",
        bounds=[2, 44],
        value_type="int",
        log_scale=False,
        config_type="range",
    ),
    ParameterConfig(
        name="n_estimators",
        bounds=[10, 500],
        value_type="int",
        log_scale=False,
        config_type="range",
    ),
    ParameterConfig(
        name="lambda_l2",
        bounds=[0.0, 100.0],
        value_type="float",
        log_scale=False,
        config_type="range",
    ),
    ParameterConfig(
        name="min_gain_to_split",
        bounds=[0.0, 0.2],
        value_type="float",
        log_scale=False,
        config_type="range",
    ),
    ParameterConfig(
        name="max_depth",
        bounds=[2, 15],
        value_type="int",
        log_scale=True,
        config_type="range",
    ),
    ParameterConfig(
        name="min_sum_hessian_in_leaf",
        bounds=[1e-3, 1200],
        value_type="float",
        log_scale=True,
        config_type="range",
    ),
]

# These are the default hyperparameters used in the original lightgbm library (https://lightgbm.readthedocs.io/en/latest/Parameters.html)
ORIGINAL_LIGHTGBM_PARAMS: dict[str, int | float] = {
    "learning_rate": 0.1,
    "min_child_samples": 20,
    "num_leaves": 31,
    "n_estimators": 100,
    "lambda_l2": 0.0,
    "min_gain_to_split": 0.0,
    "max_depth": 15,  # the original paper uses -1, which means no limit, but it usually leads to overfitting. We set it to a high value.
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


def tune_mcboost_params(
    model: methods.MCBoost,
    df_train: pd.DataFrame,
    prediction_column_name: str,
    label_column_name: str,
    df_val: pd.DataFrame | None = None,
    weight_column_name: str | None = None,
    categorical_feature_column_names: list[str] | None = None,
    numerical_feature_column_names: list[str] | None = None,
    n_trials: int = 20,
    n_warmup_random_trials: int | None = None,
    parameter_configurations: list[ParameterConfig] | None = None,
    pass_df_val_into_tuning: bool = False,
    pass_df_val_into_final_fit: bool = False,
) -> tuple[methods.MCBoost | None, pd.DataFrame]:
    # Make a description for this function
    """
    Tune the hyperparameters of an MCBoost model using Ax.

    :param model: The MCBoost model to be tuned. It could be a fitted model or an unfitted model.
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
    """

    if df_val is None:
        df_train, df_val = train_test_split(
            df_train,
            test_size=0.2,
            random_state=42,
            stratify=df_train[label_column_name],
        )
    assert df_val is not None

    if (
        model.EARLY_STOPPING_ESTIMATION_METHOD
        == methods.EstimationMethod.CROSS_VALIDATION
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
            label_column_name=label_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
        )
        # pyre-ignore[16] we assert above that df_val is not None
        sample_weight = df_val[weight_column_name] if weight_column_name else None
        return normalized_entropy(
            labels=df_val[label_column_name],
            predictions=prediction,
            sample_weight=sample_weight,
        )

    ax_client = AxClient()
    ax_client.create_experiment(
        name="mcboost_lightgbm_autotuning" + str(uuid.uuid4())[:8],
        parameters=[config.to_dict() for config in parameter_configurations],
        objectives={"normalized_entropy": ObjectiveProperties(minimize=True)},
        # If num_initialization_trials is None, the number of warm starting trials is automatically determined
        choose_generation_strategy_kwargs={
            "num_trials": n_trials
            - 1,  # -1 is because we add an initial trial with default parameters
            # +1 to account for the manually added trial with default parameters.
            "num_initialization_trials": n_warmup_random_trials + 1
            if n_warmup_random_trials is not None
            else None,
        },
    )

    # Construct a set of parameters for the first trial which contains the MCBoost defaults for every parameter that is tuned. If a default is not available
    # use the Lightgbm default
    initial_trial_parameters = {}
    mcboost_defaults = methods.MCBoost.DEFAULT_HYPERPARAMS["lightgbm_params"]
    for config in parameter_configurations:
        if config.name in mcboost_defaults:
            initial_trial_parameters[config.name] = mcboost_defaults[config.name]
        else:
            initial_trial_parameters[config.name] = ORIGINAL_LIGHTGBM_PARAMS[
                config.name
            ]

    logger.info(
        f"Adding initial configuration from MCBoost defaults to trials: {initial_trial_parameters}"
    )

    with _suppress_logger(methods.logger):
        # Attach and complete the initial trial with default hyperparameters. Note that we're only using the defaults for the parameters that are being tuned.
        # That is, this configuration does not necessarily correspond to the out-of-the-box defaults for MCBoost.
        _, initial_trial_index = ax_client.attach_trial(
            parameters=initial_trial_parameters
        )
        initial_score = _train_evaluate(initial_trial_parameters)
        ax_client.complete_trial(
            trial_index=initial_trial_index, raw_data=initial_score
        )
        logger.info(f"Initial trial completed with score: {initial_score}")

        for _ in range(n_trials - 1):
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(
                trial_index=trial_index, raw_data=_train_evaluate(parameters)
            )

    trial_results = ax_client.get_trials_data_frame().sort_values("normalized_entropy")
    best_params = ax_client.get_best_parameters()
    if best_params is not None:
        best_params = best_params[0]

    logger.info(f"Best parameters: {best_params}")
    logger.info("Fitting MCBoost model with best parameters")

    with _suppress_logger(methods.logger):
        model._set_lightgbm_params(best_params)

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
