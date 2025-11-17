# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

import copy
import logging
import os
import pickle
import sys
import time
import uuid
import warnings
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike
from typing import Any, Generator, List, Protocol

import numpy as np
import pandas as pd
import scipy
from ax.service.ax_client import AxClient, ObjectiveProperties
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold, ParameterSampler, train_test_split

from multicalibration import methods
from multicalibration.metrics import (
    MulticalibrationError,
    kuiper_calibration,
    kuiper_test,
    multi_segment_inverse_sqrt_normalized_statistic_max,
    multi_segment_kuiper_test,
    multi_segment_pvalue_geometric_mean,
    multicalibration_error,
    normalized_entropy,
)

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated

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
    """

    if df_val is None:
        df_train, df_val = train_test_split(
            df_train,
            test_size=0.2,
            random_state=42,
            stratify=df_train[label_column_name],
        )
    assert df_val is not None

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
            "num_initialization_trials": (
                n_warmup_random_trials + 1
                if n_warmup_random_trials is not None
                else None
            ),
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
    logger.info(f"Fitting MCBoost model with best parameters")

    with _suppress_logger(methods.logger):
        model._set_lightgbm_params(best_params)

    model.fit(
        df_train=df_train,
        prediction_column_name=prediction_column_name,
        label_column_name=label_column_name,
        categorical_feature_column_names=categorical_feature_column_names,
        numerical_feature_column_names=numerical_feature_column_names,
        weight_column_name=weight_column_name,
    )

    return model, trial_results


# Don't use. Kept for backward compatibility
HYPERPARAM_RANGES: dict[str, list[float] | list[int]] = {
    config.name: config.bounds for config in default_parameter_configurations
}


def sample_max_depth() -> np.ndarray:  # pyre-ignore[24]
    return np.random.choice(
        list(
            range(
                int(HYPERPARAM_RANGES["max_depth"][0]),
                int(HYPERPARAM_RANGES["max_depth"][1] + 1),
            )
        )
    )


class MaxDepthDistribution(scipy.stats.rv_discrete):
    def rvs(self, *args: Any, **kwargs: Any) -> np.ndarray:  # pyre-ignore[24]
        return sample_max_depth()


DEFAULT_PARAMETER_SPACE: dict[
    str,
    (list[bool] | scipy.stats.rv_continuous | scipy.stats.rv_discrete),
] = {
    "monotone_t": [False],
    "learning_rate": scipy.stats.uniform(*HYPERPARAM_RANGES["learning_rate"]),
    "min_child_samples": scipy.stats.randint(*HYPERPARAM_RANGES["min_child_samples"]),
    "num_leaves": scipy.stats.randint(*HYPERPARAM_RANGES["num_leaves"]),
    "max_depth": MaxDepthDistribution(),
    "n_estimators": scipy.stats.randint(*HYPERPARAM_RANGES["n_estimators"]),
    "lambda_l2": scipy.stats.uniform(*HYPERPARAM_RANGES["lambda_l2"]),
    "min_gain_to_split": scipy.stats.uniform(*HYPERPARAM_RANGES["min_gain_to_split"]),
    "min_sum_hessian_in_leaf": scipy.stats.loguniform(
        *HYPERPARAM_RANGES["min_sum_hessian_in_leaf"]
    ),
}


def _kuiper_statistic_sd_normalized(
    df: pd.DataFrame,
    label_column: str,
    score_column: str,
    weight_column: str | None = None,
    categorical_segment_columns: list[str] | None = None,
    numerical_segment_columns: list[str] | None = None,
) -> float:
    # kuiper_calibration returns a float if segments is None
    normalized_kuiper_stats = kuiper_calibration(
        labels=df[label_column].values,
        predicted_scores=df[score_column].values,
        sample_weight=df[weight_column].values if weight_column else None,
        normalization_method="kuiper_standard_deviation",
        segments=None,
    )
    assert isinstance(normalized_kuiper_stats, float)
    return normalized_kuiper_stats


def _kuiper_p_value_metric(
    df: pd.DataFrame,
    label_column: str,
    score_column: str,
    weight_column: str | None = None,
    categorical_segment_columns: list[str] | None = None,
    numerical_segment_columns: list[str] | None = None,
) -> float:
    return kuiper_test(
        labels=df[label_column].values,
        predicted_scores=df[score_column].values,
        sample_weight=df[weight_column].values if weight_column else None,
    )[1]


def _make_segments_df(
    df: pd.DataFrame,
    categorical_segment_columns: list[str] | None = None,
    numerical_segment_columns: list[str] | None = None,
) -> pd.DataFrame:
    assert categorical_segment_columns or numerical_segment_columns
    categorical_segment_columns = categorical_segment_columns or []
    numerical_segment_columns = numerical_segment_columns or []
    segments_df = df[categorical_segment_columns + numerical_segment_columns]
    return segments_df


@deprecated("Use _mce_metric instead")
def _multi_kuiper_statistic_metric(
    df: pd.DataFrame,
    label_column: str,
    score_column: str,
    weight_column: str | None = None,
    categorical_segment_columns: list[str] | None = None,
    numerical_segment_columns: list[str] | None = None,
) -> float:
    segments_df = _make_segments_df(
        df=df,
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
    )

    return multi_segment_kuiper_test(
        predictions=df[score_column].values,
        labels=df[label_column].values,
        segments_df=segments_df,
        sample_weight=df[weight_column].values if weight_column else None,
        combination_method="fisher",
        min_segment_size=10,
    )["statistic"]


def _prauc_metric(
    df: pd.DataFrame,
    label_column: str,
    score_column: str,
    weight_column: str | None = None,
    categorical_segment_columns: list[str] | None = None,
    numerical_segment_columns: list[str] | None = None,
) -> float:
    return average_precision_score(
        y_true=df[label_column].values,
        y_score=df[score_column].values,
        sample_weight=df[weight_column].values if weight_column else None,
    )


def _multi_ace_metric(
    df: pd.DataFrame,
    label_column: str,
    score_column: str,
    weight_column: str | None = None,
    categorical_segment_columns: list[str] | None = None,
    numerical_segment_columns: list[str] | None = None,
) -> float:
    segments_df = _make_segments_df(
        df=df,
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
    )

    return multicalibration_error(
        predictions=df[score_column].values,
        labels=df[label_column].values,
        segments_df=segments_df,
        sample_weight=df[weight_column].values if weight_column else None,
    )


def _multi_geom_mean_pval(
    df: pd.DataFrame,
    label_column: str,
    score_column: str,
    weight_column: str | None = None,
    categorical_segment_columns: list[str] | None = None,
    numerical_segment_columns: list[str] | None = None,
) -> float:
    segments_df = _make_segments_df(
        df=df,
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
    )
    return multi_segment_pvalue_geometric_mean(
        predictions=df[score_column].values,
        labels=df[label_column].values,
        segments_df=segments_df,
        sample_weight=df[weight_column].values if weight_column else None,
    )


def _multi_inv_sqrt_norm_stat_max(
    df: pd.DataFrame,
    label_column: str,
    score_column: str,
    weight_column: str | None = None,
    categorical_segment_columns: list[str] | None = None,
    numerical_segment_columns: list[str] | None = None,
) -> float:
    segments_df = _make_segments_df(
        df=df,
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
    )
    return multi_segment_inverse_sqrt_normalized_statistic_max(
        predictions=df[score_column].values,
        labels=df[label_column].values,
        segments_df=segments_df,
        sample_weight=df[weight_column].values if weight_column else None,
    )


def _mce_metric(
    df: pd.DataFrame,
    label_column: str,
    score_column: str,
    weight_column: str | None = None,
    categorical_segment_columns: list[str] | None = None,
    numerical_segment_columns: list[str] | None = None,
    **kwargs: Any,
) -> float:
    metric = MulticalibrationError(
        df=df,
        label_column=label_column,
        score_column=score_column,
        weight_column=weight_column,
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
        # setting explicit defaults here to make sure tuning runs are comparable
        # even if defaults are changed in the future
        max_depth=3 if "max_depth" not in kwargs else kwargs["max_depth"],
        max_values_per_segment_feature=(
            3
            if "max_values_per_segment_feature" not in kwargs
            else kwargs["max_values_per_segment_feature"]
        ),
        min_samples_per_segment=(
            10
            if "min_samples_per_segment" not in kwargs
            else kwargs["min_samples_per_segment"]
        ),
        sigma_estimation_method=(
            "kuiper_standard_deviation"
            if "sigma_estimation_method" not in kwargs
            else kwargs["sigma_estimation_method"]
        ),
        # Setting this very low to make sure CV isn't slowed down by this metric
        max_n_segments=(
            100 if "max_n_segments" not in kwargs else kwargs["max_n_segments"]
        ),
    )
    return metric.mce_absolute


class CVMetricInterface(Protocol):
    def __call__(
        self,
        df: pd.DataFrame,
        label_column: str,
        score_column: str,
        weight_column: str | None = None,
        categorical_segment_columns: list[str] | None = None,
        numerical_segment_columns: list[str] | None = None,
    ) -> float: ...


DEFAULT_CV_METRICS: dict[str, CVMetricInterface] = {
    "kuiper_statistic_sd_normalized": _kuiper_statistic_sd_normalized,
    "kuiper_p_value": _kuiper_p_value_metric,
    "prauc": _prauc_metric,
    "multi_kuiper_metric": _mce_metric,
    "multi_ace": _multi_ace_metric,
}


@deprecated("Use methods.MCBoost.tune() instead")
def tune_mcboost(
    train_df: pd.DataFrame,
    scores_column: str,
    label_column: str,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    categorical_evaluation_segment_columns: list[str] | None = None,
    numerical_evaluation_segment_columns: list[str] | None = None,
    weight_column: str | None = None,
    n_parameter_samples: int = 100,
    n_folds: int = 10,
    parameter_space: None | (
        dict[
            str,
            (Iterable[int] | scipy.stats.rv_continuous | scipy.stats.rv_discrete),
        ]
    ) = None,
    max_num_rounds: int = 100,
    metrics: dict[str, CVMetricInterface] | None = None,
    cache_file: PathLike[bytes] | PathLike[str] | None = None,
    continue_from_cache: bool = False,
    baselines: list[type[methods.BaseCalibrator]] | None = None,
    metafeatures: None | (
        dict[str, Callable[[pd.DataFrame], str | float | int]]
    ) = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Tune the MCBoost model using randomized search over a specified parameter space with cross-validation.

    :param train_df: The dataframe containing the training data.
    :param scores_column: Name of the column in `train_df` that contains the scores.
    :param label_column: Name of the column in `train_df` that contains the labels.
    :param categorical_segment_columns: List of column names in `train_df` that are categorical features.
    :param numerical_segment_columns: List of column names in `train_df` that are numerical features.
    :param categorical_evaluation_segment_columns: List of column names to be used for evaluation. If None, uses `categorical_segment_columns` by default.
    :param numerical_evaluation_segment_columns: List of column names to be used for evaluation. If None, uses `numerical_segment_columns` by default.
    :param weight_column: Name of the column in `train_df` that contains the weights. If None, all instances are considered to have equal weight.
    :param n_parameter_samples: Number of samples from the hyperparameter space.
    :param n_folds: Number of folds for the K-Fold cross-validation.
    :param parameter_space: The parameter space to explore during tuning. If None, a default parameter space is used.
    :param max_num_rounds: Maximum number of MCBoost rounds to fit in each CV iteration. Results for each n_round between 0 and the number of rounds from
        MCBoost's early stopping are stored, with a maximum of max_num_rounds.
    :param metrics: Dictionary of additional metric functions to be added to the set of default metrics (`DEFAULT_CV_METRICS`).
    :param cache_file: Path to a file used to cache intermediate results. Useful for resuming the tuning.
    :param continue_from_cache: If True, continue tuning from cached results.
    :param metafeatures: Context features to extract from the training data. If None, no context features are used.
    :param random_state: Seed for the random number generator for reproducibility.

    :return: A DataFrame containing the cross-validation results, including statistics and parameter sets.

    Notes:
    -----
    The function performs a randomized search over the specified `parameter_space` using cross-validation.
    Results from each iteration are optionally cached to allow resuming. The final output is a DataFrame
    summarizing the performance across all iterations and parameter sets.
    """

    categorical_evaluation_segment_columns = (
        categorical_evaluation_segment_columns or []
    )
    numerical_evaluation_segment_columns = numerical_evaluation_segment_columns or []
    if (
        len(categorical_evaluation_segment_columns) == 0
        and len(numerical_evaluation_segment_columns) == 0
    ):
        raise ValueError("At least one evaluation segment column must be specified.")

    metrics = metrics or DEFAULT_CV_METRICS

    if baselines is None:
        baselines = []
    if metafeatures is None:
        metafeatures = {}

    if continue_from_cache and cache_file is None:
        raise ValueError("cache_file must be specified to continue from cache.")

    if (
        not continue_from_cache
        and cache_file is not None
        and os.path.exists(cache_file)
    ):
        raise ValueError(
            "Error: cache_file is specified but continue_from_cache is False. "
            "Execution is aborted to prevent the cache file from getting overwritten. "
        )

    splitter, sampler = _set_up_cv(
        parameter_space=parameter_space,
        n_parameter_samples=n_parameter_samples,
        n_folds=n_folds,
        random_state=random_state,
    )

    fold_results, fold_params, last_completed_iteration = _check_cache(
        cache_file, continue_from_cache
    )

    for iteration, params in enumerate(sampler):
        t_start = time.time()
        if continue_from_cache and iteration <= last_completed_iteration:
            continue

        logger.info(f"CV iteration: {iteration}")
        logger.info(f"  Params: {params}")

        num_rounds = _determine_mcboost_num_rounds(
            train_df=train_df,
            scores_column=scores_column,
            label_column=label_column,
            categorical_segment_columns=categorical_segment_columns,
            numerical_segment_columns=numerical_segment_columns,
            params=params,
            weight_column=weight_column,
        )

        for i, (train_index, test_index) in enumerate(splitter.split(train_df)):
            logger.debug(f"  Fold: {i}")

            fold_train = train_df.iloc[train_index]

            # Extract context features
            context_feature_values = {}
            for feature_name, feature_func in metafeatures.items():
                context_feature_values[feature_name] = feature_func(
                    df=fold_train,
                    scores_column=scores_column,
                    label_column=label_column,
                    categorical_segment_columns=categorical_segment_columns,
                    numerical_segment_columns=numerical_segment_columns,
                    weight_column=weight_column,
                )

            logger.debug("    Fitting mcboost")
            mcboost = _fit_mcboost(
                train_df=train_df,
                scores_column=scores_column,
                label_column=label_column,
                categorical_segment_columns=categorical_segment_columns,
                numerical_segment_columns=numerical_segment_columns,
                params=params,
                num_rounds=num_rounds,
                train_index=train_index,
                weight_column=weight_column,
            )

            logger.debug("    Predicting with mcboost")
            # We use the fact that for n_rounds=n we need to run n_rounds=1 ... n_rounds=n-1
            # as well. Therefore we get results for n_rounds < max_num_rounds for free.
            fold_val = train_df.iloc[test_index]
            predictions_per_round = mcboost.predict(
                fold_val,
                scores_column,
                categorical_segment_columns,
                numerical_segment_columns,
                return_all_rounds=True,
            )

            fold_eval_df = train_df[
                categorical_evaluation_segment_columns
                + numerical_evaluation_segment_columns
                + [label_column]
            ].iloc[test_index]
            if weight_column is not None:
                fold_eval_df[weight_column] = train_df[weight_column].iloc[test_index]

            for round_, predictions in enumerate(predictions_per_round):
                logger.debug(f"      Computing metrics for round {round_}")
                fold_eval_df[scores_column] = predictions

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fold_result = _compute_metrics(
                        df=fold_eval_df,
                        label_column=label_column,
                        score_column=scores_column,
                        weight_column=weight_column,
                        categorical_segment_columns=categorical_evaluation_segment_columns,
                        numerical_segment_columns=numerical_evaluation_segment_columns,
                        metrics=metrics,
                    )

                this_fold_params = params.copy()
                this_fold_params["num_rounds"] = round_ + 1
                this_fold_params["param_iteration"] = iteration
                fold_params.append(this_fold_params)

                fold_result["num_rounds"] = round_ + 1
                fold_result["param_iteration"] = iteration
                fold_result["method"] = "MCBoost"  # pyre-ignore
                fold_result["context"] = context_feature_values  # pyre-ignore
                logger.debug(f"      Fold result: {fold_result}")
                fold_results.append(fold_result)

            # Add baselines
            if baselines:
                logger.debug("      Adding baselines")
            for baseline_model in baselines:
                model = _fit_model(
                    model_cls=baseline_model,
                    model_kwargs={},
                    train_df=train_df,
                    scores_column=scores_column,
                    label_column=label_column,
                    categorical_segment_columns=categorical_segment_columns,
                    numerical_segment_columns=numerical_segment_columns,
                    train_index=train_index,
                    weight_column=weight_column,
                )

                fold_eval_df[scores_column] = model.predict(
                    fold_val,
                    scores_column,
                    categorical_segment_columns,
                    numerical_segment_columns,
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fold_result = _compute_metrics(
                        df=fold_eval_df,
                        label_column=label_column,
                        score_column=scores_column,
                        weight_column=weight_column,
                        categorical_segment_columns=categorical_evaluation_segment_columns,
                        numerical_segment_columns=numerical_evaluation_segment_columns,
                        metrics=metrics,
                    )
                    fold_result["param_iteration"] = iteration
                    fold_result["num_rounds"] = 0
                    fold_result["method"] = baseline_model.__name__
                    fold_result["context"] = context_feature_values  # pyre-ignore
                fold_params.append({})
                fold_results.append(fold_result)

        logger.debug("  Writing cache")
        _cache_results(cache_file, fold_results, fold_params)
        logger.info(f"  Time elapsed: {time.time() - t_start:.2f}s")

    cv_results = _aggregate_metrics_and_add_parameters(fold_results, fold_params)
    return cv_results


def score_agg(
    df: pd.DataFrame,
    scores_column: str,
    label_column: str,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    score_func: Callable[[pd.Series], float],
    weight_column: str | None = None,
) -> float:
    return score_func(df[scores_column])


def label_agg(
    df: pd.DataFrame,
    scores_column: str,
    label_column: str,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    score_func: Callable[[pd.Series], float],
    weight_column: str | None = None,
) -> float:
    return score_func(df[label_column])


def prevalence(
    df: pd.DataFrame,
    scores_column: str,
    label_column: str,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    weight_column: str | None = None,
) -> float:
    prev = label_agg(
        df=df,
        scores_column=scores_column,
        label_column=label_column,
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
        score_func=lambda x: np.mean(x),
    )
    return prev


def num_positives(
    df: pd.DataFrame,
    scores_column: str,
    label_column: str,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    weight_column: str | None = None,
) -> float:
    n_pos = label_agg(
        df=df,
        scores_column=scores_column,
        label_column=label_column,
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
        score_func=lambda x: np.sum(x),
    )
    return n_pos


def num_negatives(
    df: pd.DataFrame,
    scores_column: str,
    label_column: str,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    weight_column: str | None = None,
) -> float:
    n_pos = num_positives(
        df=df,
        scores_column=scores_column,
        label_column=label_column,
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
    )
    return df.shape[0] - n_pos


def num_rows(
    df: pd.DataFrame,
    scores_column: str,
    label_column: str,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    weight_column: str | None = None,
) -> float:
    return df.shape[0]


def mean_score(
    df: pd.DataFrame,
    scores_column: str,
    label_column: str,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    weight_column: str | None = None,
) -> float:
    mean_prediction = score_agg(
        df=df,
        scores_column=scores_column,
        label_column=label_column,
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
        score_func=lambda x: np.mean(x),
    )
    return mean_prediction


def var_score(
    df: pd.DataFrame,
    scores_column: str,
    label_column: str,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    weight_column: str | None = None,
) -> float:
    mean_prediction = score_agg(
        df=df,
        scores_column=scores_column,
        label_column=label_column,
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
        score_func=lambda x: np.var(x, ddof=1),
    )
    return mean_prediction


def calibration_ratio(
    df: pd.DataFrame,
    scores_column: str,
    label_column: str,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    weight_column: str | None = None,
) -> float:
    label_mean = prevalence(
        df=df,
        scores_column=scores_column,
        label_column=label_column,
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
    )
    score_mean = mean_score(
        df=df,
        scores_column=scores_column,
        label_column=label_column,
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
    )
    return score_mean / label_mean


def numerical_features_max_abs_rank_correlation_with_residuals(
    df: pd.DataFrame,
    scores_column: str,
    label_column: str,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    weight_column: str | None = None,
) -> float:
    residuals = np.abs(df[scores_column] - df[label_column])
    max_rank_corr = 0
    for numerical_col in numerical_segment_columns:
        rank_corr = abs(
            scipy.stats.kendalltau(df[numerical_col], residuals).correlation
        )
        if rank_corr > max_rank_corr:
            max_rank_corr = rank_corr
    return max_rank_corr


def numerical_features_min_rank_correlation_pval_with_residuals(
    df: pd.DataFrame,
    scores_column: str,
    label_column: str,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    weight_column: str | None = None,
) -> float:
    residuals = np.abs(df[scores_column] - df[label_column])
    min_pvalue = 1
    for numerical_col in numerical_segment_columns:
        pval = scipy.stats.kendalltau(df[numerical_col], residuals).pvalue
        if pval < min_pvalue:
            min_pvalue = pval
    return min_pvalue


def min_valuecount_categorical_segment_columns(
    df: pd.DataFrame,
    scores_column: str,
    label_column: str,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    weight_column: str | None = None,
) -> float:
    min_value_count = df.shape[0]
    for categorical_col in categorical_segment_columns:
        value_count = df[categorical_col].value_counts().min()
        if value_count < min_value_count:
            min_value_count = value_count
    return min_value_count


def max_valuecount_categorical_segment_columns(
    df: pd.DataFrame,
    scores_column: str,
    label_column: str,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    weight_column: str | None = None,
) -> float:
    max_value_count = 0
    for categorical_col in categorical_segment_columns:
        value_count = df[categorical_col].value_counts().min()
        if value_count > max_value_count:
            max_value_count = value_count
    return max_value_count


def num_categorical_segment_columns(
    df: pd.DataFrame,
    scores_column: str,
    label_column: str,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    weight_column: str | None = None,
) -> float:
    return len(categorical_segment_columns)


def num_numerical_segment_columns(
    df: pd.DataFrame,
    scores_column: str,
    label_column: str,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    weight_column: str | None = None,
) -> float:
    return len(numerical_segment_columns)


def _check_cache(
    cache_file: PathLike[bytes] | PathLike[str] | None,
    continue_from_cache: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    last_completed_iteration = 0
    if continue_from_cache and cache_file is not None:
        with open(cache_file, "rb") as infile:
            fold_results, fold_params = pickle.load(infile)  # @nolint
            last_completed_iteration = fold_results[-1]["param_iteration"]
        logger.info(
            f"Found {last_completed_iteration} existing iterations in cache file {cache_file}"
        )
    else:
        fold_results, fold_params = [], []

    return fold_results, fold_params, last_completed_iteration


def _cache_results(
    cache_file: PathLike[bytes] | PathLike[str] | None,
    fold_results: list[dict[str, Any]],
    fold_params: list[dict[str, Any]],
) -> None:
    if cache_file is not None:
        with open(cache_file, "wb") as outfile:
            pickle.dump((fold_results, fold_params), outfile)  # @nolint


def _set_up_cv(
    parameter_space: dict[str, Any] | None,
    n_parameter_samples: int,
    n_folds: int,
    random_state: int | None = None,
) -> tuple[KFold, ParameterSampler]:
    if parameter_space is None:
        parameter_space = DEFAULT_PARAMETER_SPACE
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    sampler = ParameterSampler(
        parameter_space, n_iter=n_parameter_samples, random_state=random_state
    )
    return splitter, sampler


def _determine_mcboost_num_rounds(
    train_df: pd.DataFrame,
    scores_column: str,
    label_column: str,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    params: dict[str, Any],
    weight_column: str | None = None,
) -> int:

    lightgbm_params = {
        k: v
        for k, v in params.items()
        if k
        not in ["monotone_t", "num_rounds", "early_stopping", "patience", "n_folds"]
    }
    model_kwargs = {
        "early_stopping": True,
        "monotone_t": params["monotone_t"],
        "lightgbm_params": lightgbm_params,
    }

    mcboost_model = methods.MCBoost(**model_kwargs)
    logger.info("    Determining best number of rounds")
    num_rounds = mcboost_model.determine_best_num_rounds(
        train_df,
        label_column,
        scores_column,
        categorical_feature_column_names=categorical_segment_columns,
        numerical_feature_column_names=numerical_segment_columns,
        weight_column_name=weight_column,
    )
    logger.info("    Optimal number of rounds: {num_rounds}")
    return num_rounds


def _fit_mcboost(
    train_df: pd.DataFrame,
    scores_column: str,
    label_column: str,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    params: dict[str, Any],
    num_rounds: int,
    train_index: np.ndarray,  # pyre-ignore[24]
    weight_column: str | None = None,
) -> methods.MCBoost:

    lightgbm_params = {
        k: v
        for k, v in params.items()
        if k
        not in ["monotone_t", "num_rounds", "early_stopping", "patience", "n_folds"]
    }
    model_kwargs = {
        "early_stopping": False,
        "num_rounds": num_rounds,
        "monotone_t": params["monotone_t"],
        "lightgbm_params": lightgbm_params,
    }
    mcboost = _fit_model(
        model_cls=methods.MCBoost,
        model_kwargs=model_kwargs,
        train_df=train_df,
        scores_column=scores_column,
        label_column=label_column,
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
        weight_column=weight_column,
        train_index=train_index,
    )
    assert isinstance(
        mcboost, methods.MCBoost
    )  # always true, but necessary for type checking
    return mcboost


def _fit_model(
    model_cls: type[methods.BaseCalibrator],
    model_kwargs: dict[str, Any],
    train_df: pd.DataFrame,
    scores_column: str,
    label_column: str,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    train_index: np.ndarray,  # pyre-ignore[24]
    weight_column: str | None = None,
) -> methods.BaseCalibrator:

    fold_train = train_df.iloc[train_index]
    model = model_cls(**model_kwargs)  # pyre-ignore

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            fold_train,
            scores_column,
            label_column,
            weight_column_name=weight_column,
            categorical_feature_column_names=categorical_segment_columns,
            numerical_feature_column_names=numerical_segment_columns,
        )

    return model


def _compute_metrics(
    metrics: dict[str, CVMetricInterface],
    df: pd.DataFrame,
    label_column: str,
    score_column: str,
    categorical_segment_columns: list[str] | None = None,
    numerical_segment_columns: list[str] | None = None,
    weight_column: str | None = None,
) -> dict[str, float]:
    """
    Compute metrics including multicalibration error and precision-recall AUC.
    """

    def _failable_metric_computation(
        metric_func: CVMetricInterface, metric_name: str, **kwargs: Any
    ) -> float:
        try:
            return metric_func(**kwargs)
        except Exception as e:
            logger.error(f"Couldn't compute {metric_name}: {e}")

        return np.nan

    output = {}
    for metric_name, metric_func in metrics.items():
        output[metric_name] = _failable_metric_computation(
            metric_func,
            metric_name,
            df=df,
            label_column=label_column,
            score_column=score_column,
            weight_column=weight_column,
            categorical_segment_columns=categorical_segment_columns,
        )

    return output


def _confidence_interval(
    column: np.ndarray,  # pyre-ignore[24]
) -> tuple[float, float]:
    """
    Calculate the confidence interval for a given dataset column.
    :param column: Numpy array of data points.
    :return: Tuple containing the lower and upper bounds of the confidence interval.
    """
    confidence_level = 0.95
    degrees_freedom = column.size - 1
    sample_mean = np.mean(column)
    sample_standard_error = scipy.stats.sem(column)
    confidence_interval = scipy.stats.t.interval(
        confidence_level, degrees_freedom, sample_mean, sample_standard_error
    )

    return round(confidence_interval[0], 3), round(
        confidence_interval[1], 3
    )  # pyre-ignore


def _collect_to_list(series: pd.Series) -> list[Any]:
    return list(series)


def _aggregate_metrics_and_add_parameters(
    fold_results: list[dict[str, Any]],
    fold_params: list[dict[str, Any]],
) -> pd.DataFrame:
    cv_res_df = pd.DataFrame(fold_results)
    cv_res_context = cv_res_df[
        ["param_iteration", "num_rounds", "method", "context"]
    ].copy()
    del cv_res_df["context"]

    # Aggregate the results over folds by iteration (and num rounds since we have those for free)
    # We also keep the raw results in collect_to_list such that we keep the freedom to calculate any metric
    cv_results = cv_res_df.groupby(["param_iteration", "num_rounds", "method"]).agg(
        ["mean", "min", "max", _confidence_interval, _collect_to_list]
    )
    cv_results.columns = ["_".join(col).strip() for col in cv_results.columns.values]

    # Join in the parameter sets and the context features
    ## Aggregate the context features
    context_df = cv_res_context.groupby(
        ["param_iteration", "num_rounds", "method"]
    ).agg(_collect_to_list)
    ## remove the duplicates due to parameters being constant in folds
    param_df = pd.DataFrame(fold_params).drop_duplicates()
    # Store parameters as a dictionary as well to allow easy retrieval w/o specifying columns
    # setting the index to param_iteration as a quick way to remove it from the dicts
    param_df["param_dict"] = param_df.set_index("param_iteration").to_dict(
        orient="records"
    )  # pyre-ignore
    param_df["method"] = "MCBoost"
    param_df = param_df.set_index(["param_iteration", "num_rounds", "method"])
    return cv_results.join(param_df).join(context_df).reset_index()
