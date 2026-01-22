# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict

"""
Calibration methods for machine learning models.

This module provides implementations of various calibration techniques including
multicalibration methods (MCGrad), traditional approaches (Platt scaling, isotonic
regression), and segment-aware calibrators.

All calibrators follow a scikit-learn-style fit/predict interface defined by
:class:`~multicalibration.base.BaseCalibrator`.
"""

import json
import logging
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, cast, Dict, Generic, TypeVar

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn import isotonic, metrics as skmetrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from typing_extensions import Self

from . import _utils as utils
from .base import BaseCalibrator
from .metrics import _ScoreFunctionInterface, wrap_sklearn_metric_func

logger: logging.Logger = logging.getLogger(__name__)

from ._compat import create_kbins_discretizer, groupby_apply
# @oss-disable[end= ]: from .internal._compat import DeprecatedAttributesMixin


@dataclass(frozen=True, slots=True)
class _MCGradProcessedData:
    """Preprocessed data container for MCGrad training and prediction.

    This immutable dataclass holds all preprocessed inputs needed for fitting
    or applying an MCGrad calibration model. It supports indexing to extract
    subsets of the data (e.g., for cross-validation folds).

    :param features: 2D array of shape (n_samples, n_features) containing the
        extracted segment features (categorical encoded + numerical).
    :param predictions: 1D array of transformed predictions (e.g., logits for
        binary classification).
    :param weights: 1D array of sample weights.
    :param output_presence_mask: Boolean array indicating which samples have
        valid predictions. Samples with invalid predictions (NaN, out of bounds)
        are marked as False.
    :param categorical_feature_names: List of categorical feature column names.
    :param numerical_feature_names: List of numerical feature column names.
    :param labels: Optional 1D array of ground truth labels. Required for fitting,
        but None during prediction.
    """

    features: npt.NDArray
    predictions: npt.NDArray
    weights: npt.NDArray
    output_presence_mask: npt.NDArray
    categorical_feature_names: list[str]
    numerical_feature_names: list[str]
    labels: npt.NDArray | None = None

    def __getitem__(self, index: npt.NDArray) -> "_MCGradProcessedData":
        """Index into the data to extract a subset.

        :param index: Boolean or integer array specifying which samples to select.
        :return: A new MCGradProcessedData instance containing only the selected samples.
        """
        return _MCGradProcessedData(
            features=self.features[index],
            predictions=self.predictions[index],
            weights=self.weights[index],
            output_presence_mask=self.output_presence_mask[index],
            categorical_feature_names=self.categorical_feature_names,
            numerical_feature_names=self.numerical_feature_names,
            labels=self.labels[index] if self.labels is not None else None,
        )


# @oss-disable[end= ]: _MCBoostProcessedData = _MCGradProcessedData


class _EstimationMethod(Enum):
    """Estimation method for early stopping validation.

    Determines how the validation set is created for early stopping during
    MCGrad training.

    :cvar CROSS_VALIDATION: Use k-fold cross-validation to estimate performance.
        More robust but slower, recommended for smaller datasets.
    :cvar HOLDOUT: Use a single train/validation split. Faster but may have
        higher variance, suitable for larger datasets.
    :cvar AUTO: Automatically choose between cross-validation and holdout based
        on the effective sample size of the dataset.
    """

    CROSS_VALIDATION = 1
    HOLDOUT = 2
    AUTO = 3


class _BaseMCGrad(
    BaseCalibrator,
    ABC,
):
    """
    Abstract base class for MCGrad models. This class hosts the common functionality for all MCGrad models and defines
    an abstract interface that all MCGrad models must implement.
    """

    _SERIALIZATION_KEY = "mcgrad"
    VALID_SIZE = 0.4
    MCE_STAT_SIGN_THRESHOLD = 2.49767216
    MCE_STRONG_EVIDENCE_THRESHOLD = 4.70812972
    DEFAULT_ALLOW_MISSING_SEGMENT_FEATURE_VALUES = True
    ESS_THRESHOLD_FOR_CROSS_VALIDATION = 2500000
    # Name of the prediction feature, e.g. for feature_importance
    _PREDICTION_FEATURE_NAME = "prediction"
    MAX_NUM_ROUNDS_EARLY_STOPPING = 100
    NUM_ROUNDS_DEFAULT_NO_EARLY_STOPPING = 5

    DEFAULT_HYPERPARAMS: dict[str, Any] = {
        "monotone_t": False,
        "early_stopping": True,
        "patience": 0,
        "n_folds": 5,
    }

    @property
    @abstractmethod
    def _objective(self) -> str:
        pass

    @property
    @abstractmethod
    def _default_early_stopping_metric(self) -> tuple[_ScoreFunctionInterface, bool]:
        """Return the default early stopping metric and whether to minimize it.

        :return: A tuple of (score_function, minimize_score) where minimize_score
            is True if lower scores are better (e.g., log_loss, MSE) and False
            if higher scores are better (e.g., AUC, accuracy).
        """
        pass

    @staticmethod
    @abstractmethod
    def _transform_predictions(predictions: npt.NDArray) -> npt.NDArray:
        pass

    @staticmethod
    @abstractmethod
    def _inverse_transform_predictions(transformed: npt.NDArray) -> npt.NDArray:
        pass

    @staticmethod
    @abstractmethod
    def _compute_unshrink_factor(
        y: npt.NDArray, predictions: npt.NDArray, w: npt.NDArray | None
    ) -> float:
        pass

    @abstractmethod
    def _check_predictions(
        self, df_train: pd.DataFrame, prediction_column_name: str
    ) -> None:
        pass

    @abstractmethod
    def _check_labels(self, df_train: pd.DataFrame, label_column_name: str) -> None:
        pass

    @staticmethod
    @abstractmethod
    def _predictions_out_of_bounds(predictions: npt.NDArray) -> npt.NDArray:
        pass

    @property
    @abstractmethod
    def _cv_splitter(self) -> KFold | StratifiedKFold:
        pass

    @property
    @abstractmethod
    def _holdout_splitter(self) -> utils.TrainTestSplitWrapper:
        pass

    @property
    @abstractmethod
    def _noop_splitter(
        self,
    ) -> utils.NoopSplitterWrapper:
        pass

    def __init__(
        self,
        encode_categorical_variables: bool = True,
        monotone_t: bool | None = None,
        num_rounds: int | None = None,
        lightgbm_params: dict[str, Any] | None = None,
        early_stopping: bool | None = None,
        patience: int | None = None,
        early_stopping_use_crossvalidation: bool | None = None,
        n_folds: int | None = None,
        early_stopping_score_func: _ScoreFunctionInterface | None = None,
        early_stopping_minimize_score: bool | None = None,
        early_stopping_timeout: int | None = 8 * 60 * 60,  # 8 hours
        save_training_performance: bool = False,
        monitored_metrics_during_training: list[_ScoreFunctionInterface] | None = None,
        allow_missing_segment_feature_values: bool = DEFAULT_ALLOW_MISSING_SEGMENT_FEATURE_VALUES,
        random_state: int | np.random.Generator | None = 42,
    ) -> None:
        """
        :param encode_categorical_variables: whether to encode categorical variables using a modified label encoding (when True),
            or whether to assume that categorical variables are already manipulated into the right format prior to calling MCGrad
            (when False).
        :param monotone_t: whether to use a monotonicity constraint on the logit feature (i.e., t): value
            True implies that the decision tree is blocked from creating splits where a lower value of t
            results in a higher predicted probability.
        :param num_rounds: number of rounds boosting that is used in MCGrad. When early stopping is used, then num_rounds specifies the maximum
            number of rounds. If set to None, default values are used.
        :param lightgbm_params: the training parameters of lightgbm model. See: https://lightgbm.readthedocs.io/en/stable/Parameters.html
            if None, we will use a set of default parameters.
        :param early_stopping: whether to use early stopping based on cross-validation. When early stopping is used, then num_rounds specifies
            the maximum number of rounds that are fit, and the effective number of rounds is determined based on cross-validation.
        :param patience: the maximum number of consecutive rounds without improvement in `early_stopping_score_func`.
        :param early_stopping_use_crossvalidation: whether to use cross-validation (k-fold) for early stopping (otherwise use holdout). If set to None, then the evaluation method is determined automatically.
        :param early_stopping_score_func: the metric (default = log_loss if set to None) used to select the optimal number of rounds, when early stopping is used. It can be the Multicalibration Error (MulticalibrationError) or any SkLearn metric (SkLearnWrapper).
        :param early_stopping_minimize_score: whether the score function used for early stopping should be minimized. If set to False score is maximized.
        :param early_stopping_timeout: number of seconds after which early stopping is forced to stop and the number of rounds is determined. If set to None, then early stopping will not time out. Ignored when early stopping is disabled.
        :param n_folds: number of folds for k-fold cross-validation (used only when `early_stopping_use_crossvalidation` is `True`; or when that argument is `None` and k-fold is chosen automatically).
        :param save_training_performance: whether to save the training performance values for each round, in addition to the performance on the held-out validation set.
            This parameter is only relevant when early stopping is used. If set to False, then only the performance on the held-out validation set is saved.
        :param monitored_metrics_during_training: a list of metrics to monitor during training. This parameter is only relevant when early stopping is used.
            It includes which metrics to monitor during training, in addition to the metric used for early stopping (score_func).
        :param allow_missing_segment_feature_values: whether to allow missing values in the segment feature data. If set to True, missing values are used for training and prediction. If set to False, training with missing values will raise an Exception and prediction
            with missing values will return None.
        """
        self.random_state = random_state
        if isinstance(random_state, np.random.Generator):
            self._rng: np.random.Generator = random_state
        else:
            self._rng: np.random.Generator = np.random.default_rng(random_state)

        if early_stopping_score_func is not None:
            if early_stopping_minimize_score is None:
                raise ValueError(
                    "If using a custom score function the attribute "
                    "`early_stopping_minimize_score` has to be set."
                )
            self.early_stopping_score_func: _ScoreFunctionInterface = (
                early_stopping_score_func
            )
            self.early_stopping_minimize_score: bool = early_stopping_minimize_score
        else:
            default_metric, default_minimize = self._default_early_stopping_metric
            self.early_stopping_score_func = default_metric
            self.early_stopping_minimize_score: bool = default_minimize
            if early_stopping_minimize_score is not None:
                raise ValueError(
                    f"`early_stopping_minimize_score` is only relevant when using a "
                    f"custom score function. The default score function is "
                    f"{self.early_stopping_score_func.name} for which "
                    f"`early_stopping_minimize_score` is set to "
                    f"{self.early_stopping_minimize_score} automatically."
                )

        self._set_lightgbm_params(lightgbm_params)

        self.encode_categorical_variables = encode_categorical_variables
        self.monotone_t: bool = (
            self.DEFAULT_HYPERPARAMS["monotone_t"] if monotone_t is None else monotone_t
        )

        self.early_stopping: bool = (
            self.DEFAULT_HYPERPARAMS["early_stopping"]
            if early_stopping is None
            else early_stopping
        )

        if not self.early_stopping:
            if patience is not None:
                raise ValueError(
                    "`patience` must be None when argument `early_stopping` is disabled."
                )
            if early_stopping_use_crossvalidation is not None:
                raise ValueError(
                    "`early_stopping_use_crossvalidation` must be None when `early_stopping` is disabled."
                )
            if early_stopping_score_func is not None:
                raise ValueError(
                    "`score_func` must be None when `early_stopping` is disabled."
                )
            if early_stopping_minimize_score is not None:
                raise ValueError(
                    "`minimize` must be None when `early_stopping` is disabled"
                )
            # Override the timeout when early stopping is disabled
            early_stopping_timeout = None

        self.early_stopping_estimation_method: _EstimationMethod
        if early_stopping_use_crossvalidation is True:
            self.early_stopping_estimation_method = _EstimationMethod.CROSS_VALIDATION
        elif early_stopping_use_crossvalidation is None:
            self.early_stopping_estimation_method = _EstimationMethod.AUTO
        else:
            self.early_stopping_estimation_method = _EstimationMethod.HOLDOUT

        if self.early_stopping_estimation_method == _EstimationMethod.HOLDOUT:
            if n_folds is not None:
                raise ValueError(
                    "`n_folds` must be None when `early_stopping_use_crossvalidation` is disabled."
                )

        if num_rounds is None:
            if self.early_stopping:
                num_rounds = self.MAX_NUM_ROUNDS_EARLY_STOPPING
            else:
                num_rounds = self.NUM_ROUNDS_DEFAULT_NO_EARLY_STOPPING

        self.num_rounds: int = num_rounds

        self.patience: int = (
            self.DEFAULT_HYPERPARAMS["patience"] if patience is None else patience
        )

        self.early_stopping_timeout: int | None = early_stopping_timeout

        self.n_folds: int = (
            1  # Because we make a single train/test split when using holdout
            if (self.early_stopping_estimation_method == _EstimationMethod.HOLDOUT)
            else self.DEFAULT_HYPERPARAMS["n_folds"]
            if n_folds is None
            else n_folds
        )

        self.mr: list[lgb.Booster] = []
        self.unshrink_factors: list[float] = []
        self.enc: utils.OrdinalEncoderWithUnknownSupport | None = None

        self.save_training_performance = save_training_performance
        self._performance_metrics: Dict[str, list[float]] = defaultdict(list)
        self.monitored_metrics_during_training: list[_ScoreFunctionInterface] = (
            []
            if monitored_metrics_during_training is None
            else monitored_metrics_during_training
        )
        # Include the score function in the monitored metrics, if not there already
        if self.early_stopping_score_func.name not in [
            monitored_metric.name
            for monitored_metric in self.monitored_metrics_during_training
        ]:
            self.monitored_metrics_during_training.append(
                self.early_stopping_score_func
            )

        self.monitored_metrics_during_training = self._remove_duplicate_metrics(
            self.monitored_metrics_during_training
        )

        self.mce_below_initial: bool | None = None
        self.mce_below_strong_evidence_threshold: bool | None = None
        self.allow_missing_segment_feature_values = allow_missing_segment_feature_values
        self.categorical_feature_names: list[str] | None = None
        self.numerical_feature_names: list[str] | None = None

    def _next_seed(self) -> int:
        return int(self._rng.integers(0, 2**32 - 1))

    def _set_lightgbm_params(self, lightgbm_params: dict[str, Any] | None) -> None:
        """
        Sets or updates the LightGBM parameters for this MCGrad instance.


        The `lightgbm_params` argument and `self.lightgbm_params` attribute are not always identical.
        When tuning hyperparameters (see tuning.py), we modify existing MCGrad objects rather than creating new objects.
        This design choice allows for parameter updates during hyperparameter tuning without
        recreating the entire object, but it means the instance's parameters may differ from
        what was originally passed during initialization.

        :param lightgbm_params: Dictionary of LightGBM parameters to set or update. If None,
            the default parameters will be used.
        """
        try:
            if self.mr:
                logger.warning(
                    "Model has already been fit. To avoid inconsistent state all training state will be reset after setting lightgbm_params."
                )
                self._reset_training_state()
        except AttributeError:
            pass

        if not hasattr(self, "lightgbm_params"):
            params_to_set = self.DEFAULT_HYPERPARAMS.get("lightgbm_params", {}).copy()
        else:
            params_to_set = self.lightgbm_params.copy()

        if lightgbm_params is not None:
            params_to_set.update(lightgbm_params)

        if "num_rounds" in params_to_set:
            raise ValueError(
                "Avoid using `num_rounds` in `lightgbm_params` due to a naming "
                "conflict with `num_rounds` in MCGrad. Use any of the other aliases "
                "instead (https://lightgbm.readthedocs.io/en/latest/Parameters.html)"
            )

        self.lightgbm_params: dict[str, Any] = {
            **params_to_set,
            "objective": self._objective,
            "seed": self._next_seed(),
            "deterministic": True,
            "verbosity": -1,
        }

    def feature_importance(self) -> pd.DataFrame:
        """Returns the feature importance of the first MCGrad round.

        Importance is defined as the total gain from splits on a feature from the first round of MCGrad.

        :return: A dataframe with columns 'feature' and 'importance', sorted by importance in descending order
        """
        if (
            not self.mr
            or self.categorical_feature_names is None
            or self.numerical_feature_names is None
        ):
            raise ValueError("Model has not been fit yet.")

        feature_importance = self.mr[0].feature_importance(importance_type="gain")

        return pd.DataFrame(
            {
                # Ordering of features here relies on two things 1) that MCGrad.extract_features returns first categoricals then
                # numericals and 2) that .fit method concatenates logits to the end of the feature matrix
                # pyre-ignore[58] if either feature_names attribute is None an error is raised above
                "feature": self.categorical_feature_names
                + self.numerical_feature_names
                + [self._PREDICTION_FEATURE_NAME],
                "importance": feature_importance,
            }
        ).sort_values("importance", ascending=False)

    def _reset_training_state(self) -> None:
        self.mr = []
        self.unshrink_factors = []
        self.mce_below_initial = None
        self.mce_below_strong_evidence_threshold = None
        self._performance_metrics = defaultdict(list)
        self.enc: utils.OrdinalEncoderWithUnknownSupport | None = None
        self.categorical_feature_names = None
        self.numerical_feature_names = None

    @property
    def _mce_is_satisfactory(self) -> bool | None:
        return self.mce_below_initial and self.mce_below_strong_evidence_threshold

    @property
    def performance_metrics(self) -> dict[str, list[float]]:
        """Returns the performance metrics collected during early stopping procedure.

        Metrics are tracked for each round of MCGrad during the early stopping phase. The dictionary
        contains metric names as keys and lists of values (one per round) as values. Metrics include
        the early stopping metric and any additional monitored metrics specified during initialization.

        :return: Dictionary mapping metric names to lists of values per round
        """
        if not self._performance_metrics:  # empty
            raise ValueError(
                "Performance metrics are only available after the model has been fit with `early_stopping=True`"
            )
        return self._performance_metrics

    def _check_segment_features(
        self,
        df: pd.DataFrame,
        categorical_feature_column_names: list[str],
        numerical_feature_column_names: list[str],
    ) -> None:
        segment_df = df[
            categorical_feature_column_names + numerical_feature_column_names
        ]
        if segment_df.isnull().any().any():
            if self.allow_missing_segment_feature_values:
                logger.info(
                    f"Missing values found in segment feature data. {self.__class__.__name__} supports handling of missing data in segment features. If you want to disable native missing value support and predict None for examples with missing values in segment features, set `allow_missing_segment_feature_values=False` in the constructor of {self.__class__.__name__}. "
                )
            else:
                raise ValueError(
                    f"Missing values found in segment feature data and `allow_missing_segment_feature_values` is set to False. If you want to enable native missing value support, set `allow_missing_segment_feature_values=True` in the constructor of {self.__class__.__name__}."
                )

    def _check_input_data(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        categorical_feature_column_names: list[str] | None,
        numerical_feature_column_names: list[str] | None,
    ) -> None:
        self._check_predictions(df, prediction_column_name)
        self._check_labels(df, label_column_name)
        self._check_segment_features(
            df,
            categorical_feature_column_names or [],
            numerical_feature_column_names or [],
        )

    def _preprocess_input_data(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str | None,
        weight_column_name: str | None,
        categorical_feature_column_names: list[str],
        numerical_feature_column_names: list[str],
        is_fit_phase: bool = False,
    ) -> _MCGradProcessedData:
        """
        Prepares processed data representation by extracting features once and computing the presence mask.

        This method extracts features, transforms predictions, and computes the presence mask
        all in one go, avoiding redundant operations later.

        :param df: DataFrame containing the data
        :param prediction_column_name: Name of the prediction column
        :param label_column_name: Optional name of the label column (required for fit, optional for predict)
        :param weight_column_name: Optional name of the weight column
        :param categorical_feature_column_names: List of categorical feature column names
        :param numerical_feature_column_names: List of numerical feature column names
        :param is_fit_phase: Whether this is during fit phase (for encoder training)
        :return: MCGradProcessedData object with extracted features and metadata
        """
        logger.info(
            f"Preprocessing input data with {len(df)} rows; in_fit_phase = {is_fit_phase}"
        )
        x = self._extract_features(
            df=df,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
            is_fit_phase=is_fit_phase,
        )

        predictions = self._transform_predictions(df[prediction_column_name].values)
        y = (
            df[label_column_name].values.astype(float)
            if label_column_name is not None
            else None
        )
        w = (
            df[weight_column_name].values.astype(float)
            if weight_column_name
            else np.ones(len(df))
        )

        presence_mask = self._get_output_presence_mask(
            df,
            prediction_column_name,
            categorical_feature_column_names or [],
            numerical_feature_column_names or [],
        )

        return _MCGradProcessedData(
            features=x,
            predictions=predictions,
            weights=w,
            output_presence_mask=presence_mask,
            categorical_feature_names=categorical_feature_column_names,
            numerical_feature_names=numerical_feature_column_names,
            labels=y,
        )

    def fit(
        self,
        df_train: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        df_val: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the MCGrad calibration model on the provided training data.

        :param df_train: The dataframe containing the training data
        :param prediction_column_name: Name of the column in dataframe df that contains the uncalibrated predictions
        :param label_column_name: Name of the column in dataframe df that contains the ground truth labels
        :param weight_column_name: Name of the column in dataframe df that contains the instance weights
        :param categorical_feature_column_names: List of column names in the df that contain the categorical
            segmentation features
        :param numerical_feature_column_names: List of column names in the df that contain the numerical
            segmentation features
        :param df_val: Optional validation dataframe for early stopping. When provided with early stopping enabled,
            this validation set will be used instead of a holdout from the training data. early_stopping_use_crossvalidation has
            to be set to False for this to work.
        :param kwargs: Additional keyword arguments
        :return: The fitted calibrator instance
        """
        self._check_input_data(
            df_train,
            prediction_column_name,
            label_column_name,
            categorical_feature_column_names,
            numerical_feature_column_names,
        )

        self._reset_training_state()

        # Store feature names to be used in feature importance later
        self.categorical_feature_names = categorical_feature_column_names or []
        self.numerical_feature_names = numerical_feature_column_names or []

        preprocessed_data = self._preprocess_input_data(
            df=df_train,
            prediction_column_name=prediction_column_name,
            label_column_name=label_column_name,
            weight_column_name=weight_column_name,
            categorical_feature_column_names=categorical_feature_column_names or [],
            numerical_feature_column_names=numerical_feature_column_names or [],
            is_fit_phase=True,
        )

        preprocessed_val_data = None

        num_rounds = self.num_rounds
        if self.early_stopping:
            timeout_msg = (
                f" (timeout: {self.early_stopping_timeout}s)"
                if self.early_stopping_timeout
                else ""
            )
            logger.info(
                f"Early stopping activated, max_num_rounds={self.num_rounds}{timeout_msg}"
            )

            if df_val is not None:
                self._check_input_data(
                    df_val,
                    prediction_column_name,
                    label_column_name,
                    categorical_feature_column_names,
                    numerical_feature_column_names,
                )

                preprocessed_val_data = self._preprocess_input_data(
                    df=df_val,
                    prediction_column_name=prediction_column_name,
                    label_column_name=label_column_name,
                    weight_column_name=weight_column_name,
                    categorical_feature_column_names=categorical_feature_column_names
                    or [],
                    numerical_feature_column_names=numerical_feature_column_names or [],
                    is_fit_phase=False,  # Don't want to fit the encoder on validation data, emulate predict setup
                )

            num_rounds = self._determine_best_num_rounds(
                preprocessed_data, preprocessed_val_data
            )

            if num_rounds > 0:
                logger.info(
                    f"Fitting final {self.__class__.__name__} model with {num_rounds} rounds"
                )
        else:
            logger.info(f"Early stopping deactivated, fitting {self.num_rounds} rounds")

        predictions = preprocessed_data.predictions
        for round_idx in range(num_rounds):
            logger.info(f"Fitting round {round_idx + 1}")
            predictions = self._fit_single_round(
                x=preprocessed_data.features,
                # pyre-ignore[6] `label_column_name` is a mandatory argument and therefore passed to _preprocess_input_data
                # if lables are not available that function would have raised an error. We can therefore assume that labels are not None.
                y=preprocessed_data.labels,
                prediction=predictions,
                w=preprocessed_data.weights,
                categorical_feature_column_names=preprocessed_data.categorical_feature_names,
                numerical_feature_column_names=preprocessed_data.numerical_feature_names,
            )
        return self

    def _fit_single_round(
        self,
        x: npt.NDArray,
        y: npt.NDArray,
        prediction: npt.NDArray,
        w: npt.NDArray | None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
    ) -> npt.NDArray:
        x = np.c_[x, prediction]

        if categorical_feature_column_names is None:
            categorical_feature_column_names = []
        if numerical_feature_column_names is None:
            numerical_feature_column_names = []

        self.mr.append(
            lgb.train(
                params=self._get_lgbm_params(x),
                train_set=lgb.Dataset(
                    x,
                    label=y,
                    init_score=prediction,
                    weight=w,
                    categorical_feature=categorical_feature_column_names,
                    feature_name=categorical_feature_column_names
                    + numerical_feature_column_names
                    + [self._PREDICTION_FEATURE_NAME],
                ),
            )
        )

        new_pred = self.mr[-1].predict(x, raw_score=True)
        prediction = prediction + new_pred
        self.unshrink_factors.append(self._compute_unshrink_factor(y, prediction, w))
        prediction *= self.unshrink_factors[-1]

        return prediction

    def _get_output_presence_mask(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str],
        numerical_feature_column_names: list[str],
    ) -> npt.NDArray:
        """
        Returns a boolean mask indicating for which examples predictions are valid (i.e., not NaN).

        For examples with missing or otherwise invalid uncalibrated score as well as for examples with missing segment features (if self.allow_missing_segment_feature_values is False), predictions are not valid.
        """
        predictions = df[prediction_column_name].to_numpy()
        nan_mask = np.isnan(predictions)
        outofbounds_mask = self._predictions_out_of_bounds(predictions)
        if nan_mask.any():
            logger.warning(
                f"{self.__class__.__name__} does not support missing values in the prediction column. Found {nan_mask.sum()} missing values. {self.__class__.__name__}.predict will return np.nan for these predictions."
            )
        if outofbounds_mask.any():
            min_score = np.min(df[prediction_column_name].values)
            max_score = np.max(df[prediction_column_name].values)
            logger.warning(
                f"{self.__class__.__name__} calibrates probabilistic binary classifiers, hence predictions must be in (0,1). Found min {min_score} and max {max_score}. {self.__class__.__name__}.predict will return np.nan for these predictions."
            )
        invalid_mask = nan_mask | outofbounds_mask
        if not self.allow_missing_segment_feature_values:
            segment_feature_missing_mask = (
                df[categorical_feature_column_names + numerical_feature_column_names]
                .isnull()
                .any(axis=1)
            )
            if segment_feature_missing_mask.any():
                logger.warning(
                    f"Found {segment_feature_missing_mask.sum()} missing values in segment features. {self.__class__.__name__}.predict will return np.nan for these predictions. {self.__class__.__name__} supports handling of missing data in segment features. If you want to enable native missing value support set `allow_missing_segment_feature_values=True` in the constructor of {self.__class__.__name__}. "
                )
            invalid_mask = invalid_mask | segment_feature_missing_mask
        return np.logical_not(invalid_mask)

    @staticmethod
    def _remove_duplicate_metrics(
        monitored_metrics_during_training: list[_ScoreFunctionInterface],
    ) -> list[_ScoreFunctionInterface]:
        """
        Removes duplicate metrics from the list of monitored metrics during training.
        """
        unique_metrics = []
        for metric in monitored_metrics_during_training:
            if metric.name not in [m.name for m in unique_metrics]:
                unique_metrics.append(metric)
        return unique_metrics

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        return_all_rounds: bool = False,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Apply the MCGrad calibration model to a DataFrame.

        This requires the `fit` method to have been previously called on this calibrator object.

        :param df: The dataframe containing the data to calibrate
        :param prediction_column_name: Name of the column in dataframe df that contains the predictions
        :param categorical_feature_column_names: List of column names in the df that contain the categorical
            segmentation features
        :param numerical_feature_column_names: List of column names in the df that contain the numerical
            segmentation features
        :param return_all_rounds: If True, returns predictions for all MCGrad rounds as a 2D array of shape
            (num_rounds, num_samples). If False, returns only the final round predictions as a 1D array
        :param kwargs: Additional keyword arguments
        :return: Array of calibrated predictions. Shape depends on return_all_rounds parameter
        """
        preprocessed_data = self._preprocess_input_data(
            df=df,
            prediction_column_name=prediction_column_name,
            label_column_name=None,
            weight_column_name=None,
            categorical_feature_column_names=categorical_feature_column_names or [],
            numerical_feature_column_names=numerical_feature_column_names or [],
            is_fit_phase=False,
        )

        predictions = self._predict(
            preprocessed_data.features,
            preprocessed_data.predictions,
            return_all_rounds,
        )

        return np.where(preprocessed_data.output_presence_mask, predictions, np.nan)

    def _predict(
        self,
        x: npt.NDArray,
        transformed_predictions: npt.NDArray,
        return_all_rounds: bool = False,
    ) -> npt.NDArray:
        """
        Predicts the calibrated probabilities using the trained model.

        :param x: the segment features.
        :param transformed_predictions: the transformed (e.g., logit) predictions that we are looking to calibrate.
        """
        assert len(self.mr) == len(self.unshrink_factors)
        if len(self.mr) < 1:
            logger.warning(
                f"{self.__class__.__name__} has not been fit. Returning the uncalibrated predictions."
            )
            inverse_preds = self._inverse_transform_predictions(transformed_predictions)
            return inverse_preds.reshape(1, -1) if return_all_rounds else inverse_preds

        predictions = transformed_predictions.copy()
        x = np.c_[x, predictions]
        predictions_per_round = np.zeros((len(self.mr), len(predictions)))
        for i in range(len(self.mr)):
            new_pred = self.mr[i].predict(x, raw_score=True)
            predictions += new_pred
            predictions *= self.unshrink_factors[i]
            x[:, -1] = predictions
            predictions_per_round[i] = self._inverse_transform_predictions(predictions)

        return predictions_per_round if return_all_rounds else predictions_per_round[-1]

    def _get_lgbm_params(self, x: npt.NDArray) -> dict[str, Any]:
        lgb_params = self.lightgbm_params.copy()
        if self.monotone_t:
            score_constraint = [1]
            segment_feature_constraints = [0] * (x.shape[1] - 1)
            lgb_params["monotone_constraints"] = (
                segment_feature_constraints + score_constraint
            )
        return lgb_params

    def _extract_features(
        self,
        df: pd.DataFrame,
        categorical_feature_column_names: list[str] | None,
        numerical_feature_column_names: list[str] | None,
        is_fit_phase: bool = False,
    ) -> npt.NDArray:
        if categorical_feature_column_names:
            cat_features = df[categorical_feature_column_names].values
            if self.encode_categorical_variables:
                if is_fit_phase:
                    self.enc = utils.OrdinalEncoderWithUnknownSupport()
                    self.enc.fit(cat_features)

                if self.enc is not None:
                    cat_features = self.enc.transform(cat_features)
                else:
                    raise ValueError(
                        "Fit has to be called before encoder can be applied."
                    )
            if np.nanmax(cat_features) >= np.iinfo(np.int32).max:
                raise ValueError(
                    "All categorical feature values must be smaller than 2^32 to prevent integer overflow internal to LightGBM."
                )
            if not self.encode_categorical_variables and np.nanmin(cat_features) < 0:
                raise ValueError(
                    "All categorical feature values must be non-negative, because LightGBM treats negative categorical values as missing."
                )
        else:
            cat_features = np.empty((df.shape[0], 0))

        if numerical_feature_column_names:
            num_features = df[numerical_feature_column_names].values
        else:
            num_features = np.empty((df.shape[0], 0))

        x = np.concatenate((cat_features, num_features), axis=1)
        return x

    def _determine_train_test_splitter(
        self,
        estimation_method: _EstimationMethod,
        has_custom_validation_set: bool,
    ) -> (
        KFold
        | StratifiedKFold
        | utils.TrainTestSplitWrapper
        | utils.NoopSplitterWrapper
    ):
        if estimation_method == _EstimationMethod.CROSS_VALIDATION:
            if has_custom_validation_set:
                raise ValueError(
                    "Custom validation set was provided while cross validation was enabled for early stopping. Please set early_stopping_use_crossvalidation to False or remove df_val."
                )

            logger.info("Running early stopping using Cross Validation.")
            train_test_splitter = self._cv_splitter
        else:
            if not has_custom_validation_set:
                logger.info(
                    f"Running early stopping using holdout set of size {self.VALID_SIZE}."
                )
                train_test_splitter = self._holdout_splitter
            else:
                logger.info("Running early stopping using provided validation set.")
                train_test_splitter = self._noop_splitter

        return train_test_splitter

    def _determine_n_folds(
        self,
        estimation_method: _EstimationMethod,
    ) -> int:
        if estimation_method == _EstimationMethod.CROSS_VALIDATION:
            n_folds = self.n_folds
            logger.info(f"Using {n_folds} folds for cross-validation.")
        else:
            n_folds = 1
        return n_folds

    def _determine_best_num_rounds(
        self,
        data_train: _MCGradProcessedData,
        data_val: _MCGradProcessedData | None = None,
    ) -> int:
        logger.info("Determining optimal number of rounds")
        if data_train.labels is None:
            raise ValueError("_determine_best_num_rounds() requires labels.")

        estimation_method = self._determine_estimation_method(data_train.weights)
        train_test_splitter = self._determine_train_test_splitter(
            estimation_method,
            data_val is not None,
        )
        final_n_folds = self._determine_n_folds(estimation_method)

        patience_counter = 0

        num_rounds = 0
        best_num_rounds = 0

        mcgrad_per_fold: Dict[int, _BaseMCGrad] = {}
        predictions_per_fold: Dict[int, npt.NDArray] = {}

        best_score = -np.inf

        start_time = time.time()

        while num_rounds <= self.num_rounds and patience_counter <= self.patience:
            log_add = ""
            if num_rounds == 0:
                log_add = " (input prediction for early stopping baseline)"
            logger.info(f"Evaluating round {num_rounds}{log_add}")

            if self.early_stopping_timeout is not None and self._get_elapsed_time(
                start_time
            ) > cast(int, self.early_stopping_timeout):
                logger.warning(
                    f"Stopping early stopping upon exceeding the {self.early_stopping_timeout:,}-second timeout; "
                    + f"{self.__class__.__name__} results will likely improve by increasing `early_stopping_timeout` or setting it to None"
                )
                break

            valid_monitored_metrics_per_round = np.zeros(
                (len(self.monitored_metrics_during_training), final_n_folds),
                dtype=float,
            )
            train_monitored_metrics_per_round = np.zeros(
                (len(self.monitored_metrics_during_training), final_n_folds),
                dtype=float,
            )

            fold_num = 0
            for train_index, valid_index in train_test_splitter.split(
                data_train.features, data_train.labels
            ):
                data_train_cv = data_train[train_index]
                data_valid_cv = data_val or data_train[valid_index]

                if num_rounds == 0:
                    train_fold_preds = self._inverse_transform_predictions(
                        data_train_cv.predictions
                    )
                    valid_fold_preds = self._inverse_transform_predictions(
                        data_valid_cv.predictions
                    )
                else:
                    if fold_num not in mcgrad_per_fold:
                        mcgrad = self._create_instance_for_cv(
                            encode_categorical_variables=self.encode_categorical_variables,
                            monotone_t=self.monotone_t,
                            lightgbm_params=self.lightgbm_params,
                            early_stopping=False,
                            num_rounds=0,
                        )
                        mcgrad_per_fold[fold_num] = mcgrad
                        predictions_per_fold[fold_num] = data_train_cv.predictions

                    new_predictions = mcgrad_per_fold[
                        fold_num
                    ]._fit_single_round(
                        x=data_train_cv.features,
                        y=data_train_cv.labels,  # pyre-ignore[6]: we assert that data_train_cv.labels is not None above
                        prediction=predictions_per_fold[fold_num],
                        w=data_train_cv.weights,
                        categorical_feature_column_names=data_train_cv.categorical_feature_names,
                        numerical_feature_column_names=data_train_cv.numerical_feature_names,
                    )
                    predictions_per_fold[fold_num] = new_predictions
                    if self.save_training_performance:
                        train_fold_preds = mcgrad_per_fold[fold_num]._predict(
                            x=data_train_cv.features,
                            transformed_predictions=data_train_cv.predictions,
                            return_all_rounds=False,
                        )

                    valid_fold_preds = mcgrad_per_fold[fold_num]._predict(
                        x=data_valid_cv.features,
                        transformed_predictions=data_valid_cv.predictions,
                        return_all_rounds=False,
                    )

                for metric_idx, monitored_metric in enumerate(
                    self.monitored_metrics_during_training
                ):
                    valid_monitored_metrics_per_round[metric_idx, fold_num] = (
                        self._compute_metric_on_internal_data(
                            monitored_metric,
                            data_valid_cv,
                            valid_fold_preds,
                        )
                    )
                    if self.save_training_performance:
                        train_monitored_metrics_per_round[metric_idx, fold_num] = (
                            self._compute_metric_on_internal_data(
                                monitored_metric,
                                data_train_cv,
                                train_fold_preds,  # pyre-ignore[61]: train_fold_preds is not None whenever self.save_training_performance is True
                            )
                        )

                logger.debug(f"Evaluated on fold {fold_num}")
                fold_num += 1

            valid_mean_scores = np.mean(valid_monitored_metrics_per_round, axis=1)
            train_mean_scores = np.mean(train_monitored_metrics_per_round, axis=1)

            for metric_idx, monitored_metric in enumerate(
                self.monitored_metrics_during_training
            ):
                self._performance_metrics[
                    f"avg_valid_performance_{monitored_metric.name}"
                ].append(valid_mean_scores[metric_idx])
                if self.save_training_performance:
                    self._performance_metrics[
                        f"avg_train_performance_{monitored_metric.name}"
                    ].append(train_mean_scores[metric_idx])
                if monitored_metric.name != self.early_stopping_score_func.name:
                    logger.info(
                        f"{monitored_metric.name} on validation set: {valid_mean_scores[metric_idx]:.4f}"
                    )

            early_stopping_metric_value = self._performance_metrics[
                f"avg_valid_performance_{self.early_stopping_score_func.name}"
            ][-1]

            current_score = (
                -early_stopping_metric_value
                if self.early_stopping_minimize_score
                else early_stopping_metric_value
            )

            if current_score > best_score:
                best_score = current_score
                best_num_rounds = num_rounds
                patience_counter = 0
            else:
                patience_counter += 1

            best_early_stopping_metric_value = (
                (-best_score if best_score != -np.inf else np.inf)
                if self.early_stopping_minimize_score
                else best_score
            )
            logger.info(
                f"Round {num_rounds}: validation loss = {early_stopping_metric_value:.4f} (best: {best_early_stopping_metric_value:.4f}, patience: {patience_counter}/{self.patience})"
            )

            num_rounds += 1

        if best_num_rounds == 0:
            logger.warning(
                f"Selected 0 to be the best number of rounds for {self.__class__.__name__} for this dataset, meaning that uncalibrated predictions will be returned. This is because the optimization metric did not improve during the first round of boosting."
            )
        elif best_num_rounds == self.num_rounds:
            logger.warning(
                f"max_num_rounds might be too low: best performance was at the maximum number of rounds ({self.num_rounds})"
            )

        logger.info(f"Determined {best_num_rounds} to be best number of rounds")

        for monitored_metric in self.monitored_metrics_during_training:
            if monitored_metric.name == "Multicalibration Error<br>(mce_sigma_scale)":
                mce_at_best_num_rounds = self._performance_metrics[
                    f"avg_valid_performance_{monitored_metric.name}"
                ][best_num_rounds]
                mce_at_initial_round = self._performance_metrics[
                    f"avg_valid_performance_{monitored_metric.name}"
                ][0]

                self.mce_below_initial = mce_at_best_num_rounds < mce_at_initial_round
                self.mce_below_strong_evidence_threshold = (
                    mce_at_best_num_rounds < self.MCE_STRONG_EVIDENCE_THRESHOLD
                )

                if not self.mce_below_strong_evidence_threshold:
                    logger.warning(
                        f"The final Multicalibration Error on the validation set after using {self.__class__.__name__} is {mce_at_best_num_rounds}. This is higher than 4.0, which still indicates strong evidence for miscalibration."
                    )
                if not self.mce_below_initial:
                    logger.warning(
                        f"The final Multicalibration Error on the validation set after using {self.__class__.__name__} is {mce_at_best_num_rounds}, which is not lower than the initial Multicalibration Error of {mce_at_initial_round}. This indicates that {self.__class__.__name__} did not improve the multi-calibration of the model."
                    )

        return best_num_rounds

    def _compute_metric_on_internal_data(
        self,
        metric: _ScoreFunctionInterface,
        data: _MCGradProcessedData,
        predictions: npt.NDArray,
    ) -> float:
        """
        Compatibility wrapper for MCGradProcessedData -> ScoreFunctionInterface.
        """
        feature_columns = data.categorical_feature_names + data.numerical_feature_names
        df = pd.DataFrame(
            data.features,
            columns=feature_columns,
        )
        df["label"] = data.labels
        df["prediction"] = predictions
        df["weight"] = data.weights
        return metric(
            df=df,
            label_column="label",
            score_column="prediction",
            weight_column="weight",
        )

    def _get_elapsed_time(self, start_time: float) -> int:
        """
        Returns the elapsed time since the given start time in seconds.
        """
        return int(time.time() - start_time)

    def serialize(self) -> str:
        """Serializes the fitted MCGrad model to a JSON string.

        The serialized model includes all boosters, unshrink factors, encoder state, and configuration parameters,
        allowing the model to be saved and restored later.

        :return: JSON string containing the serialized model
        """
        serialized_boosters = [booster.model_to_string() for booster in self.mr]
        json_obj: dict[str, Any] = {
            self._SERIALIZATION_KEY: [
                {
                    "booster": serialized_booster,
                    "unshrink_factor": unshrink_factor,
                }
                for serialized_booster, unshrink_factor in zip(
                    serialized_boosters, self.unshrink_factors
                )
            ],
            "params": {
                "allow_missing_segment_feature_values": self.allow_missing_segment_feature_values,
            },
        }
        json_obj["has_encoder"] = self.encode_categorical_variables
        if hasattr(self, "enc") and self.enc is not None:
            json_obj["encoder"] = self.enc.serialize()
        return json.dumps(json_obj)

    @classmethod
    def _create_instance_for_cv(cls, **kwargs: Any) -> Self:
        return cls(**kwargs)

    @classmethod
    def deserialize(cls, model_str: str) -> Self:
        """Deserializes an MCGrad model from a JSON string.

        Reconstructs a fitted MCGrad model from a previously serialized representation.

        :param model_str: JSON string containing the serialized model
        :return: A fitted MCGrad instance with all state restored
        """
        json_obj = json.loads(model_str)
        model = cls()
        model.mr = []
        model.unshrink_factors = []

        for model_info in json_obj[cls._SERIALIZATION_KEY]:
            booster = lgb.Booster(model_str=model_info["booster"])
            model.mr.append(booster)
            model.unshrink_factors.append(model_info["unshrink_factor"])

        model.num_rounds = len(model.mr)

        model.encode_categorical_variables = json_obj["has_encoder"]
        if json_obj["has_encoder"] and "encoder" in json_obj:
            model.enc = utils.OrdinalEncoderWithUnknownSupport.deserialize(
                json_obj["encoder"]
            )

        return model

    def _compute_effective_sample_size(self, weights: npt.NDArray) -> int:
        """
        Computes the effective sample size for the given weights.
        The effective sample size is defined as square of the sum of weights over the sum of the squared weights,
        as common in the importance sampling literature (e.g., see https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-024-02412-1).

        :param weights: weights for each sample.
        :return: effective sample size.
        """
        # Compute the effective sample size using the weights
        return (weights.sum() ** 2) / np.power(weights, 2).sum()

    def _determine_estimation_method(self, weights: npt.NDArray) -> _EstimationMethod:
        """
        Returns the estimation method to use for early stopping given the arguments and the weights (when relevant).
        This is especially useful for the AUTO option, where we infer the proper estimation method to use based on the effective sample size.

        :return: the estimation method to use.
        """
        if self.early_stopping_estimation_method != _EstimationMethod.AUTO:
            return self.early_stopping_estimation_method

        if self.early_stopping_score_func.name != "log_loss":
            # Automatically infer the estimation method only when using the logistic loss, otherwise use k-fold.
            # This is because we analyzed the effective sample size specifically with log_loss.
            return _EstimationMethod.CROSS_VALIDATION

        # We use a rule-of-thumb to determine whether to use cross-validation or holdout for early stopping.
        # Namely, if the effective sample size is less than 2.5M, we use cross-validation, otherwise we use holdout.
        ess = self._compute_effective_sample_size(weights)

        if ess < self.ESS_THRESHOLD_FOR_CROSS_VALIDATION:
            logger.info(
                f"Found a relatively small effective sample size ({ess:,}), choosing k-fold for early stopping. "
                + "You can override this by explicitly setting `early_stopping_use_crossvalidation` to `False`."
            )
            return _EstimationMethod.CROSS_VALIDATION
        else:
            logger.info(
                f"Found a large enough effective sample size ({ess:,}), choosing holdout for early stopping. "
                + "You can override this by explicitly setting `early_stopping_use_crossvalidation` to `True`."
            )
            return _EstimationMethod.HOLDOUT


class MCGrad(_BaseMCGrad):
    """
    MCGrad (Multicalibration Gradient Boosting) as described in [1].

    References:

    [1] Tax, N., Perini, L., Linder, F., Haimovich, D., Karamshuk, D., Okati, N., Vojnovic, M.,
      & Apostolopoulos, P. A. (2026). MCGrad: Multicalibration at Web Scale.
      In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2026).
      https://doi.org/10.1145/3770854.3783954
    - arXiv preprint: https://arxiv.org/abs/2509.19884
    """

    UNSHRINK_LOGIT_EPSILON = 10

    DEFAULT_HYPERPARAMS: dict[str, Any] = {
        "monotone_t": False,
        "early_stopping": True,
        "patience": 0,
        "n_folds": 5,
        "lightgbm_params": {
            "learning_rate": 0.028729759162731475,
            "max_depth": 5,
            "min_child_samples": 160,
            "n_estimators": 94,
            "num_leaves": 5,
            "lambda_l2": 0.009131373863997217,
            "min_gain_to_split": 0.15007305226251808,
        },
    }

    @staticmethod
    def _predictions_out_of_bounds(predictions: npt.NDArray) -> npt.NDArray:
        return (predictions < 0) | (predictions > 1)

    @staticmethod
    def _transform_predictions(predictions: npt.NDArray) -> npt.NDArray:
        return utils.logit(predictions)

    @staticmethod
    def _inverse_transform_predictions(transformed: npt.NDArray) -> npt.NDArray:
        return utils.logistic_vectorized(transformed)

    @staticmethod
    def _compute_unshrink_factor(
        y: npt.NDArray, predictions: npt.NDArray, w: npt.NDArray | None
    ) -> float:
        return utils.unshrink(
            y, predictions, w, logit_epsilon=MCGrad.UNSHRINK_LOGIT_EPSILON
        )

    @property
    def _objective(self) -> str:
        return "binary"

    @property
    def _default_early_stopping_metric(
        self,
    ) -> tuple[_ScoreFunctionInterface, bool]:
        return wrap_sklearn_metric_func(skmetrics.log_loss), True

    def _check_predictions(
        self, df_train: pd.DataFrame, prediction_column_name: str
    ) -> None:
        predictions = df_train[prediction_column_name].to_numpy()
        if self._predictions_out_of_bounds(predictions).any():
            raise ValueError(
                "Predictions must be probabilities in the (0, 1) interval. "
                f"Found predictions outside this range: min={predictions.min()}, max={predictions.max()}"
            )
        if df_train[prediction_column_name].isnull().any():
            raise ValueError(
                f"{self.__class__.__name__} does not support missing values in the prediction column, but {df_train[prediction_column_name].isnull().sum()}"
                f" of {len(df_train[prediction_column_name])} are null."
            )

        lower_prob_bound = utils.logistic(-self.UNSHRINK_LOGIT_EPSILON)
        upper_prob_bound = utils.logistic(self.UNSHRINK_LOGIT_EPSILON)
        num_out_of_bounds = np.sum(
            (predictions < lower_prob_bound) | (predictions > upper_prob_bound)
        )
        if num_out_of_bounds > 0:
            pct_out_of_bounds = 100.0 * num_out_of_bounds / len(predictions)
            logger.warning(
                f"Found {num_out_of_bounds} ({pct_out_of_bounds:.2f}%) predictions with extreme values (boundaries: [{lower_prob_bound:.6g}, {upper_prob_bound:.6g}]). "
                f"These samples will be clipped in the unshrink step. Consider reviewing input prediction quality."
            )

    def _check_labels(self, df_train: pd.DataFrame, label_column_name: str) -> None:
        if df_train[label_column_name].isnull().any():
            raise ValueError(
                f"{self.__class__.__name__} does not support missing values in the label column, but {df_train[label_column_name].isnull().sum()}"
                f" of {len(df_train[label_column_name])} are null."
            )
        unique_labels = list(df_train[label_column_name].unique())
        labels_are_valid_int = df_train[label_column_name].isin([0, 1]).all()
        labels_are_valid_bool = df_train[label_column_name].isin([True, False]).all()
        if not (labels_are_valid_bool or labels_are_valid_int):
            raise ValueError(
                f"Labels in column `{label_column_name}` must be binary, either 0/1 or True/False. Got {unique_labels=}"
            )
        if not len(unique_labels) == 2:
            raise ValueError(
                f"Labels in column `{label_column_name}` must have at least 2 values but the data contains only 1: {unique_labels=}"
            )

    @property
    def _cv_splitter(self) -> StratifiedKFold:
        return StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self._next_seed(),
        )

    @property
    def _holdout_splitter(self) -> utils.TrainTestSplitWrapper:
        return utils.TrainTestSplitWrapper(
            test_size=self.VALID_SIZE,
            shuffle=True,
            random_state=self._next_seed(),
            stratify=True,
        )

    @property
    def _noop_splitter(
        self,
    ) -> utils.NoopSplitterWrapper:
        return utils.NoopSplitterWrapper()


class RegressionMCGrad(_BaseMCGrad):
    """
    Regression variant of MCGrad for continuous label calibration.

    Note that automatic determination of train/test split vs. cross validation is currently not supported for Regression.
    """

    DEFAULT_HYPERPARAMS: dict[str, Any] = {
        "monotone_t": False,
        "early_stopping": True,
        "patience": 0,
        "n_folds": 5,
        # All lightgbm_params set to default values of LightGBM, https://lightgbm.readthedocs.io/en/latest/Parameters.html
        "lightgbm_params": {
            "learning_rate": 0.1,
            "max_depth": -1,
            "min_child_samples": 20,
            "n_estimators": 100,
            "num_leaves": 31,
            "min_gain_to_split": 0,
        },
    }

    @staticmethod
    def _predictions_out_of_bounds(predictions: npt.NDArray) -> npt.NDArray:
        return np.isnan(predictions) | np.isinf(predictions)

    @staticmethod
    def _transform_predictions(predictions: npt.NDArray) -> npt.NDArray:
        return predictions.astype(float)

    @staticmethod
    def _inverse_transform_predictions(transformed: npt.NDArray) -> npt.NDArray:
        return transformed

    @staticmethod
    def _compute_unshrink_factor(
        y: npt.NDArray, predictions: npt.NDArray, w: npt.NDArray | None
    ) -> float:
        if w is None:
            w = np.ones_like(y)
        predictions_reshaped = predictions.reshape(-1, 1)

        solver = LinearRegression(fit_intercept=False)
        solver.fit(predictions_reshaped, y, sample_weight=w)
        # pyre-ignore[16]: `LinearRegression` has coef_ attribute after fitting
        return solver.coef_[0]

    @property
    def _objective(self) -> str:
        return "regression"

    @property
    def _default_early_stopping_metric(
        self,
    ) -> tuple[_ScoreFunctionInterface, bool]:
        return wrap_sklearn_metric_func(skmetrics.mean_squared_error), True

    def _check_predictions(
        self, df_train: pd.DataFrame, prediction_column_name: str
    ) -> None:
        predictions = df_train[prediction_column_name]
        if predictions.isnull().any():
            raise ValueError(
                f"{self.__class__.__name__} does not support missing values in the prediction column, but {predictions.isnull().sum()}"
                f" of {len(predictions)} are null."
            )
        if np.isinf(predictions).any():
            raise ValueError(
                f"{self.__class__.__name__} does not support infinite values in the prediction column, but {np.sum(np.isinf(predictions))}"
                f" of {len(predictions)} are null."
            )

    def _check_labels(self, df_train: pd.DataFrame, label_column_name: str) -> None:
        labels = df_train[label_column_name]
        if not pd.api.types.is_numeric_dtype(labels):
            raise ValueError(
                f"{self.__class__.__name__} only supports numeric labels, but {label_column_name} has type {labels.dtype}."
            )
        if labels.isnull().any() or labels.isna().any():
            raise ValueError(
                f"{self.__class__.__name__} does not support missing values in the label column, but {labels.isnull().sum()}"
                f" of {len(labels)} are null."
            )
        if np.isinf(labels).any():
            raise ValueError(
                f"{self.__class__.__name__} does not support infinite values in the prediction column, but {np.sum(np.isinf(labels))}"
                f" of {len(labels)} are null."
            )
        if labels.nunique() < 2:
            raise ValueError(
                f"{self.__class__.__name__} requires at least 2 unique values in the label column, but {label_column_name} has only {labels.nunique()}."
            )

    @property
    def _cv_splitter(self) -> KFold:
        return KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self._next_seed(),
        )

    @property
    def _holdout_splitter(self) -> utils.TrainTestSplitWrapper:
        return utils.TrainTestSplitWrapper(
            test_size=self.VALID_SIZE,
            shuffle=True,
            random_state=self._next_seed(),
            stratify=False,
        )

    @property
    def _noop_splitter(
        self,
    ) -> utils.NoopSplitterWrapper:
        return utils.NoopSplitterWrapper()


# @oss-disable[end= ]: class MCBoost(
    # @oss-disable[end= ]: MCGrad,
    # @oss-disable[end= ]: DeprecatedAttributesMixin,
# @oss-disable[end= ]: ):
    # @oss-disable[end= ]: _SERIALIZATION_KEY = "mcboost"


# @oss-disable[end= ]: class RegressionMCBoost(
    # @oss-disable[end= ]: RegressionMCGrad,
    # @oss-disable[end= ]: DeprecatedAttributesMixin,
# @oss-disable[end= ]: ):
    # @oss-disable[end= ]: _SERIALIZATION_KEY = "mcboost"


class PlattScaling(BaseCalibrator):
    """Platt scaling calibration method.

    Platt scaling fits a logistic regression model to transform uncalibrated predictions into
    calibrated probabilities. Given an uncalibrated prediction :math:`\\hat{p}`, it first converts
    to log-odds (logit): :math:`t = \\log(\\hat{p} / (1 - \\hat{p}))`, then fits the model:

    .. math::

        P(y=1 | t) = \\sigma(a \\cdot t + b)

    where :math:`\\sigma` is the sigmoid function and :math:`a, b` are learned parameters.
    This is equivalent to fitting a logistic regression with a single feature (the logit of the
    original prediction).

    References:

    - Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized
      likelihood methods. Advances in large margin classifiers, 10(3), 61-74.
    - Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning.
      International Conference on Machine Learning (ICML). pp. 625-632.
    """

    def __init__(self) -> None:
        self.log_reg: LogisticRegression | None = None

    def fit(
        self,
        df_train: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the Platt scaling model on the provided training data.

        :param df_train: The dataframe containing the training data
        :param prediction_column_name: Name of the column in dataframe df that contains the predictions
        :param label_column_name: Name of the column in dataframe df that contains the ground truth labels
        :param weight_column_name: Name of the column in dataframe df that contains the instance weights
        :param categorical_feature_column_names: Ignored for Platt scaling (no multicalibration)
        :param numerical_feature_column_names: Ignored for Platt scaling (no multicalibration)
        :param kwargs: Additional keyword arguments
        :return: The fitted calibrator instance
        """
        y = df_train[label_column_name].values.astype(float)
        y_hat = df_train[prediction_column_name].values.astype(float)
        w = df_train[weight_column_name] if weight_column_name else np.ones_like(y)

        logits = utils.logit(y_hat).reshape(-1, 1)
        if len(np.unique(y)) < 2:
            self.log_reg = None
        else:
            log_reg = LogisticRegression(C=np.inf)
            # Suppress sklearn 1.8+ UserWarning which is a known bug. Will be fixed in sklearn 1.8.1
            # See: https://github.com/scikit-learn/scikit-learn/issues/32927
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Setting penalty=None will ignore the C.*",
                    category=UserWarning,
                )
                log_reg.fit(logits, y, sample_weight=w)
            self.log_reg = log_reg
        return self

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Apply the Platt scaling model to a DataFrame.

        This requires the `fit` method to have been previously called on this calibrator object.

        :param df: The dataframe containing the data to calibrate
        :param prediction_column_name: Name of the column in dataframe df that contains the predictions
        :param categorical_feature_column_names: Ignored for Platt scaling (no multicalibration)
        :param numerical_feature_column_names: Ignored for Platt scaling (no multicalibration)
        :param kwargs: Additional keyword arguments
        :return: Array of calibrated predictions
        """
        y_hat = df[prediction_column_name].values.astype(float)

        log_reg = self.log_reg
        if log_reg is None:
            return y_hat

        logits = utils.logit(y_hat).reshape(-1, 1)
        return log_reg.predict_proba(logits)[:, 1]


class IsotonicRegression(BaseCalibrator):
    """Isotonic regression calibration method.

    Isotonic regression fits a non-decreasing step function that minimizes the mean squared error
    between calibrated predictions and true labels, subject to a monotonicity constraint.
    Given uncalibrated predictions :math:`\\hat{p}_i` and labels :math:`y_i`, it finds:

    .. math::

        \\min_{f} \\sum_{i} (y_i - f(\\hat{p}_i))^2 \\quad \\text{subject to} \\quad f(\\hat{p}_i) \\leq f(\\hat{p}_j) \\text{ whenever } \\hat{p}_i \\leq \\hat{p}_j

    The result is a piecewise-constant function that maps predictions to calibrated probabilities.
    For input values outside of the training domain, predictions are clipped to the value
    corresponding to the nearest training interval endpoint.

    References:

    - Zadrozny, B., & Elkan, C. (2001). Obtaining calibrated probability estimates from decision trees and
      naive bayesian classifiers. International Conference on Machine Learning (ICML). pp. 609-616.
    - Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning.
      International Conference on Machine Learning (ICML). pp. 625-632.
    """

    def __init__(self) -> None:
        """Initializes an IsotonicRegression calibrator.

        Creates an isotonic regression model that enforces monotonicity constraints. For input values outside
        of the training domain, predictions are set to the value corresponding to the nearest training interval endpoint.
        """
        self.isoreg = isotonic.IsotonicRegression()

    def fit(
        self,
        df_train: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the isotonic regression calibration model on the provided training data.

        :param df_train: The dataframe containing the training data
        :param prediction_column_name: Name of the column in dataframe df that contains the predictions
        :param label_column_name: Name of the column in dataframe df that contains the ground truth labels
        :param weight_column_name: Name of the column in dataframe df that contains the instance weights
        :param categorical_feature_column_names: Ignored for isotonic regression (no multicalibration)
        :param numerical_feature_column_names: Ignored for isotonic regression (no multicalibration)
        :param kwargs: Additional keyword arguments
        :return: The fitted calibrator instance
        """
        y = df_train[label_column_name].values.astype(float)
        y_hat = df_train[prediction_column_name].values.astype(float)
        w = df_train[weight_column_name] if weight_column_name else np.ones_like(y)

        # out_of_bounds=clip ensures predictions outside training domain range are clipped to nearest valid value instead of NaN
        # These are set to nearest train interval endpoints
        self.isoreg = isotonic.IsotonicRegression(out_of_bounds="clip").fit(
            y_hat, y, sample_weight=w
        )
        return self

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Apply the isotonic regression calibration model to a DataFrame.

        This requires the `fit` method to have been previously called on this calibrator object.

        :param df: The dataframe containing the data to calibrate
        :param prediction_column_name: Name of the column in dataframe df that contains the predictions
        :param categorical_feature_column_names: Ignored for isotonic regression (no multicalibration)
        :param numerical_feature_column_names: Ignored for isotonic regression (no multicalibration)
        :param kwargs: Additional keyword arguments
        :return: Array of calibrated predictions
        """
        y_hat = df[prediction_column_name].values.astype(float)
        return self.isoreg.transform(y_hat)


class MultiplicativeAdjustment(BaseCalibrator):
    """
    Calibrates predictions by applying a multiplicative correction factor.

    This method computes a scalar multiplier :math:`m` that aligns the sum of predictions with
    the sum of labels. Given predictions :math:`\\hat{p}_i`, labels :math:`y_i`, and optional
    weights :math:`w_i`, the multiplier is computed as:

    .. math::

        m = \\frac{\\sum_i w_i y_i}{\\sum_i w_i \\hat{p}_i}

    The calibrated predictions are then :math:`m \\cdot \\hat{p}_i`.
    This is useful when predictions are directionally correct but systematically over- or under-estimated.
    """

    def __init__(self, clip_to_zero_one: bool = True) -> None:
        """
        :param clip_to_zero_one: If True, clips calibrated predictions to the [0, 1] range.
        """
        self.multiplier: float | None = None
        self.clip_to_zero_one = clip_to_zero_one

    def fit(
        self,
        df_train: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the multiplicative adjustment calibration model on the provided training data.

        :param df_train: The dataframe containing the training data
        :param prediction_column_name: Name of the column in dataframe df that contains the predictions
        :param label_column_name: Name of the column in dataframe df that contains the ground truth labels
        :param weight_column_name: Name of the column in dataframe df that contains the instance weights
        :param categorical_feature_column_names: Ignored for multiplicative adjustment (no multicalibration)
        :param numerical_feature_column_names: Ignored for multiplicative adjustment (no multicalibration)
        :param kwargs: Additional keyword arguments
        :return: The fitted calibrator instance
        """
        w = (
            df_train[weight_column_name]
            if weight_column_name
            else np.ones(df_train.shape[0])
        )
        total_score = (w * df_train[prediction_column_name]).sum()
        total_positive = (w * df_train[label_column_name]).sum()
        self.multiplier = total_positive / total_score if total_score != 0 else 1.0
        return self

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Apply the multiplicative adjustment calibration model to a DataFrame.

        This requires the `fit` method to have been previously called on this calibrator object.

        :param df: The dataframe containing the data to calibrate
        :param prediction_column_name: Name of the column in dataframe df that contains the predictions
        :param categorical_feature_column_names: Ignored for multiplicative adjustment (no multicalibration)
        :param numerical_feature_column_names: Ignored for multiplicative adjustment (no multicalibration)
        :param kwargs: Additional keyword arguments
        :return: Array of calibrated predictions
        """
        preds = df[prediction_column_name].values * self.multiplier
        if self.clip_to_zero_one:
            preds = np.clip(preds, 0, 1)
        return preds


class AdditiveAdjustment(BaseCalibrator):
    """
    Calibrates predictions by adding a constant correction term.

    This method computes a scalar offset :math:`c` that aligns the weighted average of predictions
    with the weighted average of labels. Given predictions :math:`\\hat{p}_i`, labels :math:`y_i`,
    and optional weights :math:`w_i`, the offset is computed as:

    .. math::

        c = \\frac{\\sum_i w_i (y_i - \\hat{p}_i)}{\\sum_i w_i}

    The calibrated predictions are then :math:`\\hat{p}_i + c`.
    This is useful when predictions have an approximately constant bias that needs correction.
    """

    def __init__(self, clip_to_zero_one: bool = True) -> None:
        """
        :param clip_to_zero_one: If True, clips calibrated predictions to the [0, 1] range.
        """
        self.offset: float | None = None
        self.clip_to_zero_one = clip_to_zero_one

    def fit(
        self,
        df_train: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the additive adjustment calibration model on the provided training data.

        :param df_train: The dataframe containing the training data
        :param prediction_column_name: Name of the column in dataframe df that contains the predictions
        :param label_column_name: Name of the column in dataframe df that contains the ground truth labels
        :param weight_column_name: Name of the column in dataframe df that contains the instance weights
        :param categorical_feature_column_names: Ignored for additive adjustment (no multicalibration)
        :param numerical_feature_column_names: Ignored for additive adjustment (no multicalibration)
        :param kwargs: Additional keyword arguments
        :return: The fitted calibrator instance
        """
        w = (
            df_train[weight_column_name]
            if weight_column_name
            else np.ones(df_train.shape[0])
        )
        total_score = (w * df_train[prediction_column_name]).sum()
        total_positive = (w * df_train[label_column_name]).sum()
        sum_w = w.sum()
        if sum_w == 0:
            self.offset = 0.0
        else:
            self.offset = (total_positive - total_score) / sum_w
        return self

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Apply the additive adjustment calibration model to a DataFrame.

        This requires the `fit` method to have been previously called on this calibrator object.

        :param df: The dataframe containing the data to calibrate
        :param prediction_column_name: Name of the column in dataframe df that contains the predictions
        :param categorical_feature_column_names: Ignored for additive adjustment (no multicalibration)
        :param numerical_feature_column_names: Ignored for additive adjustment (no multicalibration)
        :param kwargs: Additional keyword arguments
        :return: Array of calibrated predictions
        """
        preds = df[prediction_column_name].values + self.offset
        if self.clip_to_zero_one:
            preds = np.clip(preds, 0, 1)
        return preds


class IdentityCalibrator(BaseCalibrator):
    """
    A pass-through calibrator that returns predictions unchanged. Useful as a baseline or fallback option.
    """

    def fit(
        self,
        df_train: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the identity calibrator (no-op, returns uncalibrated predictions).

        :param df_train: The dataframe containing the training data (ignored)
        :param prediction_column_name: Name of the column in dataframe df that contains the predictions (ignored)
        :param label_column_name: Name of the column in dataframe df that contains the ground truth labels (ignored)
        :param weight_column_name: Name of the column in dataframe df that contains the instance weights (ignored)
        :param categorical_feature_column_names: Ignored
        :param numerical_feature_column_names: Ignored
        :param kwargs: Additional keyword arguments (ignored)
        :return: The calibrator instance
        """
        return self

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Apply the identity calibrator (returns uncalibrated predictions).

        :param df: The dataframe containing the data
        :param prediction_column_name: Name of the column in dataframe df that contains the predictions
        :param categorical_feature_column_names: Ignored
        :param numerical_feature_column_names: Ignored
        :param kwargs: Additional keyword arguments (ignored)
        :return: Array of uncalibrated predictions
        """
        return df[prediction_column_name].values


class PlattScalingWithFeatures(BaseCalibrator):
    """
    A variant of Platt scaling that incorporates additional features alongside the log-odds.

    This calibrator fits a logistic regression model using the log-odds of the original prediction
    plus additional features derived from categorical and numerical columns. Given an uncalibrated
    prediction :math:`\\hat{p}` and feature vector :math:`\\mathbf{x}`, it fits the model:

    .. math::

        P(y=1 | \\hat{p}, \\mathbf{x}) = \\sigma(a \\cdot t + \\mathbf{w}^T \\mathbf{x} + b)

    where :math:`t = \\log(\\hat{p} / (1 - \\hat{p}))` is the logit transformation,
    :math:`\\sigma` is the sigmoid function, :math:`a` is the coefficient for the logit,
    :math:`\\mathbf{w}` are the coefficients for the features, and :math:`b` is the intercept.

    Categorical features are one-hot encoded and numerical features are discretized into 3 quantile bins
    before fitting. This allows the calibration to vary across different feature values while still
    learning a single unified model (unlike :class:`SegmentwiseCalibrator` which fits completely
    separate models per segment).
    """

    def __init__(self) -> None:
        self.log_reg: LogisticRegression | None = None
        self.logits_column_name = "__logits"
        self.ohe: OneHotEncoder | None = None
        self.kbd: KBinsDiscretizer | None = None
        self.ohe_columns: list[str] | None = None
        self.kbd_columns: list[str] | None = None
        self.features: list[str] | None = None

    def _fit_feature_encoders(
        self,
        df: pd.DataFrame,
        categorical_feature_column_names: list[str] | None,
        numerical_feature_column_names: list[str] | None,
    ) -> None:
        if categorical_feature_column_names:
            self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.ohe.fit(df[categorical_feature_column_names])
        else:
            self.ohe = None

        if numerical_feature_column_names:
            self.kbd = create_kbins_discretizer(
                encode="onehot-dense", n_bins=3, subsample=None
            )
            self.kbd.fit(df[numerical_feature_column_names])
        else:
            self.kbd = None

    def _convert_df(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None,
        numerical_feature_column_names: list[str] | None,
    ) -> pd.DataFrame:
        y_hat = df[prediction_column_name].values.astype(float)
        df[self.logits_column_name] = utils.logit(y_hat)
        if categorical_feature_column_names and self.ohe is not None:
            ohe_df = pd.DataFrame(
                self.ohe.transform(df[categorical_feature_column_names])
            )
            if hasattr(self.ohe, "get_feature_names"):
                ohe_df.columns = self.ohe.get_feature_names(  # pyre-ignore: Maintain compatibility with sklearn <1.0
                    categorical_feature_column_names
                )
            elif hasattr(self.ohe, "get_feature_names_out"):
                ohe_df.columns = self.ohe.get_feature_names_out(  # pyre-ignore
                    categorical_feature_column_names
                )
            else:
                raise ValueError(
                    "Could not obtain feature names from OneHotEncoder. Expected get_feature_names_out for sklearn >1.0 or get_feature_names for sklearn <1.0."
                )
            df = pd.concat([df, ohe_df], axis=1)
            self.ohe_columns = list(ohe_df.columns)
        else:
            self.ohe_columns = []

        if numerical_feature_column_names and self.kbd is not None:
            kbd_df = pd.DataFrame(
                self.kbd.transform(df[numerical_feature_column_names])
            )
            kbd_df.columns = [str(col) for col in kbd_df.columns]
            df = pd.concat([df, kbd_df], axis=1)
            self.kbd_columns = list(kbd_df.columns)
        else:
            self.kbd_columns = []

        return df

    def _train_model(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
    ) -> LogisticRegression | None:
        categorical_feature_column_names = self.ohe_columns or []
        numerical_feature_column_names = self.kbd_columns or []

        features = (
            [self.logits_column_name]
            + categorical_feature_column_names
            + numerical_feature_column_names
        )

        y = df[label_column_name].values.astype(float)

        w = (
            df[weight_column_name].values
            if weight_column_name
            else np.ones(df.shape[0])
        )
        if len(np.unique(y)) < 2:
            self.features = features
            return None

        log_reg = LogisticRegression(C=0.1).fit(df[features], y, sample_weight=w)
        self.features = features
        return log_reg

    def fit(
        self,
        df_train: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the Platt scaling with features model on the provided training data.

        :param df_train: The dataframe containing the training data
        :param prediction_column_name: Name of the column in dataframe df that contains the predictions
        :param label_column_name: Name of the column in dataframe df that contains the ground truth labels
        :param weight_column_name: Name of the column in dataframe df that contains the instance weights
        :param categorical_feature_column_names: List of column names in the df that contain the categorical
            segmentation features (these will be one-hot encoded)
        :param numerical_feature_column_names: List of column names in the df that contain the numerical
            segmentation features (these will be discretized into bins)
        :param kwargs: Additional keyword arguments
        :return: The fitted calibrator instance
        """
        df_train = df_train.copy().reset_index().fillna(0)
        self._fit_feature_encoders(
            df_train, categorical_feature_column_names, numerical_feature_column_names
        )

        df_train = self._convert_df(
            df_train,
            prediction_column_name,
            categorical_feature_column_names,
            numerical_feature_column_names,
        )

        log_reg = self._train_model(
            df_train,
            prediction_column_name,
            label_column_name,
            weight_column_name,
            categorical_feature_column_names,
            numerical_feature_column_names,
        )
        self.log_reg = log_reg
        return self

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Apply the Platt scaling with features model to a DataFrame.

        This requires the `fit` method to have been previously called on this calibrator object.

        :param df: The dataframe containing the data to calibrate
        :param prediction_column_name: Name of the column in dataframe df that contains the predictions
        :param categorical_feature_column_names: List of column names in the df that contain the categorical
            segmentation features (must match the features used during training)
        :param numerical_feature_column_names: List of column names in the df that contain the numerical
            segmentation features (must match the features used during training)
        :param kwargs: Additional keyword arguments
        :return: Array of calibrated predictions
        """
        df = df.copy().reset_index().fillna(0)

        df = self._convert_df(
            df=df,
            prediction_column_name=prediction_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
        )
        if self.log_reg is None:
            return df[prediction_column_name].values
        return self.log_reg.predict_proba(df[self.features])[:, 1]


# For backwards compatibility, we keep the original class name @oss-disable
# @oss-disable[end= ]: class SwissCheesePlattScaling(PlattScalingWithFeatures):
    # @oss-disable[end= ]: pass


TCalibrator = TypeVar("TCalibrator", bound=BaseCalibrator)


class SegmentwiseCalibrator(Generic[TCalibrator], BaseCalibrator):
    """
    A meta-calibrator that partitions data into segments based on categorical features and applies a separate calibration
    method to each segment. This enables more precise calibration when different segments require different calibration
    adjustments.

    Example::

        calibrator = SegmentwiseCalibrator(calibrator_class=PlattScaling)
        calibrator.fit(
            df_train,
            prediction_column_name="prediction",
            label_column_name="label",
            categorical_feature_column_names=["country"],
        )
        calibrated_predictions = calibrator.predict(
            df_test,
            prediction_column_name="prediction",
            categorical_feature_column_names=["country"],
        )

    This is equivalent to fitting a separate :class:`PlattScaling` model for each unique country value in the dataset.
    At prediction time, each sample is calibrated using the calibration model that was fit on its corresponding country
    segment. For unseen segments during prediction, the uncalibrated predictions are returned.
    """

    calibrator_per_segment: dict[str, BaseCalibrator]
    calibrator_class: type[TCalibrator]
    calibrator_kwargs: dict[str, Any]

    def __init__(
        self,
        calibrator_class: type[TCalibrator],
        calibrator_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        :param calibrator_class: The calibrator class to use for each segment (must be a subclass of BaseCalibrator)
        :param calibrator_kwargs: Optional keyword arguments to pass when instantiating calibrators for each segment
        """
        self.calibrator_class = calibrator_class
        self.calibrator_kwargs = calibrator_kwargs or {}

        # Check if calibrator_class can be instantiated with calibrator_kwargs
        try:
            self.calibrator_class(**self.calibrator_kwargs)
        except TypeError:
            raise ValueError(
                f"Unable to instantiate calibrator class {self.calibrator_class.__name__} with the provided keyword arguments: {str(calibrator_kwargs)}"
            )

        self.calibrator_per_segment = {}

    def fit(
        self,
        df_train: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit segment-specific calibration models on the provided training data.

        Data is partitioned into segments based on categorical features, and a separate calibrator is fit
        for each segment.

        :param df_train: The dataframe containing the training data
        :param prediction_column_name: Name of the column in dataframe df that contains the predictions
        :param label_column_name: Name of the column in dataframe df that contains the ground truth labels
        :param weight_column_name: Name of the column in dataframe df that contains the instance weights
        :param categorical_feature_column_names: List of column names in the df that contain the categorical
            segmentation features (passed to individual calibrators)
        :param numerical_feature_column_names: List of column names in the df that contain the numerical
            segmentation features (passed to individual calibrators)
        :param kwargs: Additional keyword arguments
        :return: The fitted calibrator instance
        """
        if categorical_feature_column_names is None:
            categorical_feature_column_names = []
        if numerical_feature_column_names is None:
            numerical_feature_column_names = []

        df_train = df_train.copy()
        df_train["segment"] = df_train[categorical_feature_column_names].apply(
            lambda row: repr(tuple(row.values)), axis=1
        )

        fit_segment_func = partial(
            self._fit_segment,
            prediction_column_name=prediction_column_name,
            label_column_name=label_column_name,
            weight_column_name=weight_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
        )
        groupby_apply(df_train.groupby("segment"), fit_segment_func)
        return self

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Apply segment-specific calibration models to a DataFrame.

        This requires the `fit` method to have been previously called on this calibrator object.
        For any unseen segments, the identity calibrator is used (returns uncalibrated predictions).

        :param df: The dataframe containing the data to calibrate
        :param prediction_column_name: Name of the column in dataframe df that contains the predictions
        :param categorical_feature_column_names: List of column names in the df that contain the categorical
            segmentation features (must match the features used during training)
        :param numerical_feature_column_names: List of column names in the df that contain the numerical
            segmentation features (must match the features used during training)
        :param kwargs: Additional keyword arguments
        :return: Array of calibrated predictions
        """
        if df.empty:
            return np.array([])

        if categorical_feature_column_names is None:
            categorical_feature_column_names = []
        if numerical_feature_column_names is None:
            numerical_feature_column_names = []

        df = df.copy()
        df["segment"] = df[categorical_feature_column_names].apply(
            lambda row: repr(tuple(row.values)), axis=1
        )

        predict_segment_func = partial(
            self._predict_segment,
            prediction_column_name=prediction_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
        )
        calibrated_scores_df = groupby_apply(
            df.groupby("segment"), predict_segment_func
        )
        return calibrated_scores_df["calibrated_scores"].sort_index(level=-1).values

    def _fit_segment(
        self,
        df_segment_train: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
    ) -> pd.DataFrame:
        # If the current segment contains only one class, we cannot fit a calibrator,
        # we fall back to the IdentityCalibrator, which we don't need to fit.
        if len(df_segment_train[label_column_name].unique()) > 1:
            calibrator = self.calibrator_class(**self.calibrator_kwargs)
            calibrator.fit(
                df_train=df_segment_train,
                prediction_column_name=prediction_column_name,
                label_column_name=label_column_name,
                weight_column_name=weight_column_name,
                categorical_feature_column_names=categorical_feature_column_names,
                numerical_feature_column_names=numerical_feature_column_names,
            )
            self.calibrator_per_segment[df_segment_train.name] = calibrator
        else:
            self.calibrator_per_segment[df_segment_train.name] = IdentityCalibrator()
        return df_segment_train  # return DataFrame to satisfy pandas apply, even though we don't use it

    def _predict_segment(
        self,
        df_segment: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str],
        numerical_feature_column_names: list[str],
    ) -> pd.DataFrame:
        # Handle edge case of unseen segment
        if df_segment.name not in self.calibrator_per_segment:
            self.calibrator_per_segment[df_segment.name] = IdentityCalibrator()
        df_segment["calibrated_scores"] = self.calibrator_per_segment[
            df_segment.name
        ].predict(
            df=df_segment,
            prediction_column_name=prediction_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
        )
        return df_segment
