# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

import json
import logging
import time

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import partial

from typing import Any, cast, Dict, Generic, TypeVar

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from multicalibration import utils
# @oss-disable[end= ]: from multicalibration.mcnet import CoreMCNet
from multicalibration.metrics import ScoreFunctionInterface, wrap_sklearn_metric_func
from numpy import typing as npt
from sklearn import isotonic, metrics as skmetrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from typing_extensions import Self

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MCBoostProcessedData:
    features: npt.NDArray
    predictions: npt.NDArray
    weights: npt.NDArray
    output_presence_mask: npt.NDArray
    categorical_feature_names: list[str]
    numerical_feature_names: list[str]
    labels: npt.NDArray | None = None

    def __getitem__(self, index: npt.NDArray) -> "MCBoostProcessedData":
        return MCBoostProcessedData(
            features=self.features[index],
            predictions=self.predictions[index],
            weights=self.weights[index],
            output_presence_mask=self.output_presence_mask[index],
            categorical_feature_names=self.categorical_feature_names,
            numerical_feature_names=self.numerical_feature_names,
            labels=self.labels[index] if self.labels is not None else None,
        )


class BaseCalibrator(ABC):
    @abstractmethod
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
        """Fit the calibration method on the provided training data.

        :param df_train: The dataframe containing the training data
        :param prediction_column_name: Name of the column in dataframe df that contains the predictions
        :param label_column_name: Name of the column in dataframe df that contains the ground truth labels
        :param weight_column_name: Name of the column in dataframe df that contains the instance weights
        :param categorical_feature_column_names: List of column names in the df that contain the categorical
                                               dimensions that are part of the segment space. This argument is ignored by methods that merely
                                               calibrate and do not multicalibrate (e.g., Isotonic regression and Platt scaling)
        :param numerical_feature_column_names: List of column names in the df that contain the numerical
                                             dimensions that are part of the segment space. This argument is ignored by methods that merely
                                             calibrate and do not multicalibrate (e.g., Isotonic regression and Platt scaling)
        :param kwargs: Additional keyword arguments
        :return: The fitted calibrator instance
        """
        pass

    @abstractmethod
    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Apply a calibration model to a DataFrame.

        This requires the `fit` method to have been previously called on this calibrator object.

        :param df: The dataframe containing the data to calibrate
        :param prediction_column_name: Name of the column in dataframe df that contains the predictions
        :param categorical_feature_column_names: List of column names in the df that contain the categorical
                                               dimensions that are part of the segment space. This argument is ignored by methods that merely
                                               calibrate and do not multicalibrate (e.g., Isotonic regression and Platt scaling)
        :param numerical_feature_column_names: List of column names in the df that contain the numerical
                                             dimensions that are part of the segment space. This argument is ignored by methods that merely
                                             calibrate and do not multicalibrate (e.g., Isotonic regression and Platt scaling)
        :param kwargs: Additional keyword arguments
        :return: Array of calibrated predictions
        """
        pass

    def fit_transform(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        is_train_set_col_name: str | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        """
        Fits the model using the training data and then applies the calibration transformation to all data.

        :param df: the dataframe containing the data to calibrate
        :param prediction_column_name: name of the column in dataframe df that contains the predictions
        :param label_column_name: name of the column in dataframe df that contains the ground truth labels
        :param weight_column_name: name of the column in dataframe df that contains the instance weights
        :param categorical_feature_column_names: list of column names in the df that contain the categorical
            dimensions that are part of the segment space. This argument is ignored by methods that merely
            calibrate and do not multicalibrate (e.g., Isotonic regression and Platt scaling).
        :param numerical_feature_column_names: list of column names in the df that contain the numerical
            dimensions that are part of the segment space. This argument is ignored by methods that merely
            calibrate and do not multicalibrate (e.g., Isotonic regression and Platt scaling).
        :param is_train_set_col_name: name of the column in the dataframe that contains a boolean indicating
            whether the row is part of the training set (0) or test set (1). If no is_train_set_col_name is
            provided, then all rows are considered part of the training set.
        """
        df_train = (
            df if is_train_set_col_name is None else df[df[is_train_set_col_name]]
        )
        self.fit(
            df_train=df_train,
            prediction_column_name=prediction_column_name,
            label_column_name=label_column_name,
            weight_column_name=weight_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
            **kwargs,
        )
        result = self.predict(
            df=df,
            prediction_column_name=prediction_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
            **kwargs,
        )
        return result


class EstimationMethod(Enum):
    CROSS_VALIDATION = 1
    HOLDOUT = 2
    AUTO = 3


class BaseMCBoost(BaseCalibrator, ABC):
    """
    Abstract base class for MCBoost models. This class hosts the common functionality for all MCBoost models and defines
    an abstract interface that all MCBoost models must implement.
    """

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
    def _default_early_stopping_metric(self) -> ScoreFunctionInterface:
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

    @staticmethod
    @abstractmethod
    def _check_predictions(df_train: pd.DataFrame, prediction_column_name: str) -> None:
        pass

    @staticmethod
    @abstractmethod
    def _check_labels(df_train: pd.DataFrame, label_column_name: str) -> None:
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
        early_stopping_score_func: ScoreFunctionInterface | None = None,
        early_stopping_minimize_score: bool | None = None,
        early_stopping_timeout: int | None = 8 * 60 * 60,  # 8 hours
        save_training_performance: bool = False,
        monitored_metrics_during_training: list[ScoreFunctionInterface] | None = None,
        allow_missing_segment_feature_values: bool = DEFAULT_ALLOW_MISSING_SEGMENT_FEATURE_VALUES,
        random_state: int | np.random.Generator | None = 42,
    ) -> None:
        """
        :param encode_categorical_variables: whether to encode categorical variables using a modified label encoding (when True),
            or whether to assume that categorical variables are already manipulated into the right format prior to calling MCBoost
            (when False).
        :param monotone_t: whether to use a monotonicity constraint on the logit feature (i.e., t): value
            True implies that the decision tree is blocked from creating splits where a lower value of t
            results in a higher predicted probability.
        :param num_rounds: number of rounds boosting that is used in MCBoost. When early stopping is used, then num_rounds specifies the maximum
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
            assert (
                early_stopping_minimize_score is not None
            ), "If using a custom score function the attribute `early_stopping_minimize_score` has to be set."
            self.early_stopping_score_func: ScoreFunctionInterface = (
                early_stopping_score_func
            )
            self.early_stopping_minimize_score: bool = early_stopping_minimize_score
        else:
            # Note: When changing the default score function, make sure to update the default value of `early_stopping_minimize_score` in the next line accordingly.
            self.early_stopping_score_func = self._default_early_stopping_metric
            self.early_stopping_minimize_score: bool = True
            assert (
                early_stopping_minimize_score is None
            ), f"`early_stopping_minimize_score` is only relevant when using a custom score function. The default score function is {self.early_stopping_score_func.name} for which `early_stopping_minimize_score` is set to {self.early_stopping_minimize_score} automatically."

        self._set_lightgbm_params(lightgbm_params)

        self.encode_categorical_variables = encode_categorical_variables
        self.MONOTONE_T: bool = (
            self.DEFAULT_HYPERPARAMS["monotone_t"] if monotone_t is None else monotone_t
        )

        self.EARLY_STOPPING: bool = (
            self.DEFAULT_HYPERPARAMS["early_stopping"]
            if early_stopping is None
            else early_stopping
        )

        if not self.EARLY_STOPPING:
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

        self.EARLY_STOPPING_ESTIMATION_METHOD: EstimationMethod = (
            EstimationMethod.CROSS_VALIDATION
            if early_stopping_use_crossvalidation
            else (
                EstimationMethod.AUTO
                if early_stopping_use_crossvalidation is None
                else EstimationMethod.HOLDOUT
            )
        )

        if self.EARLY_STOPPING_ESTIMATION_METHOD == EstimationMethod.HOLDOUT:
            if n_folds is not None:
                raise ValueError(
                    "`n_folds` must be None when `early_stopping_use_crossvalidation` is disabled."
                )

        if num_rounds is None:
            if self.EARLY_STOPPING:
                num_rounds = self.MAX_NUM_ROUNDS_EARLY_STOPPING
            else:
                num_rounds = self.NUM_ROUNDS_DEFAULT_NO_EARLY_STOPPING

        self.NUM_ROUNDS: int = num_rounds

        self.PATIENCE: int = (
            self.DEFAULT_HYPERPARAMS["patience"] if patience is None else patience
        )

        self.EARLY_STOPPING_TIMEOUT: int | None = early_stopping_timeout

        self.N_FOLDS: int = (
            1  # Because we make a single train/test split when using holdout
            if (self.EARLY_STOPPING_ESTIMATION_METHOD == EstimationMethod.HOLDOUT)
            else self.DEFAULT_HYPERPARAMS["n_folds"]
            if n_folds is None
            else n_folds
        )

        self.mr: list[lgb.Booster] = []
        self.unshrink_factors: list[float] = []
        self.enc: utils.OrdinalEncoderWithUnknownSupport | None = None

        self.save_training_performance = save_training_performance
        self._performance_metrics: Dict[str, list[float]] = defaultdict(list)
        self.monitored_metrics_during_training: list[ScoreFunctionInterface] = (
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
        Sets or updates the LightGBM parameters for this MCBoost instance.


        The `lightgbm_params` argument and `self.lightgbm_params` attribute are not always identical.
        When tuning hyperparameters (see tuning.py), we modify existing MCBoost objects rather than creating new objects.
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
                self.reset_training_state()
        except AttributeError:
            pass

        # Start with defaults if the method is used in the constructor for setting the parameters for the first time
        if not hasattr(self, "lightgbm_params"):
            params_to_set = self.DEFAULT_HYPERPARAMS["lightgbm_params"].copy()
        else:
            params_to_set = self.lightgbm_params.copy()

        if lightgbm_params is not None:
            params_to_set.update(lightgbm_params)

        assert (
            "num_rounds" not in params_to_set
        ), "avoid using `num_rounds` in `lightgbm_params` due to a naming conflict with `num_rounds` in MCBoost. Use any of the other aliases instead (https://lightgbm.readthedocs.io/en/latest/Parameters.html)"

        self.lightgbm_params: dict[str, Any] = {
            **params_to_set,
            "objective": self._objective,
            "seed": self._next_seed(),
            "deterministic": True,
            "verbosity": -1,
        }

    def feature_importance(self) -> pd.DataFrame:
        """
        Returns a dataframe with the feature importance of the final model.

        Importance is defined as the total gain from splits on a feature from the first round of MCBoost.

        :return: a dataframe with the feature importance.
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
                # Ordering of features here relies on two things 1) that MCBoost.extract_features returns first categoricals then
                # numericals and 2) that .fit method concatenates logits to the end of the feature matrix
                # pyre-ignore[58] if either feature_names attribute is None an error is raised above
                "feature": self.categorical_feature_names
                + self.numerical_feature_names
                + [self._PREDICTION_FEATURE_NAME],
                "importance": feature_importance,
            }
        ).sort_values("importance", ascending=False)

    def reset_training_state(self) -> None:
        self.mr = []
        self.unshrink_factors = []
        self.mce_below_initial = None
        self.mce_below_strong_evidence_threshold = None
        self._performance_metrics = defaultdict(list)
        self.enc: utils.OrdinalEncoderWithUnknownSupport | None = None
        self.categorical_feature_names = None
        self.numerical_feature_names = None

    @property
    def mce_is_satisfactory(self) -> bool | None:
        return self.mce_below_initial and self.mce_below_strong_evidence_threshold

    @property
    def performance_metrics(self) -> dict[str, list[float]]:
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
                    "Missing values found in segment feature data. MCBoost supports handling of missing data in segment features. If you want to disable native missing value support and predict None for examples with missing values in segment features, set `allow_missing_segment_feature_values=False` in the constructor of MCBoost. "
                )
            else:
                raise ValueError(
                    "Missing values found in segment feature data and `allow_missing_segment_feature_values` is set to False. If you want to enable native missing value support, set `allow_missing_segment_feature_values=True` in the constructor of MCBoost."
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
    ) -> MCBoostProcessedData:
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
        :return: MCBoostProcessedData object with extracted features and metadata
        """
        logger.info(
            f"Preprocessing input data with {len(df)} rows; in_fit_phase = {is_fit_phase}"
        )
        x = self.extract_features(
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

        return MCBoostProcessedData(
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
        self._check_input_data(
            df_train,
            prediction_column_name,
            label_column_name,
            categorical_feature_column_names,
            numerical_feature_column_names,
        )

        self.reset_training_state()

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

        num_rounds = self.NUM_ROUNDS
        if self.EARLY_STOPPING:
            timeout_msg = (
                f" (timeout: {self.EARLY_STOPPING_TIMEOUT}s)"
                if self.EARLY_STOPPING_TIMEOUT
                else ""
            )
            logger.info(
                f"Early stopping activated, max_num_rounds={self.NUM_ROUNDS}{timeout_msg}"
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
                logger.info(f"Fitting final MCBoost model with {num_rounds} rounds")
        else:
            logger.info(f"Early stopping deactivated, fitting {self.NUM_ROUNDS} rounds")

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
                categorical_feature_column_names=categorical_feature_column_names,
                numerical_feature_column_names=numerical_feature_column_names,
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
                params=self.get_lgbm_params(x),
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
                f"MCBoost does not support missing values in the prediction column. Found {nan_mask.sum()} missing values. MCBoost.predict will return np.nan for these predictions."
            )
        if outofbounds_mask.any():
            min_score = np.min(df[prediction_column_name].values)
            max_score = np.max(df[prediction_column_name].values)
            logger.warning(
                f"MCBoost calibrates probabilistic binary classifiers, hence predictions must be in (0,1). Found min {min_score} and max {max_score}. MCBoost.predict will return np.nan for these predictions."
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
                    f"Found {segment_feature_missing_mask.sum()} missing values in segment features. MCBoost.predict will return np.nan for these predictions. MCBoost supports handling of missing data in segment features. If you want to enable native missing value support set `allow_missing_segment_feature_values=True` in the constructor of MCBoost. "
                )
            invalid_mask = invalid_mask | segment_feature_missing_mask
        return np.logical_not(invalid_mask)

    @staticmethod
    def _remove_duplicate_metrics(
        monitored_metrics_during_training: list[ScoreFunctionInterface],
    ) -> list[ScoreFunctionInterface]:
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
                "MCBoost has not been fit. Returning the uncalibrated predictions."
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

    def get_lgbm_params(self, x: npt.NDArray) -> dict[str, Any]:
        lgb_params = self.lightgbm_params.copy()
        if self.MONOTONE_T:
            score_constraint = [1]
            segment_feature_constraints = [0] * (x.shape[1] - 1)
            lgb_params["monotone_constraints"] = (
                segment_feature_constraints + score_constraint
            )
        return lgb_params

    def extract_features(
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
        estimation_method: EstimationMethod,
        has_custom_validation_set: bool,
    ) -> (
        KFold
        | StratifiedKFold
        | utils.TrainTestSplitWrapper
        | utils.NoopSplitterWrapper
    ):
        if estimation_method == EstimationMethod.CROSS_VALIDATION:
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
        estimation_method: EstimationMethod,
    ) -> int:
        if estimation_method == EstimationMethod.CROSS_VALIDATION:
            n_folds = self.N_FOLDS
            logger.info(f"Using {n_folds} folds for cross-validation.")
        else:
            n_folds = 1
        return n_folds

    def _determine_best_num_rounds(
        self,
        data_train: MCBoostProcessedData,
        data_val: MCBoostProcessedData | None = None,
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

        mcboost_per_fold: Dict[int, BaseMCBoost] = {}
        predictions_per_fold: Dict[int, npt.NDArray] = {}

        best_score = -np.inf

        start_time = time.time()

        while num_rounds <= self.NUM_ROUNDS and patience_counter <= self.PATIENCE:
            log_add = ""
            if num_rounds == 0:
                log_add = " (input prediction for early stopping baseline)"
            logger.info(f"Evaluating round {num_rounds}{log_add}")

            if self.EARLY_STOPPING_TIMEOUT is not None and self._get_elapsed_time(
                start_time
            ) > cast(int, self.EARLY_STOPPING_TIMEOUT):
                logger.warning(
                    f"Stopping early stopping upon exceeding the {self.EARLY_STOPPING_TIMEOUT:,}-second timeout; "
                    + "MCBoost results will likely improve by increasing `early_stopping_timeout` or setting it to None"
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
                    if fold_num not in mcboost_per_fold:
                        mcboost = self._create_instance_for_cv(
                            encode_categorical_variables=self.encode_categorical_variables,
                            monotone_t=self.MONOTONE_T,
                            lightgbm_params=self.lightgbm_params,
                            early_stopping=False,
                            num_rounds=0,
                        )
                        mcboost_per_fold[fold_num] = mcboost
                        predictions_per_fold[fold_num] = data_train_cv.predictions

                    new_predictions = mcboost_per_fold[
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
                        train_fold_preds = mcboost_per_fold[fold_num]._predict(
                            x=data_train_cv.features,
                            transformed_predictions=data_train_cv.predictions,
                            return_all_rounds=False,
                        )

                    valid_fold_preds = mcboost_per_fold[fold_num]._predict(
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
                f"Round {num_rounds}: validation loss = {early_stopping_metric_value:.4f} (best: {best_early_stopping_metric_value:.4f}, patience: {patience_counter}/{self.PATIENCE})"
            )

            num_rounds += 1

        if best_num_rounds == 0:
            logger.warning(
                "Selected 0 to be the best number of rounds for MCBoost for this dataset, meaning that uncalibrated predictions will be returned. This is because the optimization metric did not improve during the first round of boosting."
            )
        elif best_num_rounds == self.NUM_ROUNDS:
            logger.warning(
                f"max_num_rounds might be too low: best performance was at the maximum number of rounds ({self.NUM_ROUNDS})"
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

                self.mce_below_initial = (
                    True if mce_at_best_num_rounds < mce_at_initial_round else False
                )
                self.mce_below_strong_evidence_threshold = (
                    True
                    if mce_at_best_num_rounds < self.MCE_STRONG_EVIDENCE_THRESHOLD
                    else False
                )

                if not self.mce_below_strong_evidence_threshold:
                    logger.warning(
                        f"The final Multicalibration Error on the validation set after using MCBoost is {mce_at_best_num_rounds}. This is higher than 4.0, which still indicates strong evidence for miscalibration."
                    )
                if not self.mce_below_initial:
                    logger.warning(
                        f"The final Multicalibration Error on the validation set after using MCBoost is {mce_at_best_num_rounds}, which is not lower than the initial Multicalibration Error of {mce_at_initial_round}. This indicates that MCBoost did not improve the multi-calibration of the model."
                    )

        return best_num_rounds

    def _compute_metric_on_internal_data(
        self,
        metric: ScoreFunctionInterface,
        data: MCBoostProcessedData,
        predictions: npt.NDArray,
    ) -> float:
        """
        Compatibility wrapper for MCBoostProcessedData -> ScoreFunctionInterface.
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
        """
        Serializes the model into a JSON string.

        :return: serialised model.
        """
        serialized_boosters = [booster.model_to_string() for booster in self.mr]
        json_obj: dict[str, Any] = {
            "mcboost": [
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
        json_obj = json.loads(model_str)
        model = cls()
        model.mr = []
        model.unshrink_factors = []

        for model_info in json_obj["mcboost"]:
            booster = lgb.Booster(model_str=model_info["booster"])
            model.mr.append(booster)
            model.unshrink_factors.append(model_info["unshrink_factor"])

        model.NUM_ROUNDS = len(model.mr)

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

    def _determine_estimation_method(self, weights: npt.NDArray) -> EstimationMethod:
        """
        Returns the estimation method to use for early stopping given the arguments and the weights (when relevant).
        This is especially useful for the AUTO option, where we infer the proper estimation method to use based on the effective sample size.

        :return: the estimation method to use.
        """
        if self.EARLY_STOPPING_ESTIMATION_METHOD != EstimationMethod.AUTO:
            return self.EARLY_STOPPING_ESTIMATION_METHOD

        if self.early_stopping_score_func.name != "log_loss":
            # Automatically infer the estimation method only when using the logistic loss, otherwise use k-fold.
            # This is because we analyzed the effective sample size specifically with log_loss.
            return EstimationMethod.CROSS_VALIDATION

        # We use a rule-of-thumb to determine whether to use cross-validation or holdout for early stopping.
        # Namely, if the effective sample size is less than 2.5M, we use cross-validation, otherwise we use holdout.
        # See N6787810 for more details.
        ess = self._compute_effective_sample_size(weights)

        if ess < self.ESS_THRESHOLD_FOR_CROSS_VALIDATION:
            logger.info(
                f"Found a relatively small effective sample size ({ess:,}), choosing k-fold for early stopping. "
                + "You can override this by explicitly setting `early_stopping_use_crossvalidation` to `False`."
            )
            return EstimationMethod.CROSS_VALIDATION
        else:
            logger.info(
                f"Found a large enough effective sample size ({ess:,}), choosing holdout for early stopping. "
                + "You can override this by explicitly setting `early_stopping_use_crossvalidation` to `True`."
            )
            return EstimationMethod.HOLDOUT


class MCBoost(BaseMCBoost):
    """
    Multicalibration boosting (MCBoost) as introduced in [1].

    References:
    [1] Hébert-Johnson, U., Kim, M., Reingold, O., & Rothblum, G. (2018). Multicalibration: Calibration for the
        (computationally-identifiable) masses. In International Conference on Machine Learning (pp. 1939-1948). PMLR.
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
            y, predictions, w, logit_epsilon=MCBoost.UNSHRINK_LOGIT_EPSILON
        )

    @property
    def _objective(self) -> str:
        return "binary"

    @property
    def _default_early_stopping_metric(self) -> ScoreFunctionInterface:
        return wrap_sklearn_metric_func(skmetrics.log_loss)

    @staticmethod
    def _check_predictions(df_train: pd.DataFrame, prediction_column_name: str) -> None:
        predictions = df_train[prediction_column_name].to_numpy()
        if MCBoost._predictions_out_of_bounds(predictions).any():
            raise ValueError(
                "Predictions must be probabilities in the (0, 1) interval. "
                f"Found predictions outside this range: min={predictions.min()}, max={predictions.max()}"
            )
        if df_train[prediction_column_name].isnull().any():
            raise ValueError(
                f"MCBoost does not support missing values in the prediction column, but {df_train[prediction_column_name].isnull().sum()}"
                f" of {len(df_train[prediction_column_name])} are null."
            )

        lower_prob_bound = utils.logistic(-MCBoost.UNSHRINK_LOGIT_EPSILON)
        upper_prob_bound = utils.logistic(MCBoost.UNSHRINK_LOGIT_EPSILON)
        num_out_of_bounds = np.sum(
            (predictions < lower_prob_bound) | (predictions > upper_prob_bound)
        )
        if num_out_of_bounds > 0:
            pct_out_of_bounds = 100.0 * num_out_of_bounds / len(predictions)
            logger.warning(
                f"Found {num_out_of_bounds} ({pct_out_of_bounds:.2f}%) predictions with extreme values (boundaries: [{lower_prob_bound:.6g}, {upper_prob_bound:.6g}]). "
                f"These samples will be clipped in the unshrink step. Consider reviewing input prediction quality."
            )

    @staticmethod
    def _check_labels(df_train: pd.DataFrame, label_column_name: str) -> None:
        if df_train[label_column_name].isnull().any():
            raise ValueError(
                f"MCBoost does not support missing values in the label column, but {df_train[label_column_name].isnull().sum()}"
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
            n_splits=self.N_FOLDS,
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


class RegressionMCBoost(BaseMCBoost):
    """
    Regression variant of MCBoost for continuous label calibration.

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
    def _default_early_stopping_metric(self) -> ScoreFunctionInterface:
        return wrap_sklearn_metric_func(skmetrics.mean_squared_error)

    @staticmethod
    def _check_predictions(df_train: pd.DataFrame, prediction_column_name: str) -> None:
        predictions = df_train[prediction_column_name]
        if predictions.isnull().any():
            raise ValueError(
                f"RegressionMCBoost does not support missing values in the prediction column, but {predictions.isnull().sum()}"
                f" of {len(predictions)} are null."
            )
        if np.isinf(predictions).any():
            raise ValueError(
                f"RegressionMCBoost does not support infinite values in the prediction column, but {np.sum(np.isinf(predictions))}"
                f" of {len(predictions)} are null."
            )

    @staticmethod
    def _check_labels(df_train: pd.DataFrame, label_column_name: str) -> None:
        labels = df_train[label_column_name]
        if not pd.api.types.is_numeric_dtype(labels):
            raise ValueError(
                f"RegressionMCBoost only supports numeric labels, but {label_column_name} has type {labels.dtype}."
            )
        if labels.isnull().any() or labels.isna().any():
            raise ValueError(
                f"RegressionMCBoost does not support missing values in the label column, but {labels.isnull().sum()}"
                f" of {len(labels)} are null."
            )
        if np.isinf(labels).any():
            raise ValueError(
                f"RegressionMCBoost does not support infinite values in the prediction column, but {np.sum(np.isinf(labels))}"
                f" of {len(labels)} are null."
            )
        if labels.nunique() < 2:
            raise ValueError(
                f"RegressionMCBoost requires at least 2 unique values in the label column, but {label_column_name} has only {labels.nunique()}."
            )

    @property
    def _cv_splitter(self) -> KFold:
        return KFold(
            n_splits=self.N_FOLDS,
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


class MCNet(BaseCalibrator):
    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        num_multicalibration_blocks: int = 10,
        num_boosting_sub_blocks: int = 25,
        batch_size: int | None = None,
        loss_fn_class: type[nn.Module] = nn.BCEWithLogitsLoss,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        num_epochs_per_block: int = 50,
        early_stopping: bool | None = None,
        num_epochs_level_patience: int | None = None,
        sub_block_level_patience: int | None = None,
        block_level_patience: int | None = None,
        validation_split_fraction: float = 0.1,
        max_n_categories: int = 10,
        l1_alpha: float | None = None,
        device: torch.device | str | None = None,
        allow_missing_segment_feature_values: bool = True,
    ) -> None:
        """MCNet: Wrapper around CoreMCNet that provides the same interface as MCBoost.

        :param hidden_layers: List of hidden layer sizes for sub-blocks (e.g., [16, 4] for 2-layer network)
        :param num_multicalibration_blocks: (Upper bound of the) Number of multicalibration blocks
        :param num_boosting_sub_blocks: (Upper bound of the) Number of MLP sub-blocks per multicalibration block
        :param batch_size: Batch size for training. If None, will be set to 0.005 * n_train
        :param loss_fn_class: Loss function class to use for training. Defaults to nn.BCEWithLogitsLoss
        :param optimizer_class: Optimizer class to use for training. Defaults to torch.optim.Adam
        :param optimizer_kwargs: Additional keyword arguments to pass to the optimizer. Defaults to {'lr':0.001}
        :param num_epochs_per_block: Number of epochs per multicalibration block
        :param early_stopping: Whether to use early stopping at all levels (epochs, sub-blocks, blocks)
        :param num_epochs_level_patience: Patience for epochs-level early stopping. If None, uses default hyperparameter value
        :param sub_block_level_patience: Patience for individual sub-block training. If None, uses default hyperparameter value
        :param block_level_patience: Patience for block-level early stopping. If None, uses default hyperparameter value
        :param validation_split_fraction: Fraction of data to use for validation (default: 0.1)
        :param max_n_categories: Maximum number of categories to use for categorical features (default: 10)
        :param l1_alpha: L1 regularization strength (None = disabled, default: None)
        :param device: Device to run the model on. Can be torch.device, string (e.g., 'cuda', 'cpu'), or None for auto-detection (default: None)
        :param allow_missing_segment_feature_values: Whether to allow missing segment feature values (default: True)
        """
        # Initialize with a placeholder - will be updated after feature preprocessing with the correct feature_dim
        self.core_mcnet: CoreMCNet | None = None
        self.hidden_layers = hidden_layers
        self.num_multicalibration_blocks = num_multicalibration_blocks
        self.num_boosting_sub_blocks = num_boosting_sub_blocks
        self.feature_processor = utils.FeatureProcessorState()
        self.batch_size = batch_size
        self.loss_fn_class = loss_fn_class
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs: dict[str, Any] = (
            {} if optimizer_kwargs is None else optimizer_kwargs
        )
        if "lr" not in self.optimizer_kwargs:
            self.optimizer_kwargs["lr"] = 0.001
        self.num_epochs_per_block = num_epochs_per_block

        if not early_stopping:
            if (
                num_epochs_level_patience is not None
                or sub_block_level_patience is not None
                or block_level_patience is not None
            ):
                raise ValueError(
                    "patience values must be None when argument `early_stopping` is disabled."
                )
        self.early_stopping = early_stopping
        self.num_epochs_level_patience = num_epochs_level_patience
        self.sub_block_level_patience = sub_block_level_patience
        self.block_level_patience = block_level_patience

        self.validation_split_fraction = validation_split_fraction
        self.max_n_categories = max_n_categories
        self.l1_alpha = l1_alpha

        if device is None:
            self.device: torch.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.allow_missing_segment_feature_values = allow_missing_segment_feature_values

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
        """Fit the MCNet model on the provided training data.

        This is a wrapper around CoreMCNet's fit that converts it to MCBoost interface.

        :param df_train: The dataframe containing the training data
        :param prediction_column_name: Name of the column in dataframe df that contains the predictions
        :param label_column_name: Name of the column in dataframe df that contains the ground truth labels
        :param weight_column_name: Name of the column in dataframe df that contains the instance weights (not currently used)
        :param categorical_feature_column_names: List of column names in the df that contain the categorical
                                               dimensions that are part of the segment space
        :param numerical_feature_column_names: List of column names in the df that contain the numerical
                                             dimensions that are part of the segment space
        """

        loss_fn = self.loss_fn_class(
            reduction="none"
        )  # Compute per-sample loss to be divided by sum weights

        if df_train[prediction_column_name].isnull().any():
            raise ValueError(
                f"MCNet does not support missing values in the prediction column, but {df_train[prediction_column_name].isnull().sum()}"
                f" of {len(df_train[prediction_column_name])} are null."
            )

        if df_train[label_column_name].isnull().any():
            raise ValueError(
                f"MCNet does not support missing values in the label column, but {df_train[label_column_name].isnull().sum()}"
                f" of {len(df_train[label_column_name])} are null."
            )

        n_samples = len(df_train)
        indices = torch.randperm(n_samples)
        n_train = int((1 - self.validation_split_fraction) * n_samples)

        if self.batch_size is None:
            self.batch_size = max(1, int(0.005 * n_train))

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        df_train_split = df_train.iloc[train_indices]
        df_val = df_train.iloc[val_indices]

        train_features, processor_state = utils.extract_segment_features(
            df=df_train_split,
            categorical_segment_cols=categorical_feature_column_names or [],
            numerical_segment_cols=numerical_feature_column_names or [],
            max_n_categories=self.max_n_categories,
            is_fit_phase=True,
            fillna=self.allow_missing_segment_feature_values,
        )

        val_features, _ = utils.extract_segment_features(
            df=df_val,
            categorical_segment_cols=categorical_feature_column_names or [],
            numerical_segment_cols=numerical_feature_column_names or [],
            processor_state=processor_state,
            is_fit_phase=False,
            fillna=self.allow_missing_segment_feature_values,
        )

        y = torch.tensor(
            df_train_split[label_column_name].values,
            dtype=torch.float32,
            device=self.device,
        )
        y_val = torch.tensor(
            df_val[label_column_name].values, dtype=torch.float32, device=self.device
        )

        if weight_column_name is not None:
            weights_train = torch.tensor(
                df_train_split[weight_column_name].values.astype(float),
                dtype=torch.float32,
                device=self.device,
            )
            weights_val = torch.tensor(
                df_val[weight_column_name].values.astype(float),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            weights_train, weights_val = None, None

        base_logits = torch.tensor(
            utils.logit(df_train_split[prediction_column_name].values),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        val_base_logits = torch.tensor(
            utils.logit(df_val[prediction_column_name].values),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        # Move feature tensors to the specified device
        x_train, x_val = train_features.to(self.device), val_features.to(self.device)
        y_train, base_logits_train = y, base_logits
        base_logits_val = val_base_logits

        self.feature_processor = processor_state

        # Create CoreMCNet now that we know the actual feature_dim after preprocessing
        self.core_mcnet = CoreMCNet(
            feature_dim=processor_state.feature_dim,
            hidden_layers=self.hidden_layers,
            num_multicalibration_blocks=self.num_multicalibration_blocks,
            num_boosting_sub_blocks=self.num_boosting_sub_blocks,
            l1_alpha=self.l1_alpha,
            early_stopping=self.early_stopping,
            num_epochs_level_patience=self.num_epochs_level_patience,
            sub_block_level_patience=self.sub_block_level_patience,
            block_level_patience=self.block_level_patience,
        ).to(self.device)

        self._print_architecture_and_parameters_before_training(
            n_training_samples=len(y_train),
            n_validation_samples=len(y_val),
        )

        assert self.core_mcnet is not None and self.batch_size is not None
        self.core_mcnet.fit(
            x=x_train,
            y=y_train,
            base_logits=base_logits_train,
            num_epochs_per_block=self.num_epochs_per_block,
            batch_size=self.batch_size,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            loss_fn=loss_fn,
            x_val=x_val,
            y_val=y_val,
            base_logits_val=base_logits_val,
            weights=weights_train,
            weights_val=weights_val,
            validation_split_fraction=self.validation_split_fraction,
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
        """Apply the MCNet calibration model to a DataFrame.

        This requires the `fit` method to have been previously called on this calibrator object.

        :param df: The dataframe containing the data to calibrate
        :param prediction_column_name: Name of the column in dataframe df that contains the predictions
        :param categorical_feature_column_names: List of column names in the df that contain the categorical
                                               dimensions that are part of the segment space
        :param numerical_feature_column_names: List of column names in the df that contain the numerical
                                             dimensions that are part of the segment space
        :return: Calibrated predictions as numpy array
        """

        assert self.core_mcnet is not None, "MCNet has not been fitted yet."

        if df[prediction_column_name].isnull().any():
            n_missing = df[prediction_column_name].isnull().sum()
            n_total = len(df[prediction_column_name])
            raise ValueError(
                f"MCNet does not support missing values in the prediction column, but {n_missing}"
                f" of {n_total} are null."
            )

        x, _ = utils.extract_segment_features(
            df=df,
            categorical_segment_cols=categorical_feature_column_names or [],
            numerical_segment_cols=numerical_feature_column_names or [],
            processor_state=self.feature_processor,
            is_fit_phase=False,
            fillna=self.allow_missing_segment_feature_values,
        )

        y_hat = df[prediction_column_name].values.astype(float)
        base_logits = torch.tensor(
            utils.logit(y_hat), dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        x = x.to(self.device)

        # pyre-ignore[16]: self.core_mcnet is not None after fit() call
        calibrated_probs = self.core_mcnet.predict_proba(x=x, base_logits=base_logits)

        return calibrated_probs.detach().cpu().numpy()

    def extract_segment_features(
        self,
        df: pd.DataFrame,
        categorical_segment_cols: list[str] | None = None,
        numerical_segment_cols: list[str] | None = None,
        is_fit_phase: bool = False,
    ) -> torch.Tensor:
        features, self.feature_processor = utils.extract_segment_features(
            df,
            categorical_segment_cols,
            numerical_segment_cols,
            self.feature_processor,
            is_fit_phase,
        )
        return features

    def get_list_num_active_boosting_sub_blocks(self) -> list[int]:
        if self.core_mcnet is None:
            raise ValueError("MCNet has not been fitted yet.")
        return self.core_mcnet.get_list_num_active_boosting_sub_blocks()

    def get_num_active_multicalibration_blocks(self) -> int:
        if self.core_mcnet is None:
            raise ValueError("MCNet has not been fitted yet.")
        return self.core_mcnet.get_num_active_multicalibration_blocks()

    def _print_architecture_and_parameters_before_training(
        self,
        n_training_samples: int,
        n_validation_samples: int,
    ) -> None:
        logger.info("=" * 80)
        logger.info("MCNet Architecture & Training Configuration".center(80))
        logger.info("=" * 80)

        # Since this function is only called after self.core_mcnet is created, we can assert it's not None
        assert (
            self.core_mcnet is not None
        ), "MCNet core should be initialized before printing architecture"

        core_mcnet = self.core_mcnet

        assert self.batch_size is not None, "Batch size should be set before training."
        batch_size = self.batch_size

        logger.info("  Model Structure:")
        logger.info(f"   └── MCNet (feature_dim={core_mcnet.feature_dim})")
        logger.info(
            f"       └── blocks: ModuleList ({core_mcnet.num_multicalibration_blocks} identical blocks)"
        )
        logger.info("           └── MulticalibrationBlock")
        logger.info(
            f"               └── sub_blocks: ModuleList ({core_mcnet.num_boosting_sub_blocks} identical sub-blocks)"
        )
        logger.info("                   └── BoostingSubBlock")
        logger.info("                       └── mlp: Sequential")

        first_block = core_mcnet.blocks[0].sub_blocks[0]
        for i, layer in enumerate(first_block.mlp):
            layer_name = layer.__class__.__name__
            if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
                layer_desc = f"{layer_name}(in_features={layer.in_features}, out_features={layer.out_features}, bias={layer.bias is not None})"
            else:
                layer_desc = f"{layer_name}()"

            is_last_layer = i == len(first_block.mlp) - 1
            layer_prefix = "└──" if is_last_layer else "├──"
            logger.info(
                f"                           {layer_prefix} ({i}): {layer_desc}"
            )

        logger.info("\n  Training Configuration:")
        logger.info(f"   ├── Epochs per sub-block: {self.num_epochs_per_block}")
        logger.info(
            f"   ├── Batch size: {batch_size} ({np.round(100*batch_size/n_training_samples,2)}% of training data)"
        )
        logger.info(
            f"   ├── Validation split: {self.validation_split_fraction:.1%} ({n_training_samples} training & {n_validation_samples} validation samples)"
        )
        logger.info(
            f"   ├── Optimizer: {self.optimizer_class.__name__}{self.optimizer_kwargs}"
        )
        logger.info(f"   ├── Loss function: {self.loss_fn_class.__name__}()")
        logger.info("   └── Early stopping patience:")
        logger.info(f"       ├── Epochs: {self.num_epochs_level_patience}")
        logger.info(f"       ├── Sub-blocks: {self.sub_block_level_patience}")
        logger.info(f"       └── Blocks: {self.block_level_patience}")

        logger.info("=" * 80)
        logger.info("")


class PlattScaling(BaseCalibrator):
    """
    Provides an implementation of Platt scaling, which is just a Logistic Regression applied to the logits.

    References:
    - Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized
        likelihood methods. Advances in large margin classifiers, 10(3), 61-74.
    - Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning.
        International Conference on Machine Learning (ICML). pp. 625-632.
    """

    def __init__(self) -> None:
        self.log_reg = LogisticRegression()

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
        y = df_train[label_column_name].values.astype(float)
        y_hat = df_train[prediction_column_name].values.astype(float)
        w = df_train[weight_column_name] if weight_column_name else np.ones_like(y)

        logits = utils.logit(y_hat).reshape(-1, 1)
        self.log_reg = LogisticRegression(penalty=None)
        self.log_reg.fit(logits, y, sample_weight=w)
        return self

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        y_hat = df[prediction_column_name].values.astype(float)

        logits = utils.logit(y_hat).reshape(-1, 1)
        return self.log_reg.predict_proba(logits)[:, 1]


class IsotonicRegression(BaseCalibrator):
    """
    Provides an implementation of Isotonic regression. For input values outside of the training
    domain, predictions are set to the value corresponding to the nearest training interval endpoint.

    References:
    - Zadrozny, B., & Elkan, C. (2001). Obtaining calibrated probability estimates from decision trees and
        naive bayesian classifiers. International Conference on Machine Learning (ICML). pp. 609-616.
    - Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning.
        International Conference on Machine Learning (ICML). pp. 625-632.
    """

    def __init__(self) -> None:
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
        y_hat = df[prediction_column_name].values.astype(float)
        return self.isoreg.transform(y_hat)


class MultiplicativeAdjustment(BaseCalibrator):
    """
    Calibrates predictions by multiplying scores with a correction factor derived from the ratio of total positive
    labels to sum of predicted scores. This helps align the overall prediction distribution with the true label distribution.
    """

    def __init__(self, clip_to_zero_one: bool = True) -> None:
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
        preds = df[prediction_column_name].values * self.multiplier
        if self.clip_to_zero_one:
            preds = np.clip(preds, 0, 1)
        return preds


class AdditiveAdjustment(BaseCalibrator):
    """
    Calibrates predictions by adding a correction term derived from the difference between total positive labels
    and sum of predicted scores. This helps align the overall prediction distribution with the true label distribution.
    """

    def __init__(self, clip_to_zero_one: bool = True) -> None:
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
        w = (
            df_train[weight_column_name]
            if weight_column_name
            else np.ones(df_train.shape[0])
        )
        total_score = (w * df_train[prediction_column_name]).sum()
        total_positive = (w * df_train[label_column_name]).sum()
        self.offset = (total_positive - total_score) / w.sum()
        return self

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
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
        return self

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        return df[prediction_column_name].values


class SwissCheesePlattScaling(BaseCalibrator):
    """
    A variant of Platt scaling that incorporates additional categorical and numerical features alongside logits.
    Numerical features are discretized into bins.
    """

    def __init__(self) -> None:
        self.log_reg = LogisticRegression()
        self.logits_column_name = "__logits"
        self.ohe: OneHotEncoder | None = None
        self.kbd: KBinsDiscretizer | None = None
        self.ohe_columns: list[str] | None = None
        self.kbd_columns: list[str] | None = None
        self.features: list[str] | None = None

    def fit_feature_encoders(
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
            self.kbd = KBinsDiscretizer(encode="onehot-dense", n_bins=3, subsample=None)
            self.kbd.fit(df[numerical_feature_column_names])
        else:
            self.kbd = None

    def convert_df(
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

    def train_model(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
    ) -> LogisticRegression:
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
        w = w.astype(float)

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
        df_train = df_train.copy().reset_index().fillna(0)
        self.fit_feature_encoders(
            df_train, categorical_feature_column_names, numerical_feature_column_names
        )

        df_train = self.convert_df(
            df_train,
            prediction_column_name,
            categorical_feature_column_names,
            numerical_feature_column_names,
        )

        log_reg = self.train_model(
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
        df = df.copy().reset_index().fillna(0)

        df = self.convert_df(
            df=df,
            prediction_column_name=prediction_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
        )
        return self.log_reg.predict_proba(df[self.features])[:, 1]


TCalibrator = TypeVar("TCalibrator", bound=BaseCalibrator)


class SegmentwiseCalibrator(Generic[TCalibrator], BaseCalibrator):
    """
    A meta-calibrator that partitions data into segments based on categorical features and applies a separate calibration
    method to each segment. This enables more precise calibration when different segments require different calibration
    adjustments.
    """

    calibrator_per_segment: dict[str, BaseCalibrator]
    calibrator_class: type[TCalibrator]
    calibrator_kwargs: dict[str, Any]

    def __init__(
        self,
        calibrator_class: type[TCalibrator],
        calibrator_kwargs: dict[str, Any] | None = None,
    ) -> None:
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
        if categorical_feature_column_names is None:
            categorical_feature_column_names = []
        if numerical_feature_column_names is None:
            numerical_feature_column_names = []

        # Create a unique identifier for each segment
        df_train = df_train.copy()
        df_train["segment"] = df_train[categorical_feature_column_names].apply(
            lambda row: "_".join(row.values.astype(str)), axis=1
        )

        fit_segment_func = partial(
            self._fit_segment,
            prediction_column_name=prediction_column_name,
            label_column_name=label_column_name,
            weight_column_name=weight_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
        )
        df_train.groupby("segment").apply(fit_segment_func)
        return self

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        if categorical_feature_column_names is None:
            categorical_feature_column_names = []
        if numerical_feature_column_names is None:
            numerical_feature_column_names = []

        # Create a unique identifier for each segment
        df = df.copy()
        df["segment"] = df[categorical_feature_column_names].apply(
            lambda row: "_".join(row.values.astype(str)), axis=1
        )

        predict_segment_func = partial(
            self._predict_segment,
            prediction_column_name=prediction_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
        )
        calibrated_scores_df = df.groupby("segment").apply(predict_segment_func)
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
