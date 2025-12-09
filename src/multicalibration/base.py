# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from numpy import typing as npt
from typing_extensions import Self


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
