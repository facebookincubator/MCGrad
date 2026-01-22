# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict

"""
Internal utility functions for calibration and multicalibration operations.

This module provides mathematical utilities, data transformation functions,
and helper classes used throughout the multicalibration library.
"""

import ast
import functools
import hashlib
import logging
import math
import os
import threading
import time
import warnings
from collections.abc import Callable, Iterator
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
import psutil
from scipy import stats
from scipy.optimize._linesearch import LineSearchWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


logger: logging.Logger = logging.getLogger(__name__)

# Default epsilon for bin boundary calculations
BIN_EPSILON: float = 1e-8

# Minimum logit epsilon to avoid extreme values close to float limits
MIN_LOGIT_EPSILON: float = 1e-304


def unshrink(
    y: npt.NDArray[Any],
    logits: npt.NDArray[Any],
    w: npt.NDArray[Any] | None = None,
    logit_epsilon: float | None = 10,
) -> float:
    """
    Compute an unshrinkage coefficient using logistic regression.

    Fits a logistic regression model without intercept to find a scaling coefficient
    that adjusts for shrinkage in the logits. Uses Newton-CG solver primarily, with
    LBFGS as fallback if Newton-CG fails to converge.

    :param y: Array of binary labels (0 or 1).
    :param logits: Array of logit values (log-odds) to unshrink.
    :param w: Optional array of sample weights. If None, uniform weights are used.
    :param logit_epsilon: Clipping bound for logits to avoid extreme coefficients.
        Set to None to disable clipping.
    :return: The unshrinkage coefficient. Returns 1 if both solvers fail.
    """
    if w is None:
        w = np.ones_like(y)
    logits = logits.reshape(-1, 1)

    # Clip logits to avoid extreme coefficient driven by outliers
    if logit_epsilon is not None:
        logits = np.clip(logits, -logit_epsilon, logit_epsilon)

    primary_solver = LogisticRegression(
        C=np.inf, fit_intercept=False, solver="newton-cg"
    )
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        # Suppress sklearn 1.8+ UserWarning which is a known bug. Will be fixed in sklearn 1.8.1
        # See: https://github.com/scikit-learn/scikit-learn/issues/32927
        warnings.filterwarnings(
            "ignore",
            message="Setting penalty=None will ignore the C and l1_ratio parameters",
            category=UserWarning,
        )
        primary_solver.fit(logits, y, sample_weight=w)
    for rec_warn in recorded_warnings:
        if isinstance(rec_warn.message, LineSearchWarning):
            logger.info(
                "Line search warning (unshrink): %s. Solution is approximately "
                "optimal - no ideal step size for the gradient descent update "
                "can be found. These warnings are generally harmless.",
                rec_warn.message,
            )
        else:
            logger.debug(rec_warn)
            warnings.warn_explicit(
                message=str(rec_warn.message),
                category=rec_warn.category,
                filename=rec_warn.filename,
                lineno=rec_warn.lineno,
                source=rec_warn.source,
            )

    # Return result if logistic regression with Newton-CG converged to a solution,
    # if not try LBFGS.
    # pyre-ignore, coef_ is available after `fit()` has been called
    if not np.isnan(primary_solver.coef_).any():
        if primary_solver.coef_[0][0] < 0.95 or primary_solver.coef_[0][0] > 1.05:
            logger.warning(
                "Unshrink is not close to 1: %s. This may create a problem "
                "with the multicalibration of the model.",
                primary_solver.coef_[0][0],
            )

        return primary_solver.coef_[0][0]

    fallback_solver = LogisticRegression(C=np.inf, fit_intercept=False, solver="lbfgs")
    # Suppress sklearn 1.8+ UserWarning which is a known bug. Will be fixed in sklearn 1.8.1
    # See: https://github.com/scikit-learn/scikit-learn/issues/32927
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting penalty=None will ignore the C and l1_ratio parameters",
            category=UserWarning,
        )
        fallback_solver.fit(logits, y, sample_weight=w)
    if not np.isnan(fallback_solver.coef_).any():
        if primary_solver.coef_[0][0] < 0.95 or primary_solver.coef_[0][0] > 1.05:
            logger.warning(
                "Unshrink is not close to 1: %s. This may create a problem "
                "with the multicalibration of the model.",
                primary_solver.coef_[0][0],
            )
        return fallback_solver.coef_[0][0]

    # If both solvers fail, return default value. Not disastrous, but requires GBDT to do more heavy-lifting.
    return 1


def logistic(logits: float) -> float:
    """
    Compute the logistic (sigmoid) function in a numerically stable way.

    Uses a computational trick to avoid overflow/underflow by choosing
    different formulations based on the sign of the input.

    :param logits: Input value in log-odds space.
    :return: Probability value in (0, 1).
    """
    if logits >= 0:
        return 1.0 / (1.0 + math.exp(-logits))
    else:
        return math.exp(logits) / (1.0 + math.exp(logits))


logistic_vectorized: Callable[[npt.NDArray[Any]], npt.NDArray[Any]] = np.vectorize(
    logistic
)


def logit(
    probs: npt.NDArray[Any], epsilon: float = MIN_LOGIT_EPSILON
) -> npt.NDArray[Any]:
    """
    Compute the logit (log-odds) of probabilities.

    :param probs: Array of probability values.
    :param epsilon: Small constant to avoid division by zero and log(0).
    :return: Array of logit values.
    """
    with np.errstate(invalid="ignore"):
        return np.log((probs + epsilon) / (1 - probs + epsilon))


def absolute_error(
    estimate: npt.NDArray[Any], reference: npt.NDArray[Any]
) -> npt.NDArray[Any]:
    """
    Compute element-wise absolute error between estimate and reference.

    :param estimate: Array of estimated values.
    :param reference: Array of reference (ground truth) values.
    :return: Array of absolute errors.
    """
    return np.abs(estimate - reference)


def proportional_error(
    estimate: npt.NDArray[Any], reference: npt.NDArray[Any]
) -> npt.NDArray[Any]:
    """
    Compute element-wise proportional (relative) error between estimate and reference.

    :param estimate: Array of estimated values.
    :param reference: Array of reference (ground truth) values.
    :return: Array of proportional errors (absolute error divided by reference).
    """
    return np.abs(estimate - reference) / reference


class BinningMethodInterface(Protocol):
    """
    Protocol defining the interface for binning methods.

    Implementations should partition predicted scores into bins and return bin boundaries.
    """

    def __call__(
        self,
        predicted_scores: npt.NDArray[Any],
        num_bins: int,
        epsilon: float = BIN_EPSILON,
    ) -> npt.NDArray[Any]: ...


def make_equispaced_bins(
    predicted_scores: npt.NDArray[Any],
    num_bins: int,
    epsilon: float = BIN_EPSILON,
    set_range_to_zero_one: bool = True,
) -> npt.NDArray[Any]:
    """
    Create bins with equal width (equispaced) for predicted scores.

    For example, with num_bins=5 and set_range_to_zero_one=True, the bins would be
    approximately [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] (before epsilon adjustments).

    :param predicted_scores: Array of predicted probability scores.
    :param num_bins: Number of bins to create.
    :param epsilon: Small offset applied to bin edges to ensure all values are captured.
    :param set_range_to_zero_one: If True, bins span [0, 1]; otherwise, bins span
        the min/max of predicted_scores.
    :return: Array of bin boundaries with length num_bins + 1.
    """
    upper_bound = max(1, predicted_scores.max())

    bins = (
        np.linspace(0, 1, num_bins + 1)
        if set_range_to_zero_one
        else np.linspace(predicted_scores.min(), predicted_scores.max(), num_bins + 1)
    )
    bins[0] = -epsilon if set_range_to_zero_one else predicted_scores.min() - epsilon
    bins[-1] = (
        upper_bound + epsilon
        if set_range_to_zero_one
        else predicted_scores.max() + epsilon
    )
    return bins


def make_equisized_bins(
    predicted_scores: npt.NDArray[Any],
    num_bins: int,
    epsilon: float = BIN_EPSILON,
    **kwargs: Any,  # noqa: ARG001
) -> npt.NDArray[Any]:
    """
    Create bins with approximately equal number of samples (quantile-based).

    :param predicted_scores: Array of predicted probability scores.
    :param num_bins: Target number of bins. Actual number may be fewer if there
        are many duplicate values.
    :param epsilon: Small offset applied to the upper bin edge.
    :param kwargs: Additional arguments (unused, for interface compatibility).
    :return: Array of bin boundaries.
    """
    upper_bound = max(1, predicted_scores.max())
    bins = np.array(
        sorted(
            pd.qcut(
                predicted_scores, q=num_bins, duplicates="drop"
            ).categories.left.tolist()
        )
        + [upper_bound + epsilon]
    )
    return bins


def positive_label_proportion(
    labels: npt.NDArray[Any],
    predictions: npt.NDArray[Any],
    bins: npt.NDArray[Any],
    sample_weight: npt.NDArray[Any] | None = None,
    alpha: float = 0.05,
    use_weights_in_sample_size: bool = False,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Computes the proportion of positive labels in each bin.

    Additionally, it computes the lower and upper bounds of the Confidence Interval
    for the proportion using the Clopper-Pearson method
    (https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval).

    :param labels: array of labels
    :param predictions: array of predictions
    :param bins: array of bin boundaries
    :param sample_weight: array of weights for each instance. If None, then all
        instances are considered to have weight 1
    :param alpha: 1-alpha is the confidence level of the CI
    :param use_weights_in_sample_size: the effective sample size of this dataset
        depends on how the weights in this dataset were generated. This should be
        set to True in the case of Option 1 below and set to False in the case of
        Option 2 below.
        Option 1. it could be the case that there once existed a dataset that for
        example had 10 rows with score 0.6 and label 1 and 100 rows with score 0.1
        and label 0 that has been turned into an aggregated dataset with one row
        with weight 10 and score 0.6 and label 1 with weight 10 and a row with
        score 0.1 and label 0 with weight 100.
        Option 2. it could also be the case that weights merely reflects the inverse
        of the sampling probability of the instance.
    :return: Tuple of four arrays:
        - label_proportion: Proportion of positive labels in each bin.
        - lower: Lower bound of the confidence interval.
        - upper: Upper bound of the confidence interval.
        - score_average: Average predicted score in each bin.
    """
    if np.any(np.isnan(predictions)):
        raise ValueError("predictions must not contain NaNs")
    sample_weight = sample_weight if sample_weight is not None else np.ones_like(labels)

    label_binned_preds = pd.DataFrame(
        {
            "label_weighted": labels * sample_weight,
            "score_weighted": predictions * sample_weight,
            "n_sample_weighted": sample_weight,
            "n_sample_unweighted": np.ones_like(labels),
            "assigned_bin": bins[np.digitize(predictions, bins)],
        }
    )

    bin_means = (
        label_binned_preds[
            [
                "assigned_bin",
                "label_weighted",
                "score_weighted",
                "n_sample_weighted",
                "n_sample_unweighted",
            ]
        ]
        .groupby("assigned_bin")
        .sum()
    )

    # Compute average label
    bin_means["label_proportion"] = (
        bin_means["label_weighted"] / bin_means["n_sample_weighted"]
    )

    # Compute average score
    bin_means["score_average"] = (
        bin_means["score_weighted"] / bin_means["n_sample_weighted"]
    )

    # Compute confidence intervals
    def _row_ci(row: pd.Series) -> pd.Series:
        if use_weights_in_sample_size:
            n_positive = row["label_weighted"]
            n = row["n_sample_weighted"]
        else:
            n = row["n_sample_unweighted"]
            n_positive = int(row["label_proportion"] * n)

        lower = stats.beta.ppf(alpha / 2, n_positive, n - n_positive + 1)
        upper = stats.beta.ppf(1 - alpha / 2, n_positive + 1, n - n_positive)
        return pd.Series({"lower": lower, "upper": upper})

    cis = bin_means.apply(_row_ci, axis=1)

    # Rather than using bin_means directly, we create a new DataFrame and update, to
    # ensure consistent shape of the output array when there exists bins without predictions.
    prop_pos_label = pd.DataFrame(
        index=bins,
        columns=["label_proportion", "score_average", "lower", "upper"],
        data=np.nan,
    )
    prop_pos_label.update(bin_means["label_proportion"])
    prop_pos_label.update(bin_means["score_average"])
    prop_pos_label.update(cis["lower"])
    prop_pos_label.update(cis["upper"])

    return (
        prop_pos_label.label_proportion.values,
        prop_pos_label.lower.values,
        prop_pos_label.upper.values,
        prop_pos_label.score_average.values,
    )


def geometric_mean(x: npt.NDArray[Any]) -> float:
    """
    Computes the geometric mean of an array of numbers. If any of the numbers are 0, then the geometric mean is 0.
    The exp-log trick is used to avoid underflow/overflow problems when computing the product of many numbers.

    :param x: array of numbers
    :return: geometric mean of the array
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.exp(np.log(x).mean())


def make_unjoined(
    x: npt.NDArray[Any], y: npt.NDArray[Any]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Converts a regular dataset to 'unjoined' format. In the unjoined format, there is always
    a row with a negative label and there will be a second row with a positive label added to
    the dataset for the same instance if is actually a positive instance. This contrasts a
    regular dataset where each instance is represented by a single row with either a positive
    or negative label.

    This method takes a regular dataset and returns an unjoined version of that dataset.

    :param x: array of features
    :param y: array of labels
    :return: tuple of arrays (x_unjoined, y_unjoined)
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of instances")
    # Find the indices where y is positive, create duplicates for those instances
    positive_indices = np.where(y == 1)[0]
    unjoined_x = np.concatenate([x, x[positive_indices]])
    # Create an array of artificial negatives
    artificial_negatives = np.zeros(len(positive_indices), dtype=y.dtype)
    unjoined_y = np.concatenate([y, artificial_negatives])
    return unjoined_x, unjoined_y


class OrdinalEncoderWithUnknownSupport(OrdinalEncoder):
    """
    Extends the scikit-learn OrdinalEncoder by addressing the issue that the transform method
    of the OrdinalEncoder raises an error if any of the categorical features contains categories
    that were never observed when fitting the encoder. This encoder assigns value -1 to all
    unknown categories.

    Note: this is only needed in scikit-learn version 0.22. In later versions, scikit-learn's
    OrdinalEncoder supports unknown categories using the handle_unknown and unknown_value arguments.
    """

    _category_map: dict[int, dict[Any, int]]
    categories_: list[npt.NDArray[Any]]

    def __init__(self, categories: str = "auto", dtype: type[Any] = np.float64) -> None:
        """
        :param categories: Categories per feature. See sklearn.OrdinalEncoder.
        :param dtype: Desired dtype of output.
        """
        super().__init__(categories=categories, dtype=dtype)
        self._category_map = {}
        self.categories_ = []

    def fit(
        self, X: npt.NDArray[Any] | pd.DataFrame, y: Any = None
    ) -> "OrdinalEncoderWithUnknownSupport":
        """
        Fit the encoder to the given data.

        :param X: Array-like of shape (n_samples, n_features).
        :param y: Ignored, present for API compatibility.
        :return: Self.
        """

        def convert_to_native(obj: Any) -> Any:
            if hasattr(obj, "item"):
                return obj.item()
            return obj

        X = X.values if isinstance(X, pd.DataFrame) else X
        super().fit(X, y)
        for i, category in enumerate(self.categories_):
            self._category_map[i] = {
                convert_to_native(value): index for index, value in enumerate(category)
            }
        return self

    def transform(self, X: npt.NDArray[Any] | pd.DataFrame) -> npt.NDArray[Any]:
        """
        Transform categorical features to ordinal integers.

        Unknown categories are encoded as -1.

        :param X: Array-like of shape (n_samples, n_features).
        :return: Transformed array with integer codes.
        :raises ValueError: If fit has not been called.
        """
        X = X.values if isinstance(X, pd.DataFrame) else X
        if not self._category_map:
            raise ValueError("The fit method should be called before transform.")
        X_transformed = np.empty(X.shape, dtype=int)
        for i in range(X.shape[1]):
            col = X[:, i]
            category_map = self._category_map[i]
            col_series = pd.Series(col)
            X_transformed[:, i] = (
                col_series.map(category_map).fillna(-1).astype(int).values
            )
        return X_transformed

    def serialize(self) -> str:
        """
        Serialize the encoder's category mapping to a string.

        :return: String representation of the category mapping.
        """
        return str(self._category_map)

    @classmethod
    def deserialize(cls, encoder_str: str) -> "OrdinalEncoderWithUnknownSupport":
        """
        Deserialize an encoder from a string.

        :param encoder_str: String representation of the category mapping.
        :return: Reconstructed encoder instance.
        """
        enc = cls()
        enc._category_map = ast.literal_eval(encoder_str)
        return enc


def hash_categorical_feature(categorical_feature: str) -> int:
    """
    Hashes a categorical feature using the last two bytes of SHA256.

    The equivalent encoding in Presto can be done with:
        FROM_BASE(SUBSTR(TO_HEX(SHA256(CAST(categorical_feature AS VARBINARY))), -4), 16)
    """
    signature = hashlib.sha256(categorical_feature.encode("utf-8")).digest().hex()
    last_four_hex_chars = signature[-4:]
    return int(last_four_hex_chars, 16)


def rank_log_discount(n_samples: int, log_base: int = 2) -> npt.NDArray[Any]:
    """
    Rank log discount function used for the rank metrics DCG and NDCG.
    More information about the function here: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Discounted_Cumulative_Gain.

    :param n_samples: number of samples
    :param log_base: base of the logarithm
    :return: array of size n_samples with the discount factor for each sample
    """
    return np.asarray(1 / (np.log(np.arange(n_samples) + 2) / np.log(log_base)))


def rank_no_discount(num_samples: int) -> npt.NDArray[Any]:
    """
    Rank discount function used for the rank metrics DCG and NDCG.
    Returns uniform discount factor of 1 for all samples.

    :param num_samples: number of samples
    :return: array of size num_samples with the value of 1 as the discount factor for each sample
    """
    return np.ones(num_samples)


class TrainTestSplitWrapper:
    def __init__(
        self,
        test_size: float = 0.4,
        shuffle: bool = False,
        random_state: int | None = None,
        stratify: bool = True,
    ) -> None:
        """
        Customized train-test split class that allows to specify the test size (fraction).
        This is useful for the case where we want to have a single split with given test size, rather than doing k-fold crossvalidation.

        :param test_size: Size of the test set as a fraction of the total dataset size.
        :param shuffle: Whether to shuffle the data before splitting.
        :param random_state: Random state for reproducibility.
        :param stratify: Whether to stratify the split based on labels.
        """
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify = stratify

    def split(
        self, X: npt.NDArray[Any], y: npt.NDArray[Any], groups: Any = None
    ) -> Iterator[tuple[npt.NDArray[Any], npt.NDArray[Any]]]:
        """
        Generate a single train/validation split.

        :param X: Feature matrix (used only for determining array length).
        :param y: Label array used for stratification if enabled.
        :param groups: Ignored, included for API compatibility.
        :yields: Tuple of (train_indices, validation_indices).
        """
        train_idx, val_idx = train_test_split(
            np.arange(len(y)),
            test_size=self.test_size,
            shuffle=self.shuffle,
            stratify=y if self.stratify else None,
            random_state=self.random_state,
        )
        yield train_idx, val_idx


class NoopSplitterWrapper:
    def __init__(
        self,
    ) -> None:
        """
        This splitter returns the training set as it is and an empty test set.
        """

    def split(
        self, X: npt.NDArray[Any], y: npt.NDArray[Any], groups: Any = None
    ) -> Any:
        """
        Return all data as training set with empty validation set.

        :param X: Feature matrix (unused).
        :param y: Label array used to determine dataset size.
        :param groups: Ignored, included for API compatibility.
        :yields: Tuple of (all indices, empty list).
        """
        yield np.arange(len(y)), []  # train_idx, val_idx


def convert_arrow_columns_to_numpy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Arrow-backed columns in a DataFrame to NumPy arrays.

    :param df: DataFrame potentially containing Arrow extension arrays.
    :return: DataFrame with all Arrow columns converted to NumPy.
    """
    for col in df.columns:
        if isinstance(df[col].values, pd.core.arrays.ArrowExtensionArray):
            df[col] = df[col].to_numpy()
    return df


def check_range(series: pd.Series, precision_type: str) -> bool:
    """
    Check if values in a series are within the valid range for a floating-point precision type.

    Also checks that the sum does not exceed the square root of the max value to avoid
    overflow during aggregation operations.

    :param series: Pandas Series of numeric values.
    :param precision_type: One of 'float16', 'float32', or 'float64'.
    :return: True if all values are within range, False otherwise.
    """
    precision_limits = {
        "float16": (np.finfo(np.float16).min, np.finfo(np.float16).max),
        "float32": (np.finfo(np.float32).min, np.finfo(np.float32).max),
        "float64": (np.finfo(np.float64).min, np.finfo(np.float64).max),
    }

    min_val, max_val = precision_limits[precision_type]
    return not (
        (series.min() < min_val)
        or (series.max() > max_val)
        or (series.sum() > math.sqrt(max_val))
    )


def log_peak_rss(
    samples_per_second: float = 10.0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator factory to log peak RSS (Resident Set Size) while a function runs.

    Spawns a background thread that periodically samples memory usage and logs
    the peak observed value along with start/end memory and duration.

    :param samples_per_second: Sampling frequency for memory monitoring.
        E.g., 10.0 samples 10 times per second, 2.0 samples twice per second.
    :return: Decorator that wraps functions with memory logging.
    :raises ValueError: If samples_per_second is not positive.

    Example usage::

        @log_peak_rss()
        def memory_intensive_function():
            ...

        @log_peak_rss(2.0)
        def another_function():
            ...
    """
    if samples_per_second <= 0:
        raise ValueError("samples_per_second must be >0")

    sample_interval: float = 1.0 / samples_per_second

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        log: logging.Logger = logging.getLogger(func.__module__)
        func_name: str = func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Construct process object per call (cheap and fork-safe)
            process: psutil.Process = psutil.Process(os.getpid())

            start_rss = process.memory_info().rss
            peak_rss = start_rss
            stop_event: threading.Event = threading.Event()

            def sampler() -> None:
                nonlocal peak_rss
                while not stop_event.is_set():
                    rss = process.memory_info().rss
                    if rss > peak_rss:
                        peak_rss = rss
                    time.sleep(sample_interval)

            t0 = time.time()
            thread = threading.Thread(target=sampler, daemon=True)
            thread.start()
            try:
                return func(*args, **kwargs)
            finally:
                stop_event.set()
                thread.join()
                end_rss = process.memory_info().rss
                log.info(
                    "%s: rss_start=%.1f MB, rss_end=%.1f MB, peak_observed=%.1f MB, "
                    "duration=%.2fs",
                    func_name,
                    start_rss / 1024**2,
                    end_rss / 1024**2,
                    peak_rss / 1024**2,
                    time.time() - t0,
                )

        return wrapper

    return decorator


def predictions_to_labels(
    data: pd.DataFrame,
    prediction_column: str,
    thresholds: pd.DataFrame,
    threshold_column: str | None = "threshold",
) -> pd.DataFrame:
    """
    Convert prediction scores to binary labels using segment-specific thresholds.

    This utility function merges a DataFrame containing predictions with a thresholds
    DataFrame that specifies decision thresholds per segment. It then creates a new
    'predicted_label' column by comparing each prediction against its segment's threshold.

    :param data: DataFrame containing predictions and segmentation columns.
    :param prediction_column: Name of the column containing prediction scores.
    :param thresholds: DataFrame with threshold values per segment. Must contain
        the segmentation columns (used for merging) and a threshold column.
    :param threshold_column: Name of the column in thresholds containing threshold values.
    :return: DataFrame with an added 'predicted_label' column (1 if prediction >= threshold, else 0).
    """
    segmentation_columns = [c for c in thresholds.columns if c != threshold_column]
    data_w_thresholds = data.copy().merge(
        thresholds, on=segmentation_columns, how="left"
    )
    data_w_thresholds["predicted_label"] = (
        data_w_thresholds[prediction_column] >= data_w_thresholds.threshold
    ).astype(int)
    return data_w_thresholds
