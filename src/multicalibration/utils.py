# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

import ast
import functools
import hashlib
import logging
import math
import os
import threading
import time
import warnings
from typing import Any, Dict, Protocol, Tuple

import numpy as np
import pandas as pd

import psutil
import torch
from scipy import stats
from scipy.optimize._linesearch import LineSearchWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

logger = logging.getLogger(__name__)


def unshrink(
    y: np.ndarray,
    logits: np.ndarray,
    w: np.ndarray | None = None,
    logit_epsilon: float | None = 10,
) -> float:
    if w is None:
        w = np.ones_like(y)
    logits = logits.reshape(-1, 1)

    # Clip logits to avoid extreme coefficient driven by outliers
    if logit_epsilon is not None:
        logits = np.clip(logits, -logit_epsilon, logit_epsilon)

    primary_solver = LogisticRegression(
        fit_intercept=False, solver="newton-cg", penalty=None
    )
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        primary_solver.fit(logits, y, sample_weight=w)
    for rec_warn in recorded_warnings:
        if isinstance(rec_warn.message, LineSearchWarning):
            logger.info(
                f"Line search warning (unshrink): {str(rec_warn.message)}. Solution is approximately optimal - no ideal step size for the gradient descent update can be found. These warnings are generally harmless."
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

    # Return result if logistic regression with Newton-CG converged to a solution, if no try LBFGS.
    # pyre-ignore, coef_ is available after `fit()` has been called
    if not np.isnan(primary_solver.coef_).any():
        if primary_solver.coef_[0][0] < 0.95 or primary_solver.coef_[0][0] > 1.05:
            logger.warning(
                f"Unshrink is not close to 1: {primary_solver.coef_[0][0]}. This may create a problem with the multicalibration of the model."
            )

        return primary_solver.coef_[0][0]

    fallback_solver = LogisticRegression(
        fit_intercept=False, solver="lbfgs", penalty=None
    )
    fallback_solver.fit(logits, y, sample_weight=w)
    if not np.isnan(fallback_solver.coef_).any():
        if primary_solver.coef_[0][0] < 0.95 or primary_solver.coef_[0][0] > 1.05:
            logger.warning(
                f"Unshrink is not close to 1: {primary_solver.coef_[0][0]}. This may create a problem with the multicalibration of the model."
            )
        return fallback_solver.coef_[0][0]

    # If both solvers fail, return default value. Not disastrous, but requires GBDT to do more heavy-lifting.
    return 1


def logistic(logits: float) -> float:
    # Numerically stable sigmoid - Computational trick to avoid overflow/underflow
    if logits >= 0:
        return 1.0 / (1.0 + math.exp(-logits))
    else:
        return math.exp(logits) / (1.0 + math.exp(logits))


logistic_vectorized = np.vectorize(logistic)


def logit(probs: np.ndarray, epsilon=1e-304) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.log((probs + epsilon) / (1 - probs + epsilon))


def absolute_error(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(estimate - reference)


def proportional_error(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(estimate - reference) / reference


class BinningMethodInterface(Protocol):
    def __call__(
        self,
        predicted_scores: np.ndarray,
        num_bins: int,
        epsilon: float = 1e-8,
    ) -> np.ndarray: ...


def make_equispaced_bins(
    predicted_scores: np.ndarray,
    num_bins: int,
    epsilon: float = 1e-8,
    set_range_to_zero_one: bool = True,
) -> np.ndarray:
    lower_bound = min(0, predicted_scores.min())
    upper_bound = max(1, predicted_scores.max())

    bins = (
        np.linspace(0, 1, num_bins + 1)
        if set_range_to_zero_one
        else np.linspace(predicted_scores.min(), predicted_scores.max(), num_bins + 1)
    )
    bins[0] = (
        lower_bound - epsilon
        if set_range_to_zero_one
        else predicted_scores.min() - epsilon
    )
    bins[-1] = (
        upper_bound + epsilon
        if set_range_to_zero_one
        else predicted_scores.max() + epsilon
    )
    return bins


def make_equisized_bins(
    predicted_scores: np.ndarray,
    num_bins: int,
    epsilon: float = 1e-8,
    **kwargs: Any,
) -> np.ndarray:
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
    labels: np.ndarray,
    predictions: np.ndarray,
    bins: np.ndarray,
    sample_weight: np.ndarray | None = None,
    alpha: float = 0.05,
    use_weights_in_sample_size: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the proportion of positive labels in each bin. Additionally, it computes the lower and upper bounds of the Confidence Interval for the proportion
    using the Clopper-Pearson method (https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval).

    :param labels: array of labels
    :param predictions: array of predictions
    :param bins: array of bin boundaries
    :param sample_weight: array of weights for each instance. If None, then all instances are considered to have weight 1
    :param alpha: 1-alpha is the confidence level of the CI
    :param use_weights_in_sample_size: the effective sample size of this dataset depends on how the weights in this dataset were generated. This
        should be set to True in the case of Option 1 below and set to False in the case of Option 2 below.
        Option 1. it could be the case that there once existed a dataset that for example had 10 rows with score 0.6 and label 1 and 100 rows
        with score 0.1 and label 0 that has been turned into an aggregated dataset with one row with weight 10 and score 0.6 and label 1 with
        weight 10 and a row with score 0.1 and label 0 with weight 100.
        Option 2. it could also be the case that weights merely reflects the inverse of the sampling probability of the instance.

    :return: array of proportions
    """
    assert not np.any(np.isnan(predictions)), "predictions must not contain NaNs"
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
    def _row_ci(row):
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


def geometric_mean(x: np.ndarray) -> float:
    """
    Computes the geometric mean of an array of numbers. If any of the numbers are 0, then the geometric mean is 0.
    The exp-log trick is used to avoid underflow/overflow problems when computing the product of many numbers.

    :param x: array of numbers
    :return: geometric mean of the array
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.exp(np.log(x).mean())


def make_unjoined(x: np.ndarray, y: np.ndarray) -> tuple[Any, Any]:
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
    assert x.shape[0] == y.shape[0], "x and y must have the same number of instances"
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

    def __init__(self, categories="auto", dtype=np.float64):
        super().__init__(categories=categories, dtype=dtype)
        self._category_map = {}

    def fit(self, X, y=None):
        X = X.values if isinstance(X, pd.DataFrame) else X
        super().fit(X, y)
        for i, category in enumerate(self.categories_):
            self._category_map[i] = {
                value: index for index, value in enumerate(category)
            }
        return self

    def transform(self, X):
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
        return str(self._category_map)

    @classmethod
    def deserialize(cls, encoder_str) -> "OrdinalEncoderWithUnknownSupport":
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


def rank_log_discount(n_samples: int, log_base: int = 2) -> np.ndarray:
    """
    Rank log discount function used for the rank metrics DCG and NDCG.
    More information about the function here: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Discounted_Cumulative_Gain.

    :param n_samples: number of samples
    :param log_base: base of the logarithm
    :return: array of size n_samples with the discount factor for each sample
    """
    return 1 / (np.log(np.arange(n_samples) + 2) / np.log(log_base))


def rank_no_discount(num_samples: int) -> np.ndarray:
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
        :param test_size: size of the test set as a fraction of the total size of the dataset.
        :param shuffle: whether to shuffle the data before splitting;
        :param random_state: random state;
        """
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify = stratify

    def split(self, X, y, groups=None):
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

    def split(self, X, y, groups=None):
        yield np.arange(len(y)), []  # train_idx, val_idx


def convert_arrow_columns_to_numpy(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if isinstance(df[col].values, pd.core.arrays.ArrowExtensionArray):
            df[col] = df[col].to_numpy()
    return df


# Check if the values in the columns are within the valid range
def check_range(series, precision_type):
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


# ========================================= #
# FEATURE PREPROCESSING FUNCTIONS FOR MCNET #
# ========================================= #


class FeatureProcessorState:
    def __init__(self):
        self.enc: OrdinalEncoderWithUnknownSupport | None = None
        self.one_hot_enc: OneHotEncoder | None = None
        self.missing_value_fill: Dict[str, float] = {}
        self.feature_means: torch.Tensor | None = None
        self.feature_stds: torch.Tensor | None = None
        self.categorical_segment_cols: list[str] = []
        self.numerical_segment_cols: list[str] = []
        self.feature_dim: int = 0
        self._is_fitted: bool = False
        self.category_mappings: Dict[str, Dict[str, str]] = {}
        self.max_n_categories: int | None = None


def extract_segment_features(
    df: pd.DataFrame,
    categorical_segment_cols: list[str] | None = None,
    numerical_segment_cols: list[str] | None = None,
    processor_state: FeatureProcessorState | None = None,
    max_n_categories: int | None = None,
    is_fit_phase: bool = False,
    fillna: bool = True,
) -> Tuple[torch.Tensor, FeatureProcessorState]:
    if processor_state is None:
        processor_state = FeatureProcessorState()
        processor_state.max_n_categories = max_n_categories

    if is_fit_phase:
        processor_state.categorical_segment_cols = categorical_segment_cols or []
        processor_state.numerical_segment_cols = numerical_segment_cols or []

    feature_list = []

    if categorical_segment_cols:
        cat_features = _process_categorical_features(
            df, categorical_segment_cols, processor_state, is_fit_phase
        )
        feature_list.append(cat_features)

    if numerical_segment_cols:
        num_features = _process_numerical_features(
            df, numerical_segment_cols, processor_state, is_fit_phase, fillna
        )
        feature_list.append(num_features)

    if feature_list:
        combined_features = torch.cat(feature_list, dim=1)
    else:
        combined_features = torch.empty((len(df), 0), dtype=torch.float32)

    if is_fit_phase:
        processor_state.feature_dim = combined_features.shape[1]
        processor_state._is_fitted = True

    return combined_features, processor_state


def collapse_categorical_features_by_frequency(
    df: pd.DataFrame,
    categorical_cols: list[str],
    max_n_categories: int,
    processor_state: FeatureProcessorState,
    is_fit_phase: bool = True,
) -> pd.DataFrame:
    """
    Collapse categorical features to max_n_categories based on their frequency.
    Keeps the most frequent categories and groups the rest into an "OTHER" category.
    """
    df_collapsed = df.copy()

    if is_fit_phase:
        processor_state.max_n_categories = max_n_categories
        processor_state.category_mappings = {}

    for col in categorical_cols:
        if col not in df_collapsed.columns:
            continue

        if is_fit_phase:
            value_counts = df_collapsed[col].value_counts()

            if len(value_counts) <= max_n_categories:
                mapping = {}
                for category in value_counts.index:
                    mapping[category] = category

                processor_state.category_mappings[col] = mapping
            else:
                top_categories = value_counts.head(max_n_categories - 1).index.tolist()

                mapping = {}
                for category in value_counts.index:
                    if category in top_categories:
                        mapping[category] = category
                    else:
                        mapping[category] = "OTHER"

                processor_state.category_mappings[col] = mapping

                df_collapsed[col] = (
                    df_collapsed[col].map(mapping).fillna(df_collapsed[col])
                )
        else:
            if col in processor_state.category_mappings:
                mapping = processor_state.category_mappings[col]
                mapped_values = df_collapsed[col].map(mapping)
                df_collapsed[col] = mapped_values.fillna("OTHER")

    return df_collapsed


def _process_categorical_features(
    df: pd.DataFrame,
    categorical_segment_cols: list[str],
    processor_state: FeatureProcessorState,
    is_fit_phase: bool,
) -> torch.Tensor:
    """Optimized categorical feature processing."""
    cat_data = df[categorical_segment_cols].copy()

    cat_data = cat_data.fillna("__MISSING__")

    if processor_state.max_n_categories is not None:
        cat_data = collapse_categorical_features_by_frequency(
            cat_data,
            categorical_segment_cols,
            processor_state.max_n_categories,
            processor_state,
            is_fit_phase,
        )

    cat_features = _encode_categorical_features(cat_data, processor_state, is_fit_phase)
    return torch.tensor(cat_features, dtype=torch.float32)


def _encode_categorical_features(
    cat_data: pd.DataFrame,
    processor_state: FeatureProcessorState,
    is_fit_phase: bool,
) -> np.ndarray:
    cat_features = cat_data.values

    if is_fit_phase:
        processor_state.enc = OrdinalEncoderWithUnknownSupport()
        if processor_state.enc is not None:
            processor_state.enc.fit(cat_features)

    if processor_state.enc is not None:
        cat_features = processor_state.enc.transform(cat_features)
    else:
        raise ValueError("Fit has to be called before encoder can be applied.")

    if np.nanmax(cat_features) >= np.iinfo(np.int32).max:
        raise ValueError(
            "All categorical feature values must be smaller than 2^32 to prevent integer overflow."
        )

    if is_fit_phase:
        processor_state.one_hot_enc = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",  # This will create zeros for unknown categories
            dtype=np.float32,
        )
        processor_state.one_hot_enc.fit(cat_features)

    if processor_state.one_hot_enc is not None:
        cat_features = processor_state.one_hot_enc.transform(cat_features)
    else:
        raise ValueError(
            "OneHotEncoder fit has to be called before transform can be applied."
        )

    return cat_features


def _process_numerical_features(
    df: pd.DataFrame,
    numerical_segment_cols: list[str],
    processor_state: FeatureProcessorState,
    is_fit_phase: bool,
    fillna: bool = True,
) -> torch.Tensor:
    num_data = torch.tensor(df[numerical_segment_cols].values, dtype=torch.float32)

    nan_mask = torch.isnan(num_data)
    has_nan = nan_mask.any()
    if has_nan:
        missing_info = []
        for i, col_name in enumerate(numerical_segment_cols):
            col_nan_count = nan_mask[:, i].sum().item()
            if col_nan_count > 0:
                missing_pct = (col_nan_count / num_data.shape[0]) * 100
                missing_info.append(
                    f"{col_name}: {missing_pct:.1f}% ({col_nan_count}/{num_data.shape[0]} rows)"
                )

        if fillna:
            logger.warning(
                f"Found missing values in numerical features, imputing with mean. "
                f"Missing data breakdown: {', '.join(missing_info)}. "
                "Set fillna=False to raise an error instead."
            )
        else:
            raise ValueError(
                f"Found missing values in numerical features. "
                f"Missing data breakdown: {', '.join(missing_info)}. "
                "Set fillna=True to replace with mean or handle NaN values beforehand."
            )

    presence_mask = (~torch.isnan(num_data)).float()
    if is_fit_phase:
        processor_state.feature_means = torch.nanmean(num_data, dim=0)

        feature_stds = []
        for i in range(num_data.shape[1]):
            col_data = num_data[:, i]
            valid_mask = ~torch.isnan(col_data)
            if valid_mask.any():
                std_val = torch.std(col_data[valid_mask])
            else:
                std_val = torch.tensor(1.0)
            feature_stds.append(std_val)

        feature_stds = torch.stack(feature_stds)

        processor_state.feature_stds = torch.where(
            feature_stds == 0.0,
            torch.ones_like(feature_stds),
            feature_stds,
        )

    num_data_filled = (
        torch.where(torch.isnan(num_data), torch.zeros_like(num_data), num_data)
        if fillna
        else num_data
    )
    if (
        processor_state.feature_means is not None
        and processor_state.feature_stds is not None
    ):
        feature_means = processor_state.feature_means
        feature_stds = processor_state.feature_stds
        num_data_filled = (num_data_filled - feature_means) / feature_stds

    combined_features = torch.cat([num_data_filled, presence_mask], dim=1)
    return combined_features


def log_peak_rss(samples_per_second=10.0):
    """
    Decorator factory to log peak RSS while a function runs.

    samples_per_second: how often to sample memory, e.g.
        @log_peak_rss()        # 10 samples per second (default)
        @log_peak_rss(2.0)     # 2 samples per second
    """
    if samples_per_second <= 0:
        raise ValueError("samples_per_second must be > 0")

    sample_interval = 1.0 / samples_per_second

    def decorator(func):
        log = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Construct process object per call (cheap and fork-safe)
            process = psutil.Process(os.getpid())

            start_rss = process.memory_info().rss
            peak_rss = start_rss
            stop_event = threading.Event()

            def sampler():
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
                    func.__name__,
                    start_rss / 1024**2,
                    end_rss / 1024**2,
                    peak_rss / 1024**2,
                    time.time() - t0,
                )

        return wrapper

    return decorator
