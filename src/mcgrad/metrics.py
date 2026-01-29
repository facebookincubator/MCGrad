# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict

"""
Calibration and multicalibration evaluation metrics.

This module provides a comprehensive suite of metrics for evaluating the calibration
quality of probabilistic predictions, with a focus on multicalibration—calibration
across multiple subpopulations.

Key metric families include:

**Calibration Error Metrics**
    Standard and adaptive calibration error measures using binning approaches.

**ECCE (Estimated Cumulative Calibration Error) Metrics**
    Statistical tests and metrics based on the ECCE statistic (also known as Kuiper calibration).

**Ranking Metrics**
    Evaluation metrics for ranked predictions (DCG, NDCG, etc.).

**Classification Metrics**
    Standard precision, recall, and threshold-based metrics.
"""

import functools
import logging
import math
import sys
import warnings
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn import metrics as skmetrics

from . import _utils as utils
from ._compat import groupby_apply
from ._segmentation import get_segment_masks
# @oss-disable[end= ]: from .internal._compat import (
    # @oss-disable[end= ]: apply_mce_transition_overrides,
    # @oss-disable[end= ]: MulticalibrationErrorCompatMixin,
# @oss-disable[end= ]: )


logger: logging.Logger = logging.getLogger(__name__)
CALIBRATION_ERROR_NUM_BINS = 40
CALIBRATION_ERROR_EPSILON = 0.0000001
DEFAULT_PRECISION_DTYPE = np.float64

# ECCE sigma distribution constants
# ECCE_SIGMA_MAX: Maximum statistic value before CDF is effectively 1.0
# ECCE_SIGMA_MIN: Minimum statistic value below which p-value is 1.0
ECCE_SIGMA_MAX: float = 8.26732673
ECCE_SIGMA_MIN: float = 1e-20


def _calibration_error(
    labels: npt.NDArray,
    predictions: npt.NDArray,
    bins: npt.NDArray,
    bin_error_func: Callable[
        [npt.NDArray, npt.NDArray], npt.NDArray
    ] = utils.absolute_error,
    adjust_unjoined: bool = False,
    sample_weight: npt.NDArray | None = None,
) -> float:
    """Calculate the calibration error.

    :param labels: Array of true labels
    :param predictions: Array of predicted scores
    :param bins: Array of bin edges for bucketing the predictions
    :param bin_error_func: Function to calculate the error between the empirically observed rate and the estimated rate
    :param adjust_unjoined: Boolean flag indicating whether the input data is "unjoined data". In unjoined data there is
                           always a row with a negative label and there will be another row with positive label if it is a positive instance.
                           This means that for positive instances there are two rows: one with a positive and one with a negative label. On
                           unjoined datasets we need to make an adjustment to get an unbiased estimate of calibration error
    :param sample_weight: Array of weights for each instance. If None, then all instances are considered to have weight 1
    :return: The calibration error as a float
    """
    sample_weight = sample_weight if sample_weight is not None else np.ones_like(labels)

    label_binned_preds = pd.DataFrame(
        {
            "label": labels,
            "label_weighted": labels * sample_weight,
            "prediction": predictions,
            "prediction_weighted": predictions * sample_weight,
            "sample_weight": sample_weight,
            "assigned_bin": bins[np.digitize(predictions, bins)],
        }
    )
    metric_input = label_binned_preds.groupby("assigned_bin").aggregate(
        {
            "label_weighted": ["sum", "size"],
            "prediction_weighted": ["sum"],
            "sample_weight": ["sum"],
        }
    )
    metric_input["label_weighted", "mean"] = (
        1.0
        * metric_input["label_weighted", "sum"]
        / metric_input["sample_weight", "sum"]
    )
    metric_input["prediction_weighted", "mean"] = (
        1.0
        * metric_input["prediction_weighted", "sum"]
        / metric_input["sample_weight", "sum"]
    )
    estimated_rate = metric_input["prediction_weighted", "mean"]

    if adjust_unjoined:
        y_pos = label_binned_preds[label_binned_preds["label"] == 1].shape[0]
        y_neg = label_binned_preds[label_binned_preds["label"] == 0].shape[0]
        y_neg_no_unjoin = y_neg - y_pos
        empirically_observed_rate = y_pos / (y_pos + y_neg_no_unjoin)
    else:
        empirically_observed_rate = metric_input["label_weighted", "mean"]

    bin_errors = bin_error_func(empirically_observed_rate, estimated_rate)
    bin_weights = (
        metric_input["sample_weight", "sum"]
        / metric_input["sample_weight", "sum"].sum()
    )
    global_error = (bin_weights * bin_errors).sum()

    return global_error


def expected_calibration_error(
    labels: npt.NDArray,
    predicted_scores: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
    num_bins: int = CALIBRATION_ERROR_NUM_BINS,
    epsilon: float = CALIBRATION_ERROR_EPSILON,
    adjust_unjoined: bool = False,
    **kwargs: Any,
) -> float:
    """
    Calculate the Expected Calibration Error (ECE).

    :param labels: Array of true labels.
    :param predicted_scores: Array of predicted probability scores.
    :param sample_weight: Optional array of sample weights.
    :param num_bins: Number of bins to use for bucketing predictions.
    :param epsilon: Small value to avoid numerical issues at bin boundaries.
    :param adjust_unjoined: Whether to adjust for unjoined data.
    :return: The expected calibration error.
    """
    bins = utils.make_equispaced_bins(predicted_scores, num_bins, epsilon)
    return _calibration_error(
        labels,
        predicted_scores,
        bins,
        adjust_unjoined=adjust_unjoined,
        sample_weight=sample_weight,
    )


def proportional_expected_calibration_error(
    labels: npt.NDArray,
    predicted_scores: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
    num_bins: int = CALIBRATION_ERROR_NUM_BINS,
    epsilon: float = CALIBRATION_ERROR_EPSILON,
    adjust_unjoined: bool = False,
    **kwargs: Any,
) -> float:
    """
    Calculate the Proportional Expected Calibration Error.

    Uses proportional error instead of absolute error for bin error calculation.

    :param labels: Array of true labels.
    :param predicted_scores: Array of predicted probability scores.
    :param sample_weight: Optional array of sample weights.
    :param num_bins: Number of bins to use for bucketing predictions.
    :param epsilon: Small value to avoid numerical issues at bin boundaries.
    :param adjust_unjoined: Whether to adjust for unjoined data.
    :return: The proportional expected calibration error.
    """
    bins = utils.make_equispaced_bins(predicted_scores, num_bins, epsilon)
    return _calibration_error(
        labels,
        predicted_scores,
        bins,
        bin_error_func=utils.proportional_error,
        adjust_unjoined=adjust_unjoined,
        sample_weight=sample_weight,
    )


def adaptive_calibration_error(
    labels: npt.NDArray,
    predicted_scores: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
    num_bins: int = CALIBRATION_ERROR_NUM_BINS,
    epsilon: float = CALIBRATION_ERROR_EPSILON,
    adjust_unjoined: bool = False,
    **kwargs: Any,
) -> float:
    """
    Calculate the Adaptive Calibration Error (ACE).

    Unlike ECE which uses equispaced bins, ACE uses bins with equal numbers of samples.

    :param labels: Array of true labels.
    :param predicted_scores: Array of predicted probability scores.
    :param sample_weight: Optional array of sample weights.
    :param num_bins: Number of bins to use for bucketing predictions.
    :param epsilon: Small value to avoid numerical issues at bin boundaries.
    :param adjust_unjoined: Whether to adjust for unjoined data.
    :return: The adaptive calibration error.
    """
    bins = utils.make_equisized_bins(predicted_scores, num_bins, epsilon)
    return _calibration_error(
        labels,
        predicted_scores,
        bins,
        adjust_unjoined=adjust_unjoined,
        sample_weight=sample_weight,
    )


def proportional_adaptive_calibration_error(
    labels: npt.NDArray,
    predicted_scores: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
    num_bins: int = CALIBRATION_ERROR_NUM_BINS,
    epsilon: float = CALIBRATION_ERROR_EPSILON,
    adjust_unjoined: bool = False,
    **kwargs: Any,
) -> float:
    """
    Calculate the Proportional Adaptive Calibration Error.

    Combines adaptive binning with proportional error calculation.

    :param labels: Array of true labels.
    :param predicted_scores: Array of predicted probability scores.
    :param sample_weight: Optional array of sample weights.
    :param num_bins: Number of bins to use for bucketing predictions.
    :param epsilon: Small value to avoid numerical issues at bin boundaries.
    :param adjust_unjoined: Whether to adjust for unjoined data.
    :return: The proportional adaptive calibration error.
    """
    bins = utils.make_equisized_bins(predicted_scores, num_bins, epsilon)
    return _calibration_error(
        labels,
        predicted_scores,
        bins,
        bin_error_func=utils.proportional_error,
        adjust_unjoined=adjust_unjoined,
        sample_weight=sample_weight,
    )


def calibration_ratio(
    labels: npt.NDArray,
    predicted_scores: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
    adjust_unjoined: bool = False,
    **kwargs: Any,
) -> float:
    """
    Calculate the calibration ratio (sum of predictions / sum of labels).

    A value of 1.0 indicates perfect calibration on aggregate.

    :param labels: Array of true labels.
    :param predicted_scores: Array of predicted probability scores.
    :param sample_weight: Optional array of sample weights.
    :param adjust_unjoined: Whether to adjust for unjoined data.
    :return: The calibration ratio. Returns np.inf if labels sum to zero but
        predictions sum to a positive value. Returns np.nan if both labels
        and predictions sum to zero.
    """
    # equal weighting if no weights given
    if sample_weight is None:
        sample_weight = np.ones_like(predicted_scores)

    # For unjoined data we only sum the predictions of the negatives to avoid double-counting the predicted scores
    # of positive instances, since each positive instance appears as both negative and positive in the data.
    unjoined_adjustment_weights = (
        1 - labels if adjust_unjoined else np.ones_like(predicted_scores)
    )

    ratio = np.sum(
        predicted_scores * sample_weight * unjoined_adjustment_weights
    ) / np.sum(labels * sample_weight)
    return ratio


def recall(
    labels: npt.NDArray,
    predicted_labels: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
    **kwargs: Any,
) -> float:
    """
    Calculate recall (true positive rate).

    :param labels: Array of true binary labels.
    :param predicted_labels: Array of predicted binary labels.
    :param sample_weight: Optional array of sample weights.
    :return: The recall score.
    """
    return skmetrics.recall_score(
        y_true=labels.astype(int),
        y_pred=predicted_labels.astype(int),
        sample_weight=sample_weight,
    )


def precision(
    labels: npt.NDArray,
    predicted_labels: npt.NDArray,
    precision_weight: npt.NDArray | None = None,
    **kwargs: Any,
) -> float:
    """
    Calculate precision (positive predictive value).

    :param labels: Array of true binary labels.
    :param predicted_labels: Array of predicted binary labels.
    :param precision_weight: Optional array of sample weights for precision calculation.
    :return: The precision score.
    """
    return skmetrics.precision_score(
        y_true=labels.astype(int),
        y_pred=predicted_labels.astype(int),
        sample_weight=precision_weight,
    )


def fpr(
    labels: npt.NDArray,
    predicted_labels: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
    **kwargs: Any,
) -> float:
    """
    Calculate the false positive rate (FPR).

    :param labels: Array of true binary labels.
    :param predicted_labels: Array of predicted binary labels.
    :param sample_weight: Optional array of sample weights.
    :return: The false positive rate.
    """
    if len(labels) == 0:
        return 0.0
    cm = skmetrics.confusion_matrix(
        y_true=labels.astype(int), y_pred=predicted_labels, sample_weight=sample_weight
    )
    if cm.shape[0] <= 1:
        return 0.0
    fp = cm[0, 1]
    tn = cm[0, 0]
    if fp + tn == 0:
        return 0.0
    return 1.0 * fp / (fp + tn)


def fpr_with_mask(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    y_mask: npt.NDArray,
    sample_weight: npt.NDArray,
    denominator: float,
) -> float | None:
    """
    Calculate the false positive rate with a mask applied.

    Only samples where `y_mask` is True are considered when counting false positives.
    This is useful for computing FPR within a specific segment or subpopulation
    while using a shared denominator across segments.

    :param y_true: Array of true binary labels.
    :param y_pred: Array of predicted binary labels.
    :param y_mask: Boolean mask array indicating which samples to include in the
        false positive count. Only samples where mask is True contribute to the numerator.
    :param sample_weight: Array of sample weights.
    :param denominator: The denominator to use for FPR calculation (typically the
        weighted count of true negatives, possibly computed over a broader population).
    :return: The false positive rate, or None if denominator is zero.
    """
    if denominator == 0:
        return None
    fp_sr_idx = (y_pred & ~y_true & y_mask).astype(int) * sample_weight
    false_positive_rate = 1.0 * fp_sr_idx.sum() / denominator
    return false_positive_rate


def _dcg_sample_scores(
    labels: npt.NDArray,
    predicted_labels: npt.NDArray,
    rank_discount: Callable[[int], npt.NDArray],
    k: int | None = None,
) -> npt.NDArray:
    """
    Calculates the DCG score for all samples: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Discounted_Cumulative_Gain

    :param labels: Array of true labels.
    :param predicted_labels: Array of predicted labels.
    :param rank_discount: Function that takes the number of samples as input and returns an array of size n_samples.
        with the discount factor for each sample
    :param k: If not None, the DCG score is calculated only for the top k samples. If None, the DCG score is calculated for all samples.
        k cannot be smaller than 1 and cannot be larger than the number of samples.
    :return: the array of size n_samples with the DCG score for each sample. If k is not None, then elements after the k-th one are 0.
    """
    discount = rank_discount(labels.shape[0])

    # check that k is valid
    if k is not None:
        if k < 1:
            raise ValueError("k cannot be less than 1")

        discount[k:] = 0

    ranking = np.argsort(predicted_labels)[::-1]
    ranked = labels[ranking]
    cumulative_gains = np.multiply(discount, ranked)
    cumulative_gains = np.cumsum(cumulative_gains)

    return cumulative_gains


def dcg_score(
    labels: npt.NDArray,
    predicted_labels: npt.NDArray,
    rank_discount: Callable[[int], npt.NDArray] = utils.rank_no_discount,
    k: int | None = None,
) -> float:
    """
    Calculates the DCG score: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Discounted_Cumulative_Gain.

    :param labels: Array of true labels.
    :param predicted_labels: Array of predicted labels.
    :param rank_discount: Function that takes the number of samples as input and returns an array of size n_samples
        with the discount factor for each sample.
    :param k: If not None, the DCG score is calculated only based on the top k samples.
        k cannot be smaller than 1 and cannot be larger than the number of samples.
    :return: the DCG score as a float, or np.nan if the input arrays are empty.
    """
    if labels.shape[0] == 0:
        return np.nan

    scores = _dcg_sample_scores(
        labels, predicted_labels, rank_discount=rank_discount, k=k
    )
    if k is not None:
        k = min(k, labels.shape[0])
        return scores[k - 1]

    return scores[-1]


def _ndcg_sample_scores(
    labels: npt.NDArray,
    predicted_labels: npt.NDArray,
    rank_discount: Callable[[int], npt.NDArray],
    k: int | None = None,
) -> npt.NDArray:
    """
    Calculates the NDCG score for all samples: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Discounted_Cumulative_Gain

    :param labels: Array of true labels.
    :param predicted_labels: Array of predicted labels.
    :param rank_discount: Function that takes the number of samples as input and returns an array of size n_samples
        with the discount factor for each sample
    :param k: If not None, the NDCG score is calculated only for the top k samples. If None, the NDCG score is calculated for all samples.
        k cannot be smaller than 1 and cannot be larger than the number of samples.
    :return: the array of size n_samples with the NDCG score for each sample. If k is not None, then elements after the k-th one are 0.
    """
    gain = _dcg_sample_scores(
        labels, predicted_labels, rank_discount=rank_discount, k=k
    )
    normalizing_gain = _dcg_sample_scores(
        labels, labels, rank_discount=rank_discount, k=k
    )
    all_irrelevant = normalizing_gain == 0
    gain[all_irrelevant] = 0
    gain[~all_irrelevant] /= normalizing_gain[~all_irrelevant]
    return gain


def ndcg_score(
    labels: npt.NDArray,
    predicted_labels: npt.NDArray,
    rank_discount: Callable[[int], npt.NDArray] = utils.rank_no_discount,
    k: int | None = None,
    sample_weight: npt.NDArray | None = None,
) -> float:
    """
    Calculates the Normalized Discounted Cumulative Gain (NDCG)

    :param labels: Array of true labels.
    :param predicted_labels: Array of predicted labels.
    :param rank_discount: Function that takes the number of samples as input and returns an array of size n_samples
        with the discount factor for each sample.
    :param k: If not None, the NDCG score is calculated only based on the top k samples.
        k cannot be smaller than 1 and cannot be larger than the number of samples.
    :param sample_weight: Optional array of sample weights. Currently unused but included for API consistency.
    :return: the NDCG score as a float in [0,1], or np.nan if the input arrays are empty.
    """
    if labels.shape[0] == 0:
        return np.nan

    if min(labels) < 0:
        raise ValueError("NDCG should not be used with negative label values")

    gain = _ndcg_sample_scores(
        labels,
        predicted_labels,
        rank_discount=rank_discount,
        k=k,
    )
    if k is not None:
        k = min(k, labels.shape[0])
        return gain[k - 1]
    return gain[-1]


def recall_at_precision(
    y_true: npt.ArrayLike,
    y_scores: npt.ArrayLike,
    precision_target: float = 0.95,
    sample_weight: npt.ArrayLike | None = None,
) -> float:
    """
    Calculate the maximum recall at a given precision threshold.

    :param y_true: Array of true binary labels.
    :param y_scores: Array of predicted probability scores.
    :param precision_target: Minimum precision threshold to achieve.
    :param sample_weight: Optional array of sample weights.
    :return: Maximum recall achievable at the precision target, or 0 if unachievable.
    """
    precisions, recalls, _ = skmetrics.precision_recall_curve(
        y_true, y_scores, sample_weight=sample_weight
    )
    recalls_at_precision = [
        recall
        for precision, recall in zip(precisions, recalls)
        if precision >= precision_target
    ]
    return max(recalls_at_precision) if recalls_at_precision else 0


def precision_at_predictive_prevalence(
    y_true: npt.NDArray,
    y_scores: npt.NDArray,
    predictive_prevalence_target: float = 0.95,
    sample_weight: npt.NDArray | None = None,
) -> float:
    """
    Calculate precision at a given predictive prevalence threshold.

    Predictive prevalence is the fraction of samples predicted as positive.

    :param y_true: Array of true binary labels.
    :param y_scores: Array of predicted probability scores.
    :param predictive_prevalence_target: Target fraction of samples to predict as positive.
    :param sample_weight: Optional array of sample weights.
    :return: Maximum precision at the target predictive prevalence, or np.nan if
        no threshold can achieve the target prevalence.
    """
    precisions, _, thresholds = skmetrics.precision_recall_curve(
        y_true, y_scores, sample_weight=sample_weight
    )
    total_population = len(y_true)
    predictive_prevalences = [
        (y_scores >= threshold).sum() / total_population for threshold in thresholds
    ]
    precision_at_target = [
        precision
        for precision, predictive_prevalence in zip(
            precisions[:-1], predictive_prevalences
        )
        if predictive_prevalence >= predictive_prevalence_target
    ]
    if not precision_at_target:
        return np.nan
    return max(precision_at_target)


def precision_at_recall(
    y_true: npt.NDArray,
    y_scores: npt.NDArray,
    recall_target: float = 0.95,
    sample_weight: npt.NDArray | None = None,
) -> float:
    """
    Calculate the maximum precision at a given recall threshold.

    :param y_true: Array of true binary labels.
    :param y_scores: Array of predicted probability scores.
    :param recall_target: Minimum recall threshold to achieve.
    :param sample_weight: Optional array of sample weights.
    :return: Maximum precision at the recall target, or 0 if unachievable.
    """
    precisions, recalls, _ = skmetrics.precision_recall_curve(
        y_true, y_scores, sample_weight=sample_weight
    )
    precision_at_recall = [
        precision
        for precision, recall in zip(precisions, recalls)
        if recall >= recall_target
    ]
    return max(precision_at_recall) if precision_at_recall else 0


def fpr_at_precision(
    y_true: npt.NDArray,
    y_scores: npt.NDArray,
    precision_target: float = 0.95,
    sample_weight: npt.NDArray | None = None,
) -> float:
    """
    Calculate the false positive rate at a given precision threshold.

    :param y_true: Array of true binary labels.
    :param y_scores: Array of predicted probability scores.
    :param precision_target: Minimum precision threshold to achieve.
    :param sample_weight: Optional array of sample weights.
    :return: False positive rate at the precision target, or NaN if unachievable.
    """
    negatives = y_scores[y_true == 0].shape[0]
    # if there are no negatives in the data, fpr is undefined
    if negatives == 0:
        return np.nan

    precisions, _, thresholds = skmetrics.precision_recall_curve(
        y_true, y_scores, sample_weight=sample_weight
    )
    thresholds_at_target_precision = [
        threshold
        for precision, threshold in zip(precisions, thresholds)
        if precision >= precision_target
    ]

    # If there are no thresholds that meet the precision target, fpr is undefined
    if not thresholds_at_target_precision:
        return np.nan

    threshold_at_precision_target = np.min(thresholds_at_target_precision)

    false_positives = np.sum(y_scores[y_true == 0] >= threshold_at_precision_target)
    false_positive_rate = false_positives / negatives

    return false_positive_rate


class _MulticalibrationRankErrorMetricsInterface(Protocol):
    def __call__(
        self,
        labels: npt.NDArray,
        predicted_labels: npt.NDArray,
        rank_discount: Callable[[int], npt.NDArray],
        k: int | None = None,
    ) -> float: ...


def multi_cg_score(
    labels: npt.NDArray,
    predictions: npt.NDArray,
    segments_df: pd.DataFrame,
    metric: _MulticalibrationRankErrorMetricsInterface = ndcg_score,
    rank_discount: Callable[[int], npt.NDArray] = utils.rank_no_discount,
    k: int | None = None,
) -> npt.NDArray:
    """
    Calculates the metric score for each segment.

    :param labels: Array of true labels.
    :param predictions: Array of predicted labels.
    :param segments_df: Dataframe with the segments to calculate the error
    :param metric: The cumulative gain metric to use. Defaults to ndcg_score.
    :param rank_discount: rank discount function of the metric. Defaults to no discount.
    :param k: If not None, the metric is calculated only based on the top k samples.
        k cannot be smaller than 1 and cannot be larger than the number of samples.
    :return: an array of size n_segments with the metric score for each segment.
    """
    if metric not in (ndcg_score, dcg_score):
        raise ValueError("Only ndcg_score and dcg_score are supported")
    segments_df = segments_df.copy()
    segmentation_cols = list(segments_df.columns)
    segments_df["label"] = labels
    segments_df["prediction"] = predictions
    segments_df["sample_weight"] = np.ones_like(labels)

    grouping_cols = (
        segmentation_cols if len(segmentation_cols) > 1 else segmentation_cols[0]
    )
    samples_per_segment = (
        segments_df.groupby(grouping_cols)["sample_weight"]
        .sum()
        .rename("segment_total_weight")
    )

    def _group_cg_score(group: pd.DataFrame) -> float:
        return metric(
            labels=group.label.values,
            predicted_labels=group.prediction.values,
            rank_discount=rank_discount,
            k=k,
        )

    segment_errors = (
        groupby_apply(segments_df.groupby(grouping_cols), _group_cg_score)
        # pyre-ignore[6]: groupby_apply returns Series when func returns scalar; Series.rename(str) is valid
        .rename("error")
        .to_frame()
        .join(samples_per_segment)
    )

    return segment_errors["error"]


def _calculate_cumulative_differences(
    labels: npt.NDArray,
    predicted_scores: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
    segments: npt.NDArray | None = None,
    precision_dtype: type[np.float16]
    | type[np.float32]
    | type[np.float64] = DEFAULT_PRECISION_DTYPE,
) -> npt.NDArray:
    """
    Calculate cumulative differences between labels and predictions.

    Used internally by ECCE calibration functions.

    :param labels: Array of binary labels.
    :param predicted_scores: Array of predicted probability scores.
    :param sample_weight: Optional array of sample weights.
    :param segments: Optional array of segment masks.
    :param precision_dtype: Data type for precision of computation.
    :return: Array of cumulative differences.
    """
    if segments is None:
        segments = np.ones(shape=(1, len(predicted_scores)), dtype=np.bool_)
        sorted_indices = np.argsort(predicted_scores)
        predicted_scores = predicted_scores[sorted_indices]
        labels = labels[sorted_indices]
        sample_weight = (
            sample_weight[sorted_indices] if sample_weight is not None else None
        )

    if not segments.shape[1] == labels.shape[0] == predicted_scores.shape[0]:
        raise ValueError("Segments must be the same length as labels/predictions.")

    if sample_weight is None:
        sample_weight = np.ones_like(predicted_scores) / predicted_scores.shape[0]

    differences = np.empty(
        shape=(np.shape(segments)[0], np.shape(segments)[1] + 1),
        dtype=precision_dtype,
    )
    differences[:, 0] = 0
    weighted_diff = np.multiply((segments * sample_weight), (labels - predicted_scores))
    normalization = (segments * sample_weight).sum(axis=1)[:, np.newaxis]
    # Division by zero only happens for empty segments, which are handled below
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_diff = np.divide(weighted_diff, normalization)
    np.cumsum(
        normalized_diff,
        axis=1,
        out=differences[:, 1:],
    )
    differences[np.isnan(differences)] = 0
    return differences


def _ecce_standard_deviation(
    predicted_scores: npt.NDArray,
    labels: npt.NDArray | None = None,
    sample_weight: npt.NDArray | None = None,
) -> float:
    """
    Calculate the ECCE standard deviation for the entire dataset.

    :param predicted_scores: Array of predicted probability scores.
    :param labels: Optional array of labels (unused in this method).
    :param sample_weight: Optional array of sample weights.
    :return: The ECCE standard deviation as a scalar.
    """
    return _ecce_standard_deviation_per_segment(
        predicted_scores, labels, sample_weight
    ).item()


def _ecce_standard_deviation_per_segment(
    predicted_scores: npt.NDArray,
    labels: npt.NDArray | None = None,
    sample_weight: npt.NDArray | None = None,
    segments: npt.NDArray | None = None,
    precision_dtype: type[np.float16]
    | type[np.float32]
    | type[np.float64] = DEFAULT_PRECISION_DTYPE,
) -> npt.NDArray:
    """
    Calculate ECCE standard deviation per segment.

    Computes the standard deviation based on the variance of predictions.

    :param predicted_scores: Array of predicted probability scores.
    :param labels: Optional array of labels (unused in this method).
    :param sample_weight: Optional array of sample weights.
    :param segments: Optional array of segment masks.
    :param precision_dtype: Data type for precision of computation.
    :return: Array of standard deviations per segment.
    """
    if sample_weight is None:
        sample_weight = np.ones_like(predicted_scores)
    if segments is None:
        segments = np.ones(shape=(1, len(predicted_scores)), dtype=np.bool_)

    if segments.shape[1] != predicted_scores.shape[0]:
        raise ValueError("Segments must be the same length as labels/predictions.")

    ecce_std_dev = np.zeros(
        shape=(np.shape(segments)[0],),
        dtype=precision_dtype,
    )
    weighted_segments = segments * np.square(sample_weight)
    variance_preds = predicted_scores * (1 - predicted_scores)
    variance_weighted_segments = np.multiply(weighted_segments, variance_preds).sum(
        axis=1
    )
    normalization_variance = np.square((segments * sample_weight).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        np.sqrt(
            np.divide(
                variance_weighted_segments,
                normalization_variance,
            ),
            out=ecce_std_dev,
        )
    ecce_std_dev[np.isnan(ecce_std_dev)] = 0
    return ecce_std_dev


def _ecce_per_segment(
    labels: npt.NDArray,
    predicted_scores: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
    segments: npt.NDArray | None = None,
    precision_dtype: type[np.float16]
    | type[np.float32]
    | type[np.float64] = DEFAULT_PRECISION_DTYPE,
) -> npt.NDArray:
    """
    Compute ECCE (unnormalized) for multiple segments efficiently.

    This is an optimized internal helper for MulticalibrationError that computes
    ECCE across multiple segments in a vectorized manner.

    :param labels: Array of binary labels (0 or 1)
    :param predicted_scores: Array of predicted probability scores
    :param sample_weight: Optional array of sample weights
    :param segments: Array of segment masks for parallel computation
    :param precision_dtype: Data type for precision of the output
    :return: Array of ECCE values per segment
    """
    differences = _calculate_cumulative_differences(
        labels, predicted_scores, sample_weight, segments, precision_dtype
    )
    if segments is None:
        differences = differences.reshape(1, -1)
    return np.ptp(differences, axis=1)


def _ecce_cdf(x: float) -> float:
    """
    Evaluate the cumulative distribution function for the ECCE statistic.

    This computes the CDF for the range (maximum minus minimum) of the
    standard Brownian motion on [0, 1], which is used to compute p-values
    for the ECCE statistic.

    :param float x: argument at which to evaluate the cumulative distribution function
                    (must be positive)
    :return: cumulative distribution function evaluated at x
    :rtype: float
    """
    if x <= 0:
        raise ValueError(f"Can only evaluate ECCE CDF at positive x, not at {x}")
    if x >= ECCE_SIGMA_MAX:
        return 1.0 - sys.float_info.epsilon

    # Compute the machine precision assuming binary numerical representations.
    eps = sys.float_info.epsilon
    # Determine how many terms to use to attain accuracy eps.
    fact = 4.0 / math.sqrt(2.0 * math.pi) * (1.0 / x + x / math.pi**2)
    kmax = math.ceil(
        1.0 / 2.0 + x / math.pi / math.sqrt(2) * math.sqrt(math.log(fact / eps))
    )

    # Sum the series.
    c = 0.0
    for k in range(kmax):
        kplus = k + 1.0 / 2.0
        c += (8.0 / x**2.0 + 2.0 / kplus**2.0 / math.pi**2.0) * math.exp(
            -2.0 * kplus**2.0 * math.pi**2.0 / x**2.0
        )
    return c


def ecce_pvalue_from_sigma(ecce_sigma: float) -> float:
    """
    Compute p-value from a sigma-scaled ECCE statistic.

    :param ecce_sigma: The ECCE statistic normalized by standard deviation.
    :return: The p-value from the ECCE test.
    """
    if ecce_sigma < ECCE_SIGMA_MIN:
        return 1.0
    if ecce_sigma > ECCE_SIGMA_MAX:
        return sys.float_info.epsilon
    return 1 - _ecce_cdf(ecce_sigma)


def ecce(
    labels: npt.NDArray,
    predicted_scores: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
) -> float:
    """
    Calculate the Estimated Cumulative Calibration Error (ECCE) [1].

    ECCE measures the maximum deviation between the cumulative distribution of
    predicted probabilities for positive and negative examples. It is equivalent
    to the unnormalized Kuiper calibration statistic.

    [1]: Arrieta-Ibarra, I., Gujral, P., Tannen, J., Tygert, M., & Xu, C. (2022).
    Metrics of calibration for probabilistic predictions. Journal of Machine
    Learning Research, 23(351), 1-54. (https://tygert.com/ece.pdf)

    :param labels: Array of true binary labels (0 or 1).
    :param predicted_scores: Array of predicted probabilities.
    :param sample_weight: Optional array of sample weights.
    :return: The ECCE value.
    """
    differences = _calculate_cumulative_differences(
        labels, predicted_scores, sample_weight
    )
    return np.ptp(differences)


def ecce_sigma(
    labels: npt.NDArray,
    predicted_scores: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
) -> float:
    """
    Calculate the ECCE normalized by standard deviation.

    This returns the ECCE statistic normalized by the standard deviation of the
    calibration error under the null hypothesis of perfect calibration.

    :param labels: Array of true binary labels (0 or 1).
    :param predicted_scores: Array of predicted probabilities.
    :param sample_weight: Optional array of sample weights.
    :return: The normalized ECCE value.
    """
    ecce_value = ecce(labels, predicted_scores, sample_weight)
    sigma = _ecce_standard_deviation(predicted_scores, labels, sample_weight)

    if sigma == 0:
        return np.inf if ecce_value != 0 else 0.0
    return ecce_value / sigma


def ecce_pvalue(
    labels: npt.NDArray,
    predicted_scores: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
) -> float:
    """
    Calculate the p-value for the ECCE statistic.

    Tests the null hypothesis that predictions are perfectly calibrated using
    the Kuiper test.

    :param labels: Array of true binary labels (0 or 1).
    :param predicted_scores: Array of predicted probabilities.
    :param sample_weight: Optional array of sample weights.
    :return: The p-value from the calibration test.
    """
    sigma = ecce_sigma(labels, predicted_scores, sample_weight)
    return ecce_pvalue_from_sigma(sigma)


def _rank_calibration_error(
    labels: npt.NDArray,
    predicted_labels: npt.NDArray,
    num_bins: int = CALIBRATION_ERROR_NUM_BINS,
    rng: np.random.RandomState | None = None,
) -> tuple[float, npt.NDArray, npt.NDArray]:
    """
    Calculates rank calibration error as proposed in: https://arxiv.org/pdf/2404.03163

    :param labels: Array of true labels.
    :param predicted_labels: Array of predicted labels.
    :param num_bins: Number of bins to use for the rank calibration error calculation.
    :return: tuple (RCE, label_cdfs, prediction_cdfs)
    """
    # break ties
    rng = np.random.RandomState(42) if rng is None else rng
    eps = rng.uniform(0, 1, labels.shape[0]) * CALIBRATION_ERROR_EPSILON
    labels = labels + eps
    predicted_labels = predicted_labels + eps

    n = labels.shape[0]
    sorted_prediction_indices = np.argsort(predicted_labels)
    sorted_predictions = predicted_labels[sorted_prediction_indices]
    sorted_labels = labels[sorted_prediction_indices]
    label_means = np.zeros(num_bins)
    prediction_means = np.zeros(num_bins)

    bin_endpoints = [round(i) for i in np.linspace(0, n, num_bins + 1)]
    for i in range(1, num_bins + 1):
        low, high = bin_endpoints[i - 1], bin_endpoints[i]
        label_means[i - 1] = np.mean(sorted_labels[low:high])
        prediction_means[i - 1] = np.mean(sorted_predictions[low:high])

    label_cdfs = np.array(
        [np.sum(label_means[i] >= label_means) / num_bins for i in range(num_bins)]
    )

    prediction_cdfs = np.array(
        [
            (np.sum([prediction_means[i] >= prediction_means])) / (num_bins)
            for i in range(num_bins)
        ]
    )
    RCE = np.sum(np.abs(label_cdfs - prediction_cdfs)) / num_bins
    return RCE, label_cdfs, prediction_cdfs


def rank_calibration_error(
    labels: npt.NDArray,
    predicted_labels: npt.NDArray,
    num_bins: int = CALIBRATION_ERROR_NUM_BINS,
) -> float:
    """
    Calculates rank calibration error as proposed in: https://arxiv.org/pdf/2404.03163

    :param labels: Array of true labels.
    :param predicted_labels: Array of predicted labels.
    :param num_bins: Number of bins to use for the rank calibration error calculation.
    :return: float indicating the rank calibration error
    """

    return _rank_calibration_error(
        labels=labels, predicted_labels=predicted_labels, num_bins=num_bins
    )[0]


def rank_multicalibration_error(
    labels: npt.NDArray,
    predicted_labels: npt.NDArray,
    segments_df: pd.DataFrame,
    num_bins: int = CALIBRATION_ERROR_NUM_BINS,
) -> float:
    """
    Calculates rank calibration error for each segment.

    :param labels: Array of true labels.
    :param predicted_labels: Array of predicted labels.
    :param segments_df: Dataframe with the segments to calculate the error
    :param num_bins: Number of bins to use for the rank calibration error calculation.
    :return: float representing the weighted average of rank calibration errors across all segments.
    """
    segments_df = segments_df.copy()
    segmentation_cols = list(segments_df.columns)
    segments_df["label"] = labels
    segments_df["predicted_labels"] = predicted_labels
    segments_df["sample_weight"] = np.ones_like(labels)

    grouping_cols = (
        segmentation_cols if len(segmentation_cols) > 1 else segmentation_cols[0]
    )
    samples_per_segment = (
        segments_df.groupby(grouping_cols)["sample_weight"]
        .sum()
        .rename("segment_total_weight")
    )

    def _group_rank_calibration_error(
        group: pd.DataFrame,
    ) -> float:
        return rank_calibration_error(
            labels=group.label.values,
            predicted_labels=group.predicted_labels.values,
            num_bins=num_bins,
        )

    segment_RCE = (
        groupby_apply(segments_df.groupby(grouping_cols), _group_rank_calibration_error)
        # pyre-ignore[6]: groupby_apply returns Series when func returns scalar; Series.rename(str) is valid
        .rename("error")
        .to_frame()
        .join(samples_per_segment)
    )
    segment_RCE["weight"] = segment_RCE["segment_total_weight"] / len(labels)
    segment_RCE["weighted_error"] = segment_RCE["error"] * segment_RCE["weight"]
    if not np.allclose(segment_RCE["weight"].sum(), 1.0):
        raise AssertionError("Segment weights do not sum to 1.0")

    return segment_RCE["weighted_error"].sum()


def _rank_multicalibration_error(
    labels: npt.NDArray,
    predicted_labels: npt.NDArray,
    segments_df: pd.DataFrame,
    num_bins: int = CALIBRATION_ERROR_NUM_BINS,
) -> pd.Series:
    """
    Calculates rank calibration error for each segment.

    :param labels: Array of true labels.
    :param predicted_labels: Array of predicted labels.
    :param segments_df: Dataframe with the segments to calculate the error
    :param num_bins: Number of bins to use for the rank calibration error calculation.
    :return: an array of size n_segments with the tuple of (RCE, label_cdfs, prediction_cdfs) for each segment.
    """
    segments_df = segments_df.copy()
    segmentation_cols = list(segments_df.columns)
    segments_df["label"] = labels
    segments_df["predicted_labels"] = predicted_labels
    segments_df["sample_weight"] = np.ones_like(labels)

    grouping_cols = (
        segmentation_cols if len(segmentation_cols) > 1 else segmentation_cols[0]
    )
    samples_per_segment = (
        segments_df.groupby(grouping_cols)["sample_weight"]
        .sum()
        .rename("segment_total_weight")
    )

    def _group_rank_calibration_error(
        group: pd.DataFrame,
    ) -> tuple[float, npt.NDArray, npt.NDArray]:
        return _rank_calibration_error(
            labels=group.label.values,
            predicted_labels=group.predicted_labels.values,
            num_bins=num_bins,
        )

    segment_RCE = (
        groupby_apply(segments_df.groupby(grouping_cols), _group_rank_calibration_error)
        # pyre-ignore[6]: groupby_apply returns Series when func returns scalar; Series.rename(str) is valid
        .rename("error")
        .to_frame()
        .join(samples_per_segment)
    )

    return segment_RCE["error"]


def normalized_entropy(
    labels: npt.NDArray,
    predicted_scores: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
) -> float:
    """
    Calculates the normalized entropy, defined as the ratio between the prediction's log loss (binary cross entropy)
    and the log loss obtained from fixed predictions equal to the test set prevalence.

    :param labels: Ground truth (correct) labels for n_samples samples.
    :param predicted_scores: Predicted probabilities, as returned by a classifier's predict_proba method.
    :returns: the normalized entropy
    """
    if sample_weight is None:
        sample_weight = np.ones_like(predicted_scores)

    prediction_log_loss = skmetrics.log_loss(
        labels, predicted_scores, sample_weight=sample_weight
    )

    prevalence = np.sum(labels * sample_weight) / np.sum(sample_weight)
    baseline_predictions = np.full_like(labels, prevalence, dtype=np.float32)
    baseline_logloss = skmetrics.log_loss(
        labels, baseline_predictions, sample_weight=sample_weight
    )

    ne = prediction_log_loss / baseline_logloss
    return ne


def calibration_free_normalized_entropy(
    labels: npt.NDArray,
    predicted_scores: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
    tolerance: float = 1e-5,
    max_iter: int = 10000,
) -> float:
    """
    Calculates the Calibration-Free normalized entropy.

    :param labels: Ground truth (correct) labels for n_samples samples.
    :param predicted_scores: Predicted probabilities, as returned by a classifier's predict_proba method.
    :param sample_weight: Optional array of sample weights for each instance.
    :param tolerance: Convergence tolerance for the iterative calibration adjustment. Defaults to 1e-5.
    :param max_iter: Maximum number of iterations for the calibration adjustment. Defaults to 10000.
    :return: the calibration-free NE.
    """
    if len(labels.shape) != 1:
        raise ValueError("y_pred must be the predicted probability for class 1 only.")

    current_calibration = calibration_ratio(labels, predicted_scores, sample_weight)

    it = 0
    while abs(current_calibration - 1) > tolerance and it < max_iter:
        predicted_scores = predicted_scores / (
            current_calibration + (1 - current_calibration) * predicted_scores
        )
        current_calibration = calibration_ratio(labels, predicted_scores, sample_weight)
        it += 1

    calib_free_ne = normalized_entropy(labels, predicted_scores)
    return calib_free_ne


DEFAULT_MCE_MAX_VALUES_PER_SEGMENT_FEATURE: int = 3
DEFAULT_MCE_MIN_DEPTH: int = 0
DEFAULT_MCE_MAX_DEPTH: int = 3
DEFAULT_MCE_MIN_SAMPLES_PER_SEGMENT: int = 10
DEFAULT_MCE_GLOBAL_NORMALIZATION: str = "prevalence_adjusted"
DEFAULT_MCE_N_SEGMENTS: int | None = 1000


class MulticalibrationError(
    # @oss-disable[end= ]: MulticalibrationErrorCompatMixin,
):
    """
    Evaluates calibration quality across multiple subpopulations (segments).

    Multicalibration error (MCE) (introduced in [1]) extends traditional calibration metrics by measuring
    calibration not just globally, but across many automatically-generated segments
    of the data. This helps identify subpopulations where a model may be poorly
    calibrated even when global calibration appears good.

    The metric is based on ECCE (Estimated Cumulative Calibration Error) [2].

    [1] Guy, I., Haimovich, D., Linder, F., Okati, N., Perini, L., Tax, N., & Tygert, M.
    (2025). Measuring multi-calibration. arXiv preprint arXiv:2506.11251.
    (https://arxiv.org/abs/2506.11251)

    [2] Arrieta-Ibarra, I., Gujral, P., Tannen, J., Tygert, M., & Xu, C. (2022).
    Metrics of calibration for probabilistic predictions. Journal of Machine Learning
    Research, 23(351), 1-54. (https://www.jmlr.org/papers/volume23/22-0658/22-0658.pdf)

    **Key Concepts**

    Segments
        Subpopulations defined by combinations of feature values. Segments are
        generated automatically from categorical and numerical features at various
        depths (single features, pairs, triplets, etc.).

    Scales
        Results are available in four scales:

        - **Absolute**: Raw ECCE value (same units as predictions, typically 0-1)
        - **Relative (%)**: Percentage of prevalence, easier to interpret across datasets
        - **Sigma**: Statistical significance, values > 2 suggest miscalibration
        - **P-value**: Probability of seeing this ECCE under perfect calibration

    MCE vs Global ECCE
        - ``global_ecce``: ECCE computed on the entire dataset (no segmentation)
        - ``mce``: Largest ECCE across all segments

    **Interpreting Results**

    - ``mce_pvalue < 0.05``: Statistically significant miscalibration detected
    - ``mce_sigma > 2``: Miscalibration exceeds 2 standard deviations
    - ``mce_relative``: Miscalibration as percentage of prevalence (e.g., 5% means
      predictions are off by 5% of the base rate in the worst segment)
    - ``mde_relative``: Approximation of the minimum detectable error - miscalibration smaller than this
      cannot be reliably detected given the sample size
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_column: str,
        score_column: str,
        weight_column: str | None = None,
        categorical_segment_columns: list[str] | None = None,
        numerical_segment_columns: list[str] | None = None,
        max_depth: int | None = DEFAULT_MCE_MAX_DEPTH,
        max_values_per_segment_feature: int = DEFAULT_MCE_MAX_VALUES_PER_SEGMENT_FEATURE,
        min_samples_per_segment: int = DEFAULT_MCE_MIN_SAMPLES_PER_SEGMENT,
        max_n_segments: int | None = DEFAULT_MCE_N_SEGMENTS,
        chunk_size: int = 50,
        precision_dtype: str = "float32",
    ) -> None:
        """
        Initialize MulticalibrationError with data and segmentation parameters.

        :param df: DataFrame containing predictions, labels, and segment features.
        :param label_column: Column name containing binary labels (0 or 1).
        :param score_column: Column name containing predicted probabilities (0 to 1).
        :param weight_column: Optional column containing sample weights. If None,
            all samples are weighted equally.
        :param categorical_segment_columns: Columns with categorical values to use
            for segmentation (e.g., country, device_type). Each unique value becomes
            a potential segment boundary.
        :param numerical_segment_columns: Columns with numerical values to use for
            segmentation. Values are automatically quantile-binned.
        :param max_depth: Maximum depth of segment combinations. Depth 0 = global only,
            depth 1 = single features, depth 2 = pairs of features, etc. Higher depths
            find more granular miscalibration but increase computation.
        :param max_values_per_segment_feature: Maximum unique values per feature to
            consider. Features with more unique values are binned.
        :param min_samples_per_segment: Minimum samples required for a segment to be
            included. Smaller segments are excluded to reduce noise.
        :param max_n_segments: Maximum total segments to evaluate. Limits computation
            time for large feature spaces. Set to None for no limit.
        :param chunk_size: Number of segments to process per batch. Larger values
            improve speed but increase memory usage.
        :param precision_dtype: Floating-point precision for computations. Options:
            'float16' (fast, less precise), 'float32' (balanced), 'float64' (precise).
        """
        self.label_column = label_column
        self.score_column = score_column
        self.weight_column = weight_column
        self.categorical_segment_columns = categorical_segment_columns
        self.numerical_segment_columns = numerical_segment_columns
        self.max_depth = max_depth
        self.max_values_per_segment_feature = max_values_per_segment_feature
        self.min_samples_per_segment = min_samples_per_segment
        self.df: pd.DataFrame = df.copy(deep=False)
        self.df.sort_values(by=score_column, inplace=True)
        self.df.reset_index(inplace=True)

        if max_n_segments and chunk_size > max_n_segments:
            logger.warning(
                f"The chunk size {chunk_size} cannot be greater than max number of segments {max_n_segments}. "
                f"Setting speedup chunk size to {max_n_segments}."
            )
            chunk_size = max_n_segments

        self.chunk_size = chunk_size
        self.max_n_segments = max_n_segments

        if precision_dtype not in ["float16", "float32", "float64"]:
            raise ValueError(
                f"Invalid precision type: {precision_dtype}. Must be one of ['float16', 'float32', 'float64']."
            )
        self.precision_dtype: type[np.float16] | type[np.float32] | type[np.float64] = (
            getattr(np, precision_dtype)
        )

        self.df[self.score_column] = self.df[self.score_column].astype(
            self.precision_dtype
        )
        if self.weight_column is not None:
            if utils.check_range(self.df[self.weight_column], precision_dtype):
                self.df[self.weight_column] = self.df[self.weight_column].astype(
                    self.precision_dtype
                )
            else:
                logger.info(
                    f"Sample weights are not in range for {precision_dtype}. Keeping their initial type {self.df[self.weight_column].dtype}."
                )

        # Motivation for total_number_segments: chunks of segments with less than chunk_size elements are topped up with zeros
        # Such zeros are not needed for the computation of the metric and must be removed (lines: 1548, 1663)
        self.total_number_segments: int = -1  # initialized as -1

    def __str__(self) -> str:
        return f"""{self.mce_relative}% (sigmas={self.mce_sigma}, p={self.mce_pvalue}, mde={self.mde_relative})"""

    def __format__(self, format_spec: str) -> str:
        formatted_mce_relative = format(self.mce_relative, format_spec)
        formatted_p_value = format(self.mce_pvalue, format_spec)
        formatted_mde = format(self.mde_relative, format_spec)
        formatted_mce_sigma = format(self.mce_sigma, format_spec)
        return f"""{formatted_mce_relative}% (sigmas={formatted_mce_sigma}, p={formatted_p_value}, mde={formatted_mde})"""

    @functools.cached_property
    def _segments(self) -> tuple[npt.NDArray[np.bool_], pd.DataFrame]:
        segments_masks = []
        segments_feature_values = pd.DataFrame(
            columns=["segment_column", "value", "idx_segment"]
        )
        tot_segments: int = 0
        segments_generator = get_segment_masks(
            df=self.df,
            categorical_segment_columns=self.categorical_segment_columns,
            numerical_segment_columns=self.numerical_segment_columns,
            max_depth=self.max_depth,
            max_values_per_segment_feature=self.max_values_per_segment_feature,
            min_samples_per_segment=self.min_samples_per_segment,
            chunk_size=self.chunk_size,
        )
        for (
            segments_chunk_mask,
            size_chunk_mask,
            segment_chunk_feature_values,
        ) in segments_generator:
            if self.max_n_segments is not None and tot_segments >= self.max_n_segments:
                break

            segments_masks.append(segments_chunk_mask)
            segments_feature_values = pd.concat(
                [
                    segments_feature_values,
                    segment_chunk_feature_values,
                ],
                ignore_index=True,
            )
            tot_segments += size_chunk_mask

        segments = np.stack(segments_masks, axis=0)
        self.total_number_segments = tot_segments
        return segments, segments_feature_values

    @functools.cached_property
    def _segments_indices(self) -> pd.Series:
        segments_2d = self._segments[0].reshape(-1, self._segments[0].shape[2])
        indices = np.argwhere(segments_2d)
        index_series = pd.Series(indices[:, 1], index=indices[:, 0])

        return index_series

    @functools.cached_property
    def segments_ecce(
        self,
    ) -> npt.NDArray[np.float16 | np.float32 | np.float64]:
        """
        ECCE values per segment on absolute scale.

        Returns an array where each element is the Estimated Cumulative Calibration
        Error for one segment. The first element (index 0) is always the global
        segment (entire dataset).

        Values are in the same units as predictions (typically 0-1). Larger values
        indicate worse calibration in that segment.

        :return: Array of shape (n_segments,) with ECCE values.
        """
        segments = self._segments[0]
        statistics = np.zeros(
            self.total_number_segments,
            dtype=self.precision_dtype,
        )

        for i, segment in enumerate(segments):
            statistics[
                self.chunk_size * i : min(
                    self.chunk_size * (i + 1),
                    self.total_number_segments,
                )
            ] = _ecce_per_segment(
                labels=self.df[self.label_column].values,
                predicted_scores=self.df[self.score_column].values,
                sample_weight=(
                    None
                    if self.weight_column is None
                    else self.df[self.weight_column].values
                ),
                segments=segment[: self.total_number_segments - self.chunk_size * i,],
                precision_dtype=self.precision_dtype,
            )
        return statistics

    @functools.cached_property
    def segments_ecce_relative(
        self,
    ) -> npt.NDArray[np.float16 | np.float32 | np.float64]:
        """
        ECCE values per segment on relative (prevalence-adjusted percentage) scale.

        Values are expressed as a percentage of the label prevalence, making them
        easier to interpret across datasets with different base rates. For example,
        a value of 10 means the calibration error is 10% of the prevalence.

        :return: Array of shape (n_segments,) with relative ECCE values (%).
        """
        return self.segments_ecce_sigma * self._global_ecce_std / self._prevalence * 100

    @functools.cached_property
    def global_ecce(self) -> float:
        """
        Global ECCE on absolute scale.

        ECCE computed on the entire dataset without segmentation. This is equivalent
        to ``segments_ecce[0]`` since the first segment is always the global segment.

        Use this to assess overall model calibration before looking at segment-level
        miscalibration.

        :return: Global ECCE value.
        """
        return self.segments_ecce[0]

    @functools.cached_property
    def global_ecce_relative(self) -> float:
        """
        Global ECCE on relative (prevalence-adjusted percentage) scale.

        ECCE computed on the entire dataset without segmentation. This is equivalent
        to ``segments_ecce_relative[0]`` since the first segment is always the global segment.

        Use this to assess overall model calibration before looking at segment-level
        miscalibration.

        :return: Global ECCE as percentage of prevalence.
        """
        return self.segments_ecce_relative[0]

    @functools.cached_property
    def global_ecce_sigma(self) -> float:
        """
        Global ECCE on sigma scale.

        Indicates the statistical significance of the global ECCE value. Values above
        5 indicate strong evidence of miscalibration.

        :return: Global ECCE in standard deviations.
        """
        return self.segments_ecce_sigma[0]

    @functools.cached_property
    def global_ecce_pvalue(self) -> float:
        """
        P-value for global ECCE calibration test.

        The probability of observing this ECCE value (or larger) if the model
        were perfectly calibrated.

        :return: p-value between 0 and 1.
        """
        return self.segments_ecce_pvalue[0]

    @functools.cached_property
    def segments_ecce_sigma(
        self,
    ) -> npt.NDArray[np.float16 | np.float32 | np.float64]:
        """
        ECCE values per segment on sigma scale.

        Each value represents how many standard deviations the observed ECCE
        is from zero (perfect calibration). Values above 5 indicate strong
        evidence of miscalibration. You can plot the distribution of these
        values with `plotting.plot_segment_calibration_errors`.

        :return: Array of shape (n_segments,).
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            statistics = np.where(
                (self.segments_ecce != 0) & (self._segments_ecce_std == 0),
                np.inf,
                np.where(
                    self._segments_ecce_std == 0,
                    0,
                    self.segments_ecce / self._segments_ecce_std,
                ),
            )
        return statistics

    @functools.cached_property
    def segments_ecce_pvalue(self) -> npt.NDArray[np.float16 | np.float32 | np.float64]:
        """
        p-values per segment for ECCE calibration test.

        Each value is the probability of observing the corresponding ECCE
        (or larger) if the model were perfectly calibrated in that segment.

        :return: Array of shape (n_segments,) with p-values between 0 and 1.
        """
        ecce_pvalue_vec = np.vectorize(ecce_pvalue_from_sigma)
        p_values = ecce_pvalue_vec(self.segments_ecce_sigma)
        return p_values

    @functools.cached_property
    def _segments_ecce_std(self) -> npt.NDArray[np.float16 | np.float32 | np.float64]:
        segments = self._segments[0]
        sigmas = np.zeros(self.total_number_segments, dtype=self.precision_dtype)
        for i, segment in enumerate(segments):
            sigmas[
                self.chunk_size * i : min(
                    self.chunk_size * (i + 1),
                    self.total_number_segments,
                )
            ] = _ecce_standard_deviation_per_segment(
                predicted_scores=self.df[self.score_column].values,
                labels=self.df[self.label_column].values,
                sample_weight=(
                    None
                    if self.weight_column is None
                    else self.df[self.weight_column].values
                ),
                segments=segment[: self.total_number_segments - self.chunk_size * i,],
                precision_dtype=self.precision_dtype,
            )
        return sigmas

    @functools.cached_property
    def _global_ecce_std(self) -> float:
        if "_segments_ecce_std" in self.__dict__:
            return self._segments_ecce_std[0]
        std = _ecce_standard_deviation_per_segment(
            predicted_scores=self.df[self.score_column].values,
            labels=self.df[self.label_column].values,
            sample_weight=(
                None
                if self.weight_column is None
                else self.df[self.weight_column].values
            ),
            segments=np.ones(shape=(1, len(self.df)), dtype=np.bool_),
            precision_dtype=self.precision_dtype,
        )
        return std.item()

    @functools.cached_property
    def mce_sigma(self) -> float:
        """
        Multicalibration error on sigma scale.

        The largest ECCE-sigma across all segments. This identifies the segment with
        the most statistically significant miscalibration. Values above 5 indicate
        significant miscalibration.

        :return: Maximum segment ECCE-sigma.
        """
        return np.max(self.segments_ecce_sigma)

    @functools.cached_property
    def mce(self) -> float:
        """
        Multicalibration error on absolute scale.

        The largest ECCE across all segments, converted to absolute scale.
        This represents the worst-case calibration error found in any segment.

        :return: Maximum ECCE value.
        """
        return self.mce_sigma * self._global_ecce_std

    @functools.cached_property
    def mce_relative(self) -> float:
        """
        Multicalibration error on relative (prevalence-adjusted percentage) scale.

        The MCE expressed as a percentage of label prevalence. For example, if
        prevalence is 10% and mce_relative is 20, the worst segment has predictions
        off by 2 percentage points (20% of 10%).

        This is often the most interpretable metric for comparing calibration
        across different datasets or use cases.

        :return: MCE as percentage of prevalence.
        """
        # Compute directly to avoid recursion with transition override
        mce_abs = self.mce_sigma * self._global_ecce_std
        return mce_abs / self._prevalence * 100

    @functools.cached_property
    def _prevalence(self) -> float:
        p = (
            (self.df[self.label_column] * self.df[self.weight_column]).sum()
            / (self.df[self.weight_column].sum())
            if self.weight_column is not None
            else self.df[self.label_column].mean()
        )
        return min(p, 1 - p)

    @functools.cached_property
    def mce_pvalue(self) -> float:
        """
        p-value for the multicalibration error.

        The probability of observing this MCE (or larger) if the model were
        perfectly calibrated across all segments. This is the minimum p-value
        across all segments.

        Note that this p-value is not adjusted for multiple testing, so the
        Type I error rate (concluding there is miscalibration when there is
        none) will be higher in practice. We expect any required adjustment to
        be small because the hypotheses are highly correlated (many segments
        overlap). We therefore did not apply common corrections such as
        Bonferroni, as they would be overly conservative and could substantially
        increase Type II errors (failing to detect miscalibration when it exists).

        :return: p-value between 0 and 1.
        """
        if "segments_ecce_pvalue" in self.__dict__:
            return np.min(self.segments_ecce_pvalue)
        return ecce_pvalue_from_sigma(self.mce_sigma)

    @functools.cached_property
    def mde(self) -> float:
        """Minimum detectable error on absolute (probability) scale.

        The MDE represents the smallest calibration error that would be
        statistically detectable at approximately 5 sigma significance,
        given the sample size. Expressed as an absolute probability difference.
        """
        return 5 * self._global_ecce_std

    @functools.cached_property
    def mde_relative(self) -> float:
        """
        Minimum detectable error on relative (prevalence-adjusted percentage) scale.

        The smallest calibration error that can be reliably detected given the
        sample size and variance in the data. Miscalibration smaller than this
        value may not be statistically significant even if present.

        Based on a 5-sigma detection threshold (very high confidence).

        :return: MDE as percentage of prevalence.
        """
        return 5 * self._global_ecce_std / self._prevalence * 100


# Apply transition period overrides for mce/global_ecce/mde (internal only)
# @oss-disable[end= ]: apply_mce_transition_overrides(MulticalibrationError)


class _ScoreFunctionInterface(Protocol):
    name: str

    def __call__(
        self,
        df: pd.DataFrame,
        label_column: str,
        score_column: str,
        weight_column: str | None,
    ) -> float: ...


def wrap_sklearn_metric_func(
    func: Callable[..., float],
) -> _ScoreFunctionInterface:
    """
    Wrap an sklearn-style metric function for use with the evaluation framework.

    :param func: A function with signature (y_true, y_pred, sample_weight=None) -> float.
    :return: A ScoreFunctionInterface-compatible wrapper.
    """

    class WrappedFuncSkLearn(_ScoreFunctionInterface):
        name = func.__name__

        def __call__(
            self,
            df: pd.DataFrame,
            label_column: str,
            score_column: str,
            weight_column: str | None,
        ) -> float:
            y_true = df[label_column].values
            y_pred = df[score_column].values
            sample_weight = df[weight_column].values if weight_column else None
            return func(y_true, y_pred, sample_weight=sample_weight)

    return WrappedFuncSkLearn()


def wrap_multicalibration_error_metric(
    categorical_segment_columns: list[str] | None = None,
    numerical_segment_columns: list[str] | None = None,
    max_depth: int = DEFAULT_MCE_MAX_DEPTH,
    max_values_per_segment_feature: int = DEFAULT_MCE_MAX_VALUES_PER_SEGMENT_FEATURE,
    min_samples_per_segment: int = DEFAULT_MCE_MIN_SAMPLES_PER_SEGMENT,
    max_n_segments: int | None = DEFAULT_MCE_N_SEGMENTS,
    metric_version: str = "mce_relative",
) -> _ScoreFunctionInterface:
    """
    Create a wrapped MulticalibrationError metric for use with the evaluation framework.

    :param categorical_segment_columns: Columns to use for categorical segmentation.
    :param numerical_segment_columns: Columns to use for numerical segmentation.
    :param max_depth: Maximum depth for segment generation.
    :param max_values_per_segment_feature: Max unique values per segment feature.
    :param min_samples_per_segment: Minimum samples required per segment.
    :param max_n_segments: Maximum number of segments to generate.
    :param metric_version: Which metric to return. Options:
        - 'mce_relative': relative (prevalence-adjusted percentage) scale (default)
        - 'mce': absolute scale
        - 'mce_sigma': sigma (z-score) scale
        - 'mce_pvalue': p-value
        Legacy names are also supported but deprecated:
        - 'mce_sigma_scale' -> 'mce_sigma'
        - 'mce_absolute' -> 'mce'
        - 'p_value' -> 'mce_pvalue'
    :return: A ScoreFunctionInterface-compatible wrapper.
    """
    if categorical_segment_columns is None and numerical_segment_columns is None:
        raise ValueError(
            "No segment columns provided. Please provide either "
            "categorical_segment_columns or numerical_segment_columns."
        )

    # Map legacy names to new names
    legacy_to_new = {
        "mce_sigma_scale": "mce_sigma",
        "mce_absolute": "mce",
        "p_value": "mce_pvalue",
    }
    if metric_version in legacy_to_new:
        warnings.warn(
            f"metric_version='{metric_version}' is deprecated. "
            f"Use '{legacy_to_new[metric_version]}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        metric_version = legacy_to_new[metric_version]

    valid_versions = ("mce", "mce_relative", "mce_sigma", "mce_pvalue")
    if metric_version not in valid_versions:
        raise ValueError(
            f"`metric_version` has to be one of {list(valid_versions)}. "
            f"Got `{metric_version}`."
        )

    class WrappedFuncMCE(_ScoreFunctionInterface):
        name = f"Multicalibration Error<br>({metric_version})"

        def __init__(
            self,
            categorical_segment_columns: list[str] | None,
            numerical_segment_columns: list[str] | None,
            max_depth: int = DEFAULT_MCE_MAX_DEPTH,
            max_values_per_segment_feature: int = DEFAULT_MCE_MAX_VALUES_PER_SEGMENT_FEATURE,
            min_samples_per_segment: int = DEFAULT_MCE_MIN_SAMPLES_PER_SEGMENT,
            max_n_segments: int | None = DEFAULT_MCE_N_SEGMENTS,
        ):
            self.categorical_segment_columns = categorical_segment_columns
            self.numerical_segment_columns = numerical_segment_columns
            self.max_depth = max_depth
            self.max_values_per_segment_feature = max_values_per_segment_feature
            self.min_samples_per_segment = min_samples_per_segment
            self.max_n_segments = max_n_segments

        def __call__(
            self,
            df: pd.DataFrame,
            label_column: str,
            score_column: str,
            weight_column: str | None,
        ) -> float:
            mce = MulticalibrationError(
                df,
                label_column,
                score_column,
                weight_column,
                categorical_segment_columns=self.categorical_segment_columns,
                numerical_segment_columns=self.numerical_segment_columns,
                max_depth=self.max_depth,
                max_values_per_segment_feature=self.max_values_per_segment_feature,
                min_samples_per_segment=self.min_samples_per_segment,
                max_n_segments=self.max_n_segments,
            )
            return getattr(mce, metric_version)

    return WrappedFuncMCE(
        categorical_segment_columns,
        numerical_segment_columns,
        max_depth,
        max_values_per_segment_feature,
        min_samples_per_segment,
        max_n_segments,
    )
