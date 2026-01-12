# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict

import functools
import logging
import math
import sys
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import pandas as pd
from multicalibration import utils
from multicalibration.segmentation import get_segment_masks
from numpy import typing as npt
from scipy import stats
from sklearn import metrics as skmetrics


logger: logging.Logger = logging.getLogger(__name__)
CALIBRATION_ERROR_NUM_BINS = 40
CALIBRATION_ERROR_EPSILON = 0.0000001
DEFAULT_PRECISION_DTYPE = np.float64

# Kuiper distribution constants
# KUIPER_STATISTIC_MAX: Maximum statistic value before CDF is effectively 1.0
# KUIPER_STATISTIC_MIN: Minimum statistic value below which p-value is 1.0
KUIPER_STATISTIC_MAX: float = 8.26732673
KUIPER_STATISTIC_MIN: float = 1e-20


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
    :return: The calibration ratio.
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
    :return: the DCG score as a float.
    """
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
    :return: the NDCG score as a float in [0,1].
    """
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
    :return: Maximum precision at the target predictive prevalence.
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


def predictions_to_labels(
    data: pd.DataFrame,
    prediction_column: str,
    thresholds: pd.DataFrame,
    threshold_column: str | None = "threshold",
) -> pd.DataFrame:
    """
    Convert prediction scores to binary labels using segment-specific thresholds.

    :param data: DataFrame containing predictions and segmentation columns.
    :param prediction_column: Name of the column containing prediction scores.
    :param thresholds: DataFrame with threshold values per segment.
    :param threshold_column: Name of the column in thresholds containing threshold values.
    :return: DataFrame with an added 'predicted_label' column.
    """
    segmentation_columns = [c for c in thresholds.columns if c != threshold_column]
    data_w_thresholds = data.copy().merge(
        thresholds, on=segmentation_columns, how="left"
    )
    data_w_thresholds["predicted_label"] = (
        data_w_thresholds[prediction_column] >= data_w_thresholds.threshold
    ).astype(int)
    return data_w_thresholds


class MulticalibrationErrorMetricInterface(Protocol):
    def __call__(
        self,
        labels: npt.NDArray,
        predicted_scores: npt.NDArray,
        sample_weight: npt.NDArray,
        num_bins: int,
        epsilon: float,
    ) -> float: ...


def multicalibration_error(
    labels: npt.NDArray,
    predictions: npt.NDArray,
    segments_df: pd.DataFrame,
    sample_weight: npt.NDArray | None = None,
    metric: MulticalibrationErrorMetricInterface = adaptive_calibration_error,
    num_bins: int = 40,
    epsilon: float = 0.0000001,
) -> float:
    """
    Calculate the multicalibration error across multiple segments.

    Computes a weighted average of calibration errors for each segment.

    :param labels: Array of true labels.
    :param predictions: Array of predicted probability scores.
    :param segments_df: DataFrame defining the segmentation columns.
    :param sample_weight: Optional array of sample weights.
    :param metric: Calibration error metric to use per segment.
    :param num_bins: Number of bins for the calibration error calculation.
    :param epsilon: Small value to avoid numerical issues.
    :return: Weighted average calibration error across all segments.
    """
    segments_df = segments_df.copy()
    segmentation_cols = list(segments_df.columns)
    segments_df["label"] = labels
    segments_df["prediction"] = predictions
    segments_df["sample_weight"] = (
        sample_weight if sample_weight is not None else np.ones_like(labels)
    )

    total_weight = segments_df.sample_weight.sum()

    # Handle the case when there are no segmentation columns, in which case
    # we compute the error for the entire dataset as a single segment
    if not segmentation_cols:
        return metric(
            labels=labels,
            predicted_scores=predictions,
            sample_weight=sample_weight
            if sample_weight is not None
            else np.ones_like(labels),
            num_bins=num_bins,
            epsilon=epsilon,
        )

    grouping_cols = (
        segmentation_cols if len(segmentation_cols) > 1 else segmentation_cols[0]
    )
    samples_per_segment = (
        segments_df.groupby(grouping_cols)["sample_weight"]
        .sum()
        .rename("segment_total_weight")
    )

    def _group_calibration_error(group: pd.DataFrame) -> float:
        return metric(
            labels=group.label.values,
            predicted_scores=group.prediction.values,
            sample_weight=group.sample_weight.values,
            num_bins=num_bins,
            epsilon=epsilon,
        )

    segment_errors = (
        segments_df.groupby(grouping_cols)
        .apply(_group_calibration_error)
        .rename("error")
        .to_frame()
        .join(samples_per_segment)
    )
    segment_errors["weight"] = segment_errors["segment_total_weight"] / total_weight
    segment_errors["weighted_error"] = (
        segment_errors["error"] * segment_errors["weight"]
    )
    return segment_errors["weighted_error"].sum()


class MulticalibrationRankErrorMetricsInterface(Protocol):
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
    metric: MulticalibrationRankErrorMetricsInterface = ndcg_score,
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
        segments_df.groupby(grouping_cols)
        .apply(_group_cg_score)
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
    precision_dtype: np.float16 | np.float32 | np.float64 = DEFAULT_PRECISION_DTYPE,
) -> npt.NDArray:
    """
    Calculate cumulative differences between labels and predictions.

    Used internally by Kuiper calibration functions.

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


class KuiperNormalizationInterface(Protocol):
    def __call__(
        self,
        predicted_scores: npt.NDArray,
        labels: npt.NDArray | None,
        sample_weight: npt.NDArray | None,
        segments: npt.NDArray | None,
        precision_dtype: np.float16 | np.float32 | np.float64,
    ) -> npt.NDArray: ...


def kuiper_standard_deviation(
    predicted_scores: npt.NDArray,
    labels: npt.NDArray | None = None,
    sample_weight: npt.NDArray | None = None,
) -> float:
    """
    Calculate the Kuiper standard deviation for the entire dataset.

    :param predicted_scores: Array of predicted probability scores.
    :param labels: Optional array of labels (unused in this method).
    :param sample_weight: Optional array of sample weights.
    :return: The Kuiper standard deviation as a scalar.
    """
    return kuiper_standard_deviation_per_segment(
        predicted_scores, labels, sample_weight
    ).item()


def kuiper_upper_bound_standard_deviation_per_segment(
    predicted_scores: npt.NDArray,
    labels: npt.NDArray | None = None,
    sample_weight: npt.NDArray | None = None,
    segments: npt.NDArray | None = None,
    precision_dtype: np.float16 | np.float32 | np.float64 = DEFAULT_PRECISION_DTYPE,
) -> npt.NDArray:
    """
    Calculate an upper bound on Kuiper standard deviation per segment.

    Uses a conservative estimate: 1 / (2 * sqrt(n)) for each segment.

    :param predicted_scores: Array of predicted probability scores.
    :param labels: Optional array of labels (unused in this method).
    :param sample_weight: Optional array of sample weights.
    :param segments: Optional array of segment masks.
    :param precision_dtype: Data type for precision of computation.
    :return: Array of upper bound standard deviations per segment.
    """
    if sample_weight is None:
        sample_weight = np.ones_like(predicted_scores)
    if segments is None:
        segments = np.ones(shape=(1, len(predicted_scores)), dtype=np.bool_)
    with np.errstate(divide="ignore"):
        kuiper_ub = np.divide(1, (2 * np.sqrt((segments * sample_weight).sum(axis=1))))
    kuiper_ub[np.isinf(kuiper_ub)] = 0
    return kuiper_ub


def kuiper_upper_bound_standard_deviation(
    predicted_scores: npt.NDArray,
    labels: npt.NDArray | None = None,
    sample_weight: npt.NDArray | None = None,
) -> float:
    """
    Calculate an upper bound on Kuiper standard deviation for the entire dataset.

    :param predicted_scores: Array of predicted probability scores.
    :param labels: Optional array of labels (unused in this method).
    :param sample_weight: Optional array of sample weights.
    :return: The upper bound standard deviation as a scalar.
    """
    if sample_weight is None:
        sample_weight = np.ones_like(predicted_scores)

    return kuiper_upper_bound_standard_deviation_per_segment(
        predicted_scores, labels, sample_weight
    ).item()


def kuiper_standard_deviation_per_segment(
    predicted_scores: npt.NDArray,
    labels: npt.NDArray | None = None,
    sample_weight: npt.NDArray | None = None,
    segments: npt.NDArray | None = None,
    precision_dtype: np.float16 | np.float32 | np.float64 = DEFAULT_PRECISION_DTYPE,
) -> npt.NDArray:
    """
    Calculate Kuiper standard deviation per segment.

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

    kuip_std_dev = np.zeros(
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
            out=kuip_std_dev,
        )
    kuip_std_dev[np.isnan(kuip_std_dev)] = 0
    return kuip_std_dev


def kuiper_label_based_standard_deviation_per_segment(
    predicted_scores: npt.NDArray,
    labels: npt.NDArray | None,
    sample_weight: npt.NDArray | None = None,
    segments: npt.NDArray | None = None,
    precision_dtype: np.float16 | np.float32 | np.float64 = DEFAULT_PRECISION_DTYPE,
) -> npt.NDArray:
    """
    Calculate label-based Kuiper standard deviation per segment.

    Uses differences between labels and predictions to estimate variance.

    :param predicted_scores: Array of predicted probability scores.
    :param labels: Array of binary labels (required for this method).
    :param sample_weight: Optional array of sample weights.
    :param segments: Optional array of segment masks.
    :param precision_dtype: Data type for precision of computation.
    """
    if sample_weight is None:
        sample_weight = np.ones_like(predicted_scores)
    if segments is None:
        segments = np.ones(shape=(1, len(predicted_scores)), dtype=np.bool_)

    if labels is not None:
        return np.array(
            [
                kuiper_label_based_standard_deviation(
                    predicted_scores[np.where(segment)[0]],
                    labels[np.where(segment)[0]],
                    sample_weight[np.where(segment)[0]],
                )
                for segment in segments
            ]
        )
    raise ValueError("Labels are required for this method")


def kuiper_label_based_standard_deviation(
    predicted_scores: npt.NDArray,
    labels: npt.NDArray | None,
    sample_weight: npt.NDArray | None = None,
) -> float:
    """
    Calculate label-based Kuiper standard deviation for the entire dataset.

    :param predicted_scores: Array of predicted probability scores.
    :param labels: Array of binary labels (required for this method).
    :param sample_weight: Optional array of sample weights.
    :return: The label-based standard deviation as a scalar.
    """
    if sample_weight is None:
        sample_weight = np.ones_like(predicted_scores)

    if labels is not None:
        return np.sqrt(
            np.sum(
                (
                    labels[:-1]
                    - predicted_scores[:-1]
                    - labels[1:]
                    + predicted_scores[1:]
                )
                ** 2
                * (sample_weight[:-1] + sample_weight[1:]) ** 2
            )
            / (
                8
                * sample_weight.sum()
                * (
                    0.5 * sample_weight[0]
                    + 0.5 * sample_weight[-1]
                    + np.sum(sample_weight[1:-1])
                )
            )
        )

    raise ValueError("Labels are required for this method")


def identity_per_segment(
    predicted_scores: npt.NDArray,
    labels: npt.NDArray | None = None,
    sample_weight: npt.NDArray | None = None,
    segments: npt.NDArray | None = None,
    precision_dtype: np.float16 | np.float32 | np.float64 = DEFAULT_PRECISION_DTYPE,
) -> npt.NDArray:
    """
    Return an array of ones (identity normalization).

    Used when no normalization is desired for Kuiper calibration.

    :param predicted_scores: Array of predicted probability scores (unused).
    :param labels: Optional array of labels (unused).
    :param sample_weight: Optional array of sample weights (unused).
    :param segments: Optional array of segment masks.
    :param precision_dtype: Data type for precision of computation (unused).
    :return: Array of ones with length equal to number of segments.
    """
    if segments is None:
        return np.ones(1)
    return np.ones(segments.shape[0])


def kuiper_calibration_per_segment(
    labels: npt.NDArray,
    predicted_scores: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
    normalization_method: str | None = None,
    segments: npt.NDArray | None = None,
    precision_dtype: np.float16 | np.float32 | np.float64 = DEFAULT_PRECISION_DTYPE,
) -> npt.NDArray:
    """
    Calculates Kuiper calibration distance between responses and scores.

    For details, see:
    Mark Tygert. (2024, January 10). Conditioning on and controlling for
    variates via cumulative differences: measuring calibration, reliability,
    biases, and other treatment effects. Zenodo.
    https://doi.org/10.5281/zenodo.10481097

    :param labels: Array of binary labels (0 or 1)
    :param predicted_scores: Array of predicted probability scores. (floats between 0 and 1)
    :param sample_weight: Optional array of sample weights (non-negative floats)
    :param normalization_method: Optional function to calculate a normalization constant.
            See for example kuiper_sd or inverse_sqrt_sample_size, methods need to follow the same interface.
    :param segments: Optional array of segments to parallelize the computation of the kuiper calibration distance.
    :param precision_dtype: Optional dtype for the precision of the output. Defaults to np.float64.
    :return: Kuiper calibration distance
    """

    normalization_func = _normalization_method_assignment(normalization_method)

    denominator = normalization_func(
        predicted_scores=predicted_scores,
        labels=labels,
        sample_weight=sample_weight,
        segments=segments,
        precision_dtype=precision_dtype,
    )

    differences = _calculate_cumulative_differences(
        labels, predicted_scores, sample_weight, segments, precision_dtype
    )
    if segments is None:
        differences = differences.reshape(1, -1)

    c_range = np.ptp(differences, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            (denominator == 0) & (c_range != 0),
            np.inf,
            np.where(denominator == 0, 0, c_range / denominator),
        )


def kuiper_calibration(
    labels: npt.NDArray,
    predicted_scores: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
    normalization_method: str | None = None,
    segments: npt.NDArray | None = None,
    precision_dtype: np.float16 | np.float32 | np.float64 = DEFAULT_PRECISION_DTYPE,
) -> float:
    """
    Calculates Kuiper calibration distance between responses and scores.

    For details, see:
    Mark Tygert. (2024, January 10). Conditioning on and controlling for
    variates via cumulative differences: measuring calibration, reliability,
    biases, and other treatment effects. Zenodo.
    https://doi.org/10.5281/zenodo.10481097

    :param labels: Array of binary labels (0 or 1)
    :param predicted_scores: Array of predicted probability scores (floats between 0 and 1)
    :param sample_weight: Optional array of sample weights (non-negative floats)
    :param normalization_method: Optional method name for calculating a normalization constant.
        See kuiper_standard_deviation_per_segment or kuiper_upper_bound_standard_deviation_per_segment.
    :param segments: Optional array of segments to parallelize the computation of the kuiper calibration distance.
    :param precision_dtype: Data type for precision of computation. Defaults to np.float64.
    :return: Kuiper calibration distance
    """

    return kuiper_calibration_per_segment(
        labels,
        predicted_scores,
        sample_weight,
        normalization_method,
        segments,
        precision_dtype,
    ).item()


def kuiper_distribution(x: float) -> float:
    """
    Evaluates the cumulative distribution function for the range
    (maximum minus minimum) of the standard Brownian motion on [0, 1].

    :param float x: argument at which to evaluate the cumulative distribution function
                    (must be positive)
    :return: cumulative distribution function evaluated at x
    :rtype: float
    """
    if x <= 0:
        raise ValueError(
            f"Can only evaluate cumulative Kuiper distribution at positive x, not at {x}"
        )
    # If x goes to infinity, c tends to 1.0
    if x >= KUIPER_STATISTIC_MAX:
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


def _normalization_method_assignment(
    method: str | None,
) -> KuiperNormalizationInterface:
    methods = {
        "kuiper_standard_deviation": kuiper_standard_deviation_per_segment,
        "kuiper_upper_bound_standard_deviation": kuiper_upper_bound_standard_deviation_per_segment,
        "kuiper_label_based_standard_deviation": kuiper_label_based_standard_deviation_per_segment,
        None: identity_per_segment,
    }
    if method not in methods:
        raise ValueError(
            f"Unknown normalization method {method}. Available methods are {list(methods)}"
        )
    return methods[method]


def kuiper_test(
    labels: npt.NDArray,
    predicted_scores: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
) -> tuple[float, float]:
    """
    Calculates the Kuiper test statistic and p-value for the Kuiper calibration
    distance. This test is used to assess how well the predicted probabilities
    of a binary classifier are calibrated.

    :param labels: Array of true binary labels (0 or 1).
    :param predicted_scores: Array of predicted probabilities, corresponding to the likelihood of the label being 1.
    :param sample_weight: Optional array of weights for the samples, must be the same length as labels and predicted_scores.
    :return: A tuple containing the Kuiper statistic and the corresponding p-value.
    """

    kuiper_stat = kuiper_calibration(
        labels,
        predicted_scores,
        sample_weight,
        normalization_method="kuiper_standard_deviation",
    )
    if kuiper_stat < KUIPER_STATISTIC_MIN:
        pval = 1.0
    elif kuiper_stat > KUIPER_STATISTIC_MAX:
        pval = sys.float_info.epsilon
    else:
        pval = 1 - kuiper_distribution(kuiper_stat)

    return kuiper_stat, pval


def kuiper_statistic(
    labels: npt.NDArray,
    predicted_scores: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
) -> float:
    """
    Calculate the Kuiper test statistic.

    :param labels: Array of true binary labels.
    :param predicted_scores: Array of predicted probability scores.
    :param sample_weight: Optional array of sample weights.
    :return: The Kuiper statistic.
    """
    return kuiper_test(
        labels,
        predicted_scores,
        sample_weight,
    )[0]


def kuiper_pvalue(
    labels: npt.NDArray,
    predicted_scores: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
) -> float:
    """
    Calculate the Kuiper test p-value.

    :param labels: Array of true binary labels.
    :param predicted_scores: Array of predicted probability scores.
    :param sample_weight: Optional array of sample weights.
    :return: The p-value from the Kuiper test.
    """
    return kuiper_test(
        labels,
        predicted_scores,
        sample_weight,
    )[1]


def kuiper_func_per_segment(
    labels: npt.NDArray,
    predictions: npt.NDArray,
    segments_df: pd.DataFrame,
    func: Callable[[pd.DataFrame], float | npt.NDArray | None],
    output_series_name: str,
    sample_weight: npt.NDArray | None = None,
    min_segment_size: int = 2,
) -> pd.Series:
    """
    Apply a function to each segment and return results as a Series.

    :param labels: Array of true binary labels.
    :param predictions: Array of predicted probability scores.
    :param segments_df: DataFrame defining the segmentation columns.
    :param func: Function to apply to each segment's DataFrame.
    :param output_series_name: Name for the resulting Series.
    :param sample_weight: Optional array of sample weights.
    :param min_segment_size: Minimum samples required per segment.
    :return: Series with function results indexed by segment.
    """
    segments_df = segments_df.copy()
    segmentation_cols = list(segments_df.columns)
    segments_df["label"] = labels
    segments_df["prediction"] = predictions
    segments_df["sample_weight"] = (
        sample_weight if sample_weight is not None else np.ones_like(labels)
    )

    grouping_cols = (
        segmentation_cols if len(segmentation_cols) > 1 else segmentation_cols[0]
    )

    segment_p_values = (
        segments_df.groupby(grouping_cols).apply(func).rename(output_series_name)
    ).dropna()

    return segment_p_values


def kuiper_pvalue_per_segment(
    labels: npt.NDArray,
    predictions: npt.NDArray,
    segments_df: pd.DataFrame,
    sample_weight: npt.NDArray | None = None,
    min_segment_size: int = 2,
) -> pd.Series:
    """
    Calculate Kuiper p-values for each segment.

    :param labels: Array of true binary labels.
    :param predictions: Array of predicted probability scores.
    :param segments_df: DataFrame defining the segmentation columns.
    :param sample_weight: Optional array of sample weights.
    :param min_segment_size: Minimum samples required per segment.
    :return: Series of p-values indexed by segment.
    """

    def _group_kuiper_p_value(group: pd.DataFrame) -> float | None:
        if len(group) < min_segment_size:
            return None
        return kuiper_test(
            labels=group.label.values,
            predicted_scores=group.prediction.values,
            sample_weight=group.sample_weight.values,
        )[1]

    segment_p_values = kuiper_func_per_segment(
        labels=labels,
        predictions=predictions,
        segments_df=segments_df,
        func=_group_kuiper_p_value,
        output_series_name="p_value",
        sample_weight=sample_weight,
        min_segment_size=min_segment_size,
    )

    return segment_p_values


def kuiper_statistic_per_segment(
    labels: npt.NDArray,
    predictions: npt.NDArray,
    segments_df: pd.DataFrame,
    sample_weight: npt.NDArray | None = None,
    min_segment_size: int = 2,
    standardization_method: str | None = "kuiper_standard_deviation",
) -> pd.Series:
    """
    Calculate kuiper statistics for each segment in the data.

    :param labels: Array of true binary labels (0 or 1)
    :param predictions: Array of predicted probabilities corresponding to the likelihood of label being 1
    :param segments_df: DataFrame containing the segment definitions to calculate statistics for
    :param sample_weight: Optional array of weights for each instance. If None, all instances have weight 1
    :param min_segment_size: Minimum number of samples required in a segment to calculate statistics. Defaults to 2
    :param standardization_method: Optional string specifying the method used for standardizing the kuiper statistic. Defaults to kuiper_standard_deviation
    :return: series containing kuiper statistics for each segment
    """

    def _group_kuiper_statistic_value(group: pd.DataFrame) -> float | None:
        if len(group) < min_segment_size:
            return None
        return kuiper_test(
            labels=group.label.values,
            predicted_scores=group.prediction.values,
            sample_weight=group.sample_weight.values,
        )[0]

    segment_p_values = kuiper_func_per_segment(
        labels=labels,
        predictions=predictions,
        segments_df=segments_df,
        func=_group_kuiper_statistic_value,
        output_series_name="kuiper_statistic",
        sample_weight=sample_weight,
        min_segment_size=min_segment_size,
    )

    return segment_p_values


def multi_segment_pvalue_geometric_mean(
    labels: npt.NDArray,
    predictions: npt.NDArray,
    segments_df: pd.DataFrame,
    sample_weight: npt.NDArray | None = None,
    min_segment_size: int = 2,
) -> float:
    """
    Calculate the geometric mean of Kuiper p-values across segments.

    :param labels: Array of true binary labels.
    :param predictions: Array of predicted probability scores.
    :param segments_df: DataFrame defining the segmentation columns.
    :param sample_weight: Optional array of sample weights.
    :param min_segment_size: Minimum samples required per segment.
    :return: Geometric mean of segment p-values.
    """
    segment_p_values = kuiper_pvalue_per_segment(
        labels=labels,
        predictions=predictions,
        segments_df=segments_df,
        sample_weight=sample_weight,
        min_segment_size=min_segment_size,
    )
    result = utils.geometric_mean(segment_p_values.to_numpy(np.float64))
    return result


def multi_segment_inverse_sqrt_normalized_statistic_max(
    labels: npt.NDArray,
    predictions: npt.NDArray,
    segments_df: pd.DataFrame,
    sample_weight: npt.NDArray | None = None,
    min_segment_size: int = 2,
) -> float:
    """
    Calculate the max Kuiper statistic across segments with inverse-sqrt normalization.

    :param labels: Array of true binary labels.
    :param predictions: Array of predicted probability scores.
    :param segments_df: DataFrame defining the segmentation columns.
    :param sample_weight: Optional array of sample weights.
    :param min_segment_size: Minimum samples required per segment.
    :return: Maximum normalized Kuiper statistic across all segments.
    """
    segment_statistics = kuiper_statistic_per_segment(
        labels=labels,
        predictions=predictions,
        segments_df=segments_df,
        sample_weight=sample_weight,
        min_segment_size=min_segment_size,
        standardization_method="kuiper_upper_bound_standard_deviation",
    )
    result = segment_statistics.to_numpy(np.float64).max()
    return result


def multi_segment_kuiper_test(
    labels: npt.NDArray,
    predictions: npt.NDArray,
    segments_df: pd.DataFrame,
    sample_weight: npt.NDArray | None = None,
    alpha: float = 0.05,
    combination_method: str = "poisson",
    min_segment_size: int = 2,
) -> dict[str, float]:
    """
    Perform a combined Kuiper test across multiple segments.

    :param labels: Array of true binary labels.
    :param predictions: Array of predicted probability scores.
    :param segments_df: DataFrame defining the segmentation columns.
    :param sample_weight: Optional array of sample weights.
    :param alpha: Significance level for the test.
    :param combination_method: Method to combine p-values ('poisson' or scipy methods).
    :param min_segment_size: Minimum samples required per segment.
    :return: Dictionary with n_segments, statistic, p_value, and segment_p_values.
    """
    segment_p_values = kuiper_pvalue_per_segment(
        labels=labels,
        predictions=predictions,
        segments_df=segments_df,
        sample_weight=sample_weight,
        min_segment_size=min_segment_size,
    )

    n_tests = len(segment_p_values)

    combined_p_value = None
    if combination_method == "poisson":
        statistic = np.sum(segment_p_values.values < alpha)
        combined_p_value = 1 - stats.poisson.cdf(statistic - 1, n_tests * alpha)
    else:
        statistic, combined_p_value = stats.combine_pvalues(
            segment_p_values, method=combination_method
        )
    return {
        "n_segments": n_tests,
        "statistic": statistic,
        "p_value": combined_p_value,
        "segment_p_values": segment_p_values,
    }


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
        segments_df.groupby(grouping_cols)
        .apply(_group_rank_calibration_error)
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
        segments_df.groupby(grouping_cols)
        .apply(_group_rank_calibration_error)
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


DEFAULT_MULTI_KUIPER_NORMALIZATION_METHOD: str = "kuiper_standard_deviation"
DEFAULT_MULTI_KUIPER_MAX_VALUES_PER_SEGMENT_FEATURE: int = 3
DEFAULT_MULTI_KUIPER_MIN_DEPTH: int = 0
DEFAULT_MULTI_KUIPER_MAX_DEPTH: int = 3
DEFAULT_MULTI_KUIPER_MIN_SAMPLES_PER_SEGMENT: int = 10
DEFAULT_MULTI_KUIPER_GLOBAL_NORMALIZATION: str = "prevalence_adjusted"
DEFAULT_MULTI_KUIPER_N_SEGMENTS: int | None = 1000


class MulticalibrationError:
    def __init__(
        self,
        df: pd.DataFrame,
        label_column: str,
        score_column: str,
        weight_column: str | None = None,
        categorical_segment_columns: list[str] | None = None,
        numerical_segment_columns: list[str] | None = None,
        max_depth: int | None = DEFAULT_MULTI_KUIPER_MAX_DEPTH,
        max_values_per_segment_feature: int = DEFAULT_MULTI_KUIPER_MAX_VALUES_PER_SEGMENT_FEATURE,
        min_samples_per_segment: int = DEFAULT_MULTI_KUIPER_MIN_SAMPLES_PER_SEGMENT,
        sigma_estimation_method: str | None = DEFAULT_MULTI_KUIPER_NORMALIZATION_METHOD,
        max_n_segments: int | None = DEFAULT_MULTI_KUIPER_N_SEGMENTS,
        chunk_size: int = 50,
        precision_dtype: str = "float32",
    ) -> None:
        """
        Calculates the multicalibration error with respect to a set of segments for a given dataset.

        :param df: A pandas DataFrame containing the data.
        :param label_column: The name of the column in `df` that contains the true labels.
        :param score_column: The name of the column in `df` that contains the predicted scores.
        :param weight_column: An optional column in `df` that contains sample weights.
        :param categorical_segment_columns: An optional list of column names in `df` to be used for categorical segmentation.
        :param numerical_segment_columns: An optional list of column names in `df` to be used for numerical segmentation.
        :param max_depth: The maximum depth for segment generation.
        :param max_values_per_segment_feature: The maximum number of unique values per segment feature.
        :param min_samples_per_segment: The minimum number of samples required per segment.
        :param sigma_estimation_method: The method used for sigma estimation.
        :param max_n_segments: The maximum number of segments to generate.
        :param chunk_size: Size of chunks of segments to process per iteration of the algorithm. Larger values improve runtime but increase memory usage (OOM errors are possible).
        :param precision_dtype: The precision type for the metric. Can be 'float16', 'float32', or 'float64'.
        """
        self.label_column = label_column
        self.score_column = score_column
        self.weight_column = weight_column
        self.categorical_segment_columns = categorical_segment_columns
        self.numerical_segment_columns = numerical_segment_columns
        self.max_depth = max_depth
        self.max_values_per_segment_feature = max_values_per_segment_feature
        self.min_samples_per_segment = min_samples_per_segment
        self.estimate_sigma: KuiperNormalizationInterface = (
            _normalization_method_assignment(sigma_estimation_method)
        )
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
        self.precision_dtype: np.float16 | np.float32 | np.float64 = getattr(
            np, precision_dtype
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
        return f"""{self.mce}% (sigmas={self.mce_sigma_scale}, p={self.p_value}, mde={self.mde})"""

    def __format__(self, format_spec: str) -> str:
        # Use the format specifier to format each attribute
        formatted_mce_relative = format(self.mce, format_spec)
        formatted_p_value = format(self.p_value, format_spec)
        formatted_mde = format(self.mde, format_spec)
        formatted_mce_sigma_scale = format(self.mce_sigma_scale, format_spec)
        return f"""{formatted_mce_relative}% (sigmas={formatted_mce_sigma_scale}, p={formatted_p_value}, mde={formatted_mde})"""

    @functools.cached_property
    def segments(self) -> tuple[npt.NDArray[np.bool_], pd.DataFrame]:
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
    def segment_indices(self) -> pd.Series:
        segments_2d = self.segments[0].reshape(-1, self.segments[0].shape[2])
        indices = np.argwhere(segments_2d)
        index_series = pd.Series(indices[:, 1], index=indices[:, 0])

        return index_series

    @functools.cached_property
    def segment_ecces_absolute(
        self,
    ) -> npt.NDArray[np.float16 | np.float32 | np.float64]:
        segments = self.segments[0]
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
            ] = kuiper_calibration_per_segment(
                labels=self.df[self.label_column].values,
                predicted_scores=self.df[self.score_column].values,
                sample_weight=(
                    None
                    if self.weight_column is None
                    else self.df[self.weight_column].values
                ),
                normalization_method=None,
                segments=segment[: self.total_number_segments - self.chunk_size * i,],
            )
        return statistics

    @functools.cached_property
    def segment_ecces(self) -> npt.NDArray[np.float16 | np.float32 | np.float64]:
        return self.segment_ecces_sigma_scale * self.sigma_0 / self.prevalence * 100

    @functools.cached_property
    def global_ecce(self) -> float:
        return self.segment_ecces[0]

    @functools.cached_property
    def global_ecce_sigma_scale(self) -> float:
        return self.segment_ecces_sigma_scale[0]

    @functools.cached_property
    def global_ecce_p_value(self) -> float:
        return self.segment_p_values[0]

    @functools.cached_property
    def segment_ecces_sigma_scale(
        self,
    ) -> npt.NDArray[np.float16 | np.float32 | np.float64]:
        with np.errstate(divide="ignore", invalid="ignore"):
            statistics = np.where(
                (self.segment_ecces_absolute != 0) & (self.segment_sigmas == 0),
                np.inf,
                np.where(
                    self.segment_sigmas == 0,
                    0,
                    self.segment_ecces_absolute / self.segment_sigmas,
                ),
            )
        return statistics

    @functools.cached_property
    def segment_p_values(self) -> npt.NDArray[np.float16 | np.float32 | np.float64]:
        kuiper_distribution_vec = np.vectorize(kuiper_distribution)
        p_values = np.ones_like(
            self.segment_ecces_sigma_scale
        ) - kuiper_distribution_vec(self.segment_ecces_sigma_scale)
        return p_values

    @functools.cached_property
    def segment_sigmas(self) -> npt.NDArray[np.float16 | np.float32 | np.float64]:
        segments = self.segments[0]
        sigmas = np.zeros(self.total_number_segments, dtype=self.precision_dtype)
        for i, segment in enumerate(segments):
            sigmas[
                self.chunk_size * i : min(
                    self.chunk_size * (i + 1),
                    self.total_number_segments,
                )
            ] = self.estimate_sigma(
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
    def sigma_0(self) -> float:
        if "segment_sigmas" in self.__dict__:
            return self.segment_sigmas[0]
        sigma_0 = self.estimate_sigma(
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
        return sigma_0.item()

    @functools.cached_property
    def mce_sigma_scale(self) -> float:
        return np.max(self.segment_ecces_sigma_scale)

    @functools.cached_property
    def mce_absolute(self) -> float:
        return self.mce_sigma_scale * self.sigma_0

    @functools.cached_property
    def prevalence(self) -> float:
        p = (
            (self.df[self.label_column] * self.df[self.weight_column]).sum()
            / (self.df[self.weight_column].sum())
            if self.weight_column is not None
            else self.df[self.label_column].mean()
        )
        return min(p, 1 - p)

    @functools.cached_property
    def mce(self) -> float:
        return self.mce_absolute / self.prevalence * 100

    @functools.cached_property
    def p_value(self) -> float:
        if "segment_p_values" in self.__dict__:
            return np.min(self.segment_p_values)
        return 1 - kuiper_distribution(self.mce_sigma_scale)

    @functools.cached_property
    def mde(self) -> float:
        # This is a rough, conservative approximation of the MDE. We divide by the prevalence
        # and multiply by 100 to get the MDE in the same unit as the MCE metric
        return 5 * self.sigma_0 / self.prevalence * 100


class ScoreFunctionInterface(Protocol):
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
) -> ScoreFunctionInterface:
    """
    Wrap an sklearn-style metric function for use with the evaluation framework.

    :param func: A function with signature (y_true, y_pred, sample_weight=None) -> float.
    :return: A ScoreFunctionInterface-compatible wrapper.
    """

    class WrappedFuncSkLearn(ScoreFunctionInterface):
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
    max_depth: int = DEFAULT_MULTI_KUIPER_MAX_DEPTH,
    max_values_per_segment_feature: int = DEFAULT_MULTI_KUIPER_MAX_VALUES_PER_SEGMENT_FEATURE,
    min_samples_per_segment: int = DEFAULT_MULTI_KUIPER_MIN_SAMPLES_PER_SEGMENT,
    sigma_estimation_method: str = DEFAULT_MULTI_KUIPER_NORMALIZATION_METHOD,
    max_n_segments: int | None = DEFAULT_MULTI_KUIPER_N_SEGMENTS,
    metric_version: str = "mce",
) -> ScoreFunctionInterface:
    """
    Create a wrapped MulticalibrationError metric for use with the evaluation framework.

    :param categorical_segment_columns: Columns to use for categorical segmentation.
    :param numerical_segment_columns: Columns to use for numerical segmentation.
    :param max_depth: Maximum depth for segment generation.
    :param max_values_per_segment_feature: Max unique values per segment feature.
    :param min_samples_per_segment: Minimum samples required per segment.
    :param sigma_estimation_method: Method for sigma estimation.
    :param max_n_segments: Maximum number of segments to generate.
    :param metric_version: Which metric to return ('mce', 'mce_sigma_scale', 'mce_absolute', 'p_value').
    :return: A ScoreFunctionInterface-compatible wrapper.
    """
    if categorical_segment_columns is None and numerical_segment_columns is None:
        raise ValueError(
            "No segment columns provided. Please provide either "
            "categorical_segment_columns or numerical_segment_columns."
        )
    valid_versions = ("mce", "mce_sigma_scale", "mce_absolute", "p_value")
    if metric_version not in valid_versions:
        raise ValueError(
            f"`metric_version` has to be one of {list(valid_versions)}. "
            f"Got `{metric_version}`."
        )

    class WrappedFuncMCE(ScoreFunctionInterface):
        name = f"Multicalibration Error<br>({metric_version})"

        def __init__(
            self,
            categorical_segment_columns: list[str] | None,
            numerical_segment_columns: list[str] | None,
            max_depth: int = DEFAULT_MULTI_KUIPER_MAX_DEPTH,
            max_values_per_segment_feature: int = DEFAULT_MULTI_KUIPER_MAX_VALUES_PER_SEGMENT_FEATURE,
            min_samples_per_segment: int = DEFAULT_MULTI_KUIPER_MIN_SAMPLES_PER_SEGMENT,
            sigma_estimation_method: str = DEFAULT_MULTI_KUIPER_NORMALIZATION_METHOD,
            max_n_segments: int | None = DEFAULT_MULTI_KUIPER_N_SEGMENTS,
        ):
            self.categorical_segment_columns = categorical_segment_columns
            self.numerical_segment_columns = numerical_segment_columns
            self.max_depth = max_depth
            self.max_values_per_segment_feature = max_values_per_segment_feature
            self.min_samples_per_segment = min_samples_per_segment
            self.sigma_estimation_method = sigma_estimation_method
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
                sigma_estimation_method=self.sigma_estimation_method,
                max_n_segments=self.max_n_segments,
            )
            return getattr(mce, metric_version)

    return WrappedFuncMCE(
        categorical_segment_columns,
        numerical_segment_columns,
        max_depth,
        max_values_per_segment_feature,
        min_samples_per_segment,
        sigma_estimation_method,
        max_n_segments,
    )
