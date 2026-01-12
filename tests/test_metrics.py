# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

import warnings
from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
import pytest
import scipy
import sklearn.metrics as skmetrics
from multicalibration import metrics, utils
from multicalibration.metrics import (
    fpr,
    kuiper_standard_deviation,
    kuiper_test,
    wrap_multicalibration_error_metric,
    wrap_sklearn_metric_func,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.mark.parametrize(
    "rank_discount",
    [
        utils.rank_log_discount,
        partial(utils.rank_log_discount, log_base=3),
        utils.rank_no_discount,
    ],
)
def test_dcg_sample_scores_are_lower_than_ideal(rank_discount: Callable, rng):
    y_true, y_score = rng.random_sample((2, 100))
    ideal = metrics._dcg_sample_scores(
        y_true,
        y_true,
        rank_discount=rank_discount,
    )
    score = metrics._dcg_sample_scores(
        y_true,
        y_score,
        rank_discount=rank_discount,
    )
    assert (score <= ideal).all()


@pytest.mark.parametrize(
    "rank_discount",
    [
        utils.rank_log_discount,
        partial(utils.rank_log_discount, log_base=3),
        utils.rank_no_discount,
    ],
)
def test_dcg_sample_scores_gives_expected_result(rank_discount: Callable, rng):
    y_true, y_score = rng.random_sample((2, 100))
    scores = metrics._dcg_sample_scores(
        y_true,
        y_score,
        rank_discount=rank_discount,
    )
    assert scores.shape == (y_true.shape[0],)

    discount = rank_discount(y_true.shape[0])
    assert np.allclose(scores, np.cumsum(y_true[np.argsort(y_score)[::-1]] * discount))


@pytest.mark.parametrize(
    "rank_discount",
    [
        utils.rank_log_discount,
        partial(utils.rank_log_discount, log_base=3),
        utils.rank_no_discount,
    ],
)
def test_dcg_score_is_lower_than_ideal(rank_discount: Callable, rng):
    y_true, y_score = rng.random_sample((2, 100))
    ideal = metrics.dcg_score(y_true, y_true, rank_discount=rank_discount)
    score = metrics.dcg_score(y_true, y_score, rank_discount=rank_discount)
    assert score <= ideal


@pytest.mark.parametrize(
    "rank_discount",
    [
        utils.rank_log_discount,
        partial(utils.rank_log_discount, log_base=3),
        utils.rank_no_discount,
    ],
)
def test_dcg_score_is_smaller_when_k_is_specified(rank_discount: Callable, rng):
    y_true, y_score = rng.random_sample((2, 100))
    score = metrics.dcg_score(y_true, y_score, rank_discount=rank_discount)
    assert (
        metrics.dcg_score(y_true, y_score, k=10, rank_discount=rank_discount) <= score
    )


@pytest.mark.parametrize(
    "rank_discount",
    [
        utils.rank_log_discount,
        partial(utils.rank_log_discount, log_base=3),
        utils.rank_no_discount,
    ],
)
def test_dcg_score_is_correct_when_k_larger_than_array_length(
    rank_discount: Callable, rng
):
    y_true, y_score = rng.random_sample((2, 100))
    score = metrics.dcg_score(
        y_true, y_score, k=len(y_true) + 100, rank_discount=rank_discount
    )
    assert np.allclose(
        metrics.dcg_score(y_true, y_score, rank_discount=rank_discount), score
    )


@pytest.mark.parametrize(
    "rank_discount",
    [
        utils.rank_log_discount,
        partial(utils.rank_log_discount, log_base=3),
        utils.rank_no_discount,
    ],
)
def test_dcg_score_gives_expected_result(rank_discount: Callable, rng):
    y_true, y_score = rng.random_sample((2, 100))
    scores = metrics.dcg_score(
        y_true,
        y_score,
        rank_discount=rank_discount,
    )
    discount = rank_discount(y_true.shape[0])
    assert np.allclose(
        scores,
        np.cumsum(y_true[np.argsort(y_score)[::-1]] * discount)[-1],
    )


@pytest.mark.parametrize(
    "rank_discount",
    [
        utils.rank_log_discount,
        partial(utils.rank_log_discount, log_base=3),
        utils.rank_no_discount,
    ],
)
def test_ndcg_sample_score_is_lower_than_ideal(rank_discount: Callable, rng):
    y_true, y_score = rng.random_sample((2, 100))
    ideal = metrics._ndcg_sample_scores(
        y_true,
        y_true,
        rank_discount=rank_discount,
    )
    score = metrics._ndcg_sample_scores(
        y_true,
        y_score,
        rank_discount=rank_discount,
    )
    assert (score <= ideal).all()


@pytest.mark.parametrize(
    "rank_discount",
    [
        utils.rank_log_discount,
        partial(utils.rank_log_discount, log_base=3),
        utils.rank_no_discount,
    ],
)
def test_ndcg_sample_ideal_is_one_for_all_samples(rank_discount: Callable, rng):
    y_true, _ = rng.random_sample((2, 100))
    ideal = metrics._ndcg_sample_scores(
        y_true,
        y_true,
        rank_discount=rank_discount,
    )
    all_zero = y_true == 0

    # ideal NDCG is always 1 except for those label values that are 0
    assert np.allclose(ideal[~all_zero], np.ones((~all_zero).sum()))

    # for the label values that are 0, the ideal NDCG is 0
    assert np.allclose(ideal[all_zero], np.zeros(all_zero.sum()))


@pytest.mark.parametrize(
    "rank_discount",
    [
        utils.rank_log_discount,
        partial(utils.rank_log_discount, log_base=3),
        utils.rank_no_discount,
    ],
)
def test_ndcg_sample_score_gives_correct_result(rank_discount: Callable, rng):
    y_true, y_score = rng.random_sample((2, 100))
    k = 10
    score = metrics._ndcg_sample_scores(
        y_true,
        y_score,
        k=k,
        rank_discount=rank_discount,
    )

    assert score.shape == (y_true.shape[0],)

    all_zero = y_true == 0

    # by definition of NDCG, the score should be the ratio of the DCG and the ideal DCG
    assert np.allclose(
        score[~all_zero],
        metrics._dcg_sample_scores(y_true, y_score, k=k, rank_discount=rank_discount)[
            ~all_zero
        ]
        / metrics._dcg_sample_scores(y_true, y_true, k=k, rank_discount=rank_discount)[
            ~all_zero
        ],
    )

    # for the label values that are 0, NDCG is 0
    assert np.allclose(score[all_zero], np.zeros(all_zero.sum()))


@pytest.mark.parametrize(
    "rank_discount",
    [
        utils.rank_log_discount,
        partial(utils.rank_log_discount, log_base=3),
        utils.rank_no_discount,
    ],
)
def test_ndcg_score_is_lower_than_ideal(rank_discount: Callable, rng):
    y_true, y_score = rng.random_sample((2, 100))
    k = 10
    ideal = metrics.ndcg_score(y_true, y_true, rank_discount=rank_discount, k=k)
    score = metrics.ndcg_score(y_true, y_score, rank_discount=rank_discount, k=k)
    assert score <= ideal


@pytest.mark.parametrize(
    "rank_discount",
    [
        utils.rank_log_discount,
        partial(utils.rank_log_discount, log_base=3),
        utils.rank_no_discount,
    ],
)
def test_ndcg_score_ideal_is_one_for_all_samples(rank_discount: Callable, rng):
    y_true, y_score = rng.random_sample((2, 100))

    # ideal NDCG should always be 1, for any k
    assert np.allclose(
        metrics.ndcg_score(y_true, y_true, rank_discount=rank_discount), 1.0
    )
    assert np.allclose(
        metrics.ndcg_score(y_true, y_true, k=10, rank_discount=rank_discount), 1.0
    )


@pytest.mark.parametrize(
    "rank_discount",
    [
        utils.rank_log_discount,
        partial(utils.rank_log_discount, log_base=3),
        utils.rank_no_discount,
    ],
)
def test_ndcg_score_when_ranking_is_preserved_is_one(rank_discount: Callable, rng):
    _, y_score = rng.random_sample((2, 100))
    y_true = y_score * 2.0
    assert np.allclose(
        metrics.ndcg_score(y_true, y_score, k=10, rank_discount=rank_discount), 1.0
    )


@pytest.mark.parametrize(
    "rank_discount",
    [
        utils.rank_log_discount,
        partial(utils.rank_log_discount, log_base=3),
        utils.rank_no_discount,
    ],
)
def test_ndcg_score_gives_expected_result(rank_discount: Callable, rng):
    y_true, y_score = rng.random_sample((2, 100))
    k = 10
    score = metrics.ndcg_score(
        y_true,
        y_score,
        k=k,
        rank_discount=rank_discount,
    )
    discount = rank_discount(y_true.shape[0])
    assert np.allclose(
        score,
        np.cumsum(y_true[np.argsort(y_score)[::-1]] * discount)[k - 1]
        / np.cumsum(np.sort(y_true)[::-1] * discount)[k - 1],
    )


@pytest.mark.parametrize(
    "rank_discount",
    [
        utils.rank_log_discount,
        partial(utils.rank_log_discount, log_base=3),
        utils.rank_no_discount,
    ],
)
def test_ndcg_score_is_correct_when_k_larger_than_array_length(
    rank_discount: Callable, rng
):
    y_true, y_score = rng.random_sample((2, 100))
    score = metrics.ndcg_score(
        y_true, y_score, k=len(y_true) + 100, rank_discount=rank_discount
    )
    assert np.allclose(
        metrics.ndcg_score(y_true, y_score, rank_discount=rank_discount), score
    )


@pytest.mark.parametrize(
    "rank_discount",
    [
        utils.rank_log_discount,
        partial(utils.rank_log_discount, log_base=3),
        utils.rank_no_discount,
    ],
)
def test_multi_cg_gives_same_result_as_cg_per_segment(rank_discount, rng):
    n_samples = 1000
    k = 100
    df = pd.DataFrame(index=range(n_samples))
    df["label"] = rng.random_sample(n_samples)
    df["score"] = rng.random_sample(n_samples)
    df["segment_1"] = rng.choice(["A", "B"], size=len(df))
    df["segment_2"] = rng.choice(["C", "D"], size=len(df))

    min_segments_size = df.groupby(by=["segment_1", "segment_2"]).count().values.min()
    k = min(k, min_segments_size)

    multi_cg_scores = metrics.multi_cg_score(
        labels=df["label"],
        predictions=df["score"],
        segments_df=df[["segment_1", "segment_2"]],
        metric=metrics.ndcg_score,
        rank_discount=rank_discount,
        k=k,
    )

    groups = df.groupby(by=["segment_1", "segment_2"])
    for segment_1 in ["A", "B"]:
        for segment_2 in ["C", "D"]:
            group = groups.get_group((segment_1, segment_2))
            segment_result = metrics.ndcg_score(
                labels=group.label.values,
                predicted_labels=group.score.values,
                rank_discount=rank_discount,
                k=k,
            )
            assert np.allclose(segment_result, multi_cg_scores[segment_1, segment_2])


def test_predictions_to_labels_gives_expected_result():
    data = pd.DataFrame(
        {
            "prediction": [0.1, 0.2, 0.4, 0.8, 0.9],
            "segment_a": ["a", "a", "b", "b", "a"],
            "segment_b": ["i", "j", "i", "j", "i"],
        }
    )
    thresholds = pd.DataFrame(
        {
            "segment_a": ["a", "a", "b", "b"],
            "segment_b": ["i", "j", "i", "j"],
            "threshold": [0.8, 0.7, 0.6, 0.5],
        }
    )
    expected = pd.DataFrame(
        {
            "prediction": [0.1, 0.2, 0.4, 0.8, 0.9],
            "segment_a": ["a", "a", "b", "b", "a"],
            "segment_b": ["i", "j", "i", "j", "i"],
            "threshold": [0.8, 0.7, 0.6, 0.5, 0.8],
            "predicted_label": [0, 0, 0, 1, 1],
        }
    )

    data_with_predicted_labels_and_thresholds = metrics.predictions_to_labels(
        data=data,
        prediction_column="prediction",
        thresholds=thresholds,
        threshold_column="threshold",
    )

    pd.testing.assert_frame_equal(data_with_predicted_labels_and_thresholds, expected)


def test_multicalibration_error_without_weights_gives_expected_results():
    # Create some sample data
    _labels = np.linspace(0, 1, 100)
    _predictions = 1 - np.linspace(0, 1, 100)
    segments_df = pd.DataFrame({"segment": np.array(["A"] * 50 + ["B"] * 50)})

    # Define a custom metric function
    def custom_metric(
        labels, predicted_scores, num_bins=40, epsilon=0.0000001, sample_weight=None
    ):
        return np.abs(
            np.mean(predicted_scores * sample_weight) - np.mean(labels * sample_weight)
        )

    # Test the multicalibration_error method with the custom metric function
    error = metrics.multicalibration_error(
        _labels, _predictions, segments_df, metric=custom_metric
    )
    assert pytest.approx(error, 0.001) == 0.5050505050505052


@pytest.mark.parametrize(
    "labels, predictions, segments_df, sample_weight, metric, expected_error",
    [
        (
            np.linspace(0, 1, 100),
            1 - np.linspace(0, 1, 100),
            pd.DataFrame({"segment": np.array(["A"] * 50 + ["B"] * 50)}),
            np.ones(100),
            metrics.expected_calibration_error,
            0.5050505050505052,
        ),
        (
            np.linspace(0, 1, 100),
            1 - np.linspace(0, 1, 100),
            pd.DataFrame({"segment": np.array(["A"] * 50 + ["B"] * 50)}),
            np.concatenate((np.ones(50), np.ones(50) * 2)),
            metrics.proportional_expected_calibration_error,
            6.73198676283903,
        ),
    ],
)
def test_multicalibration_error_with_weights_gives_expected_results(
    labels, predictions, segments_df, sample_weight, metric, expected_error
):
    error = metrics.multicalibration_error(
        labels,
        predictions,
        segments_df,
        sample_weight=sample_weight,
        metric=metric,
    )
    assert pytest.approx(error, 0.001) == expected_error


@pytest.mark.parametrize(
    "labels, predictions, bins, sample_weight, expected_output, use_weights_in_sample_size",
    [
        (
            np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            np.array([0.05, 0.2, 0.4, 0.6, 0.8, 1.01]),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            (
                np.array([np.nan, 0, 0.5, 0.5, 1.0, 2 / 3]),
                np.array([np.nan, np.nan, 0.012579, 0.012579, 0.158114, 0.09429932]),
                np.array([np.nan, 0.975, 0.98742088, 0.98742088, np.nan, 0.99159624]),
                np.array([np.nan, 0.1, 0.25, 0.45, 0.65, 0.9]),
            ),
            True,
        ),
        (
            np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            np.array([0.2, 0.4, 0.6, 0.8, 1.01]),
            np.array([1, 2, 2, 3, 1, 3, 3, 2, 1, 1]),
            (
                np.array([0, 0.5, 3 / 4, 1.0, 0.5]),
                np.array([np.nan, 0.06758599, 0.19412045, 0.54074187, 0.06758599]),
                np.array([0.975, 0.93241401, 0.99369054, np.nan, 0.93241401]),
                np.array([0.1, 0.25, 0.425, 0.65, 0.875]),
            ),
            True,
        ),
        (
            np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            np.array([0.05, 0.2, 0.4, 0.6, 0.8, 1.01]),
            np.array([1, 2, 2, 3, 1, 3, 3, 2, 1, 1]),
            (
                np.array([np.nan, 0, 0.5, 0.75, 1, 0.5]),
                np.array(
                    [np.nan, np.nan, 0.01257912, 0.01257912, 0.15811388, 0.00840376]
                ),
                np.array([np.nan, 0.975, 0.98742088, 0.98742088, np.nan, 0.90570068]),
                np.array([np.nan, 0.1, 0.25, 0.425, 0.65, 0.875]),
            ),
            False,
        ),
    ],
)
def test_positive_label_proportion_with_weights_gives_expected_output(
    labels,
    predictions,
    bins,
    sample_weight,
    expected_output,
    use_weights_in_sample_size,
):
    result = utils.positive_label_proportion(
        labels=labels,
        predictions=predictions,
        bins=bins,
        sample_weight=sample_weight,
        use_weights_in_sample_size=use_weights_in_sample_size,
    )
    for res, eo in zip(result, expected_output):
        assert np.allclose(res[np.isfinite(res)], eo[np.isfinite(eo)])
        assert np.array_equal(np.isnan(res), np.isnan(eo))


def test_fpr():
    assert fpr(np.array([]), np.array([])) == 0.0

    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    predicted_labels = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    # Single-class data may trigger sklearn warning about confusion matrix shape
    # (depends on sklearn version)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A single label was found")
        assert fpr(labels, predicted_labels) == 0.0

    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    predicted_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    assert fpr(labels, predicted_labels) == 0.0

    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    predicted_labels = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    assert fpr(labels, predicted_labels) == 0.0

    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    predicted_labels = np.array([1, 1, 1, 1, 1, 1, 1, 1])

    assert fpr(labels, predicted_labels) == 1.0

    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    predicted_labels = np.array([1, 0, 1, 0, 0, 0, 0, 0])

    assert fpr(labels, predicted_labels) == 0.5

    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    predicted_labels = np.array([1, 0, 0, 0, 0, 0, 0, 0])

    assert fpr(labels, predicted_labels) == 0.25


@pytest.mark.parametrize(
    "scores, labels, sample_weight, expected_result",
    [
        ([0.6, 0.8, 0.2, 0.4], [0, 1, 0, 1], None, 0.15),
        ([0.6, 0.8, 0.2, 0.4], [1, 1, 1, 1], None, 0.5),
        ([0.6, 0.8, 0.2, 0.4], [0, 1, 0, 0], None, 0.3),
        ([0.6, 0.8, 0.2, 0.4], [0, 1, 0, 0], [0.4, 0.3, 0.2, 0.1], 0.32),
        ([0.0, 0.0, 0.0, 0.0], [0, 1, 0, 0], None, 0.25),
        ([1.0, 1.0, 1.0, 1.0], [0, 1, 0, 0], None, 0.75),
    ],
)
def test_kuiper_calibration_gives_expected_result(
    scores, labels, sample_weight, expected_result
):
    scores, labels = np.array(scores), np.array(labels)
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)
    calibration_metric = metrics.kuiper_calibration(labels, scores, sample_weight)

    # Check that the offset is correctly calculated
    np.testing.assert_allclose(calibration_metric, expected_result)


@pytest.mark.parametrize(
    "scores, labels, sample_weight, expected_result",
    [
        ([0.6, 0.8, 0.2, 0.4], [0, 1, 0, 1], None, 0.67082),
        ([0.6, 0.8, 0.2, 0.4], [1, 1, 1, 1], None, 2.23607),
        ([0.6, 0.8, 0.2, 0.4], [0, 1, 0, 0], None, 1.34164),
        ([0.6, 0.8, 0.2, 0.4], [0, 1, 0, 0], [0.4, 0.3, 0.2, 0.1], 1.289317),
    ],
)
def test_kuiper_calibration_standardized_gives_expected_result(
    scores, labels, sample_weight, expected_result
):
    scores, labels = np.array(scores), np.array(labels)
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)
    calibration_metric = metrics.kuiper_calibration(
        labels,
        scores,
        sample_weight,
        normalization_method="kuiper_standard_deviation",
    )

    # Check that the offset is correctly calculated
    np.testing.assert_allclose(calibration_metric, expected_result, atol=1e-5)


@pytest.mark.parametrize(
    "scores, labels, sample_weight, expected_result",
    [
        # These have zero total variance and zero cumulative differences, but test statistic 0 makes sense given that all predictions are correct
        ([0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0], None, 0),
        ([1.0, 1.0, 1.0, 1.0], [1, 1, 1, 1], None, 0),
        # These have zero total variance and zero cumulative differences, but test statistic math.inf makes sense given that all predictions are correct
        ([0.0, 0.0, 0.0, 0.0], [1, 1, 1, 1], None, np.inf),
        ([1.0, 1.0, 1.0, 1.0], [0, 0, 0, 0], None, np.inf),
    ],
)
def test_kuiper_calibration_standardized_gives_expected_result_for_scores_resulting_in_zero_kuiper_stat_variance(
    scores, labels, sample_weight, expected_result
):
    scores, labels = np.array(scores), np.array(labels)
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)
    calibration_metric = metrics.kuiper_calibration(
        labels,
        scores,
        sample_weight,
        normalization_method="kuiper_standard_deviation",
    )

    # Check that the offset is correctly calculated
    np.testing.assert_allclose(calibration_metric, expected_result)


@pytest.mark.parametrize(
    "labels, predicted_scores, sample_weight, num_bins, expected",
    [
        (
            np.array([0, 1, 1, 0, 1]),
            np.array([0.1, 0.9, 0.8, 0.2, 0.7]),
            np.array([1, 1, 1, 1, 1]),
            5,
            0.556303,
        ),
        (
            np.array([0, 1, 1, 0, 1]),
            np.array([0.1, 0.9, 0.8, 0.2, 0.7]),
            np.array([2, 1, 1, 1, 1]),
            5,
            0.630252,
        ),
        (
            np.array([0, 1, 1, 0, 1]),
            np.array([0.1, 0.9, 0.8, 0.2, 0.7]),
            np.array([2, 1, 2, 1, 1]),
            5,
            0.575510,
        ),
        (
            np.array([0, 1, 0, 0, 1]),
            np.array([0.2, 0.8, 0.3, 0.1, 0.9]),
            np.array([1, 1, 1, 1, 1]),
            5,
            0.670588,
        ),
    ],
)
def test_proportional_expected_calibration_error_gives_expected_result(
    labels, predicted_scores, sample_weight, num_bins, expected
):
    result = metrics.proportional_expected_calibration_error(
        labels, predicted_scores, sample_weight, num_bins
    )
    assert np.isclose(result, expected, atol=1e-6)


@pytest.mark.parametrize(
    "labels, predictions",
    [
        (
            np.array([0, 0, 1, 0, 1, 1]),
            np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9]),
        ),
        (
            np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 1]),
            np.array([0.01, 0.01, 0.01, 0.5, 0.5, 0.5, 0.5, 0.99, 0.99, 0.99]),
        ),
    ],
)
def test_adaptive_calibration_error_with_unjoined_data_gives_expected_result(
    labels, predictions
):
    # Calculate the calibration error for the original data
    original_error = metrics.adaptive_calibration_error(labels, predictions, num_bins=1)

    # Calculate the calibration error for the unjoined data, with adjustment
    unjoined_predictions, unjoined_labels = utils.make_unjoined(predictions, labels)
    unjoined_error = metrics.adaptive_calibration_error(
        unjoined_labels, unjoined_predictions, adjust_unjoined=True, num_bins=1
    )
    # Check that the two errors are close (may not be equal due to shifts in prediction distribution)
    assert np.isclose(original_error, unjoined_error, atol=1e-1)


@pytest.mark.parametrize(
    "labels, predicted_scores, sample_weight, expected",
    [
        # Test unweighted
        (np.array([0, 1, 0, 1, 1]), np.array([0.1, 0.9, 0.2, 0.8, 0.7]), None, 0.9),
        # Test with all weights 1
        (
            np.array([0, 1, 0, 1, 1]),
            np.array([0.1, 0.9, 0.2, 0.8, 0.7]),
            np.array([1, 1, 1, 1, 1]),
            0.9,
        ),
        # Test with weights
        (
            np.array([0, 1, 0, 1, 1]),
            np.array([0.1, 0.9, 0.2, 0.8, 0.7]),
            np.array([1, 1, 1, 1, 10]),
            0.75,
        ),
    ],
)
def test_calibration_ratio_gives_correct_results(
    labels, predicted_scores, sample_weight, expected
):
    result = metrics.calibration_ratio(
        labels, predicted_scores, sample_weight, adjust_unjoined=False
    )
    assert np.isclose(result, expected, atol=1e-6)


@pytest.mark.parametrize(
    "labels, predicted_scores, sample_weight, expected",
    [
        # Test unweighted
        (np.array([0, 1, 0, 1, 1]), np.array([0.1, 0.9, 0.2, 0.8, 0.7]), None, 0.9),
        # Test with all weights 1
        (
            np.array([0, 1, 0, 1, 1]),
            np.array([0.1, 0.9, 0.2, 0.8, 0.7]),
            np.array([1, 1, 1, 1, 1]),
            0.9,
        ),
        # Test with weights
        (
            np.array([0, 1, 0, 1, 1]),
            np.array([0.1, 0.9, 0.2, 0.8, 0.7]),
            np.array([1, 1, 1, 1, 10]),
            0.75,
        ),
    ],
)
def test_calibration_ratio__with_unjoined_adjustment_gives_correct_results(
    labels, predicted_scores, sample_weight, expected
):
    predicted_scores_unjoined, labels_unjoined = utils.make_unjoined(
        predicted_scores, labels
    )

    weights_unjoined = None
    if sample_weight is not None:
        weights_unjoined, _ = utils.make_unjoined(sample_weight, labels)

    result_unjoined = metrics.calibration_ratio(
        labels_unjoined,
        predicted_scores_unjoined,
        weights_unjoined,
        adjust_unjoined=True,
    )
    assert np.isclose(expected, result_unjoined, atol=1e-6)


@pytest.mark.parametrize(
    "predicted_scores, expected",
    [
        (np.array([0.5, 0.5]), 0.125),
        (np.array([1.0, 1.0]), 0),
        (np.array([0.0, 0.0]), 0),
    ],
)
def test_that_kuiper_total_variance_is_correct(predicted_scores, expected):
    assert kuiper_standard_deviation(predicted_scores) == np.sqrt(expected)


def test_that_kuiper_test_detects_miscalibration(rng):
    # Test that the Kuiper test detects miscalibration
    n = 100
    predictions = rng.uniform(low=0.0, high=1.0, size=n)
    miscalibrated_predictions = (predictions + 0.9) / 1.9
    labels = scipy.stats.binom.rvs(1, predictions, size=n, random_state=rng)

    calibrated_stat, calibrated_p_value = kuiper_test(
        labels=labels, predicted_scores=predictions
    )
    miscalibrated_stat, miscalibrated_p_value = kuiper_test(
        labels=labels, predicted_scores=miscalibrated_predictions
    )

    assert miscalibrated_p_value < 0.1


def test_RCE_is_scale_invariant(rng):
    y_true, y_score_1 = rng.random_sample((2, 100))
    y_score_2 = y_score_1 * 2
    RCE_1 = metrics.rank_calibration_error(y_true, y_score_1)
    RCE_2 = metrics.rank_calibration_error(y_true, y_score_2)
    assert np.allclose(RCE_1, RCE_2)


def test_RCE_is_zero_when_ranking_is_preserved(rng):
    y_true = rng.random_sample(100)
    y_score = y_true * 4.0
    RCE = metrics.rank_calibration_error(y_true, y_score)
    assert np.allclose(RCE, 0.0)


def test_multi_RCE_is_equal_for_groups_with_similar_ranking_quality(rng):
    y_true = rng.random_sample(100)
    df = pd.DataFrame({"group": ["A"] * 50 + ["B"] * 50})
    df["y_score"] = np.zeros(100)
    df["y_true"] = y_true
    df.loc[df.group == "A", "y_score"] = df.loc[df.group == "A", "y_true"] * 2.0
    df.loc[df.group == "B", "y_score"] = df.loc[df.group == "B", "y_true"] * 4.0
    RCEs = metrics._rank_multicalibration_error(
        labels=df["y_true"], predicted_labels=df["y_score"], segments_df=df[["group"]]
    )

    assert np.allclose(RCEs["A"][0], RCEs["B"][0])


def test_multi_RCE_is_more_for_groups_with_worse_ranking_quality(rng):
    y_true = rng.random_sample(100)
    df = pd.DataFrame({"group": ["A"] * 50 + ["B"] * 50})
    df["y_score"] = np.zeros(100)
    df["y_true"] = y_true
    df.loc[df.group == "A", "y_score"] = df.loc[df.group == "A", "y_true"] * 2.0
    df.loc[df.group == "B", "y_score"] = df.loc[df.group == "B", "y_true"] * -2.0
    RCEs = metrics._rank_multicalibration_error(
        labels=df["y_true"], predicted_labels=df["y_score"], segments_df=df[["group"]]
    )

    assert RCEs["A"][0] < RCEs["B"][0]


def test_normalized_entropy_gives_expected_result():
    y_pred = np.array([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
    y_true = np.array([0, 0, 1, 0, 1, 0, 1])
    result = metrics.normalized_entropy(y_true, y_pred)
    expected = 1.0218659880033287
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "y_pred,y_true,sample_weight,expected",
    [
        (
            np.array([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2]),
            np.array([0, 0, 1, 0, 1, 0, 1]),
            np.array([2, 2, 2, 2, 2, 2, 2]),
            1.0218659880033287,
        ),
        (
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            np.array([0, 0, 0, 0, 1, 0, 1]),
            np.array([5, 4, 2, 2, 2, 2, 2]),
            0.7247193682397917,
        ),
    ],
)
def test_normalized_entropy_with_sample_weights_gives_expected_result(
    y_pred, y_true, sample_weight, expected
):
    result = metrics.normalized_entropy(y_true, y_pred, sample_weight=sample_weight)
    assert result == pytest.approx(expected)


def test_califree_ne_gives_same_result_as_ne_when_calibration_ratio_is_one():
    # Hardcoded labels and predictions with equal sums
    labels = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    predictions = np.array([0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
    # Calculate normalized entropy and calibration-free normalized entropy
    ne = metrics.normalized_entropy(labels, predictions)
    califree_ne = metrics.calibration_free_normalized_entropy(labels, predictions)
    assert ne == califree_ne


@pytest.mark.parametrize("logit_shift", [(1), (0.1), (1.1), (10)])
def test_califree_ne_is_invariant_to_logit_shifts(logit_shift):
    # Hardcoded labels and predictions with equal sums
    labels = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    predictions = np.array([0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
    shifted_predictions = utils.logistic_vectorized(
        utils.logit(predictions) + logit_shift
    )
    # Calculate normalized entropy and calibration-free normalized entropy
    califree_ne_original = metrics.calibration_free_normalized_entropy(
        labels, predictions
    )
    califree_ne_shifted = metrics.calibration_free_normalized_entropy(
        labels, shifted_predictions
    )
    assert pytest.approx(califree_ne_shifted, 0.01) == califree_ne_original


@pytest.mark.parametrize("target_precision, expected_fpr", [(0.9, 0.0), (0.8, 0.2)])
def test_fpr_at_precision(target_precision, expected_fpr):
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.55])
    # At target precision 0.9, the threshold is 0.55 (precision is 1.000, but for next threshold 0.50 it is 0.833), fpr now is 0/5 = 0.0
    # At target precision 0.8, the threshold is 0.50 (precision is 0.833, but for next threshold 0.40 it is 0.714), fpr now is 1/5 = 0.2
    calculated_fpr = metrics.fpr_at_precision(
        y_true, y_scores, precision_target=target_precision
    )
    # Check if the calculated FPR is as expected based on calculation by hand above
    assert pytest.approx(calculated_fpr, 0.01) == expected_fpr


def test_fpr_at_precision_gives_nan_when_no_negatives():
    y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.55])
    calculated_fpr = metrics.fpr_at_precision(y_true, y_scores, precision_target=0.9)
    # y_true has no negatives, so FPR is undefined
    assert np.isnan(calculated_fpr)


def test_fpr_at_precision_gives_nan_when_target_precision_is_unreachable():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.55])
    calculated_fpr = metrics.fpr_at_precision(y_true, y_scores, precision_target=0.9)
    # Top 3 indices in y_scores are all y_true=0, so target precision 0.9 is unreachable and fpr is undefined
    assert np.isnan(calculated_fpr)


@pytest.mark.parametrize(
    "y_true, y_scores, predictive_prevalence_target, expected_precision",
    [
        # All True Positives
        (np.array([1, 1, 1, 1]), np.array([0.9, 0.8, 0.7, 0.6]), 0.95, 1.0),
        # All False Positives
        (np.array([0, 0, 0, 0]), np.array([0.9, 0.8, 0.7, 0.6]), 0.95, 0.0),
        # Mixed Predictions
        (
            np.array([1, 0, 1, 0, 1, 0]),
            np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
            0.5,
            2 / 3,
        ),
        # No Predictions Meet Target
        (
            np.array([1, 0, 1, 0, 1, 0]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            0.95,
            3 / 6,
        ),
    ],
)
def test_precision_at_predictive_prevalence(
    y_true, y_scores, predictive_prevalence_target, expected_precision
):
    assert (
        metrics.precision_at_predictive_prevalence(
            y_true, y_scores, predictive_prevalence_target
        )
        == expected_precision
    )


@pytest.mark.parametrize(
    "y_true, y_scores, recall_target, sample_weight, expected",
    [
        # Basic test cases
        (
            np.array([0, 1, 1, 0, 1, 0, 1]),
            np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.6]),
            0.6,
            None,
            0.8,
        ),
        (np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]), 1.0, None, 2 / 3),
        (np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]), 0.5, None, 1.0),
        # All ones in y_true
        (np.array([1, 1, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4]), 0.5, None, 1.0),
        # Recall target greater than possible recall
        (np.array([0, 1, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]), 1.1, None, 0),
        # Recall target of zero
        (np.array([0, 1, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]), 0.0, None, 1.0),
        # Identical scores
        (np.array([0, 1, 1, 0]), np.array([0.5, 0.5, 0.5, 0.5]), 0.5, None, 0.5),
        # Single element arrays
        (np.array([1]), np.array([0.8]), 0.5, None, 1.0),
        # Negative scores
        (np.array([0, 1, 1, 0]), np.array([-0.1, -0.4, -0.35, -0.8]), 0.5, None, 2 / 3),
        # Test with sample weight
        (
            np.array([0, 0, 1, 1]),
            np.array([0.1, 0.4, 0.35, 0.8]),
            0.5,
            np.array([1, 1, 2, 2]),
            1.0,
        ),
    ],
)
def test_precision_at_recall(y_true, y_scores, recall_target, sample_weight, expected):
    result = metrics.precision_at_recall(y_true, y_scores, recall_target, sample_weight)
    assert result == pytest.approx(expected)


def test_that_multicalibrationerror_returns_correct_value_on_perfectly_calibrated_data():
    test_df = pd.DataFrame(
        {
            "prediction": [0.0, 1.0, 0.0, 1.0],
            "label": [0, 1, 0, 1],
            "segment_A": ["a", "a", "b", "b"],
        }
    )

    mce = metrics.MulticalibrationError(
        df=test_df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=["segment_A"],
        min_samples_per_segment=1,
    )
    assert mce.mce == 0


def test_that_multicalibrationerror_is_equal_to_ecce_metric_on_single_segment():
    test_df = pd.DataFrame(
        {
            "prediction": [0.1, 0.8, 0.3, 0.7],
            "label": [0, 1, 0, 1],
            "segment_A": ["a", "a", "b", "b"],
        }
    )

    global_kuiper_metric = metrics.kuiper_calibration(
        labels=test_df.label.values,
        predicted_scores=test_df.prediction.values,
        normalization_method=None,
    )

    mce = metrics.MulticalibrationError(
        df=test_df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=["segment_A"],
        max_depth=0,
        min_samples_per_segment=1,
        sigma_estimation_method=None,
    )
    assert np.isclose(mce.mce_absolute, global_kuiper_metric, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "metric_func",
    [
        skmetrics.average_precision_score,
        skmetrics.roc_auc_score,
        skmetrics.log_loss,
        skmetrics.mean_squared_error,
        metrics.normalized_entropy,
        metrics.calibration_ratio,
        metrics.adaptive_calibration_error,
        metrics.expected_calibration_error,
        metrics.kuiper_calibration,
    ],
)
def test_wrap_sklearn_metric_func_does_not_raise_an_error_with_any_of_our_main_metrics(
    metric_func,
):
    wrapped_func = wrap_sklearn_metric_func(metric_func)

    df = pd.DataFrame(
        {
            "label": [0, 1, 0, 1],
            "score": [0.2, 0.8, 0.4, 0.6],
            "sample_weight": [1, 1, 1, 1],
        }
    )
    wrapped_func(
        df=df, label_column="label", score_column="score", weight_column="sample_weight"
    )


def test_mce_wrapper_with_variant_mce_sigma_scale_has_the_right_name():
    # This test is required to throw warnings when mce_sigma_scale is large after training mcgrad
    # See the function determine_best_num_rounds in multicalibration.methods.py
    assert (
        wrap_multicalibration_error_metric(
            categorical_segment_columns="",
            numerical_segment_columns="",
            metric_version="mce_sigma_scale",
        ).name
        == "Multicalibration Error<br>(mce_sigma_scale)"
    ), (
        "The name of the mce_sigma_scale variant should be Multicalibration Error<br>(mce_sigma_scale)."
    )


def test_mce_can_deal_with_infrequent_values_in_int_categorical_columns(rng):
    # Check if the MCE can deal with infrequent values in int categorical columns,
    # i.e. a column passed as categorical but containing integers

    df = pd.DataFrame(
        {
            "ft": rng.randint(0, 3, 100),
            "prediction": rng.rand(100),
            "label": rng.randint(0, 2, 100),
        }
    )

    mce = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=["ft"],
        max_depth=1,
        min_samples_per_segment=1,
        max_values_per_segment_feature=2,
    )
    try:
        mce.mce
    except TypeError:
        raise AssertionError(
            "MCE cannot properly deal with infrequent values in int categorical columns."
        )


def test_that_mce_returns_correct_prevalence_with_and_without_weights():
    df = pd.DataFrame(
        {
            "prediction": [0.0, 1.0, 0.0, 1.0],
            "label": [0, 1, 0, 1],
            "weight": [1, 1, 7, 1],
            "segment_A": ["a", "a", "b", "b"],
        }
    )

    weighted_mce = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        weight_column="weight",
        categorical_segment_columns=["segment_A"],
        min_samples_per_segment=1,
    )
    assert weighted_mce.prevalence == 0.2

    unweighted_mce = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        weight_column=None,
        categorical_segment_columns=["segment_A"],
        min_samples_per_segment=1,
    )
    assert unweighted_mce.prevalence == 0.5


def test_mce_speedup_returns_values_equal_for_different_chunk_sizes(rng):
    # Changing chunk size should return the same ecce absolutes and sigmas
    n_cat_fts = 3
    n_num_fts = 3
    n_samples = 100

    df = pd.DataFrame(
        {
            **{
                f"segment_A_{t}": rng.randint(0, 10, n_samples)
                for t in range(n_cat_fts)
            },
            **{f"segment_B_{t}": rng.rand(n_samples) for t in range(n_num_fts)},
            "prediction": rng.rand(n_samples),
            "weights": rng.rand(n_samples),
            "label": rng.randint(0, 2, n_samples),
        }
    )

    df = df.astype(
        {
            "prediction": "float32",
            "label": "int32",
            **{f"segment_A_{t}": "int32" for t in range(n_cat_fts)},
            **{f"segment_B_{t}": "float32" for t in range(n_num_fts)},
            "weights": "float32",
        }
    )

    categorical_segment_columns = [f"segment_A_{t}" for t in range(n_cat_fts)]
    numerical_segment_columns = [f"segment_B_{t}" for t in range(n_num_fts)]

    mce_chunk25 = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
        weight_column="weights",
        max_n_segments=None,
        chunk_size=25,
    )

    mce_chunk7 = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
        weight_column="weights",
        max_n_segments=None,
        chunk_size=7,
    )

    assert np.equal(
        mce_chunk25.segment_ecces_absolute, mce_chunk7.segment_ecces_absolute
    ).all()
    assert np.equal(mce_chunk25.segment_sigmas, mce_chunk7.segment_sigmas).all()


def test_mce_sorting_does_not_modify_original_df(rng):
    # Check that the original df remains unchanged after being passed into the MCE metric and locally sorted
    n_cat_fts = 3
    n_num_fts = 3
    n_samples = 100

    df = pd.DataFrame(
        {
            **{
                f"segment_A_{t}": rng.randint(0, 3, n_samples) for t in range(n_cat_fts)
            },
            **{f"segment_B_{t}": rng.rand(n_samples) for t in range(n_num_fts)},
            "prediction": rng.rand(n_samples),
            "weights": rng.rand(n_samples),
            "label": rng.randint(0, 2, n_samples),
        }
    )

    df = df.astype(
        {
            "prediction": "float32",
            "label": "int32",
            **{f"segment_A_{t}": "int32" for t in range(n_cat_fts)},
            **{f"segment_B_{t}": "float32" for t in range(n_num_fts)},
            "weights": "float32",
        }
    )

    categorical_segment_columns = [f"segment_A_{t}" for t in range(n_cat_fts)]
    numerical_segment_columns = [f"segment_B_{t}" for t in range(n_num_fts)]
    dfthatgoesintoMCE = df.copy(deep=True)

    metrics.MulticalibrationError(
        df=dfthatgoesintoMCE,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
        weight_column="weights",
        chunk_size=50,
    )

    assert dfthatgoesintoMCE.equals(df)


def test_mce_reducing_precision_dtype_returns_correct_value_upto_third_digit(rng):
    n_cat_fts = 3
    n_num_fts = 3
    n_samples = 1000

    FLOAT16_TOLERANCE = 2e-03
    FLOAT32_TOLERANCE = 2e-07

    df = pd.DataFrame(
        {
            **{
                f"segment_A_{t}": rng.randint(0, 3, n_samples) for t in range(n_cat_fts)
            },
            **{f"segment_B_{t}": rng.rand(n_samples) for t in range(n_num_fts)},
            "prediction": rng.rand(n_samples),
            "weights": rng.rand(n_samples),
            "label": rng.randint(0, 2, n_samples),
        }
    )

    df = df.astype(
        {
            "prediction": "float64",
            "label": "int64",
            **{f"segment_A_{t}": "int64" for t in range(n_cat_fts)},
            **{f"segment_B_{t}": "float64" for t in range(n_num_fts)},
            "weights": "float64",
        }
    )

    categorical_segment_columns = [f"segment_A_{t}" for t in range(n_cat_fts)]
    numerical_segment_columns = [f"segment_B_{t}" for t in range(n_num_fts)]

    mce_float16 = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
        weight_column="weights",
        max_n_segments=1000,
        max_values_per_segment_feature=3,
        min_samples_per_segment=1,
        chunk_size=100,
        precision_dtype="float16",
    )

    mce_float32 = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
        weight_column="weights",
        max_n_segments=1000,
        max_values_per_segment_feature=3,
        min_samples_per_segment=1,
        chunk_size=100,
        precision_dtype="float32",
    )

    mce_float64 = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
        weight_column="weights",
        max_n_segments=1000,
        max_values_per_segment_feature=3,
        min_samples_per_segment=1,
        chunk_size=100,
        precision_dtype="float64",
    )
    attrs_to_check = [
        "segment_ecces_absolute",
        "segment_ecces_sigma_scale",
        "segment_sigmas",
        "sigma_0",
        "mce",
        "mce_sigma_scale",
        "mde",
    ]
    for attr_name in attrs_to_check:
        # float16 should be within 2e-3 of float64
        assert np.isclose(
            getattr(mce_float16, attr_name),
            getattr(mce_float64, attr_name),
            rtol=FLOAT16_TOLERANCE,
        ).all()
        # float32 should be within 2e-7 of float64
        assert np.isclose(
            getattr(mce_float32, attr_name),
            getattr(mce_float64, attr_name),
            rtol=FLOAT32_TOLERANCE,
        ).all()


def test_equivalence_kuiper_label_based_standard_deviation_per_segment():
    predicted_scores = np.random.rand(100)
    labels = np.random.rand(100)
    sample_weight = np.random.rand(100)

    seg_idx_list = [
        [1, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        [2, 6, 11, 16, 21, 26, 31, 36, 41, 46],
        [3, 7, 12, 17, 22, 27, 32, 37, 42, 47],
        [4, 8, 13, 18, 23, 28, 33, 38, 43, 48],
        [5, 9, 14, 19, 24, 29, 34, 39, 44, 49],
    ]

    segments = np.zeros(shape=(5, 100), dtype=np.bool_)
    std_dev_original = np.zeros(shape=(5,), dtype=np.float64)
    for i, seg_idx in enumerate(seg_idx_list):
        std_dev_original[i] = metrics.kuiper_label_based_standard_deviation(
            predicted_scores[seg_idx], labels[seg_idx], sample_weight[seg_idx]
        )
        segments[i, seg_idx] = 1

    std_dev_segment_based = metrics.kuiper_label_based_standard_deviation_per_segment(
        predicted_scores, labels, sample_weight, segments
    ).reshape(-1)
    assert np.allclose(std_dev_original, std_dev_segment_based)


def test_segment_feature_values_has_the_correct_features_used_for_segment_generation(
    rng,
):
    """
    Test that the segment_feature_values dataframe returned by MulticalibrationError.segments
    has the expected structure and values when using one categorical and one numerical feature.
    """
    n_samples = 100
    df = pd.DataFrame(
        {
            "categorical_feature": rng.choice(["A", "B"], size=n_samples),
            "numerical_feature": rng.randint(1, 3, size=n_samples),
            "label": rng.randint(0, 2, size=n_samples),
            "prediction": rng.random(size=n_samples),
        }
    )

    mce = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=["categorical_feature"],
        numerical_segment_columns=["numerical_feature"],
        max_depth=2,
        max_values_per_segment_feature=3,
        min_samples_per_segment=1,
    )

    _, segment_feature_values = mce.segments

    # Verify the structure of the segment_feature_values dataframe
    assert isinstance(segment_feature_values, pd.DataFrame), (
        "segment_feature_values should be a pandas DataFrame"
    )
    assert set(segment_feature_values.columns) == {
        "segment_column",
        "value",
        "idx_segment",
    }, "segment_feature_values should have columns: segment_column, value, idx_segment"

    # Check that we have the expected number of segments
    # For depth=0: 1 segment (whole dataset). However this corresponds to "empty dataframe", i.e. idx 0 should not be in the df.
    # For depth=1: 2 segments for categorical feature + 2 segments for numerical feature = 4 segments
    # For depth=2: 2 (categorical) * 2 (numerical) = 4 segments
    # Total: 4 + 4 = 8 segments
    unique_segment_indices = segment_feature_values["idx_segment"].unique()
    assert len(unique_segment_indices) == 8, (
        f"Expected 8 segments, got {len(unique_segment_indices)}"
    )

    # Check that the segment_feature_values dataframe contains the expected values
    # For categorical feature, we should have values "A" and "B"
    categorical_values = segment_feature_values[
        segment_feature_values["segment_column"] == "categorical_feature"
    ]["value"].unique()
    assert set(categorical_values) == {
        "A",
        "B",
    }, f"Expected categorical values {{'A', 'B'}}, got {set(categorical_values)}"

    # For numerical feature, we should have exactly 2 unique values after collapsing
    numerical_values = segment_feature_values[
        segment_feature_values["segment_column"] == "numerical_feature"
    ]["value"].unique()
    assert len(numerical_values) == 2, (
        f"Expected 2 unique numerical values, got {len(numerical_values)}"
    )

    # Test that idx 0 results in the empty dataframe
    assert (
        len(segment_feature_values[segment_feature_values["idx_segment"] == 0]) == 0
    ), (
        f"Expected an empty dataframe for index 0, got {segment_feature_values[segment_feature_values['idx_segment'] == 0]}"
    )

    # Get the actual numerical values for use in the next check
    numerical_values_list = sorted(numerical_values)

    # Check depth=2 combinations (categorical * numerical)
    # We should have all combinations of categorical and numerical values
    depth_2_segments = []
    for cat_val in ["A", "B"]:
        for num_val in numerical_values_list:
            # Find segments that contain both this categorical and numerical value
            cat_segments = segment_feature_values[
                (segment_feature_values["segment_column"] == "categorical_feature")
                & (segment_feature_values["value"] == cat_val)
            ]["idx_segment"].unique()

            num_segments = segment_feature_values[
                (segment_feature_values["segment_column"] == "numerical_feature")
                & (segment_feature_values["value"] == num_val)
            ]["idx_segment"].unique()

            # Find the intersection of these segment indices
            common_segments = set(cat_segments).intersection(set(num_segments))

            # There should be exactly one segment with this combination
            assert len(common_segments) == 1, (
                f"Expected 1 segment for combination {cat_val}/{num_val}, got {len(common_segments)}"
            )
            depth_2_segments.extend(common_segments)

    # Ensure we found all 4 depth=2 segments
    assert len(depth_2_segments) == 4, (
        f"Expected 4 depth = 2 segments, got {len(depth_2_segments)}"
    )


def test_precision_dtype_is_maintained_in_multicalibration_error_methods(rng):
    """
    Test that the precision_dtype parameter is maintained when running specific methods
    in the MulticalibrationError class.
    """
    n_cat_fts = 2
    n_num_fts = 2
    n_samples = 100

    df = pd.DataFrame(
        {
            **{
                f"segment_A_{t}": rng.randint(0, 3, n_samples) for t in range(n_cat_fts)
            },
            **{f"segment_B_{t}": rng.rand(n_samples) for t in range(n_num_fts)},
            "prediction": rng.rand(n_samples),
            "weights": rng.rand(n_samples),
            "label": rng.randint(0, 2, n_samples),
        }
    )

    categorical_segment_columns = [f"segment_A_{t}" for t in range(n_cat_fts)]
    numerical_segment_columns = [f"segment_B_{t}" for t in range(n_num_fts)]

    mce_float16 = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
        weight_column="weights",
        max_n_segments=10,
        max_values_per_segment_feature=3,
        min_samples_per_segment=1,
        chunk_size=5,
        precision_dtype="float16",
    )

    mce_float32 = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
        weight_column="weights",
        max_n_segments=10,
        max_values_per_segment_feature=3,
        min_samples_per_segment=1,
        chunk_size=5,
        precision_dtype="float32",
    )

    mce_float64 = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
        weight_column="weights",
        max_n_segments=10,
        max_values_per_segment_feature=3,
        min_samples_per_segment=1,
        chunk_size=5,
        precision_dtype="float64",
    )

    assert mce_float16.segment_ecces_sigma_scale.dtype == np.float16
    assert mce_float32.segment_ecces_sigma_scale.dtype == np.float32
    assert mce_float64.segment_ecces_sigma_scale.dtype == np.float64

    assert mce_float16.segment_sigmas.dtype == np.float16
    assert mce_float32.segment_sigmas.dtype == np.float32
    assert mce_float64.segment_sigmas.dtype == np.float64

    assert mce_float16.segment_ecces_absolute.dtype == np.float16
    assert mce_float32.segment_ecces_absolute.dtype == np.float32
    assert mce_float64.segment_ecces_absolute.dtype == np.float64

    assert mce_float16.mce_sigma_scale.dtype == np.float16
    assert mce_float32.mce_sigma_scale.dtype == np.float32
    assert mce_float64.mce_sigma_scale.dtype == np.float64


def test_precision_dtype_is_extended_for_large_weights(rng):
    n_cat_fts = 2
    n_num_fts = 2
    n_samples = 100

    df = pd.DataFrame(
        {
            **{
                f"segment_A_{t}": rng.randint(0, 3, n_samples) for t in range(n_cat_fts)
            },
            **{f"segment_B_{t}": rng.rand(n_samples) for t in range(n_num_fts)},
            "prediction": rng.rand(n_samples),
            "weights": 1e6 * rng.rand(n_samples),
            "label": rng.randint(0, 2, n_samples),
        }
    )

    categorical_segment_columns = [f"segment_A_{t}" for t in range(n_cat_fts)]
    numerical_segment_columns = [f"segment_B_{t}" for t in range(n_num_fts)]

    mce_float16 = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
        weight_column="weights",
        max_n_segments=10,
        max_values_per_segment_feature=3,
        min_samples_per_segment=1,
        chunk_size=5,
        precision_dtype="float16",
    )

    assert mce_float16.df["weights"].dtype == np.float64
    mce_float16.df["weights"] = mce_float16.df["weights"].replace(
        [np.inf, -np.inf], np.nan
    )
    assert len(mce_float16.df.dropna(subset=["weights"], how="all")) == len(df)


def test_ecce_and_standard_deviation_return_zero_for_empty_segment(rng):
    n_samples = 100

    df = pd.DataFrame(
        {
            "prediction": rng.rand(n_samples),
            "weights": np.ones(n_samples),
            "label": rng.randint(0, 2, n_samples),
        }
    )

    masks = np.array(
        [[False for i in range(n_samples)]],
        dtype=bool,
    )

    c_jk = metrics.kuiper_calibration_per_segment(
        labels=df["label"].values,
        predicted_scores=df["prediction"].values,
        sample_weight=df["weights"].values,
        segments=masks,
    )[0]

    k_ub = metrics.kuiper_upper_bound_standard_deviation_per_segment(
        labels=df["label"].values,
        predicted_scores=df["prediction"].values,
        sample_weight=df["weights"].values,
        segments=masks,
    )[0]

    k_std = metrics.kuiper_standard_deviation_per_segment(
        labels=df["label"].values,
        predicted_scores=df["prediction"].values,
        sample_weight=df["weights"].values,
        segments=masks,
    )[0]
    assert c_jk == k_ub == k_std, "Expected 0 for empty segment"


def test_predictions_to_labels_does_not_modify_input_dataframes():
    """Verify predictions_to_labels does not modify input DataFrames."""
    data = pd.DataFrame(
        {
            "prediction": [0.1, 0.2, 0.4, 0.8, 0.9],
            "segment_a": ["a", "a", "b", "b", "a"],
            "segment_b": ["i", "j", "i", "j", "i"],
        }
    )
    thresholds = pd.DataFrame(
        {
            "segment_a": ["a", "a", "b", "b"],
            "segment_b": ["i", "j", "i", "j"],
            "threshold": [0.8, 0.7, 0.6, 0.5],
        }
    )

    data_original = data.copy()
    thresholds_original = thresholds.copy()

    _ = metrics.predictions_to_labels(
        data=data, prediction_column="prediction", thresholds=thresholds
    )

    pd.testing.assert_frame_equal(data, data_original)
    pd.testing.assert_frame_equal(thresholds, thresholds_original)


def test_multicalibration_error_does_not_modify_segments_df():
    """Verify multicalibration_error does not modify input segments_df."""
    labels = np.linspace(0, 1, 100)
    predictions = 1 - np.linspace(0, 1, 100)
    segments_df = pd.DataFrame({"segment": np.array(["A"] * 50 + ["B"] * 50)})

    segments_df_original = segments_df.copy()

    _ = metrics.multicalibration_error(
        labels, predictions, segments_df, metric=metrics.expected_calibration_error
    )

    pd.testing.assert_frame_equal(segments_df, segments_df_original)


def test_multi_cg_score_does_not_modify_segments_df(rng):
    """Verify multi_cg_score does not modify input segments_df."""
    n_samples = 100
    df = pd.DataFrame(index=range(n_samples))
    df["label"] = rng.random_sample(n_samples)
    df["score"] = rng.random_sample(n_samples)
    df["segment_1"] = rng.choice(["A", "B"], size=len(df))
    df["segment_2"] = rng.choice(["C", "D"], size=len(df))

    segments_df = df[["segment_1", "segment_2"]].copy()
    segments_df_original = segments_df.copy()

    _ = metrics.multi_cg_score(
        labels=df["label"].values,
        predictions=df["score"].values,
        segments_df=segments_df,
        metric=metrics.ndcg_score,
        rank_discount=utils.rank_no_discount,
        k=10,
    )

    pd.testing.assert_frame_equal(segments_df, segments_df_original)


def test_rank_multicalibration_error_does_not_modify_segments_df(rng):
    """Verify rank_multicalibration_error does not modify input segments_df."""
    y_true = rng.random_sample(100)
    df = pd.DataFrame({"group": ["A"] * 50 + ["B"] * 50})
    df["y_score"] = np.zeros(100)
    df["y_true"] = y_true
    df.loc[df.group == "A", "y_score"] = df.loc[df.group == "A", "y_true"] * 2.0
    df.loc[df.group == "B", "y_score"] = df.loc[df.group == "B", "y_true"] * 4.0

    segments_df = df[["group"]].copy()
    segments_df_original = segments_df.copy()

    _ = metrics.rank_multicalibration_error(
        labels=df["y_true"].values,
        predicted_labels=df["y_score"].values,
        segments_df=segments_df,
        num_bins=40,
    )

    pd.testing.assert_frame_equal(segments_df, segments_df_original)


@pytest.mark.parametrize(
    "metric_func",
    [
        metrics.kuiper_pvalue_per_segment,
        metrics.kuiper_statistic_per_segment,
    ],
)
def test_kuiper_per_segment_does_not_modify_segments_df(rng, metric_func):
    n_samples = 100
    labels = rng.randint(0, 2, n_samples)
    predictions = rng.random_sample(n_samples)
    segments_df = pd.DataFrame({"segment": rng.choice(["A", "B"], size=n_samples)})

    segments_df_original = segments_df.copy()

    _ = metric_func(labels=labels, predictions=predictions, segments_df=segments_df)

    pd.testing.assert_frame_equal(segments_df, segments_df_original)


@pytest.mark.parametrize(
    "metric_func,metric_kwargs,use_sample_weight",
    [
        (metrics.expected_calibration_error, {}, True),
        (metrics.expected_calibration_error, {}, False),
        (metrics.calibration_ratio, {}, True),
        (metrics.calibration_ratio, {}, False),
        (
            metrics.kuiper_calibration,
            {"normalization_method": "kuiper_standard_deviation"},
            True,
        ),
        (
            metrics.kuiper_calibration,
            {"normalization_method": "kuiper_standard_deviation"},
            False,
        ),
        (metrics.normalized_entropy, {}, True),
        (metrics.normalized_entropy, {}, False),
    ],
)
def test_metric_does_not_modify_input_arrays(
    rng, metric_func, metric_kwargs, use_sample_weight
):
    labels = rng.randint(0, 2, 100).astype(np.float64)
    predicted_scores = rng.random_sample(100)
    sample_weight = rng.random_sample(100) if use_sample_weight else None

    labels_original = labels.copy()
    predicted_scores_original = predicted_scores.copy()
    sample_weight_original = sample_weight.copy() if sample_weight is not None else None

    _ = metric_func(
        labels=labels,
        predicted_scores=predicted_scores,
        sample_weight=sample_weight,
        **metric_kwargs,
    )

    np.testing.assert_array_equal(labels, labels_original)
    np.testing.assert_array_equal(predicted_scores, predicted_scores_original)
    if sample_weight is not None:
        np.testing.assert_array_equal(sample_weight, sample_weight_original)


@pytest.mark.parametrize(
    "metric_func,metric_kwargs",
    [
        (metrics.dcg_score, {"rank_discount": utils.rank_no_discount}),
        (metrics.ndcg_score, {"rank_discount": utils.rank_no_discount}),
        (metrics.rank_calibration_error, {}),
    ],
)
def test_ranking_metric_does_not_modify_input_arrays(rng, metric_func, metric_kwargs):
    labels = rng.random_sample(100)
    predicted_labels = rng.random_sample(100)

    labels_original = labels.copy()
    predicted_labels_original = predicted_labels.copy()

    _ = metric_func(labels=labels, predicted_labels=predicted_labels, **metric_kwargs)

    np.testing.assert_array_equal(labels, labels_original)
    np.testing.assert_array_equal(predicted_labels, predicted_labels_original)


@pytest.mark.parametrize(
    "predictions,labels,sample_weight",
    [
        # Minimum viable case (2 points)
        (np.array([0.1, 0.9]), np.array([0, 1]), None),
        # Edge case: all same predictions
        (np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1]), None),
        # Edge case: boundary values (0 and 1)
        (np.array([0.0, 1.0, 0.5]), np.array([0, 1, 0]), None),
        # With sample weights
        (np.array([0.1, 0.5, 0.9]), np.array([0, 1, 1]), np.array([1.0, 2.0, 1.0])),
        # Edge case: very small differences
        (np.array([0.500, 0.501, 0.502]), np.array([0, 1, 0]), None),
    ],
)
def test_kuiper_label_based_standard_deviation_edge_cases(
    predictions, labels, sample_weight
):
    result = metrics.kuiper_label_based_standard_deviation(
        predictions, labels, sample_weight
    )
    assert result >= 0
    assert not np.isnan(result)


def test_kuiper_label_based_standard_deviation_perfect_calibration():
    # Perfect calibration should give near-zero deviation
    predictions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    labels = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    result = metrics.kuiper_label_based_standard_deviation(predictions, labels)
    assert result == pytest.approx(0, abs=1e-10)


def test_kuiper_label_based_standard_deviation_requires_labels():
    with pytest.raises(ValueError, match="Labels are required"):
        metrics.kuiper_label_based_standard_deviation(
            np.array([0.1, 0.5, 0.9]), labels=None
        )


def test_calibration_free_normalized_entropy_higher_for_reversed_predictions():
    labels = np.array([0, 0, 0, 1, 1, 1])
    well_calibrated = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    poorly_calibrated = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

    result_good = metrics.calibration_free_normalized_entropy(
        labels=labels, predicted_scores=well_calibrated
    )
    result_bad = metrics.calibration_free_normalized_entropy(
        labels=labels, predicted_scores=poorly_calibrated
    )

    assert result_bad > result_good


def test_rank_calibration_error_zero_for_perfect_ranking():
    labels = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    perfect_predictions = labels * 2.0

    # Mean of empty slice warning may be emitted by numpy when computing bin statistics
    # (depends on whether np.errstate() is active in the implementation)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result = metrics._rank_calibration_error(
            labels=labels, predicted_labels=perfect_predictions
        )

    assert isinstance(result[0], (float, np.floating))
    assert result[0] == pytest.approx(0, abs=1e-10)


def test_rank_calibration_error_higher_for_reversed_ranking():
    labels = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    reversed_predictions = 1.0 - labels

    # Mean of empty slice warning may be emitted by numpy when computing bin statistics
    # (depends on whether np.errstate() is active in the implementation)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result_reversed = metrics._rank_calibration_error(
            labels=labels, predicted_labels=reversed_predictions
        )
        result_perfect = metrics._rank_calibration_error(
            labels=labels, predicted_labels=labels
        )

    assert result_reversed[0] > result_perfect[0]


def test_proportional_adaptive_calibration_error_gives_expected_result():
    labels = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    predicted_scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.85, 0.15, 0.25])

    result = metrics.proportional_adaptive_calibration_error(
        labels=labels,
        predicted_scores=predicted_scores,
        num_bins=5,
    )

    assert isinstance(result, (float, np.floating))
    assert result >= 0


def test_recall_gives_expected_result():
    labels = np.array([0, 1, 0, 1, 1, 0, 1, 1])
    predicted_labels = np.array([0, 1, 0, 1, 0, 0, 1, 1])

    result = metrics.recall(labels=labels, predicted_labels=predicted_labels)

    # Expected: 4 true positives out of 5 actual positives = 0.8
    assert result == pytest.approx(0.8)


def test_recall_with_sample_weight():
    labels = np.array([0, 1, 0, 1, 1])
    predicted_labels = np.array([0, 1, 0, 0, 1])
    sample_weight = np.array([1, 2, 1, 1, 1])

    result = metrics.recall(
        labels=labels, predicted_labels=predicted_labels, sample_weight=sample_weight
    )

    # With weights: 2 TP (weight=2+1=3) out of 3 actual positives (weight=2+1+1=4) = 0.75
    assert result == pytest.approx(0.75)


def test_precision_gives_expected_result():
    labels = np.array([0, 1, 0, 1, 1, 0, 1, 1])
    predicted_labels = np.array([0, 1, 1, 1, 0, 0, 1, 1])

    result = metrics.precision(labels=labels, predicted_labels=predicted_labels)

    # Expected: 4 true positives out of 5 predicted positives = 0.8
    assert result == pytest.approx(0.8)


def test_precision_with_sample_weight():
    labels = np.array([0, 1, 0, 1, 1])
    predicted_labels = np.array([0, 1, 1, 0, 1])
    precision_weight = np.array([1, 2, 1, 1, 1])

    result = metrics.precision(
        labels=labels,
        predicted_labels=predicted_labels,
        precision_weight=precision_weight,
    )

    # With weights: 2 TP (weight=2+1=3) out of 3 predicted positives (weight=2+1+1=4) = 0.75
    assert result == pytest.approx(0.75)


def test_fpr_edge_case_when_fp_plus_tn_equals_zero():
    """Test fpr edge case where fp + tn = 0 (all positives predicted as positive)"""
    labels = np.array([1, 1, 1, 1])
    predicted_labels = np.array([1, 1, 1, 1])

    # Single-class data may trigger sklearn warning about confusion matrix shape
    # (depends on sklearn version)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A single label was found")
        result = metrics.fpr(labels=labels, predicted_labels=predicted_labels)

    # When there are no negatives in labels, fpr should return 0.0
    assert result == 0.0


def test_fpr_with_mask_returns_none_when_denominator_zero():
    y_true = np.array([1, 1, 1, 1], dtype=bool)
    y_pred = np.array([1, 0, 1, 0], dtype=bool)
    y_mask = np.array([1, 1, 1, 1], dtype=bool)
    sample_weight = np.array([1.0, 1.0, 1.0, 1.0])

    result = metrics.fpr_with_mask(
        y_true=y_true,
        y_pred=y_pred,
        y_mask=y_mask,
        sample_weight=sample_weight,
        denominator=0.0,
    )

    assert result is None


def test_fpr_with_mask_calculates_fpr_correctly_with_nonzero_denominator():
    y_true = np.array([0, 1, 0, 1], dtype=bool)
    y_pred = np.array([1, 1, 1, 0], dtype=bool)
    y_mask = np.array([1, 1, 1, 1], dtype=bool)
    sample_weight = np.array([1.0, 1.0, 1.0, 1.0])
    denominator = 2.0

    result = metrics.fpr_with_mask(
        y_true=y_true,
        y_pred=y_pred,
        y_mask=y_mask,
        sample_weight=sample_weight,
        denominator=denominator,
    )

    # False positives: indices 0 and 2 where y_pred=1, y_true=0, y_mask=1
    # fp_sr_idx sum = 2.0, fpr = 2.0 / 2.0 = 1.0
    assert result == pytest.approx(1.0)


def test_dcg_sample_scores_raises_error_when_k_less_than_1():
    labels = np.array([1, 2, 3, 4, 5])
    predicted_labels = np.array([5, 4, 3, 2, 1])

    with pytest.raises(ValueError, match="k cannot be less than 1"):
        metrics._dcg_sample_scores(
            labels=labels,
            predicted_labels=predicted_labels,
            rank_discount=utils.rank_no_discount,
            k=0,
        )


def test_ndcg_score_raises_error_with_negative_labels():
    labels = np.array([1, 2, -1, 3, 4])
    predicted_labels = np.array([5, 4, 3, 2, 1])

    with pytest.raises(
        ValueError, match="NDCG should not be used with negative label values"
    ):
        metrics.ndcg_score(
            labels=labels,
            predicted_labels=predicted_labels,
            rank_discount=utils.rank_no_discount,
        )


def test_recall_at_precision_returns_zero_when_target_unreachable():
    y_true = np.array([0, 0, 1, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    # With mostly negatives scoring high, precision target of 0.95 is unreachable
    result = metrics.recall_at_precision(
        y_true=y_true, y_scores=y_scores, precision_target=0.95
    )

    assert result == 0


def test_recall_at_precision_with_sample_weight():
    y_true = np.array([0, 1, 1, 0, 1])
    y_scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    sample_weight = np.array([1, 2, 1, 1, 1])

    result = metrics.recall_at_precision(
        y_true=y_true,
        y_scores=y_scores,
        precision_target=0.8,
        sample_weight=sample_weight,
    )

    assert 0 <= result <= 1
    assert isinstance(result, (float, np.floating))


def test_calculate_cumulative_differences_raises_error_with_mismatched_segments():
    labels = np.array([0, 1, 1, 0, 1])
    predicted_scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    # Create segments with wrong shape (3 samples instead of 5)
    segments = np.ones(shape=(2, 3), dtype=np.bool_)

    with pytest.raises(
        ValueError, match="Segments must be the same length as labels/predictions"
    ):
        metrics._calculate_cumulative_differences(
            labels=labels,
            predicted_scores=predicted_scores,
            segments=segments,
        )


def test_kuiper_upper_bound_standard_deviation_per_segment_with_default_sample_weight():
    predicted_scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])

    result = metrics.kuiper_upper_bound_standard_deviation_per_segment(
        predicted_scores=predicted_scores,
        sample_weight=None,
    )

    assert isinstance(result, np.ndarray)
    assert len(result) == 1
    assert result[0] >= 0


def test_kuiper_upper_bound_standard_deviation_per_segment_with_default_segments():
    predicted_scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
    sample_weight = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    result = metrics.kuiper_upper_bound_standard_deviation_per_segment(
        predicted_scores=predicted_scores,
        sample_weight=sample_weight,
        segments=None,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert result[0] >= 0


def test_kuiper_upper_bound_standard_deviation_wrapper_gives_expected_result():
    predicted_scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])

    result = metrics.kuiper_upper_bound_standard_deviation(
        predicted_scores=predicted_scores,
    )

    assert isinstance(result, (float, np.floating))
    assert result >= 0


def test_kuiper_upper_bound_standard_deviation_with_sample_weight():
    predicted_scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
    sample_weight = np.array([2.0, 1.0, 1.0, 1.0, 1.0])

    result = metrics.kuiper_upper_bound_standard_deviation(
        predicted_scores=predicted_scores,
        sample_weight=sample_weight,
    )

    assert isinstance(result, (float, np.floating))
    assert result >= 0


def test_kuiper_standard_deviation_per_segment_raises_error_with_mismatched_segments():
    predicted_scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
    # Create segments with wrong shape (3 samples instead of 5)
    segments = np.ones(shape=(2, 3), dtype=np.bool_)

    with pytest.raises(
        ValueError, match="Segments must be the same length as labels/predictions"
    ):
        metrics.kuiper_standard_deviation_per_segment(
            predicted_scores=predicted_scores,
            segments=segments,
        )


def test_kuiper_label_based_standard_deviation_per_segment_with_default_sample_weight():
    predicted_scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
    labels = np.array([0, 1, 1, 0, 1])

    result = metrics.kuiper_label_based_standard_deviation_per_segment(
        predicted_scores=predicted_scores,
        labels=labels,
        sample_weight=None,
    )

    assert isinstance(result, np.ndarray)
    assert len(result) == 1
    assert result[0] >= 0


def test_kuiper_label_based_standard_deviation_per_segment_with_default_segments():
    predicted_scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
    labels = np.array([0, 1, 1, 0, 1])
    sample_weight = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    result = metrics.kuiper_label_based_standard_deviation_per_segment(
        predicted_scores=predicted_scores,
        labels=labels,
        sample_weight=sample_weight,
        segments=None,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert result[0] >= 0


def test_kuiper_label_based_standard_deviation_per_segment_raises_error_when_labels_none():
    predicted_scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])

    with pytest.raises(ValueError, match="Labels are required for this method"):
        metrics.kuiper_label_based_standard_deviation_per_segment(
            predicted_scores=predicted_scores,
            labels=None,
        )


def test_kuiper_distribution_returns_near_one_for_large_x():
    result = metrics.kuiper_distribution(8.3)

    assert result == pytest.approx(1.0)

    result_large = metrics.kuiper_distribution(100.0)
    assert result_large == pytest.approx(1.0)


def test_normalization_method_assignment_raises_error_for_unknown_method():
    with pytest.raises(
        ValueError, match="Unknown normalization method.*Available methods"
    ):
        metrics._normalization_method_assignment("invalid_method")


def test_kuiper_test_returns_pvalue_one_for_very_small_statistic():
    # Create perfectly calibrated predictions
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    predicted_scores = np.array([0.0, 0.0, 1.0, 1.0])

    statistic, pval = metrics.kuiper_test(
        labels=labels, predicted_scores=predicted_scores
    )

    # With perfect calibration, statistic should be very small and pval should be ~1.0
    assert pval == pytest.approx(1.0, abs=0.1)


def test_kuiper_test_returns_epsilon_pvalue_for_very_large_statistic():
    # Create extremely miscalibrated predictions to force large statistic
    labels = np.array([0] * 50 + [1] * 50)
    predicted_scores = np.array([0.99] * 50 + [0.01] * 50)

    statistic, pval = metrics.kuiper_test(
        labels=labels, predicted_scores=predicted_scores
    )

    # With extreme miscalibration, pval should be very small
    assert pval < 0.01
    assert statistic > 5  # Should be a large statistic


def test_kuiper_statistic_wrapper_returns_statistic_only():
    labels = np.array([0, 1, 1, 0, 1])
    predicted_scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7])

    statistic = metrics.kuiper_statistic(
        labels=labels, predicted_scores=predicted_scores
    )

    assert isinstance(statistic, (float, np.floating))
    assert statistic >= 0


def test_kuiper_pvalue_wrapper_returns_pvalue_only():
    labels = np.array([0, 1, 1, 0, 1])
    predicted_scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7])

    pvalue = metrics.kuiper_pvalue(labels=labels, predicted_scores=predicted_scores)

    assert isinstance(pvalue, (float, np.floating))
    assert 0 <= pvalue <= 1


def test_kuiper_pvalue_per_segment_calculates_pvalues_for_each_segment():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    predictions = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8])
    segments_df = pd.DataFrame({"segment": ["A", "A", "A", "A", "B", "B", "B", "B"]})

    result = metrics.kuiper_pvalue_per_segment(
        labels=labels, predictions=predictions, segments_df=segments_df
    )

    assert isinstance(result, pd.Series)
    assert len(result) == 2
    assert all(0 <= pval <= 1 for pval in result.values)


def test_kuiper_statistic_per_segment_calculates_statistics_for_each_segment():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    predictions = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8])
    segments_df = pd.DataFrame({"segment": ["A", "A", "A", "A", "B", "B", "B", "B"]})

    result = metrics.kuiper_statistic_per_segment(
        labels=labels, predictions=predictions, segments_df=segments_df
    )

    assert isinstance(result, pd.Series)
    assert len(result) == 2
    assert all(stat >= 0 for stat in result.values)


def test_multi_segment_pvalue_geometric_mean_calculates_geometric_mean():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    predictions = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8])
    segments_df = pd.DataFrame({"segment": ["A", "A", "A", "A", "B", "B", "B", "B"]})

    result = metrics.multi_segment_pvalue_geometric_mean(
        labels=labels, predictions=predictions, segments_df=segments_df
    )

    assert isinstance(result, (float, np.floating))
    assert 0 <= result <= 1


def test_multi_segment_inverse_sqrt_normalized_statistic_max_returns_max():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    predictions = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8])
    segments_df = pd.DataFrame({"segment": ["A", "A", "A", "A", "B", "B", "B", "B"]})

    result = metrics.multi_segment_inverse_sqrt_normalized_statistic_max(
        labels=labels, predictions=predictions, segments_df=segments_df
    )

    assert isinstance(result, (float, np.floating))
    assert result >= 0


def test_multi_segment_kuiper_test_with_poisson_combination():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    predictions = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8])
    segments_df = pd.DataFrame({"segment": ["A", "A", "A", "A", "B", "B", "B", "B"]})

    result = metrics.multi_segment_kuiper_test(
        labels=labels,
        predictions=predictions,
        segments_df=segments_df,
        combination_method="poisson",
    )

    assert isinstance(result, dict)
    assert "n_segments" in result
    assert "statistic" in result
    assert "p_value" in result
    assert "segment_p_values" in result

    assert result["n_segments"] == 2
    assert 0 <= result["p_value"] <= 1


def test_multi_segment_kuiper_test_with_fisher_combination():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    predictions = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8])
    segments_df = pd.DataFrame({"segment": ["A", "A", "A", "A", "B", "B", "B", "B"]})

    result = metrics.multi_segment_kuiper_test(
        labels=labels,
        predictions=predictions,
        segments_df=segments_df,
        combination_method="fisher",
    )

    assert isinstance(result, dict)
    assert "n_segments" in result
    assert "statistic" in result
    assert "p_value" in result
    assert result["n_segments"] == 2
    assert 0 <= result["p_value"] <= 1


def test_rank_multicalibration_error_calculates_weighted_average():
    labels = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0] * 2)
    predicted_labels = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0] * 2)
    segments_df = pd.DataFrame({"segment": ["A"] * 6 + ["B"] * 6})

    result = metrics.rank_multicalibration_error(
        labels=labels,
        predicted_labels=predicted_labels,
        segments_df=segments_df,
        num_bins=3,
    )

    assert isinstance(result, (float, np.floating))
    assert result >= 0


def test_multicalibration_error_raises_error_for_invalid_precision_dtype():
    df = pd.DataFrame(
        {
            "prediction": [0.1, 0.5, 0.9],
            "label": [0, 1, 1],
            "segment": ["A", "A", "A"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Invalid precision type.*Must be one of \\['float16', 'float32', 'float64'\\]",
    ):
        metrics.MulticalibrationError(
            df=df,
            label_column="label",
            score_column="prediction",
            categorical_segment_columns=["segment"],
            precision_dtype="float128",  # Invalid precision type
        )


def test_multicalibration_error_str_method_returns_formatted_string():
    df = pd.DataFrame(
        {
            "prediction": [0.1, 0.8, 0.2, 0.9],
            "label": [0, 1, 0, 1],
            "segment": ["a", "a", "b", "b"],
        }
    )

    mce = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=["segment"],
        min_samples_per_segment=1,
    )

    result = str(mce)

    # Should contain expected components: mce%, sigmas, p, mde
    assert "%" in result
    assert "sigmas=" in result
    assert "p=" in result
    assert "mde=" in result


def test_multicalibration_error_format_method_with_format_spec():
    df = pd.DataFrame(
        {
            "prediction": [0.1, 0.8, 0.2, 0.9],
            "label": [0, 1, 0, 1],
            "segment": ["a", "a", "b", "b"],
        }
    )

    mce = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=["segment"],
        min_samples_per_segment=1,
    )

    # Test with .2f format spec
    result = format(mce, ".2f")

    # Should contain formatted values with 2 decimal places
    assert "%" in result
    assert "sigmas=" in result
    assert "p=" in result
    assert "mde=" in result


def test_multicalibration_error_segment_indices_returns_series(rng):
    df = pd.DataFrame(
        {
            "prediction": rng.rand(20),
            "label": rng.randint(0, 2, 20),
            "segment_A": rng.choice(["X", "Y"], size=20),
        }
    )

    mce = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=["segment_A"],
        min_samples_per_segment=1,
        max_depth=1,
    )

    indices = mce.segment_indices

    assert isinstance(indices, pd.Series)
    assert len(indices) > 0


def test_multicalibration_error_global_ecce_returns_first_segment_ecce():
    """Test MulticalibrationError.global_ecce returns segment_ecces[0]"""
    df = pd.DataFrame(
        {
            "prediction": [0.1, 0.9, 0.2, 0.8, 0.4, 0.6],
            "label": [0, 1, 0, 1, 0, 1],
            "segment": ["a", "a", "b", "b", "c", "c"],
        }
    )

    mce = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=["segment"],
        min_samples_per_segment=1,
        max_depth=1,
    )

    # Access global_ecce property
    global_ecce = mce.global_ecce

    # Should equal segment_ecces[0]
    assert global_ecce == mce.segment_ecces[0]
    assert isinstance(global_ecce, (float, np.floating))


def test_multicalibration_error_global_ecce_sigma_scale_returns_first_segment():
    """Test MulticalibrationError.global_ecce_sigma_scale returns segment_ecces_sigma_scale[0]"""
    df = pd.DataFrame(
        {
            "prediction": [0.1, 0.9, 0.2, 0.8, 0.4, 0.6],
            "label": [0, 1, 0, 1, 0, 1],
            "segment": ["a", "a", "b", "b", "c", "c"],
        }
    )

    mce = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=["segment"],
        min_samples_per_segment=1,
        max_depth=1,
    )

    global_ecce_sigma_scale = mce.global_ecce_sigma_scale

    assert global_ecce_sigma_scale == mce.segment_ecces_sigma_scale[0]
    assert isinstance(global_ecce_sigma_scale, (float, np.floating))


def test_multicalibration_error_global_ecce_p_value_returns_first_segment():
    df = pd.DataFrame(
        {
            "prediction": [0.1, 0.9, 0.2, 0.8, 0.4, 0.6],
            "label": [0, 1, 0, 1, 0, 1],
            "segment": ["a", "a", "b", "b", "c", "c"],
        }
    )

    mce = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=["segment"],
        min_samples_per_segment=1,
        max_depth=1,
    )

    # Access global_ecce_p_value property
    global_ecce_p_value = mce.global_ecce_p_value

    # Should equal segment_p_values[0]
    assert global_ecce_p_value == mce.segment_p_values[0]
    assert 0 <= global_ecce_p_value <= 1


def test_multicalibration_error_sigma_0_fallback_when_segment_sigmas_not_computed():
    df = pd.DataFrame(
        {
            "prediction": [0.1, 0.5, 0.9, 0.2, 0.8],
            "label": [0, 1, 1, 0, 1],
            "segment": ["A", "A", "A", "A", "A"],
        }
    )

    mce = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=["segment"],
        min_samples_per_segment=1,
        max_depth=0,
    )

    # Access sigma_0 before segment_sigmas is computed
    # This should trigger the fallback path
    sigma_0 = mce.sigma_0

    # Should return a valid float value
    assert isinstance(sigma_0, (float, np.floating))
    assert sigma_0 >= 0


def test_multicalibration_error_p_value_fallback_when_segment_p_values_not_computed():
    df = pd.DataFrame(
        {
            "prediction": [0.1, 0.5, 0.9, 0.2, 0.8],
            "label": [0, 1, 1, 0, 1],
            "segment": ["A", "A", "A", "A", "A"],
        }
    )

    mce = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=["segment"],
        min_samples_per_segment=1,
        max_depth=0,
    )

    # Access p_value before segment_p_values is computed
    # This should trigger the fallback path using kuiper_distribution
    p_value = mce.p_value

    # Should return a valid p-value
    assert isinstance(p_value, (float, np.floating))
    assert 0 <= p_value <= 1
