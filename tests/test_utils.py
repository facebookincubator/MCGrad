# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe


import math

import numpy as np

import pandas as pd
import pyarrow as pa
import pytest

from multicalibration import utils
from pandas.core.arrays import ArrowExtensionArray


@pytest.fixture
def rng():
    return np.random.RandomState(42)


def test_make_equispaced_bins_gives_expected_result_when_data_between_zero_and_one_when_set_between_zero_one():
    data = np.zeros(5)
    result = utils.make_equispaced_bins(data, 2)
    expected = np.array([-1.0e-8, 0.5, 1.0 + 1.0e-8])
    assert np.allclose(result, expected, atol=1e-5)


def test_make_equispaced_bins_gives_expected_result_when_data_not_between_zero_and_one_when_set_between_zero_one():
    data = np.zeros(5) + 10
    result = utils.make_equispaced_bins(data, 2)
    expected = np.array([-1.0e-8, 0.5, 10.0 + 1.0e-8])
    assert np.allclose(result, expected, atol=1e-5)


def test_make_equispaced_bins_gives_expected_result():
    data = np.array([0.7, 1.4, 2.5, 6.2, 9.7, 2.1])
    bins = utils.make_equispaced_bins(data, 3, set_range_to_zero_one=False)

    assert np.allclose(
        bins, np.array([0.7 - 1.0e-8, 3.7, 6.7, 9.7 + 1.0e-8]), atol=1e-5
    )


def test_make_equispaced_bins_gives_similar_results_for_data_with_similar_range():
    data_1 = np.array([0.7, 100.7, 2, 2, 2, 2])
    data_2 = np.array([0.7, 100.7, 100, 100, 100, 100])

    bins_1 = utils.make_equispaced_bins(data_1, 2, set_range_to_zero_one=False)
    bins_2 = utils.make_equispaced_bins(data_2, 2, set_range_to_zero_one=False)

    assert np.allclose(
        bins_1, np.array([0.7 - 1.0e-8, 50.7, 100.7 + 1.0e-8]), atol=1e-5
    )
    assert np.allclose(bins_1, bins_2, atol=1e-5)


def test_make_equispaced_bins_gives_similar_results_for_data_with_similar_range_when_set_to_zero_one():
    data_1 = np.array([0.7, 0.9, 0.2, 0.2, 0.2, 0.2])
    data_2 = np.array([0.7, 0.9, 0.9, 0.9, 0.9, 0.9])

    bins_1 = utils.make_equispaced_bins(data_1, 2)
    bins_2 = utils.make_equispaced_bins(data_2, 2)

    assert np.allclose(bins_1, np.array([-1.0e-8, 0.5, 1.0 + 1.0e-8]), atol=1e-5)
    assert np.allclose(bins_1, bins_2, atol=1e-5)


@pytest.mark.parametrize(
    "labels,predictions,expected_result",
    [
        (
            np.array([False, True, False, True, True]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            2.8354,
        ),
        (np.array([0, 1, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 2.8354),
        (np.array([0, 1, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 2.8354),
    ],
)
def test_unshrink(labels: list[int], predictions: list[float], expected_result: float):
    # pyre-fixme
    assert pytest.approx(utils.unshrink(labels, predictions), 0.0001) == expected_result


@pytest.mark.parametrize(
    "log_odds, expected",
    [
        (0, 0.5),
        (1, 1 / (1 + math.exp(-1))),
        (-1, 1 / (1 + math.exp(1))),
        (100, 1.0),
        (-100, 3.720075976020836e-44),
        (1e20, 1.0),
        (-1e20, 0.0),
        (-710, 4.47e-309),
    ],
)
def test_logistic(log_odds, expected):
    result = utils.logistic(log_odds)
    assert math.isclose(result, expected, abs_tol=1e-310)


@pytest.mark.parametrize(
    "probs, expected",
    [
        (
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.log(
                np.array([0.1, 0.2, 0.3, 0.4, 0.5])
                / (1 - np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
            ),
        ),
        (
            np.array([0.6, 0.7, 0.8, 0.9]),
            np.log(
                np.array([0.6, 0.7, 0.8, 0.9]) / (1 - np.array([0.6, 0.7, 0.8, 0.9]))
            ),
        ),
    ],
)
def test_logit(probs, expected):
    result = utils.logit(probs)
    np.testing.assert_allclose(result, expected, rtol=1e-9)


@pytest.mark.parametrize(
    "probabilities", [(np.linspace(0.1, 0.9, num=10)), (np.linspace(0.1, 0.9, num=100))]
)
def test_logistic_is_inverse_function_of_logit(probabilities):
    vectorized_logistic = np.vectorize(utils.logistic)
    result = vectorized_logistic(utils.logit(probabilities))
    np.testing.assert_allclose(result, probabilities, rtol=1e-9)


@pytest.mark.parametrize(
    "log_odds,",
    [
        (np.array([-2, -1, 0, 1, 2])),
        (np.zeros(100)),
    ],
)
def test_logit_is_inverse_function_of_logistic(log_odds):
    vectorized_logistic = np.vectorize(utils.logistic)
    result = utils.logit(vectorized_logistic(log_odds))
    np.testing.assert_allclose(result, log_odds, rtol=1e-9)


def test_logits_and_probs_conversions_maintain_same_scale_with_clipping():
    probabilities = np.array([0, 1e-400, 1e-350, 1e-200, 1e-100, 0.1, 0.2, 0.5, 0.99])
    logits = utils.logit(probs=probabilities)

    recovered_probs = utils.logistic_vectorized(logits)

    assert np.all(recovered_probs > 0.0), "Recovered probabilities should be > 0"

    moderate_prob_mask = [False, False, False, False, False, True, True, True, True]
    if np.any(moderate_prob_mask):
        np.testing.assert_allclose(
            recovered_probs[moderate_prob_mask],
            probabilities[moderate_prob_mask],
            rtol=1e-15,
            err_msg="Moderate probabilities should be recovered accurately",
        )

    expected_min_prob = utils.logistic(utils.logit(probs=0))
    extreme_low_mask = probabilities < expected_min_prob

    if np.any(extreme_low_mask):
        np.testing.assert_allclose(
            recovered_probs[extreme_low_mask],
            expected_min_prob,
            atol=1e-25,
            rtol=1e-10,
            err_msg="Extremely low probabilities should be clipped to minimum bound",
        )


def test_OrdinalEncoderWithUnknownSupport_fit_transform_known_categories():
    encoder = utils.OrdinalEncoderWithUnknownSupport()
    df = pd.DataFrame(
        {
            "City": ["Paris", "Tokyo", "Amsterdam", "Paris", "Amsterdam"],
            "Gender": ["Male", "Female", "Male", "Female", "Male"],
        }
    )
    transformed = encoder.fit_transform(df.values)
    expected = np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_array_equal(transformed, expected)


def test_OrdinalEncoderWithUnknownSupport_transform_known_categories():
    encoder = utils.OrdinalEncoderWithUnknownSupport()
    df = pd.DataFrame(
        {
            "City": ["Paris", "Tokyo", "Amsterdam", "Paris", "Amsterdam"],
            "Gender": ["Male", "Female", "Male", "Female", "Male"],
        }
    )
    encoder.fit(df)
    transformed = encoder.transform(df.values)
    expected = np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_array_equal(transformed, expected)


def test_OrdinalEncoderWithUnknownSupport_transform_unknown_categories():
    encoder = utils.OrdinalEncoderWithUnknownSupport()
    df_a = pd.DataFrame(
        {
            "City": ["Paris", "Tokyo", "Amsterdam", "Paris", "Amsterdam"],
            "Gender": ["Male", "Female", "Male", "Female", "Male"],
        }
    )
    df_b = pd.DataFrame(
        {
            "City": ["Paris", "Copenhagen", "Tallinn", "Tokyo", "Amsterdam"],
            "Gender": ["Male", "Female", "Male", "Female", "Male"],
        }
    )
    encoder.fit(df_a.values)
    transformed = encoder.transform(df_b.values)
    expected = np.array([[1.0, 1.0], [-1.0, 0.0], [-1.0, 1.0], [2.0, 0.0], [0.0, 1.0]])
    np.testing.assert_array_equal(transformed, expected)


def test_encoder_serialize_deserialize():
    df = pd.DataFrame({"City": ["Paris", "Tokyo", "Amsterdam", "Paris", "Amsterdam"]})

    encoder = utils.OrdinalEncoderWithUnknownSupport()
    encoder.fit(df)

    serialized = encoder.serialize()
    deserialized = utils.OrdinalEncoderWithUnknownSupport.deserialize(serialized)
    assert deserialized._category_map == encoder._category_map

    df_test = pd.DataFrame({"City": ["Paris", "Copenhagen", "Tokyo"]})
    original_transformed = encoder.transform(df_test)
    deserialized_transformed = deserialized.transform(df_test)
    np.testing.assert_array_equal(deserialized_transformed, original_transformed)


def test_weighted_unshrink_gives_expected_result():
    # unshrink with duplicates should give same result as without duplicates but with weights
    y = np.array([0, 1, 0, 0, 0, 0])
    t = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    weights = np.array([1, 3, 1, 1, 1, 2])

    y_unweighted = np.repeat(y, weights)
    t_unweighted = np.repeat(t, weights)

    unshrink_factor_weighted = utils.unshrink(y, t, weights)
    unshrink_factor_unweighted = utils.unshrink(y_unweighted, t_unweighted)

    assert np.isclose(unshrink_factor_weighted, unshrink_factor_unweighted)


@pytest.mark.parametrize(
    "x, y, expected_x, expected_y",
    [
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([0, 1, 0]),
            np.array([[1, 2], [3, 4], [5, 6], [3, 4]]),
            np.array([0, 1, 0, 0]),
        ),
        (
            np.array([[7, 8], [9, 10], [11, 12]]),
            np.array([1, 0, 1]),
            np.array([[7, 8], [9, 10], [11, 12], [7, 8], [11, 12]]),
            np.array([1, 0, 1, 0, 0]),
        ),
        (
            np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9]),
            np.array([0, 0, 1, 0, 1, 1]),
            np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9, 0.5, 0.9, 0.9]),
            np.array([0, 0, 1, 0, 1, 1, 0, 0, 0]),
        ),
    ],
)
def test_make_unjoined_gives_expected_result(x, y, expected_x, expected_y):
    unjoined_x, unjoined_y = utils.make_unjoined(x, y)
    assert np.array_equal(
        unjoined_x, expected_x
    ), "The unjoined features are not as expected."
    assert np.array_equal(
        unjoined_y, expected_y
    ), "The unjoined labels are not as expected."


@pytest.mark.parametrize(
    "categorical_feature,expected_result",
    [
        ("INDIAN", 31255),
        ("SUB_SAHARAN_AFRICA", 46892),
        ("VIETNAMESE", 22530),
    ],
)
def test_hash_categorical_feature(categorical_feature, expected_result):
    """
    This unit test checks for equivalence with @jiayuanm's Hack implementation in D56290586.
    Reference values created in Hack kernel notebook N5246895.
    """
    actual_result = utils.hash_categorical_feature(categorical_feature)
    assert actual_result == expected_result


@pytest.mark.parametrize(
    "test_input, expected",
    [
        # Typical array
        (
            np.array([1, 2, 3, 4, 5]),
            2.605171,
        ),
        # Check resilience to overflow with array with large values
        (np.array([100001, 20000, 50000000, 2, 60000]), 26051.762950),
        # Check resilience to underflow with array with small values
        (
            np.array([0.0001, 0.00002, 0.00003, 0.00004, 0.00001, 0.00001, 0.00001]),
            0.0000218,
        ),
        # Geometric mean of any array of constants is the constant itself, test with long arrays
        (np.repeat(100000000000000, 10000), 100000000000000),
        (np.repeat(0.00001, 10000), 0.00001),
        # Any array containing zero has geom mean zero
        (np.array([0, 1, 2, 3]), 0),
    ],
)
def test_geometric_mean_gives_correct_result(test_input, expected):
    assert np.isclose(utils.geometric_mean(test_input), expected, atol=1e-6)


@pytest.mark.parametrize(
    "test_input",
    [
        np.array([]),
        np.array([-1, -2, -3]),
        np.array([1, 2, 3, 4, 5, -0.001]),
    ],  # Empty array  # Negative numbers
)
def test_geometric_mean_gives_nan_when_geometric_mean_is_undefined(test_input):
    result = utils.geometric_mean(test_input)
    assert np.isnan(result), f"Test failed for undefined input: {test_input}"


def test_convert_arrow_to_numpy_empty_dataframe_remains_empty():
    df = pd.DataFrame()
    result_df = utils.convert_arrow_columns_to_numpy(df)
    assert result_df.empty


def test_convert_arrow_to_numpy_single_column_converts_to_numpy_array():
    arrow_array = pa.array([1, 2, 3])
    df = pd.DataFrame({"col1": pd.Series(arrow_array, dtype=pd.ArrowDtype(pa.int64()))})
    assert isinstance(df["col1"].values, ArrowExtensionArray)

    result_df = utils.convert_arrow_columns_to_numpy(df)
    assert isinstance(result_df["col1"].values, np.ndarray)
    assert (result_df["col1"].values == np.array([1, 2, 3])).all()


def test_convert_arrow_to_numpy_single_row_converts_to_numpy_array():
    arrow_array = pa.array([1])
    df = pd.DataFrame({"col1": pd.Series(arrow_array, dtype=pd.ArrowDtype(pa.int64()))})
    assert isinstance(df["col1"].values, ArrowExtensionArray)

    result_df = utils.convert_arrow_columns_to_numpy(df)
    assert isinstance(result_df["col1"].values, np.ndarray)
    assert (result_df["col1"].values == np.array([1])).all()


def test_convert_arrow_to_numpy_with_null_values_converts_correctly():
    arrow_array = pa.array([1, None, 3], type=pa.int32())
    df = pd.DataFrame({"col1": pd.Series(arrow_array, dtype=pd.ArrowDtype(pa.int32()))})
    assert isinstance(df["col1"].values, ArrowExtensionArray)

    result_df = utils.convert_arrow_columns_to_numpy(df)
    assert isinstance(result_df["col1"].values, np.ndarray)

    expected_values = np.array([1, pd.NA, 3])
    # Custom comparison to handle pd.NA, because numpy.testing.assert_array_equal considers pd.NA unequal to pd.NA
    for actual, expected in zip(result_df["col1"].values, expected_values):
        if pd.isna(expected):
            assert pd.isna(actual), f"Expected NA, but got {actual}"
        else:
            assert actual == expected, f"Expected {expected}, but got {actual}"


def test_convert_arrow_to_numpy_with_unsupported_type_remains_unchanged():
    df = pd.DataFrame({"col1": [object(), object(), object()]})
    assert df["col1"].dtype == object
    result_df = utils.convert_arrow_columns_to_numpy(df)
    assert isinstance(result_df["col1"].values, np.ndarray)
    assert result_df["col1"].dtype == object


def test_logistic_vectorized_returns_valid_probabilities():
    log_odds = np.array([-10, -1, 0, 1, 10])
    result = utils.logistic_vectorized(log_odds)
    assert np.all(result > 0) and np.all(result < 1)


def test_logistic_vectorized_with_extreme_values():
    log_odds = np.array([-1000, -100, 100, 1000])
    result = utils.logistic_vectorized(log_odds)
    assert result[0] < 1e-300
    assert result[1] < 1e-40
    assert result[2] > 0.999  # Very close to 1
    assert result[3] > 0.999  # Very close to 1
