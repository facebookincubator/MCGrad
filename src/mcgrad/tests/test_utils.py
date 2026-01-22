# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

import math
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pandas.core.arrays import ArrowExtensionArray

from .. import _utils as utils


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
def test_unshrink(labels, predictions, expected_result):
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


def test_encoder_serialize_deserialize_with_integer_category_keys():
    df = pd.DataFrame({"Code": [100, 200, 300, 100, 200]})

    encoder = utils.OrdinalEncoderWithUnknownSupport()
    encoder.fit(df)

    serialized = encoder.serialize()
    deserialized = utils.OrdinalEncoderWithUnknownSupport.deserialize(serialized)

    assert deserialized._category_map == encoder._category_map
    for col_idx, inner_map in deserialized._category_map.items():
        assert isinstance(col_idx, int)
        for key in inner_map.keys():
            assert isinstance(key, int)

    df_test = pd.DataFrame({"Code": [100, 999, 300]})
    original_transformed = encoder.transform(df_test)
    deserialized_transformed = deserialized.transform(df_test)
    np.testing.assert_array_equal(deserialized_transformed, original_transformed)


def test_encoder_serialize_deserialize_with_mixed_columns():
    df = pd.DataFrame(
        {
            "City": ["Paris", "Tokyo", "Amsterdam"],
            "Code": [100, 200, 300],
        }
    )

    encoder = utils.OrdinalEncoderWithUnknownSupport()
    encoder.fit(df)

    serialized = encoder.serialize()
    deserialized = utils.OrdinalEncoderWithUnknownSupport.deserialize(serialized)

    assert deserialized._category_map == encoder._category_map

    df_test = pd.DataFrame({"City": ["Paris", "Berlin"], "Code": [100, 999]})
    original_transformed = encoder.transform(df_test)
    deserialized_transformed = deserialized.transform(df_test)
    np.testing.assert_array_equal(deserialized_transformed, original_transformed)


def test_encoder_deserialize_legacy_format():
    legacy_str = "{0: {'Paris': 0, 'Tokyo': 1, 'Amsterdam': 2}}"
    deserialized = utils.OrdinalEncoderWithUnknownSupport.deserialize(legacy_str)

    assert deserialized._category_map == {0: {"Paris": 0, "Tokyo": 1, "Amsterdam": 2}}

    df_test = pd.DataFrame({"City": ["Paris", "Copenhagen", "Tokyo"]})
    transformed = deserialized.transform(df_test)
    expected = np.array([[0], [-1], [1]])
    np.testing.assert_array_equal(transformed, expected)


def test_encoder_serialize_deserialize_with_numpy_scalar_keys():
    df = pd.DataFrame({"Code": np.array([100, 200, 300, 100, 200], dtype=np.int64)})

    encoder = utils.OrdinalEncoderWithUnknownSupport()
    encoder.fit(df)

    for col_idx, inner_map in encoder._category_map.items():
        assert isinstance(col_idx, int)
        for key, val in inner_map.items():
            assert isinstance(key, int), f"Expected native int, got {type(key)}"
            assert isinstance(val, int)

    serialized = encoder.serialize()
    deserialized = utils.OrdinalEncoderWithUnknownSupport.deserialize(serialized)

    assert deserialized._category_map == encoder._category_map
    assert deserialized._category_map == {0: {100: 0, 200: 1, 300: 2}}


def test_encoder_serialize_deserialize_preserves_numeric_string_keys():
    df = pd.DataFrame({"ProductCode": ["100", "200", "300", "100", "200"]})

    encoder = utils.OrdinalEncoderWithUnknownSupport()
    encoder.fit(df)

    serialized = encoder.serialize()
    deserialized = utils.OrdinalEncoderWithUnknownSupport.deserialize(serialized)

    assert deserialized._category_map == encoder._category_map
    for col_idx, inner_map in deserialized._category_map.items():
        assert isinstance(col_idx, int)
        for key in inner_map.keys():
            assert isinstance(key, str), f"Expected string key, got {type(key)}: {key}"

    df_test = pd.DataFrame({"ProductCode": ["100", "999", "300"]})
    original_transformed = encoder.transform(df_test)
    deserialized_transformed = deserialized.transform(df_test)
    np.testing.assert_array_equal(deserialized_transformed, original_transformed)
    assert deserialized_transformed[1, 0] == -1


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
    assert np.array_equal(unjoined_x, expected_x), (
        "The unjoined features are not as expected."
    )
    assert np.array_equal(unjoined_y, expected_y), (
        "The unjoined labels are not as expected."
    )


@pytest.mark.parametrize(
    "categorical_feature,expected_result",
    [
        ("TOKYO", 54410),
        ("AMSTERDAM", 42395),
        ("JAKARTA", 21470),
    ],
)
def test_hash_categorical_feature(categorical_feature, expected_result):
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
    # These edge cases may trigger numpy warnings for log of negative/zero or mean of empty slice
    # (depends on whether np.errstate() is active in the implementation)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
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


def test_OrdinalEncoderWithUnknownSupport_transform_before_fit_raises_error():
    encoder = utils.OrdinalEncoderWithUnknownSupport()
    df = pd.DataFrame({"City": ["Paris", "Tokyo"]})
    with pytest.raises(
        ValueError, match="fit method should be called before transform"
    ):
        encoder.transform(df.values)


def test_positive_label_proportion_does_not_modify_input_arrays(rng):
    labels = rng.randint(0, 2, 100).astype(float)
    predictions = rng.uniform(0.1, 0.9, 100)
    bins = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    sample_weight = rng.uniform(0.5, 2.0, 100)

    labels_original = labels.copy()
    predictions_original = predictions.copy()
    bins_original = bins.copy()
    sample_weight_original = sample_weight.copy()

    _ = utils.positive_label_proportion(
        labels=labels,
        predictions=predictions,
        bins=bins,
        sample_weight=sample_weight,
    )

    np.testing.assert_array_equal(labels, labels_original)
    np.testing.assert_array_equal(predictions, predictions_original)
    np.testing.assert_array_equal(bins, bins_original)
    np.testing.assert_array_equal(sample_weight, sample_weight_original)


def test_ordinal_encoder_fit_does_not_modify_input_array():
    data = np.array([["Paris", "Male"], ["Tokyo", "Female"], ["Amsterdam", "Male"]])
    data_original = data.copy()

    encoder = utils.OrdinalEncoderWithUnknownSupport()
    encoder.fit(data)

    np.testing.assert_array_equal(data, data_original)


def test_ordinal_encoder_fit_does_not_modify_input_dataframe():
    df = pd.DataFrame(
        {
            "City": ["Paris", "Tokyo", "Amsterdam"],
            "Gender": ["Male", "Female", "Male"],
        }
    )
    df_original = df.copy()

    encoder = utils.OrdinalEncoderWithUnknownSupport()
    encoder.fit(df)

    pd.testing.assert_frame_equal(df, df_original)


def test_ordinal_encoder_transform_does_not_modify_input_array():
    train_data = np.array(
        [["Paris", "Male"], ["Tokyo", "Female"], ["Amsterdam", "Male"]]
    )
    test_data = np.array([["Paris", "Female"], ["Copenhagen", "Male"]])
    test_data_original = test_data.copy()

    encoder = utils.OrdinalEncoderWithUnknownSupport()
    encoder.fit(train_data)
    _ = encoder.transform(test_data)

    np.testing.assert_array_equal(test_data, test_data_original)


def test_ordinal_encoder_transform_does_not_modify_input_dataframe():
    df_train = pd.DataFrame(
        {
            "City": ["Paris", "Tokyo", "Amsterdam"],
            "Gender": ["Male", "Female", "Male"],
        }
    )
    df_test = pd.DataFrame(
        {
            "City": ["Paris", "Copenhagen"],
            "Gender": ["Female", "Male"],
        }
    )
    df_test_original = df_test.copy()

    encoder = utils.OrdinalEncoderWithUnknownSupport()
    encoder.fit(df_train)
    _ = encoder.transform(df_test)

    pd.testing.assert_frame_equal(df_test, df_test_original)


def test_train_test_split_wrapper_split_does_not_modify_input_arrays(rng):
    X = rng.rand(100, 5)
    y = rng.randint(0, 2, 100)

    X_original = X.copy()
    y_original = y.copy()

    splitter = utils.TrainTestSplitWrapper(
        test_size=0.3, shuffle=True, random_state=42, stratify=True
    )
    for _, _ in splitter.split(X, y):
        pass

    np.testing.assert_array_equal(X, X_original)
    np.testing.assert_array_equal(y, y_original)


def test_make_equispaced_bins_does_not_modify_input_array(rng):
    predicted_scores = rng.uniform(0.1, 0.9, 100)
    predicted_scores_original = predicted_scores.copy()

    _ = utils.make_equispaced_bins(predicted_scores, num_bins=10)

    np.testing.assert_array_equal(predicted_scores, predicted_scores_original)


def test_make_equisized_bins_does_not_modify_input_array(rng):
    predicted_scores = rng.uniform(0.1, 0.9, 100)
    predicted_scores_original = predicted_scores.copy()

    _ = utils.make_equisized_bins(predicted_scores, num_bins=5)

    np.testing.assert_array_equal(predicted_scores, predicted_scores_original)


def test_unshrink_does_not_modify_input_arrays(rng):
    y = rng.randint(0, 2, 50).astype(float)
    logits = rng.uniform(-2, 2, 50)
    w = rng.uniform(0.5, 2.0, 50)

    y_original = y.copy()
    logits_original = logits.copy()
    w_original = w.copy()

    _ = utils.unshrink(y, logits, w)

    np.testing.assert_array_equal(y, y_original)
    np.testing.assert_array_equal(logits, logits_original)
    np.testing.assert_array_equal(w, w_original)


def test_logit_does_not_modify_input_array(rng):
    probs = rng.uniform(0.1, 0.9, 100)
    probs_original = probs.copy()

    _ = utils.logit(probs)

    np.testing.assert_array_equal(probs, probs_original)


def test_absolute_error_does_not_modify_input_arrays(rng):
    estimate = rng.uniform(0, 100, 50)
    reference = rng.uniform(0, 100, 50)

    estimate_original = estimate.copy()
    reference_original = reference.copy()

    _ = utils.absolute_error(estimate, reference)

    np.testing.assert_array_equal(estimate, estimate_original)
    np.testing.assert_array_equal(reference, reference_original)


def test_proportional_error_does_not_modify_input_arrays(rng):
    estimate = rng.uniform(1, 100, 50)
    reference = rng.uniform(1, 100, 50)

    estimate_original = estimate.copy()
    reference_original = reference.copy()

    _ = utils.proportional_error(estimate, reference)

    np.testing.assert_array_equal(estimate, estimate_original)
    np.testing.assert_array_equal(reference, reference_original)


def test_make_unjoined_does_not_modify_input_arrays(rng):
    x = rng.uniform(0, 1, (50, 3))
    y = rng.randint(0, 2, 50)

    x_original = x.copy()
    y_original = y.copy()

    _, _ = utils.make_unjoined(x, y)

    np.testing.assert_array_equal(x, x_original)
    np.testing.assert_array_equal(y, y_original)


def test_noop_splitter_wrapper_split_does_not_modify_input_arrays(rng):
    X = rng.rand(50, 5)
    y = rng.randint(0, 2, 50)

    X_original = X.copy()
    y_original = y.copy()

    splitter = utils.NoopSplitterWrapper()
    for _, _ in splitter.split(X, y):
        pass

    np.testing.assert_array_equal(X, X_original)
    np.testing.assert_array_equal(y, y_original)


def test_geometric_mean_does_not_modify_input_array(rng):
    x = rng.uniform(0.1, 100, 50)
    x_original = x.copy()

    _ = utils.geometric_mean(x)

    np.testing.assert_array_equal(x, x_original)


def test_logistic_vectorized_does_not_modify_input_array(rng):
    log_odds = rng.uniform(-5, 5, 100)
    log_odds_original = log_odds.copy()

    _ = utils.logistic_vectorized(log_odds)

    np.testing.assert_array_equal(log_odds, log_odds_original)


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

    data_with_predicted_labels_and_thresholds = utils.predictions_to_labels(
        data=data,
        prediction_column="prediction",
        thresholds=thresholds,
        threshold_column="threshold",
    )

    pd.testing.assert_frame_equal(data_with_predicted_labels_and_thresholds, expected)


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

    _ = utils.predictions_to_labels(
        data=data, prediction_column="prediction", thresholds=thresholds
    )

    pd.testing.assert_frame_equal(data, data_original)
    pd.testing.assert_frame_equal(thresholds, thresholds_original)
