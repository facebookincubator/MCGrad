# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

import numpy as np
import pandas as pd
import pytest

from .. import _segmentation as segmentation


def _count_segments(generator):
    first_chunk = next(generator)[0]
    segment_count = np.count_nonzero(first_chunk.sum(axis=1))
    return first_chunk, segment_count


def test_that_get_segment_masks_returns_full_data_at_depth_zero():
    test_df = pd.DataFrame(
        {
            "segment_A": [0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
        }
    )
    generator = segmentation.get_segment_masks(
        test_df,
        categorical_segment_columns=["segment_A"],
        max_depth=0,
        min_samples_per_segment=1,
        chunk_size=5,
    )
    masks, num_segments = _count_segments(generator)
    assert num_segments == 1
    assert np.array_equal(np.where(masks[0])[0], np.arange(10))


def test_that_get_segment_masks_works_as_expected_with_nans():
    # Expected behavior is that NaN values are treated as a separate segment
    test_df = pd.DataFrame({"segment_A": ["a", np.nan, "b", np.nan]})
    generator = segmentation.get_segment_masks(
        test_df,
        categorical_segment_columns=["segment_A"],
        max_depth=1,
        min_samples_per_segment=1,
        chunk_size=5,
    )
    masks, num_segments = _count_segments(generator)
    assert num_segments == 4  # one for full data, one for NA, a, b each
    assert np.array_equal(np.where(masks[1])[0], np.array([0]))
    assert np.array_equal(np.where(masks[2])[0], np.array([1, 3]))
    assert np.array_equal(np.where(masks[3])[0], np.array([2]))


def test_that_get_segment_masks_works_with_single_segment_feature():
    test_df = pd.DataFrame(
        {
            "segment_A": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    )
    max_values_per_segment_feature = 3
    generator = segmentation.get_segment_masks(
        test_df,
        categorical_segment_columns=[],
        numerical_segment_columns=["segment_A"],
        max_depth=1,
        max_values_per_segment_feature=max_values_per_segment_feature,
        min_samples_per_segment=1,
        chunk_size=10,
    )
    _, num_segments = _count_segments(generator)
    # max_values_per_segment_feature + 1 because we have the full data as the first segment
    assert num_segments == (max_values_per_segment_feature + 1)


def test_that_get_segment_masks_returns_correct_number_of_segments_when_using_min_depth():
    test_df = pd.DataFrame({"segment_A": ["a", "b"]})
    generator = segmentation.get_segment_masks(
        test_df,
        categorical_segment_columns=["segment_A"],
        min_depth=1,
        max_depth=1,
        min_samples_per_segment=1,
        chunk_size=5,
    )
    masks, num_segments = _count_segments(generator)
    assert num_segments == 2  # one for a, b each
    assert np.array_equal(np.where(masks[0])[0], np.array([0]))
    assert np.array_equal(np.where(masks[1])[0], np.array([1]))


def test_that_get_segment_masks_returns_whole_dataset_if_no_features_are_specified():
    test_df = pd.DataFrame({"segment_A": ["a", "b"]})
    generator = segmentation.get_segment_masks(
        test_df, min_samples_per_segment=1, chunk_size=5
    )
    masks, num_segments = _count_segments(generator)
    assert num_segments == 1  # whole dataset as single segment
    assert np.array_equal(np.where(masks[0])[0], np.array([0, 1]))


def test_that_get_segment_masks_works_as_expected_with_nans_in_numerical_feature():
    # Expected behavior is that NaN values are treated as a separate segment
    df = pd.DataFrame({"segment_A": [0.1, None, 0.3, None, 0.4, 0.5, 0.6]})
    generator = segmentation.get_segment_masks(
        df,
        numerical_segment_columns=["segment_A"],
        max_depth=1,
        min_samples_per_segment=1,
        max_values_per_segment_feature=2,
        chunk_size=10,
    )
    masks, num_segments = _count_segments(generator)
    # one for root segment and one for the two allowed values + one for the nans
    assert num_segments == 4
    assert np.array_equal(np.where(masks[0])[0], np.array([0, 1, 2, 3, 4, 5, 6]))
    assert np.array_equal(np.where(masks[1])[0], np.array([0, 2, 4]))
    assert np.array_equal(np.where(masks[2])[0], np.array([1, 3]))
    assert np.array_equal(np.where(masks[3])[0], np.array([5, 6]))


def test_that_get_segment_masks_collapses_numerical_feature_correctly():
    test_df = pd.DataFrame({"segment_A": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]})
    generator = segmentation.get_segment_masks(
        test_df,
        numerical_segment_columns=["segment_A"],
        max_values_per_segment_feature=2,
        min_samples_per_segment=1,
        chunk_size=5,
    )
    masks, num_segments = _count_segments(generator)
    assert num_segments == 3  # one for the root and one for the two bins each
    assert np.array_equal(np.where(masks[0])[0], np.array([0, 1, 2, 3, 4, 5]))
    assert np.array_equal(np.where(masks[1])[0], np.array([0, 1, 2]))
    assert np.array_equal(np.where(masks[2])[0], np.array([3, 4, 5]))


def test_that_get_segment_masks_collapses_categorical_feature_correctly():
    test_df = pd.DataFrame({"segment_A": ["a", "a", "b", "b", "c", "d"]})
    generator = segmentation.get_segment_masks(
        test_df,
        categorical_segment_columns=["segment_A"],
        max_values_per_segment_feature=3,
        min_samples_per_segment=1,
        chunk_size=5,
    )
    masks, num_segments = _count_segments(generator)
    # one for the root and one for 'a', 'b', 'other' ('c' + 'd') respectively
    assert num_segments == 4
    assert np.array_equal(np.where(masks[0])[0], np.array([0, 1, 2, 3, 4, 5]))  # full
    assert np.array_equal(np.where(masks[1])[0], np.array([0, 1]))
    assert np.array_equal(np.where(masks[2])[0], np.array([2, 3]))
    assert np.array_equal(np.where(masks[3])[0], np.array([4, 5]))


def test_that_get_segment_masks_collapses_categorical_feature_correctly_when_missing_values_exist():
    test_df = pd.DataFrame({"segment_A": ["a", "a", None, None, "c", "d"]})
    generator = segmentation.get_segment_masks(
        test_df,
        categorical_segment_columns=["segment_A"],
        max_values_per_segment_feature=3,
        min_samples_per_segment=1,
        chunk_size=10,
    )
    masks, num_segments = _count_segments(generator)
    # one for the root and one for 'a', None, 'other' ('c' + 'd') respectively
    assert num_segments == 4
    assert np.array_equal(np.where(masks[0])[0], np.array([0, 1, 2, 3, 4, 5]))
    assert np.array_equal(np.where(masks[1])[0], np.array([0, 1]))
    assert np.array_equal(np.where(masks[2])[0], np.array([2, 3]))
    assert np.array_equal(np.where(masks[3])[0], np.array([4, 5]))


def test_that_get_segment_masks_returns_correct_number_of_segments():
    test_df = pd.DataFrame(
        {
            "segment_A": ["a", "a", "a", "b", "b", "b"],
            "segment_B": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
            "segment_C": [0, 0, 1, 1, 1, np.nan],
        }
    )
    generator = segmentation.get_segment_masks(
        test_df,
        categorical_segment_columns=["segment_A", "segment_C"],
        numerical_segment_columns=["segment_B"],
        max_values_per_segment_feature=2,
        max_depth=None,
        min_samples_per_segment=1,
        chunk_size=25,
    )
    expected_n_segments = 1 + 12 + 8
    _, num_segments = _count_segments(generator)
    assert num_segments == expected_n_segments


def test_that_collapse_infrequent_values_is_identity_when_unique_values_lt_max_values():
    test_array = pd.Series(["a", "a", "b", "c"])
    results = segmentation.collapse_infrequent_values(test_array, max_unique_values=4)
    assert np.array_equal(results, test_array)


def test_that_collapse_infrequent_values_collapses_all_values_to_collapse_value_if_max_unique_is_1():
    test_array = pd.Series(["a", "a", "b", "c"])
    results = segmentation.collapse_infrequent_values(test_array, max_unique_values=1)
    expected = np.array([segmentation.CATEGORICAL_COLLAPSE_VALUE] * 4)
    assert np.array_equal(results, expected)


@pytest.mark.parametrize(
    "test_array, expected",
    [
        # Typical array
        (
            pd.Series(["a", "a", "b", "c"]),
            np.array(
                [
                    "a",
                    "a",
                    segmentation.CATEGORICAL_COLLAPSE_VALUE,
                    segmentation.CATEGORICAL_COLLAPSE_VALUE,
                ]
            ),
        ),
        # Check invariant to input order
        (
            pd.Series(["b", "c", "a", "a"]),
            np.array(
                [
                    segmentation.CATEGORICAL_COLLAPSE_VALUE,
                    segmentation.CATEGORICAL_COLLAPSE_VALUE,
                    "a",
                    "a",
                ]
            ),
        ),
        # Invariant to repetition
        (
            pd.Series(["b", "c", "a", "a"] * 2),
            np.array(
                [
                    segmentation.CATEGORICAL_COLLAPSE_VALUE,
                    segmentation.CATEGORICAL_COLLAPSE_VALUE,
                    "a",
                    "a",
                ]
                * 2
            ),
        ),
    ],
)
def test_that_collapse_infrequent_values_collapses_correctly_for_happy_path(
    test_array, expected
):
    results = segmentation.collapse_infrequent_values(
        values=test_array,
        max_unique_values=2,
    )
    assert np.array_equal(results, expected)


def test_that_collapse_numeric_values_returns_identity_for_unique_values_lt_max_values():
    test_array = pd.Series([0.1, 0.2, 0.3, 0.4])
    results = segmentation.collapse_numeric_values(test_array, max_unique_values=4)
    assert np.array_equal(results, test_array)


def test_that_collapse_numeric_values_returns_identity_for_unique_values_lt_max_values_when_missing_values_exist():
    test_array = pd.Series([0.1, 0.2, None, 0.4])
    results = segmentation.collapse_numeric_values(test_array, max_unique_values=4)
    assert np.array_equal(results, test_array, equal_nan=True)


def test_that_collapse_numeric_values_returns_correct_number_of_values():
    test_array = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    results = segmentation.collapse_numeric_values(test_array, max_unique_values=3)
    assert len(np.unique(results)) == 3


def test_that_collapse_numeric_values_missing_values_do_not_affect_other_rows():
    test_array = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    test_array_with_nan = pd.Series([None, 1, 2, 3, 4, None, 5, 6, 7, 8, 9, 10, None])
    result_without_nan = segmentation.collapse_numeric_values(
        test_array, max_unique_values=3
    )
    result_with_nan = segmentation.collapse_numeric_values(
        test_array_with_nan, max_unique_values=3
    )
    assert np.all(
        result_without_nan.values
        == result_with_nan[~result_with_nan.isna()].astype(int).values
    )


def test_that_collapse_numeric_values_returns_correct_number_of_values_with_max_values_1():
    test_array = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    results = segmentation.collapse_numeric_values(test_array, max_unique_values=1)
    assert len(np.unique(results)) == 1
    assert np.array_equal(results, np.array([0] * len(test_array)))


def test_that_get_segment_masks_works_with_arbitrary_input_index():
    test_df = pd.DataFrame(
        {"segment_A": ["a", "a", "b", "b", "c", "d"]}, index=[3, 5, 2, 0, 4, 1]
    )
    generator = segmentation.get_segment_masks(
        test_df,
        categorical_segment_columns=["segment_A"],
        max_values_per_segment_feature=4,
        min_samples_per_segment=1,
        chunk_size=10,
    )
    masks, num_segments = _count_segments(generator)
    assert num_segments == 5
    assert np.array_equal(np.where(masks[0])[0], np.array([0, 1, 2, 3, 4, 5]))  # full
    assert np.array_equal(np.where(masks[1])[0], np.array([0, 1]))  # segment_A = 'a'
    assert np.array_equal(np.where(masks[2])[0], np.array([2, 3]))  # segment_A = 'b'
    assert np.array_equal(np.where(masks[3])[0], np.array([4]))  # segment_A = 'c'
    assert np.array_equal(np.where(masks[4])[0], np.array([5]))  # segment_A = 'd'


def test_that_get_segment_masks_works_with_arbitrary_input_index_when_missing_values_exist():
    test_df = pd.DataFrame(
        {"segment_A": ["a", "a", None, None, "c", "d"]}, index=[3, 5, 2, 0, 4, 1]
    )
    generator = segmentation.get_segment_masks(
        test_df,
        categorical_segment_columns=["segment_A"],
        max_values_per_segment_feature=4,
        min_samples_per_segment=1,
        chunk_size=10,
    )
    masks, num_segments = _count_segments(generator)
    assert num_segments == 5
    assert np.array_equal(np.where(masks[0])[0], np.array([0, 1, 2, 3, 4, 5]))  # full
    assert np.array_equal(np.where(masks[1])[0], np.array([0, 1]))  # segment_A = 'a'
    assert np.array_equal(np.where(masks[2])[0], np.array([2, 3]))  # segment_A = None
    assert np.array_equal(np.where(masks[3])[0], np.array([4]))  # segment_A = 'c'
    assert np.array_equal(np.where(masks[4])[0], np.array([5]))  # segment_A = 'd'


def test_extract_masks_returns_empty_mask_when_key_not_found():
    precomputed_masks = {("column_A", "value1"): np.array([True, False, True])}
    selected_values = (("value2", "column_A"),)
    df_length = 3

    result = segmentation._extract_masks(precomputed_masks, selected_values, df_length)

    # Result should be an all-False mask when key is not found
    expected = np.zeros(df_length, dtype=np.bool_)
    assert np.array_equal(result, expected)


def test_concat_feature_values_with_non_empty_dataframes():
    df1 = pd.DataFrame({"segment_column": ["A"], "value": [1], "idx_segment": [0]})
    df2 = pd.DataFrame({"segment_column": ["B"], "value": [2], "idx_segment": [1]})

    result = segmentation._concat_feature_values([df1, df2])

    assert len(result) == 2
    assert list(result.columns) == ["segment_column", "value", "idx_segment"]
    assert result["segment_column"].tolist() == ["A", "B"]
    assert result["idx_segment"].tolist() == [0, 1]


def test_concat_feature_values_with_empty_list():
    result = segmentation._concat_feature_values([])

    assert result.empty
    assert list(result.columns) == ["segment_column", "value", "idx_segment"]


def test_concat_feature_values_with_all_empty_dataframes():
    df1 = pd.DataFrame(columns=["segment_column", "value", "idx_segment"])
    df2 = pd.DataFrame(columns=["segment_column", "value", "idx_segment"])

    result = segmentation._concat_feature_values([df1, df2])

    assert result.empty
    assert list(result.columns) == ["segment_column", "value", "idx_segment"]


def test_concat_feature_values_with_mixed_empty_and_non_empty():
    empty_df = pd.DataFrame(columns=["segment_column", "value", "idx_segment"])
    non_empty_df = pd.DataFrame(
        {"segment_column": ["A", "B"], "value": [1, 2], "idx_segment": [0, 0]}
    )

    result = segmentation._concat_feature_values([empty_df, non_empty_df, empty_df])

    assert len(result) == 2
    assert result["segment_column"].tolist() == ["A", "B"]


def test_get_segment_masks_does_not_modify_input_dataframe():
    df = pd.DataFrame(
        {
            "segment_A": ["a", "a", "b", "b", "c", "d"],
            "segment_B": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        }
    )
    df_original = df.copy()

    generator = segmentation.get_segment_masks(
        df,
        categorical_segment_columns=["segment_A"],
        numerical_segment_columns=["segment_B"],
        max_values_per_segment_feature=3,
        min_samples_per_segment=1,
        chunk_size=10,
    )
    # Consume the generator to ensure all processing is done
    _ = list(generator)

    pd.testing.assert_frame_equal(df, df_original)


def test_get_segment_masks_does_not_modify_input_dataframe_with_missing_values():
    df = pd.DataFrame(
        {
            "segment_A": ["a", "a", None, None, "c", "d"],
            "segment_B": [0.1, None, 0.3, None, 0.8, 0.9],
        }
    )
    df_original = df.copy()

    generator = segmentation.get_segment_masks(
        df,
        categorical_segment_columns=["segment_A"],
        numerical_segment_columns=["segment_B"],
        max_values_per_segment_feature=3,
        min_samples_per_segment=1,
        chunk_size=10,
    )
    # Consume the generator to ensure all processing is done
    _ = list(generator)

    pd.testing.assert_frame_equal(df, df_original)


def test_collapse_infrequent_values_does_not_modify_input_series():
    values = pd.Series(["a", "a", "b", "c", "d", "e"])
    values_original = values.copy()

    _ = segmentation.collapse_infrequent_values(values, max_unique_values=2)

    pd.testing.assert_series_equal(values, values_original)


def test_collapse_numeric_values_does_not_modify_input_series():
    values = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    values_original = values.copy()

    _ = segmentation.collapse_numeric_values(values, max_unique_values=3)

    pd.testing.assert_series_equal(values, values_original)


def test_collapse_numeric_values_does_not_modify_input_series_with_missing_values():
    values = pd.Series([0.1, None, 0.3, None, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    values_original = values.copy()

    _ = segmentation.collapse_numeric_values(values, max_unique_values=3)

    pd.testing.assert_series_equal(values, values_original)


def test_replace_missing_values_does_not_modify_input_dataframe():
    df = pd.DataFrame(
        {
            "cat_col": ["a", None, "b", None],
            "num_col": [0.1, None, 0.3, None],
        }
    )
    df_original = df.copy()

    _ = segmentation.replace_missing_values(
        df,
        categorical_segment_columns=["cat_col"],
        numerical_segment_columns=["num_col"],
    )

    pd.testing.assert_frame_equal(df, df_original)
