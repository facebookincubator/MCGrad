# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict

"""
Internal utilities for creating data segments for multicalibration analysis.

This module provides functions for partitioning data into segments based on
categorical and numerical features.
"""

import itertools
import logging
from collections.abc import Generator

import numpy as np
import pandas as pd
from numpy import typing as npt
from pandas.api.types import is_numeric_dtype


logger: logging.Logger = logging.getLogger(__name__)

CATEGORICAL_COLLAPSE_VALUE: str = "__OTHER"
NA_SEGMENT_VALUE_CATEGORICAL: str = "__NA"
NA_SEGMENT_VALUE_NUMERICAL: float = np.nan


def _concat_feature_values(feature_values_list: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate feature value DataFrames, handling empty entries."""
    non_empty_dfs = [df for df in feature_values_list if not df.empty]
    if non_empty_dfs:
        return pd.concat(non_empty_dfs, ignore_index=True)
    return pd.DataFrame(columns=["segment_column", "value", "idx_segment"])


def get_segment_masks(
    df: pd.DataFrame,
    categorical_segment_columns: list[str] | None = None,
    numerical_segment_columns: list[str] | None = None,
    min_depth: int = 0,
    max_depth: int | None = 3,
    max_values_per_segment_feature: int = 3,
    min_samples_per_segment: int = 10,
    chunk_size: int = 50,
) -> Generator[
    tuple[npt.NDArray[np.bool_], int, pd.DataFrame],
    None,
    None,
]:
    """
    Generate boolean masks for dataframe segments.

    Segments are based on combinations of categorical and numerical
    segmentation feature values.

    :param df: The dataframe to segment.
    :param categorical_segment_columns: Column names that are categorical.
    :param numerical_segment_columns: Column names that are numerical.
    :param min_depth: Minimum depth of combinations for creating segments.
    :param max_depth: Maximum depth of combinations for creating segments.
        If None, segmentation continues until all combinations are considered.
    :param max_values_per_segment_feature: Maximum unique values (or bins for
        numerical columns) to retain per segment feature before collapsing
        others into a distinct category (or bin).
    :param min_samples_per_segment: Minimum samples per segment to be returned.
        Segments with fewer samples will be discarded.
    :param chunk_size: The number of segments to return in each chunk.
    :return: A generator yielding tuples of (chunk, n_segments_in_chunk,
        feature_values_df), where chunk is an array of booleans corresponding
        to whether a sample belongs to the segment.

    Notes:
        - If both `categorical_segment_columns` and `numerical_segment_columns`
          are None, all samples are yielded as a single segment.
        - Missing values in segment columns are replaced with a predefined
          constant and a warning is logged.
    """
    if categorical_segment_columns is None and numerical_segment_columns is None:
        yield (
            np.ones((1, len(df)), dtype=np.bool_),
            1,
            pd.DataFrame(columns=["segment_column", "value", "idx_segment"]),
        )
    else:
        categorical_segment_columns = categorical_segment_columns or []
        numerical_segment_columns = numerical_segment_columns or []

        max_depth = (
            len(categorical_segment_columns + numerical_segment_columns)
            if max_depth is None
            else max_depth
        )

        df_subset = replace_missing_values(
            df, categorical_segment_columns, numerical_segment_columns
        )
        precomputed_masks, labeled_segment_column_unique_values = (
            _generate_segmentation_value_masks(
                df_subset,
                categorical_segment_columns,
                numerical_segment_columns,
                max_values=max_values_per_segment_feature,
            )
        )
        chunk = np.zeros((chunk_size, len(df)), dtype=np.bool_)
        feature_values_list: list[pd.DataFrame] = []
        n_segments_in_chunk = 0
        idx_segment = 0
        for depth in range(min_depth, max_depth + 1):
            for subset in itertools.combinations(
                labeled_segment_column_unique_values, depth
            ):
                for product in itertools.product(*subset):
                    mask = _extract_masks(precomputed_masks, product, len(df_subset))
                    if mask.sum() >= min_samples_per_segment:
                        chunk[n_segments_in_chunk] = mask
                        n_segments_in_chunk += 1
                        feature_values_list.append(
                            _format_segment_feature_values(product, idx_segment)
                        )
                        idx_segment += 1
                        if n_segments_in_chunk == chunk_size:
                            df_feature_values = _concat_feature_values(
                                feature_values_list
                            )
                            yield chunk.copy(), n_segments_in_chunk, df_feature_values
                            chunk[:] = 0
                            n_segments_in_chunk = 0
                            feature_values_list = []

        # Yield remaining segments if any
        if n_segments_in_chunk > 0:
            df_feature_values = _concat_feature_values(feature_values_list)
            yield chunk.copy(), n_segments_in_chunk, df_feature_values


def _format_segment_feature_values(
    product: tuple[tuple[int | float | str, str], ...],
    idx_segment: int,
) -> pd.DataFrame:
    df_feature_values = pd.DataFrame(product, columns=["value", "segment_column"])
    df_feature_values["idx_segment"] = idx_segment
    return df_feature_values[["segment_column", "value", "idx_segment"]]


def _label_values_with_colname(
    unique_values: npt.NDArray,
    colname: str,
) -> list[tuple[int | float | str, str]]:
    return [(value, colname) for value in unique_values]


def _extract_masks(
    precomputed_masks: dict[tuple[str, int | float | str], npt.NDArray[np.bool_]],
    selected_values: tuple[tuple[int | float | str, str]],
    df_length: int,
) -> npt.NDArray[np.bool_]:
    """Computes segment mask using precomputed boolean arrays."""
    combined_mask = np.ones(df_length, dtype=np.bool_)
    for value, column in selected_values:
        key = (
            (column, "nan")
            if isinstance(value, float) and np.isnan(value)
            else (column, value)
        )
        if key not in precomputed_masks:
            return np.zeros(df_length, dtype=np.bool_)
        combined_mask &= precomputed_masks[key]
    return combined_mask


def collapse_infrequent_values(
    values: pd.Series,
    max_unique_values: int,
    categorical_collapse_value: str = CATEGORICAL_COLLAPSE_VALUE,
) -> pd.Series:
    """Collapse infrequent categorical values into a single category.

    Retains the ``max_unique_values - 1`` most frequent values and replaces
    all other values with ``categorical_collapse_value``.

    :param values: Series of categorical values.
    :param max_unique_values: Maximum number of unique values to retain.
    :param categorical_collapse_value: The value to use for collapsed categories.
    :return: Series with infrequent values replaced.
    """
    n_unique = values.nunique()
    if n_unique <= max_unique_values:
        return values

    last_k_to_collapse = n_unique - max_unique_values + 1
    values_to_collapse = values.value_counts().index[-last_k_to_collapse:]

    # Important to use the numpy properties to speedup by >10x over dictionaries
    collapse_mask = values.isin(values_to_collapse)
    transformed_values = values.where(~collapse_mask, categorical_collapse_value)

    return transformed_values


def collapse_numeric_values(
    values: pd.Series,
    max_unique_values: int,
) -> pd.Series:
    """Bin numerical values into quantile-based bins.

    If the number of unique values exceeds ``max_unique_values``, bins the
    values into ``max_unique_values`` quantile-based bins using :func:`pd.cut`.

    :param values: Series of numerical values.
    :param max_unique_values: Maximum number of bins to create.
    :return: Series with bin labels (integers) or original values if already
        within the limit.
    """
    if values.nunique() <= max_unique_values:
        return values

    quantiles = np.linspace(0, 1, max_unique_values + 1)
    bin_edges = values.quantile(quantiles)
    transformed_values = pd.cut(
        values,
        bins=bin_edges,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )
    return transformed_values


def replace_missing_values(
    df: pd.DataFrame,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
) -> pd.DataFrame:
    """Replace missing values in segment columns with sentinel values.

    Categorical columns receive :const:`NA_SEGMENT_VALUE_CATEGORICAL` and
    numerical columns receive :const:`NA_SEGMENT_VALUE_NUMERICAL`.

    :param df: DataFrame containing the segment columns.
    :param categorical_segment_columns: List of categorical column names.
    :param numerical_segment_columns: List of numerical column names.
    :return: DataFrame subset with only segment columns and missing values replaced.
    """
    df_subset = df[categorical_segment_columns + numerical_segment_columns]
    if df_subset.isnull().values.any():
        # At this point df_subset is a reference to df. To avoid modifying the original df,
        # we need to make a copy before replacing missing values.
        df_subset = df_subset.copy(deep=False)
        for col in categorical_segment_columns:
            df_subset[col] = df_subset[col].fillna(NA_SEGMENT_VALUE_CATEGORICAL)
        for col in numerical_segment_columns:
            df_subset[col] = df_subset[col].fillna(NA_SEGMENT_VALUE_NUMERICAL)
        logger.debug(
            "Missing values found in the data. Replaced with %s for categorical "
            "and %s for numerical data. Missing values are treated as an "
            "additional segment feature value and are not counted towards the "
            "specified max_values_per_segment_feature limit.",
            NA_SEGMENT_VALUE_CATEGORICAL,
            NA_SEGMENT_VALUE_NUMERICAL,
        )
    return df_subset


def _generate_segmentation_value_masks(
    df: pd.DataFrame,
    categorical_segment_columns: list[str],
    numerical_segment_columns: list[str],
    max_values: int,
) -> tuple[
    dict[tuple[str, int | float | str], npt.NDArray[np.bool_]],
    list[list[tuple[int | float | str, str]]],
]:
    labeled_segment_column_unique_values = []
    value_masks = {}
    for col in categorical_segment_columns:
        replacement_value = CATEGORICAL_COLLAPSE_VALUE
        if is_numeric_dtype(df[col]):
            replacement_value = np.nanmin(df[col]) - 1
        transformed_values = collapse_infrequent_values(
            values=df[col],
            max_unique_values=max_values,
            categorical_collapse_value=replacement_value,
        )
        unique_values = np.array(transformed_values.unique())
        labeled_segment_column_unique_values.append(
            _label_values_with_colname(unique_values, col)
        )
        for value in unique_values:
            value_masks[(col, value)] = (transformed_values == value).values

    for col in numerical_segment_columns:
        transformed_values = collapse_numeric_values(
            df[col],
            max_unique_values=max_values,
        )
        unique_values = np.array(transformed_values.unique())
        labeled_segment_column_unique_values.append(
            _label_values_with_colname(unique_values, col)
        )
        for value in unique_values:
            # Because np.nan == np.nan returns False, we need to check for NaN separately and save them as string "nan".
            if np.isnan(value):
                value_masks[(col, "nan")] = transformed_values.isna().values
            else:
                value_masks[(col, value)] = (transformed_values == value).values
    return value_masks, labeled_segment_column_unique_values
