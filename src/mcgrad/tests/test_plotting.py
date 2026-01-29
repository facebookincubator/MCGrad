# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from .. import methods, metrics, plotting


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def sample_data(rng):
    """Fixture providing sample classification data for plotting tests."""
    n_samples = 100
    return {
        "scores": rng.rand(n_samples),
        "labels": rng.randint(0, 2, n_samples),
        "weights": rng.rand(n_samples) + 0.5,
    }


@pytest.fixture
def sample_df(rng):
    """Fixture providing a DataFrame with segment information."""
    n_cat_fts = 2
    n_num_fts = 2
    n_samples = 100

    return pd.DataFrame(
        {
            **{
                f"segment_A_{t}": rng.randint(0, 3, n_samples) for t in range(n_cat_fts)
            },
            **{f"segment_B_{t}": rng.rand(n_samples) for t in range(n_num_fts)},
            "prediction": rng.rand(n_samples),
            "weights": rng.rand(n_samples),
            "label": rng.randint(0, 2, n_samples),
        }
    ).astype(
        {
            "prediction": "float32",
            "label": "int32",
            **{f"segment_A_{t}": "int32" for t in range(n_cat_fts)},
            **{f"segment_B_{t}": "float64" for t in range(n_num_fts)},
            "weights": "float64",
        }
    )


@pytest.fixture
def mce_with_all_segments(sample_df):
    """Fixture providing MulticalibrationError with categorical and numerical segments."""
    return metrics.MulticalibrationError(
        df=sample_df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=[f"segment_A_{t}" for t in range(2)],
        numerical_segment_columns=[f"segment_B_{t}" for t in range(2)],
        weight_column="weights",
    )


def test_plot_segment_calibration_errors_basic(mce_with_all_segments):
    fig = plotting.plot_segment_calibration_errors(
        mce=mce_with_all_segments, quantity="segments_ecce_sigma"
    )
    assert fig is not None


@pytest.mark.parametrize(
    "quantity",
    ["segments_ecce_relative", "segments_ecce_pvalue", "segments_ecce"],
)
def test_plot_segment_calibration_errors_quantities(mce_with_all_segments, quantity):
    fig = plotting.plot_segment_calibration_errors(
        mce=mce_with_all_segments, quantity=quantity
    )
    assert fig is not None


def test_plot_segment_calibration_errors_raises_on_invalid_quantity(
    mce_with_all_segments,
):
    with pytest.raises(ValueError, match="Invalid quantity"):
        plotting.plot_segment_calibration_errors(
            mce=mce_with_all_segments, quantity="invalid_quantity"
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"sample_weight_col": "weights"},
        {"binning_method": "equisized"},
        {"plot_incomplete_cis": False},
    ],
    ids=["basic", "with_weights", "equisized", "incomplete_cis"],
)
def test_plot_global_calibration_curve(sample_df, kwargs):
    fig = plotting.plot_global_calibration_curve(
        data=sample_df,
        score_col="prediction",
        label_col="label",
        num_bins=10,
        **kwargs,
    )
    assert fig is not None


def test_plot_global_calibration_curve_invalid_binning_raises_error(sample_df):
    with pytest.raises(ValueError, match="Invalid binning_method"):
        plotting.plot_global_calibration_curve(
            data=sample_df,
            score_col="prediction",
            label_col="label",
            binning_method="invalid",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"sample_weight_col": "weights"},
        {"binning_method": "equisized"},
    ],
    ids=["basic", "with_weights", "equisized"],
)
def test_plot_calibration_curve_by_segment(sample_df, kwargs):
    fig = plotting.plot_calibration_curve_by_segment(
        data=sample_df,
        group_var="segment_A_0",
        score_col="prediction",
        label_col="label",
        num_bins=5,
        **kwargs,
    )
    assert fig is not None


def test_plot_calibration_curve_by_segment_empty_data():
    empty_df = pd.DataFrame({"group": [], "score": [], "label": []})

    fig = plotting.plot_calibration_curve_by_segment(
        data=empty_df,
        group_var="group",
        score_col="score",
        label_col="label",
        num_bins=5,
    )

    assert fig is not None


@pytest.fixture
def mcgrad_training_df(rng):
    """Fixture providing training data for MCGrad model tests."""
    n_samples = 200
    return pd.DataFrame(
        {
            "prediction": rng.rand(n_samples),
            "label": rng.randint(0, 2, n_samples),
            "feature": rng.choice(["a", "b", "c"], n_samples),
        }
    )


def _fit_mcgrad_model(
    df: pd.DataFrame,
    early_stopping: bool = True,
    save_training_performance: bool = False,
) -> methods.MCGrad:
    if early_stopping:
        model = methods.MCGrad(
            num_rounds=2,
            early_stopping=True,
            patience=1,
            save_training_performance=save_training_performance,
            lightgbm_params={"max_depth": 2, "n_estimators": 2},
        )
    else:
        model = methods.MCGrad(
            num_rounds=2,
            early_stopping=False,
            save_training_performance=save_training_performance,
            lightgbm_params={"max_depth": 2, "n_estimators": 2},
        )
    model.fit(
        df_train=df,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["feature"],
    )
    return model


def test_plot_learning_curve_basic(mcgrad_training_df):
    model = _fit_mcgrad_model(mcgrad_training_df)
    fig = plotting.plot_learning_curve(model)
    assert fig is not None


def test_plot_learning_curve_raises_without_early_stopping(mcgrad_training_df):
    model = _fit_mcgrad_model(mcgrad_training_df, early_stopping=False)
    with pytest.raises(
        ValueError,
        match="Learning curve can only be plotted for models trained with early_stopping=True",
    ):
        plotting.plot_learning_curve(model)


def test_plot_learning_curve_with_show_all(mcgrad_training_df):
    model = _fit_mcgrad_model(mcgrad_training_df, save_training_performance=True)
    fig = plotting.plot_learning_curve(model, show_all=True)
    assert fig is not None


def test_plot_global_calibration_curve_does_not_modify_input_dataframe(sample_df):
    df_original = sample_df.copy()

    _ = plotting.plot_global_calibration_curve(
        data=sample_df,
        score_col="prediction",
        label_col="label",
        num_bins=10,
        sample_weight_col="weights",
    )

    pd.testing.assert_frame_equal(sample_df, df_original)


def test_plot_calibration_curve_by_segment_does_not_modify_input_dataframe(sample_df):
    df_original = sample_df.copy()

    _ = plotting.plot_calibration_curve_by_segment(
        data=sample_df,
        group_var="segment_A_0",
        score_col="prediction",
        label_col="label",
        num_bins=5,
        sample_weight_col="weights",
    )

    pd.testing.assert_frame_equal(sample_df, df_original)


def test_plot_calibration_curve_by_segment_with_integer_groups(rng):
    n_samples = 100
    expected_groups = [1, 2, 3]

    df = pd.DataFrame(
        {
            "group": rng.choice(expected_groups, n_samples),
            "score": rng.rand(n_samples),
            "label": rng.randint(0, 2, n_samples),
        }
    )

    fig = plotting.plot_calibration_curve_by_segment(
        data=df,
        group_var="group",
        score_col="score",
        label_col="label",
        num_bins=5,
    )

    assert fig is not None

    # Verify that each group's subplot contains data points
    # If group filtering failed (due to string/int type mismatch), subplots would be empty
    scatter_data = [trace for trace in fig.data if isinstance(trace, go.Scatter)]

    # Each group should have calibrated and/or miscalibrated points
    # Extract group labels from scatter names (format: "group_value (calibrated/miscalibrated)")
    groups_with_data = set()
    for trace in scatter_data:
        if trace.name and len(trace.x) > 0:
            # Extract group value from name like "1 (calibrated)" -> "1"
            group_str = trace.name.split(" (")[0]
            groups_with_data.add(group_str)

    # Verify all expected groups have data in their subplots
    expected_group_strs = {str(g) for g in expected_groups}
    assert groups_with_data == expected_group_strs, (
        f"Expected all groups {expected_group_strs} to have data, "
        f"but only {groups_with_data} had data points. "
        "This suggests group filtering failed due to type mismatch."
    )
