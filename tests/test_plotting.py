# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

import numpy as np
import pandas as pd
import pytest

from multicalibration import methods, metrics, plotting


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


def test_plot_segment_calibration_errors_does_not_raise_errors_with_valid_inputs(
    sample_df,
):
    categorical_segment_columns = [f"segment_A_{t}" for t in range(2)]
    numerical_segment_columns = [f"segment_B_{t}" for t in range(2)]

    mce = metrics.MulticalibrationError(
        df=sample_df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
        weight_column="weights",
    )

    fig = plotting.plot_segment_calibration_errors(
        mce=mce, quantity="segment_ecces_sigma_scale"
    )
    assert fig is not None


def test_plot_segment_calibration_errors_with_different_quantities(sample_df):
    categorical_segment_columns = [f"segment_A_{t}" for t in range(2)]
    numerical_segment_columns = [f"segment_B_{t}" for t in range(2)]

    mce = metrics.MulticalibrationError(
        df=sample_df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
        weight_column="weights",
    )

    for quantity in ["segment_ecces", "segment_p_values", "segment_sigmas"]:
        fig = plotting.plot_segment_calibration_errors(mce=mce, quantity=quantity)
        assert fig is not None


def test_plot_segment_calibration_errors_raises_on_invalid_quantity(sample_df):
    categorical_segment_columns = [f"segment_A_{t}" for t in range(2)]

    mce = metrics.MulticalibrationError(
        df=sample_df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=categorical_segment_columns,
        weight_column="weights",
    )

    with pytest.raises(ValueError, match="Invalid quantity"):
        plotting.plot_segment_calibration_errors(mce=mce, quantity="invalid_quantity")


def test_plot_global_calibration_curve_basic(sample_df):
    fig = plotting.plot_global_calibration_curve(
        data=sample_df,
        score_col="prediction",
        label_col="label",
        num_bins=10,
    )

    assert fig is not None


def test_plot_global_calibration_curve_with_weights(sample_df):
    fig = plotting.plot_global_calibration_curve(
        data=sample_df,
        score_col="prediction",
        label_col="label",
        num_bins=10,
        sample_weight_col="weights",
    )

    assert fig is not None


def test_plot_global_calibration_curve_equisized_binning(sample_df):
    fig = plotting.plot_global_calibration_curve(
        data=sample_df,
        score_col="prediction",
        label_col="label",
        num_bins=10,
        binning_method="equisized",
    )

    assert fig is not None


def test_plot_global_calibration_curve_invalid_binning_raises_error(sample_df):
    with pytest.raises(AssertionError):
        plotting.plot_global_calibration_curve(
            data=sample_df,
            score_col="prediction",
            label_col="label",
            binning_method="invalid",
        )


def test_plot_global_calibration_curve_incomplete_cis(sample_df):
    fig = plotting.plot_global_calibration_curve(
        data=sample_df,
        score_col="prediction",
        label_col="label",
        plot_incomplete_cis=False,
    )

    assert fig is not None


def test_plot_calibration_curve_by_segment_basic(sample_df):
    fig = plotting.plot_calibration_curve_by_segment(
        data=sample_df,
        group_var="segment_A_0",
        score_col="prediction",
        label_col="label",
        num_bins=5,
    )

    assert fig is not None


def test_plot_calibration_curve_by_segment_with_weights(sample_df):
    fig = plotting.plot_calibration_curve_by_segment(
        data=sample_df,
        group_var="segment_A_0",
        score_col="prediction",
        label_col="label",
        num_bins=5,
        sample_weight_col="weights",
    )

    assert fig is not None


def test_plot_calibration_curve_by_segment_equisized_binning(sample_df):
    fig = plotting.plot_calibration_curve_by_segment(
        data=sample_df,
        group_var="segment_A_0",
        score_col="prediction",
        label_col="label",
        num_bins=5,
        binning_method="equisized",
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


def test_plot_learning_curve_with_early_stopping(rng):
    n_samples = 200
    predictions = rng.rand(n_samples)
    labels = rng.randint(0, 2, n_samples)

    df = pd.DataFrame(
        {
            "prediction": predictions,
            "label": labels,
            "feature": rng.choice(["a", "b", "c"], n_samples),
        }
    )

    model = methods.MCGrad(
        num_rounds=3,
        early_stopping=True,
        patience=1,
        lightgbm_params={"max_depth": 2, "n_estimators": 2},
    )
    model.fit(
        df_train=df,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["feature"],
    )

    fig = plotting.plot_learning_curve(model)

    assert fig is not None


def test_plot_learning_curve_raises_without_early_stopping(rng):
    n_samples = 100
    predictions = rng.rand(n_samples)
    labels = rng.randint(0, 2, n_samples)

    df = pd.DataFrame(
        {
            "prediction": predictions,
            "label": labels,
            "feature": rng.choice(["a", "b"], n_samples),
        }
    )

    model = methods.MCGrad(
        num_rounds=2,
        early_stopping=False,
        lightgbm_params={"max_depth": 2, "n_estimators": 2},
    )
    model.fit(
        df_train=df,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["feature"],
    )

    with pytest.raises(
        ValueError,
        match="Learning curve can only be plotted for models that have been trained with early_stopping=True",
    ):
        plotting.plot_learning_curve(model)


def test_plot_learning_curve_with_show_all(rng):
    n_samples = 200
    predictions = rng.rand(n_samples)
    labels = rng.randint(0, 2, n_samples)

    df = pd.DataFrame(
        {
            "prediction": predictions,
            "label": labels,
            "feature": rng.choice(["a", "b", "c"], n_samples),
        }
    )

    model = methods.MCGrad(
        num_rounds=3,
        early_stopping=True,
        patience=1,
        save_training_performance=True,
        lightgbm_params={"max_depth": 2, "n_estimators": 2},
    )
    model.fit(
        df_train=df,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["feature"],
    )

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
