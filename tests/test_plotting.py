# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

import numpy as np
import pandas as pd
import pytest

from multicalibration import metrics, plotting


@pytest.fixture
def rng():
    return np.random.RandomState(42)


def test_plot_segment_calibration_errors_does_not_raise_errors_with_valid_inputs(rng):
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

    df = df.astype(
        {
            "prediction": "float32",
            "label": "int32",
            **{f"segment_A_{t}": "int32" for t in range(n_cat_fts)},
            **{f"segment_B_{t}": "float64" for t in range(n_num_fts)},
            "weights": "float64",
        }
    )

    categorical_segment_columns = [f"segment_A_{t}" for t in range(n_cat_fts)]
    numerical_segment_columns = [f"segment_B_{t}" for t in range(n_num_fts)]

    mce = metrics.MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=categorical_segment_columns,
        numerical_segment_columns=numerical_segment_columns,
        weight_column="weights",
    )
    try:
        _ = plotting.plot_segment_calibration_errors(
            mce=mce, quantity="segment_ecces_sigma_scale"
        )
    except Exception:
        raise AssertionError(
            "The function plot_segment_calibration_errors from plotting.py failed to generate a plot."
        )
