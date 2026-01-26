# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import scipy
import sklearn.metrics as skmetrics
from sklearn.model_selection import KFold, StratifiedKFold

from .. import _utils as utils, methods
from ..metrics import (
    _ScoreFunctionInterface,
    wrap_multicalibration_error_metric,
    wrap_sklearn_metric_func,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


def generate_test_data(n):
    return pd.DataFrame(
        {
            "City": np.array(["Paris", "Tokyo", "Amsterdam", "Paris", "Amsterdam"])[:n],
            "Gender": np.array(["Male", "Female", "Male", "Female", "Male"])[:n],
            "Prediction": np.array([0.1, 0.2, 0.3, 0.4, 0.5])[:n],
            "Label": np.array([0, 1, 0, 1, 0])[:n],
        }
    )


@pytest.mark.parametrize("num_rounds", [(1), (2), (6), (10), (16)])
@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
@pytest.mark.parametrize(
    "calibrator_kwargs",
    [
        {
            "early_stopping": False,
            "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
        },
        {
            "early_stopping": True,
            "early_stopping_use_crossvalidation": True,
            "n_folds": 2,
            "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
        },
    ],
)
def test_deserialized_mcgrad_fits_correct_num_rounds_when_no_early_stopping(
    num_rounds, calibrator_class, calibrator_kwargs
):
    df_train = generate_test_data(5)
    model = calibrator_class(
        num_rounds=num_rounds,
        **calibrator_kwargs,
    )
    model.fit(
        df_train=df_train,
        prediction_column_name="Prediction",
        label_column_name="Label",
        categorical_feature_column_names=["City", "Gender"],
    )
    if calibrator_kwargs["early_stopping"]:
        assert len(model.mr) <= num_rounds
    else:
        assert len(model.mr) == num_rounds
    serialized = model.serialize()
    deserialized = calibrator_class.deserialize(serialized)
    if calibrator_kwargs["early_stopping"]:
        assert len(deserialized.mr) <= num_rounds
    else:
        assert len(deserialized.mr) == num_rounds


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
@pytest.mark.parametrize(
    "calibrator_kwargs",
    [
        # Pass MCGrad params that minimize test runtime
        {
            "num_rounds": 2,
            "early_stopping": False,
            "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
        },
        {
            "num_rounds": 2,
            "early_stopping": True,
            "early_stopping_use_crossvalidation": True,
            "n_folds": 2,  # More than 2 folds would require more input data
            # Pass MCGrad params that minimize test runtime
            "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
        },
    ],
)
def test_mcgrad_serialize_deserialize_encode_categorical(
    calibrator_class, calibrator_kwargs
):
    df_train = generate_test_data(5)
    model = calibrator_class(**calibrator_kwargs)
    model.fit(
        df_train=df_train,
        prediction_column_name="Prediction",
        label_column_name="Label",
        categorical_feature_column_names=["City", "Gender"],
    )
    serialized = model.serialize()

    deserialized = calibrator_class.deserialize(serialized)

    df_test = generate_test_data(3)
    original_scores = model.predict(
        df=df_test,
        prediction_column_name="Prediction",
        categorical_feature_column_names=["City", "Gender"],
    )
    deserialized_scores = deserialized.predict(
        df=df_test,
        prediction_column_name="Prediction",
        categorical_feature_column_names=["City", "Gender"],
    )

    # Deserialized model should have an encoder and should be configured to use one
    assert hasattr(deserialized, "enc")
    assert deserialized.encode_categorical_variables
    # Deserialized model and original model should give the same results
    np.testing.assert_array_equal(deserialized_scores, original_scores)


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
@pytest.mark.parametrize(
    "calibrator_kwargs",
    [
        # Pass MCGrad params that minimize test runtime
        {
            "num_rounds": 2,
            "early_stopping": False,
            "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
        },
        {
            "num_rounds": 2,
            "early_stopping": True,
            "early_stopping_use_crossvalidation": True,
            "n_folds": 2,  # More than 2 folds would require more input data
            # Pass MCGrad params that minimize test runtime
            "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
        },
    ],
)
def test_mcgrad_serialize_deserialize_no_encode_categorical(
    calibrator_class, calibrator_kwargs
):
    df_train = generate_test_data(5)
    city_codebook = {"Paris": 0, "Tokyo": 1, "Amsterdam": 2, "Copenhagen": 3}
    gender_codebook = {
        "Male": 0,
        "Female": 1,
    }
    df_train["City"] = df_train["City"].apply(lambda x: city_codebook[x])
    df_train["Gender"] = df_train["Gender"].apply(lambda x: gender_codebook[x])

    calibrator_kwargs = calibrator_kwargs.copy()
    calibrator_kwargs["encode_categorical_variables"] = False
    model = calibrator_class(**calibrator_kwargs)
    model.fit(
        df_train=df_train,
        prediction_column_name="Prediction",
        label_column_name="Label",
        categorical_feature_column_names=["City", "Gender"],
    )
    serialized = model.serialize()

    deserialized = calibrator_class.deserialize(serialized)

    df_test = generate_test_data(3)
    df_test["City"] = df_test["City"].apply(lambda x: city_codebook[x])
    df_test["Gender"] = df_test["Gender"].apply(lambda x: gender_codebook[x])
    original_scores = model.predict(
        df=df_test,
        prediction_column_name="Prediction",
        categorical_feature_column_names=["City", "Gender"],
    )
    deserialized_scores = deserialized.predict(
        df=df_test,
        prediction_column_name="Prediction",
        categorical_feature_column_names=["City", "Gender"],
    )
    # Deserialized model should not have an encoder
    assert deserialized.enc is None
    assert not deserialized.encode_categorical_variables
    # Deserialized model and original model should give the same results
    np.testing.assert_array_equal(deserialized_scores, original_scores)


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
@pytest.mark.parametrize("max_num_rounds", [(1), (2), (6), (10), (16)])
def test_deserialized_mcgrad_has_at_most_max_num_rounds(
    max_num_rounds, calibrator_class
):
    df_train = pd.DataFrame(
        {
            "City": ["Paris", "Tokyo", "Amsterdam", "Paris", "Amsterdam"],
            "Gender": ["Male", "Female", "Male", "Female", "Male"],
            "Prediction": [0.1, 0.2, 0.3, 0.4, 0.5],
            "Label": [0, 1, 0, 1, 0],
        }
    )
    model = calibrator_class(
        early_stopping=True,
        early_stopping_use_crossvalidation=True,
        num_rounds=max_num_rounds,
        lightgbm_params={"max_depth": 2, "n_estimators": 2},
        n_folds=2,
    )
    model.fit(
        df_train=df_train,
        prediction_column_name="Prediction",
        label_column_name="Label",
        categorical_feature_column_names=["City", "Gender"],
    )
    assert model.num_rounds <= max_num_rounds
    serialized = model.serialize()
    deserialized = calibrator_class.deserialize(serialized)
    assert len(deserialized.mr) == len(model.mr)


@pytest.mark.parametrize(
    "calibrator_class, calibrator_kwargs",
    [
        (methods.PlattScaling, {}),
        (methods.IsotonicRegression, {}),
        # Pass MCGrad params that minimize test runtime
        (
            methods.MCGrad,
            {"num_rounds": 2, "lightgbm_params": {"max_depth": 2, "n_estimators": 2}},
        ),
        (
            methods.MCGrad,
            {
                "num_rounds": 2,
                "early_stopping": True,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
        (
            methods.MCGrad,
            {
                "num_rounds": 2,
                "early_stopping": False,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
        (methods.IdentityCalibrator, {}),
        (methods.MultiplicativeAdjustment, {}),
        (methods.AdditiveAdjustment, {}),
        (
            methods.RegressionMCGrad,
            {
                "num_rounds": 2,
                "early_stopping": True,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
        (
            methods.RegressionMCGrad,
            {
                "num_rounds": 2,
                "early_stopping": False,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
    ],
)
def test_fit_transform_with_train_col_identical_to_fit_then_predict(
    calibrator_class, calibrator_kwargs, rng
):
    df = pd.DataFrame(
        {
            "prediction": np.linspace(0, 1, 100),
            "label": rng.choice([0, 1], 100),
            "is_train_set": np.concatenate([np.full(50, False), np.full(50, True)]),
        }
    )
    result_fit_transform = calibrator_class(**calibrator_kwargs).fit_transform(
        df,
        prediction_column_name="prediction",
        label_column_name="label",
        is_train_set_col_name="is_train_set",
    )

    calibrator = calibrator_class(**calibrator_kwargs)
    calibrator.fit(df[df.is_train_set], "prediction", "label")
    result_fit_predict = calibrator.predict(df[~df.is_train_set], "prediction")

    assert np.allclose(result_fit_transform[~df.is_train_set], result_fit_predict), (
        "fit_transform does not give the same result as fit followed by predict"
    )


@pytest.mark.parametrize(
    "calibrator_class, calibrator_kwargs",
    [
        (methods.PlattScaling, {}),
        (methods.IsotonicRegression, {}),
        # Pass MCGrad params that minimize test runtime
        (
            methods.MCGrad,
            {"num_rounds": 2, "lightgbm_params": {"max_depth": 2, "n_estimators": 2}},
        ),
        (
            methods.MCGrad,
            {
                "num_rounds": 2,
                "early_stopping": True,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
        (
            methods.MCGrad,
            {
                "num_rounds": 2,
                "early_stopping": False,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
        (methods.IdentityCalibrator, {}),
        (methods.MultiplicativeAdjustment, {}),
        (methods.AdditiveAdjustment, {}),
        (
            methods.RegressionMCGrad,
            {
                "num_rounds": 2,
                "early_stopping": True,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
        (
            methods.RegressionMCGrad,
            {
                "num_rounds": 2,
                "early_stopping": False,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
    ],
)
def test_fit_transform_no_train_col_identical_to_fit_then_predict(
    calibrator_class, calibrator_kwargs, rng
):
    df = pd.DataFrame(
        {
            "prediction": np.linspace(0, 1, 100),
            "label": rng.choice([0, 1], 100),
        }
    )
    result_fit_transform = calibrator_class(**calibrator_kwargs).fit_transform(
        df, prediction_column_name="prediction", label_column_name="label"
    )

    calibrator = calibrator_class(**calibrator_kwargs)
    calibrator.fit(df, "prediction", "label")
    result_fit_predict = calibrator.predict(df, "prediction")

    assert np.allclose(result_fit_transform, result_fit_predict), (
        "fit_transform does not give the same result as fit followed by predict"
    )


def test_segmentwise_calibrator_raises_when_incompatible_calibrator_kwargs_are_passed():
    with pytest.raises(ValueError):
        methods.SegmentwiseCalibrator(methods.PlattScaling, {"non_existent_arg": 42})


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.PlattScaling,
        methods.IsotonicRegression,
        methods.IdentityCalibrator,
        methods.MultiplicativeAdjustment,
        methods.AdditiveAdjustment,
    ],
)
def test_segmentwise_calibrator_equivalent_to_calibrator_per_segment(
    calibrator_class, rng
):
    df = pd.DataFrame(index=range(100))
    df["label"] = rng.choice([0, 1], size=len(df))
    # Create an 'uncalibrated_score' column that is correlated with 'label'
    # We thereofre add some random noise to the 'label' values to create the 'uncalibrated_score' values
    # We use the logistic function to map the scores to the (0, 1) interval
    noise = rng.normal(0, 0.1, size=len(df))
    df["uncalibrated_score"] = 1 / (1 + np.exp(-(df["label"] + noise)))

    df["is_train"] = rng.choice([0, 1], size=len(df)).astype(bool)
    df["segment"] = rng.choice(["A", "B"], size=len(df))

    segmentwise_calibrator = methods.SegmentwiseCalibrator(calibrator_class)
    df["segmentwise_results"] = segmentwise_calibrator.fit_transform(
        df,
        prediction_column_name="uncalibrated_score",
        label_column_name="label",
        is_train_set_col_name="is_train",
        categorical_feature_column_names=["segment"],
    )

    # Compare to the results per segment individually
    for segment in ["A", "B"]:
        segment_df = df[df["segment"] == segment]
        segment_results = calibrator_class().fit_transform(
            segment_df,
            prediction_column_name="uncalibrated_score",
            label_column_name="label",
            is_train_set_col_name="is_train",
        )
        assert np.allclose(
            df[df["segment"] == segment]["segmentwise_results"].values, segment_results
        )
        assert np.array_equal(
            df[df["segment"] == segment]["segmentwise_results"].index, segment_df.index
        )


def test_segmentwise_calibrator_with_additive_adjustment_gives_expected_results(rng):
    df = pd.DataFrame(index=range(1000))
    df["label"] = rng.choice(range(1000), size=len(df))
    df["uncalibrated_score"] = rng.choice(range(1000), size=len(df)).astype(float)
    df["segment_1"] = rng.choice(["A", "B"], size=len(df))
    df["segment_2"] = rng.choice(["C", "D"], size=len(df))

    segmentwise_calibrator = methods.SegmentwiseCalibrator(
        methods.AdditiveAdjustment, calibrator_kwargs={"clip_to_zero_one": False}
    )
    df["segmentwise_results"] = segmentwise_calibrator.fit_transform(
        df,
        prediction_column_name="uncalibrated_score",
        label_column_name="label",
        categorical_feature_column_names=["segment_1", "segment_2"],
    )
    prediction_means = df.groupby(by=["segment_1", "segment_2"])["uncalibrated_score"]
    label_means = df.groupby(by=["segment_1", "segment_2"])["label"]
    df["mean_adjusted"] = df["uncalibrated_score"]
    for segment_1 in ["A", "B"]:
        for segment_2 in ["C", "D"]:
            df.loc[
                (df.segment_1 == segment_1) & (df.segment_2 == segment_2),
                "mean_adjusted",
            ] += (
                label_means.get_group((segment_1, segment_2)).mean()
                - prediction_means.get_group((segment_1, segment_2)).mean()
            )

    assert np.allclose(df["segmentwise_results"].values, df["mean_adjusted"].values)


@pytest.mark.parametrize(
    "scores, labels, expected_multiplier",
    [
        ([0.2, 0.4, 0.6, 0.8], [0, 1, 0, 1], 1),
        ([0.2, 0.4, 0.6, 0.8], [1, 1, 1, 1], 2),
        ([0.2, 0.4, 0.6, 0.8], [0, 0, 0, 1], 0.5),
    ],
)
def test_multiplicative_adjustment_gives_expected_result(
    scores, labels, expected_multiplier
):
    df = pd.DataFrame({"prediction": scores, "label": labels})
    calibrator = methods.MultiplicativeAdjustment(clip_to_zero_one=False)
    calibrator.fit(df, "prediction", "label")

    # Check that the multiplier is correctly calculated
    assert calibrator.multiplier == expected_multiplier
    np.testing.assert_array_equal(
        calibrator.predict(df, "prediction"),
        df["prediction"] * expected_multiplier,
    )


@pytest.mark.parametrize(
    "scores, labels, expected_multiplier, expected_predictions",
    [
        ([0.2, 0.4, 0.6, 0.8], [0, 1, 0, 1], 1, [0.2, 0.4, 0.6, 0.8]),
        ([0.2, 0.4, 0.6, 0.8], [1, 1, 1, 1], 2, [0.4, 0.8, 1.0, 1.0]),
        ([0.2, 0.4, 0.6, 0.8], [0, 0, 0, 1], 0.5, [0.1, 0.2, 0.3, 0.4]),
    ],
)
def test_multiplicative_adjustment_with_clip_gives_expected_result(
    scores, labels, expected_multiplier, expected_predictions
):
    df = pd.DataFrame({"prediction": scores, "label": labels})
    calibrator = methods.MultiplicativeAdjustment(clip_to_zero_one=True)
    calibrator.fit(df, "prediction", "label")

    # Check that the multiplier is correctly calculated
    assert calibrator.multiplier == expected_multiplier
    np.testing.assert_allclose(
        calibrator.predict(df, "prediction"),
        expected_predictions,
    )


@pytest.mark.parametrize(
    "scores, labels, expected_offset",
    [
        ([0.2, 0.4, 0.6, 0.8], [0, 1, 0, 1], 0),
        ([0.2, 0.4, 0.6, 0.8], [1, 1, 1, 1], 0.5),
        ([0.2, 0.4, 0.6, 0.8], [0, 0, 0, 1], -0.25),
    ],
)
def test_additive_adjustment_calibrator_gives_expected_result(
    scores, labels, expected_offset
):
    # Create a simple DataFrame for testing
    df = pd.DataFrame({"prediction": scores, "label": labels})
    calibrator = methods.AdditiveAdjustment(clip_to_zero_one=False)
    calibrator.fit(df, "prediction", "label")

    # Check that the offset is correctly calculated
    assert calibrator.offset == expected_offset
    np.testing.assert_array_equal(
        calibrator.predict(df, "prediction"),
        df["prediction"] + expected_offset,
    )


@pytest.mark.parametrize(
    "scores, labels, expected_offset, expected_predictions",
    [
        ([0.2, 0.4, 0.6, 0.8], [0, 1, 0, 1], 0, [0.2, 0.4, 0.6, 0.8]),
        ([0.2, 0.4, 0.6, 0.8], [1, 1, 1, 1], 0.5, [0.7, 0.9, 1.0, 1.0]),
        ([0.2, 0.4, 0.6, 0.8], [0, 0, 0, 1], -0.25, [0, 0.15, 0.35, 0.55]),
    ],
)
def test_additive_adjustment_calibrator_with_clip_gives_expected_result(
    scores, labels, expected_offset, expected_predictions
):
    # Create a simple DataFrame for testing
    df = pd.DataFrame({"prediction": scores, "label": labels})
    calibrator = methods.AdditiveAdjustment(clip_to_zero_one=True)
    calibrator.fit(df, "prediction", "label")

    # Check that the offset is correctly calculated
    assert calibrator.offset == expected_offset
    np.testing.assert_allclose(
        calibrator.predict(df, "prediction"),
        expected_predictions,
    )


@pytest.mark.parametrize(
    "calibrator_class, calibrator_kwargs",
    [
        (
            methods.MCGrad,
            {
                "num_rounds": 2,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
                "early_stopping": False,
            },
        ),
        (
            methods.RegressionMCGrad,
            {
                "num_rounds": 2,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
                "early_stopping": False,
            },
        ),
        (methods.PlattScaling, {}),
        (methods.IsotonicRegression, {}),
        (methods.MultiplicativeAdjustment, {}),
        (methods.AdditiveAdjustment, {}),
        (methods.IdentityCalibrator, {}),
        (methods.PlattScalingWithFeatures, {}),
    ],
)
def test_calibration_methods_use_weight_column_correctly(
    calibrator_class, calibrator_kwargs
):
    # Create an unweighted dataset with duplicates: positive 0.2 occurs 3x and negative 0.6 occurs 2x
    df_train_unweighted = pd.DataFrame(
        {
            "prediction": [0.1, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.8],
            "label": [0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
        }
    )
    # Create a weighted dataset -> duplicates turn into weights
    df_train_weighted = pd.DataFrame(
        {
            "prediction": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
            "label": [0, 1, 0, 0, 0, 0, 1],
            "weight": [1, 3, 1, 1, 1, 2, 1],
        }
    )
    model_unweighted = calibrator_class(**calibrator_kwargs)
    model_unweighted.fit(df_train_unweighted, "prediction", "label")

    model_weighted = calibrator_class(**calibrator_kwargs)
    model_weighted.fit(df_train_weighted, "prediction", "label", "weight")
    # Adding this assertion because there were test cases earlier where a model passed
    # just because in both cases it was fit with 0 rounds
    # TODO: understand why early stopping is not equivalent with/without weights
    if isinstance(model_unweighted, methods._BaseMCGrad):
        assert len(model_weighted.mr) > 0
        assert len(model_unweighted.mr) > 0

    df_test = pd.DataFrame({"prediction": [0.15, 0.25, 0.35, 0.45, 0.55]})
    predictions_unweighted = model_unweighted.predict(df_test, "prediction")
    predictions_weighted = model_weighted.predict(df_test, "prediction")
    assert np.allclose(predictions_unweighted, predictions_weighted)


@pytest.mark.parametrize(
    "calibrator_class, predictions",
    [
        (methods.MCGrad, np.array([-0.001, 0.1])),
        (methods.MCGrad, np.array([0.1, 1.0001])),
        (methods.MCGrad, np.array([-0.001, 1.0001])),
        (methods.RegressionMCGrad, np.array([-0.001, float("inf")])),
        (methods.RegressionMCGrad, np.array([0.1, -1 * float("inf")])),
        (methods.RegressionMCGrad, np.array([-0.001, None])),
    ],
)
def test_mcgrad_raises_when_predictions_invalid(calibrator_class, predictions):
    df = pd.DataFrame({"prediction": predictions, "label": np.array([0, 1])})
    mcgrad = calibrator_class()
    with pytest.raises(ValueError):
        mcgrad.fit(
            df_train=df, prediction_column_name="prediction", label_column_name="label"
        )


@pytest.mark.parametrize(
    "calibrator_class, calibrator_kwargs",
    [
        (
            methods.MCGrad,
            # Pass MCGrad params that minimize test runtime
            {
                "num_rounds": 2,
                "early_stopping": False,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
        (
            methods.MCGrad,
            {
                "num_rounds": 2,
                "early_stopping": True,
                "early_stopping_use_crossvalidation": True,
                "n_folds": 2,  # More than 2 folds would require more input data
                # Pass MCGrad params that minimize test runtime
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
    ],
)
def test_mcgrad_runs_without_errors_when_scores_in_zero_one(
    calibrator_class, calibrator_kwargs
):
    df = pd.DataFrame(
        {
            "prediction": np.array([0.001, 0.999, 0.01, 0.99]),
            "label": np.array([0, 1, 0, 1]),
        }
    )
    mcgrad = calibrator_class(**calibrator_kwargs)
    mcgrad.fit(
        df_train=df, prediction_column_name="prediction", label_column_name="label"
    )


@pytest.mark.parametrize(
    "calibrator_class, calibrator_kwargs",
    [
        (
            methods.MCGrad,
            # Pass MCGrad params that minimize test runtime
            {
                "num_rounds": 2,
                "early_stopping": False,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
        (
            methods.MCGrad,
            {
                "num_rounds": 2,
                "early_stopping": True,
                "early_stopping_use_crossvalidation": True,
                "n_folds": 2,  # More than 2 folds would require more input data
                # Pass MCGrad params that minimize test runtime
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
    ],
)
def test_mcgrad_runs_without_errors_when_scores_are_exactly_zero_or_one(
    calibrator_class, calibrator_kwargs
):
    df = pd.DataFrame(
        {"prediction": np.array([0, 1, 0, 1]), "label": np.array([0, 1, 0, 1])}
    )
    mcgrad = calibrator_class(**calibrator_kwargs)
    mcgrad.fit(
        df_train=df, prediction_column_name="prediction", label_column_name="label"
    )


@pytest.mark.parametrize("calibrator_class", [methods.MCGrad, methods.RegressionMCGrad])
@pytest.mark.parametrize("num_rounds", [(1), (2), (4)])
def test_mcgrad_predict_returns_correct_number_of_rounds_and_consistent_final_prediction_when_no_early_stopping(
    num_rounds, calibrator_class, rng
):
    n = 10
    predictions = rng.uniform(low=0.0, high=1.0, size=n)
    labels = scipy.stats.binom.rvs(1, predictions, size=n, random_state=rng)

    df = pd.DataFrame({"prediction": predictions, "label": labels})
    calibrator = calibrator_class(
        num_rounds=num_rounds,
        early_stopping=False,
        lightgbm_params={"max_depth": 2, "n_estimators": 2},
    )
    calibrator.fit(df, "prediction", "label")

    # Check that the multiplier is correctly calculated
    rounds_predictions = calibrator.predict(df, "prediction", return_all_rounds=True)
    final_predictions = calibrator.predict(df, "prediction", return_all_rounds=False)
    assert rounds_predictions.shape == (num_rounds, len(df))
    assert final_predictions.shape == (len(df),)
    assert np.array_equal(rounds_predictions[-1], final_predictions)


@pytest.mark.parametrize("calibrator_class", [methods.MCGrad, methods.RegressionMCGrad])
@pytest.mark.parametrize("num_rounds", [(1), (2), (4)])
def test_mcgrad_predict_returns_correct_number_of_rounds_and_consistent_final_prediction_when_early_stopping_is_used(
    num_rounds, calibrator_class, rng
):
    n = 10
    predictions = rng.uniform(low=0.0, high=1.0, size=n)
    labels = scipy.stats.binom.rvs(1, predictions, size=n, random_state=rng)

    df = pd.DataFrame({"prediction": predictions, "label": labels})
    calibrator = calibrator_class(
        num_rounds=num_rounds,
        early_stopping=True,
        early_stopping_use_crossvalidation=True,
        n_folds=2,
        lightgbm_params={"max_depth": 2, "n_estimators": 2},
    )
    calibrator.fit(df, "prediction", "label")

    # Check that the multiplier is correctly calculated
    rounds_predictions = calibrator.predict(df, "prediction", return_all_rounds=True)
    final_predictions = calibrator.predict(df, "prediction", return_all_rounds=False)
    assert rounds_predictions.shape == (max(1, len(calibrator.mr)), len(df))
    assert final_predictions.shape == (len(df),)
    assert np.array_equal(rounds_predictions[-1], final_predictions)


@pytest.mark.parametrize("calibrator_class", [methods.MCGrad, methods.RegressionMCGrad])
def test_mcgrad_predict_returns_correct_number_of_rounds_and_consistent_final_prediction(
    calibrator_class,
    rng,
):
    n = 10
    max_num_rounds = 2

    predictions = rng.uniform(low=0.0, high=1.0, size=n)
    labels = scipy.stats.binom.rvs(1, predictions, size=n, random_state=rng)

    df = pd.DataFrame({"prediction": predictions, "label": labels})
    calibrator = calibrator_class(
        num_rounds=max_num_rounds,
        early_stopping=True,
        early_stopping_use_crossvalidation=True,
        lightgbm_params={"max_depth": 2, "n_estimators": 2},
        n_folds=2,
    )
    calibrator.fit(df, "prediction", "label")

    # Check that the multiplier is correctly calculated
    rounds_predictions = calibrator.predict(df, "prediction", return_all_rounds=True)
    final_predictions = calibrator.predict(df, "prediction", return_all_rounds=False)
    assert rounds_predictions.shape[0] <= max_num_rounds
    assert rounds_predictions.shape[1] == len(df)
    assert final_predictions.shape == (len(df),)
    assert np.array_equal(rounds_predictions[-1], final_predictions)


@pytest.mark.parametrize(
    "calibrator_class, input_params, expected_params, objective",
    [
        # if nothing is passed we take all defaults
        (
            methods.MCGrad,
            None,
            methods.MCGrad.DEFAULT_HYPERPARAMS["lightgbm_params"],
            "binary",
        ),
        # if only one is passed we take the default for the others
        (
            methods.MCGrad,
            {"max_depth": -1},
            {
                k: v if k != "max_depth" else -1
                for k, v in methods.MCGrad.DEFAULT_HYPERPARAMS[
                    "lightgbm_params"
                ].items()
            },
            "binary",
        ),
        # if a parameter is passed that is not present in the default we take the default + the passed parameter
        (
            methods.MCGrad,
            {"_OTHER": -1},
            methods.MCGrad.DEFAULT_HYPERPARAMS["lightgbm_params"] | {"_OTHER": -1},
            "binary",
        ),
        # if nothing is passed we take all defaults
        (
            methods.RegressionMCGrad,
            None,
            methods.RegressionMCGrad.DEFAULT_HYPERPARAMS["lightgbm_params"],
            "regression",
        ),
        # if only one is passed we take the default for the others
        (
            methods.RegressionMCGrad,
            {"max_depth": -1},
            {
                k: v if k != "max_depth" else -1
                for k, v in methods.RegressionMCGrad.DEFAULT_HYPERPARAMS[
                    "lightgbm_params"
                ].items()
            },
            "regression",
        ),
        # if a parameter is passed that is not present in the default we take the default + the passed parameter
        (
            methods.RegressionMCGrad,
            {"_OTHER": -1},
            methods.RegressionMCGrad.DEFAULT_HYPERPARAMS["lightgbm_params"]
            | {"_OTHER": -1},
            "regression",
        ),
    ],
)
def test_that_default_lightgbm_params_are_applied_correctly_for_mcgrad(
    calibrator_class, input_params, expected_params, objective
):
    model = calibrator_class(lightgbm_params=input_params)

    # These are always added in the MCGrad init
    expected_params |= {
        "objective": objective,
        "deterministic": True,
        "verbosity": -1,
    }

    assert "seed" in model.lightgbm_params
    assert isinstance(model.lightgbm_params["seed"], int)

    for key, value in expected_params.items():
        assert model.lightgbm_params[key] == value


@pytest.mark.parametrize(
    "calibrator_class, input_params, expected_params, objective",
    [
        # if nothing is passed we take all defaults
        (
            methods.MCGrad,
            None,
            methods.MCGrad.DEFAULT_HYPERPARAMS["lightgbm_params"],
            "binary",
        ),
        # if only one is passed we take the default for the others
        (
            methods.MCGrad,
            {"max_depth": -1},
            {
                k: v if k != "max_depth" else -1
                for k, v in methods.MCGrad.DEFAULT_HYPERPARAMS[
                    "lightgbm_params"
                ].items()
            },
            "binary",
        ),
        # if a parameter is passed that is not present in the default we take the default + the passed parameter
        (
            methods.MCGrad,
            {"_OTHER": -1},
            methods.MCGrad.DEFAULT_HYPERPARAMS["lightgbm_params"] | {"_OTHER": -1},
            "binary",
        ),
        # if nothing is passed we take all defaults
        (
            methods.RegressionMCGrad,
            None,
            methods.RegressionMCGrad.DEFAULT_HYPERPARAMS["lightgbm_params"],
            "regression",
        ),
        # if only one is passed we take the default for the others
        (
            methods.RegressionMCGrad,
            {"max_depth": -1},
            {
                k: v if k != "max_depth" else -1
                for k, v in methods.RegressionMCGrad.DEFAULT_HYPERPARAMS[
                    "lightgbm_params"
                ].items()
            },
            "regression",
        ),
        # if a parameter is passed that is not present in the default we take the default + the passed parameter
        (
            methods.RegressionMCGrad,
            {"_OTHER": -1},
            methods.RegressionMCGrad.DEFAULT_HYPERPARAMS["lightgbm_params"]
            | {"_OTHER": -1},
            "regression",
        ),
    ],
)
def test_that_lightgbm_params_are_applied_correctly_after_resetting_them(
    calibrator_class, input_params, expected_params, objective
):
    model = calibrator_class()
    model._set_lightgbm_params(lightgbm_params=input_params)

    # These are always added in the MCGrad init
    expected_params |= {
        "objective": objective,
        "deterministic": True,
        "verbosity": -1,
    }

    assert "seed" in model.lightgbm_params
    assert isinstance(model.lightgbm_params["seed"], int)

    for key, value in expected_params.items():
        assert model.lightgbm_params[key] == value


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_with_random_generator_as_random_state(calibrator_class):
    rng = np.random.default_rng(42)
    model = calibrator_class(random_state=rng)

    assert model._rng is rng


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_reproducibility_with_same_random_state(calibrator_class):
    model1 = calibrator_class(random_state=42)
    model2 = calibrator_class(random_state=42)

    seed1 = model1.lightgbm_params["seed"]
    seed2 = model2.lightgbm_params["seed"]

    assert seed1 == seed2, "Same random_state should produce same initial seed"

    next_seed1 = model1._next_seed()
    next_seed2 = model2._next_seed()
    assert next_seed1 == next_seed2, (
        "Same random_state should produce same seed sequence"
    )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_different_random_states_produce_different_seeds(calibrator_class):
    model1 = calibrator_class(random_state=42)
    model2 = calibrator_class(random_state=123)

    seed1 = model1.lightgbm_params["seed"]
    seed2 = model2.lightgbm_params["seed"]

    assert seed1 != seed2, "Different random_state should produce different seeds"


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_raises_when_custom_score_func_without_minimize_score(calibrator_class):
    custom_score_func = wrap_sklearn_metric_func(skmetrics.roc_auc_score)
    with pytest.raises(
        ValueError,
        match="`early_stopping_minimize_score` has to be set",
    ):
        calibrator_class(
            early_stopping=True,
            early_stopping_score_func=custom_score_func,
            early_stopping_minimize_score=None,
        )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_raises_when_patience_set_without_early_stopping(calibrator_class):
    with pytest.raises(
        ValueError,
        match="`patience` must be None when argument `early_stopping` is disabled",
    ):
        calibrator_class(
            early_stopping=False,
            patience=5,
        )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_raises_when_crossvalidation_set_without_early_stopping(
    calibrator_class,
):
    with pytest.raises(
        ValueError,
        match="`early_stopping_use_crossvalidation` must be None when `early_stopping` is disabled",
    ):
        calibrator_class(
            early_stopping=False,
            early_stopping_use_crossvalidation=True,
        )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_raises_when_score_func_set_without_early_stopping(calibrator_class):
    custom_score_func = wrap_sklearn_metric_func(skmetrics.roc_auc_score)
    with pytest.raises(
        ValueError,
        match="`score_func` must be None when `early_stopping` is disabled",
    ):
        calibrator_class(
            early_stopping=False,
            early_stopping_score_func=custom_score_func,
            early_stopping_minimize_score=False,
        )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_raises_when_minimize_score_without_custom_score_func_early_stopping_enabled(
    calibrator_class,
):
    with pytest.raises(
        ValueError,
        match="`early_stopping_minimize_score` is only relevant when using a custom score function",
    ):
        calibrator_class(
            early_stopping=True,
            early_stopping_score_func=None,
            early_stopping_minimize_score=True,
        )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_raises_when_n_folds_set_with_holdout(calibrator_class):
    with pytest.raises(
        ValueError,
        match="`n_folds` must be None when `early_stopping_use_crossvalidation` is disabled",
    ):
        calibrator_class(
            early_stopping=True,
            early_stopping_use_crossvalidation=False,
            n_folds=5,
        )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_default_num_rounds_when_early_stopping_disabled(calibrator_class):
    model = calibrator_class(early_stopping=False, num_rounds=None)
    assert model.num_rounds == calibrator_class.NUM_ROUNDS_DEFAULT_NO_EARLY_STOPPING


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_raises_when_num_rounds_in_lightgbm_params(calibrator_class):
    with pytest.raises(
        ValueError,
        match="Avoid using `num_rounds` in `lightgbm_params`",
    ):
        calibrator_class(
            lightgbm_params={"num_rounds": 10},
        )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_feature_importance_raises_when_not_fit(calibrator_class):
    model = calibrator_class()
    with pytest.raises(ValueError, match="Model has not been fit yet"):
        model.feature_importance()


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_performance_metrics_raises_when_not_fit_with_early_stopping(
    calibrator_class,
):
    model = calibrator_class(early_stopping=False)
    with pytest.raises(
        ValueError,
        match="Performance metrics are only available after the model has been fit with `early_stopping=True`",
    ):
        _ = model.performance_metrics


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_monotone_t_constraint_applied_correctly(calibrator_class, rng):
    df_train = pd.DataFrame(
        {
            "cat_feature": rng.choice(["A", "B", "C"], 50),
            "num_feature": rng.rand(50),
            "prediction": rng.rand(50),
            "label": rng.randint(0, 2, 50),
        }
    )

    model = calibrator_class(
        monotone_t=True,
        early_stopping=False,
        num_rounds=1,
        lightgbm_params={"max_depth": 2, "n_estimators": 2},
    )
    model.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
    )

    assert model.monotone_t is True
    assert len(model.mr) == 1


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_get_lgbm_params_with_monotone_t(calibrator_class):
    model = calibrator_class(monotone_t=True)

    x = np.array([[1, 2, 3], [4, 5, 6]])

    lgbm_params = model._get_lgbm_params(x)

    assert "monotone_constraints" in lgbm_params
    expected_constraints = [0, 0, 1]
    assert lgbm_params["monotone_constraints"] == expected_constraints


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_extract_features_raises_when_encoder_not_fit(calibrator_class):
    model = calibrator_class(encode_categorical_variables=True)

    df = pd.DataFrame({"cat_feature": ["A", "B", "C"]})

    with pytest.raises(ValueError, match="Fit has to be called before encoder"):
        model._extract_features(
            df=df,
            categorical_feature_column_names=["cat_feature"],
            numerical_feature_column_names=None,
            is_fit_phase=False,
        )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_feature_importance_returns_correct_dataframe(calibrator_class, rng):
    df_train = pd.DataFrame(
        {
            "cat_feature": rng.choice(["A", "B", "C"], 50),
            "num_feature": rng.rand(50),
            "prediction": rng.rand(50),
            "label": rng.randint(0, 2, 50),
        }
    )

    model = calibrator_class(
        early_stopping=False,
        num_rounds=2,
        lightgbm_params={"max_depth": 2, "n_estimators": 2},
    )
    model.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
    )

    importance_df = model.feature_importance()

    assert isinstance(importance_df, pd.DataFrame)
    assert "feature" in importance_df.columns
    assert "importance" in importance_df.columns
    expected_features = {"cat_feature", "num_feature", "prediction"}
    assert set(importance_df["feature"].values) == expected_features
    assert importance_df["importance"].iloc[0] >= importance_df["importance"].iloc[-1]


@pytest.mark.parametrize("calibrator_class", [methods.MCGrad, methods.RegressionMCGrad])
@pytest.mark.parametrize("num_rounds", [(2), (6)])
def test_early_stopping_stops_at_max_num_rounds(num_rounds: int, calibrator_class, rng):
    df_train = pd.DataFrame(
        {
            "feature1": rng.rand(50),
            "prediction": rng.rand(50),
            "label": rng.randint(0, 2, 50),
        }
    )

    # We create a dummy score function that always returns increasing values
    score_increments = np.linspace(0, 0.01, 100)  # 100 steps of tiny increments
    score_index = [0]  # Use a list to allow modification within the closure

    def dummy_score_func(
        y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        score = score_increments[score_index[0]]
        score_index[0] += 1
        return score

    # Early stopping should now not kick in, because the score is always increasing
    # It should still stop at the maximum number of rounds, which is specified by num_rounds
    mcgrad = calibrator_class(
        num_rounds=num_rounds,
        early_stopping=True,
        early_stopping_score_func=wrap_sklearn_metric_func(dummy_score_func),
        early_stopping_minimize_score=False,
        save_training_performance=True,
        lightgbm_params={"n_estimators": 1},
    )
    mcgrad.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=[],
        numerical_feature_column_names=["feature1"],
    )
    assert (
        num_rounds
        == len(mcgrad.mr)
        == len(mcgrad._performance_metrics["avg_valid_performance_dummy_score_func"])
        - 1
        == len(mcgrad._performance_metrics["avg_train_performance_dummy_score_func"])
        - 1
    ), "Early stopping exceeded the maximum number of rounds."


@pytest.mark.parametrize("calibrator_class", [methods.MCGrad, methods.RegressionMCGrad])
def test_fit_with_provided_df_val_runs_without_errors(calibrator_class, rng):
    # Setup: create training and validation datasets

    df_train = pd.DataFrame(
        {
            "feature1": rng.rand(50),
            "prediction": rng.rand(50),
            "label": rng.randint(0, 2, 50),
        }
    )
    df_val = pd.DataFrame(
        {
            "feature1": rng.rand(30),
            "prediction": rng.rand(30),
            "label": rng.randint(0, 2, 30),
        }
    )

    num_rounds = 2

    # We create a dummy score function that always returns increasing values
    score_increments = np.linspace(0, 0.01, 100)  # 100 steps of tiny increments
    score_index = [0]  # Use a list to allow modification within the closure

    def dummy_score_func(
        y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        score = score_increments[score_index[0]]
        score_index[0] += 1
        return score

    # Execute: fit model with early stopping using provided validation set
    mcgrad = calibrator_class(
        num_rounds=num_rounds,
        early_stopping=True,
        early_stopping_score_func=wrap_sklearn_metric_func(dummy_score_func),
        early_stopping_minimize_score=False,
        early_stopping_use_crossvalidation=False,
        save_training_performance=True,
        lightgbm_params={"n_estimators": 1},
    )

    mcgrad.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=[],
        numerical_feature_column_names=["feature1"],
        df_val=df_val,
    )

    assert (
        num_rounds
        == len(mcgrad.mr)
        == len(mcgrad._performance_metrics["avg_valid_performance_dummy_score_func"])
        - 1
        == len(mcgrad._performance_metrics["avg_train_performance_dummy_score_func"])
        - 1
    ), "Early stopping exceeded the maximum number of rounds."

    # Assert: model can make predictions
    predictions_val = mcgrad.predict(
        df_val,
        prediction_column_name="prediction",
        categorical_feature_column_names=[],
        numerical_feature_column_names=["feature1"],
    )

    assert len(predictions_val) > 0, (
        f"Predictions should not be empty, but got {len(predictions_val)}"
    )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mce_correctly_setup_in_mcgrad(calibrator_class, rng):
    # Check if the MCE is the right metric by looking at the name of the score function

    df_train = pd.DataFrame(
        {
            "feature1": rng.randint(0, 3, 50),
            "feature2": rng.rand(50),
            "prediction": rng.rand(50),
            "label": rng.randint(0, 2, 50),
        }
    )

    # Create an MCGrad object using MCE with the expected parameters and fit MCGrad
    mcgrad = calibrator_class(
        early_stopping_score_func=wrap_multicalibration_error_metric(
            categorical_segment_columns=["feature1"],
            numerical_segment_columns=["feature2"],
        ),
        early_stopping_minimize_score=True,
        num_rounds=1,
        lightgbm_params={
            "num_leaves": 2,
            "n_estimators": 1,
            "max_depth": 2,
        },
    ).fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["feature1"],
        numerical_feature_column_names=["feature2"],
    )

    assert "Multicalibration Error" in mcgrad.early_stopping_score_func.name, (
        "Name of the MCE metric does not contain Multicalibration Error or is not properly set up."
    )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mce_parameters_correctly_setup_in_mcgrad(calibrator_class, rng):
    # Check if the MCE's parameters are correctly set in the MCGrad object

    df_train = pd.DataFrame(
        {
            "feature1": rng.randint(0, 3, 50),
            "feature2": rng.rand(50),
            "prediction": rng.rand(50),
            "label": rng.randint(0, 2, 50),
        }
    )

    # Define the expected parameters
    expected_params = {
        "max_depth": 7,
        "min_samples_per_segment": 8,
        "max_values_per_segment_feature": 9,
        "max_n_segments": 10,
    }

    # Create an MCGrad object using MCE with the expected parameters and fit MCGrad
    mcgrad = calibrator_class(
        early_stopping_score_func=wrap_multicalibration_error_metric(
            categorical_segment_columns=["feature1"],
            numerical_segment_columns=["feature2"],
            **expected_params,
        ),
        early_stopping_minimize_score=True,
        early_stopping=True,
        num_rounds=1,
        lightgbm_params={
            "num_leaves": 2,
            "n_estimators": 1,
            "max_depth": 2,
        },
    ).fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["feature1"],
        numerical_feature_column_names=["feature2"],
    )

    # Check if the MCE's parameters are correctly set in the MCGrad object
    for param, expected_value in expected_params.items():
        assert getattr(mcgrad.early_stopping_score_func, param) == expected_value, (
            f"Parameter {param} not set correctly"
        )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_calls_score_func_during_early_stopping(calibrator_class, rng):
    # Check if the score function is called during early stopping

    mock_roc_auc_score = Mock(spec="skmetrics.roc_auc_score")
    mock_roc_auc_score.name = "roc_auc_score"
    mock_roc_auc_score.return_value = 0.5
    df_train = pd.DataFrame(
        {
            "feature1": rng.randint(0, 3, 50),
            "feature2": rng.rand(50),
            "prediction": rng.rand(50),
            "label": rng.randint(0, 2, 50),
        }
    )
    mcgrad = calibrator_class(
        early_stopping_score_func=mock_roc_auc_score,
        early_stopping_minimize_score=False,
        num_rounds=1,
        lightgbm_params={
            "num_leaves": 2,
            "n_estimators": 1,
            "max_depth": 2,
        },
    )
    mcgrad.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["feature1"],
        numerical_feature_column_names=["feature2"],
    )
    mock_roc_auc_score.assert_called()


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_early_stopping_with_multicalibration_error_metric(calibrator_class, rng):
    df_train = pd.DataFrame(
        {
            "feature1": rng.randint(0, 3, 50),
            "feature2": rng.rand(50),
            "prediction": rng.rand(50),
            "label": rng.randint(0, 2, 50),
        }
    )

    mcgrad = calibrator_class(
        num_rounds=5,
        early_stopping=True,
        early_stopping_score_func=wrap_multicalibration_error_metric(
            categorical_segment_columns=["feature1"],
            numerical_segment_columns=["feature2"],
        ),
        early_stopping_minimize_score=True,
        lightgbm_params={"max_depth": 2, "n_estimators": 2},
    )
    mcgrad.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["feature1"],
        numerical_feature_column_names=["feature2"],
    )

    assert len(mcgrad.mr) <= 5
    assert "Multicalibration Error" in mcgrad.early_stopping_score_func.name


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_performance_metrics_dictionary_size_matches_number_of_rounds(
    calibrator_class, rng
):
    # Check if the performance metrics dictionary has the correct number of elements

    df_train = pd.DataFrame(
        {
            "feature1": rng.randint(0, 3, 50),
            "feature2": rng.rand(50),
            "prediction": rng.rand(50),
            "label": rng.randint(0, 2, 50),
        }
    )

    mcgrad = calibrator_class(
        early_stopping_score_func=wrap_sklearn_metric_func(
            skmetrics.average_precision_score
        ),
        early_stopping_minimize_score=False,
        num_rounds=3,
        early_stopping=True,
        save_training_performance=True,
        patience=0,
        lightgbm_params={
            "num_leaves": 2,
            "n_estimators": 1,
            "max_depth": 2,
        },
    ).fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["feature1"],
        numerical_feature_column_names=["feature2"],
    )
    extra_evaluation_due_to_early_stopping = (
        1 if len(mcgrad.mr) < mcgrad.num_rounds else 0
    )

    # +1 because we also have the initial performance
    # + extra_evaluation_due_to_early_stopping because we also have the performance after the last round
    assert (
        len(
            mcgrad._performance_metrics[
                f"avg_valid_performance_{mcgrad.early_stopping_score_func.name}"
            ]
        )
        == 1 + len(mcgrad.mr) + extra_evaluation_due_to_early_stopping
    ), (
        f"The (validation) performance metrics dictionary should have {1 + len(mcgrad.mr) + extra_evaluation_due_to_early_stopping} elements"
    )
    assert (
        len(
            mcgrad._performance_metrics[
                f"avg_train_performance_{mcgrad.early_stopping_score_func.name}"
            ]
        )
        == 1 + len(mcgrad.mr) + extra_evaluation_due_to_early_stopping
    ), (
        f"The (training) performance metrics dictionary should have {1 + len(mcgrad.mr) + extra_evaluation_due_to_early_stopping} elements"
    )


def test_categorical_features_used_correctly_in_mcgrad_regressor():
    # Create a dataset that can be perfectly fit only when categorical features are used appropriately, rather than ordinally

    uncalibrated_col = "uncalibrated"
    categorical_segment_cols = ["X"]
    numerical_segment_cols = []
    truth_col = "y"

    LEN = 100
    categorical_col = np.array(
        ([0] * (LEN // 4)) + ([2] * (LEN // 4)) + ([1] * (LEN // 2))
    )
    y = np.array(([1] * (LEN // 2)) + ([0] * (LEN // 2)))
    uncalibrated = np.array([0] * LEN)

    # Notice that to solve it perfectly (with a shallow tree), the model has to treat 0,2 differently from 1, in a single split

    # Merge the columns into a dataframe
    df_train = pd.DataFrame(
        np.c_[categorical_col, y, uncalibrated],
        columns=[categorical_segment_cols[0], truth_col, uncalibrated_col],
    )

    h = methods.RegressionMCGrad(
        num_rounds=1,
        lightgbm_params={
            "min_child_samples": 1,
            "num_leaves": 2,
            "n_estimators": 1,
            "max_depth": 2,
        },
    )

    # Assert incorrect usage gives bad results
    h.fit(
        df_train,
        uncalibrated_col,
        truth_col,
        categorical_feature_column_names=[],
        numerical_feature_column_names=numerical_segment_cols
        + categorical_segment_cols,
    )

    assert (
        (
            h.predict(
                df_train,
                uncalibrated_col,
                [],
                numerical_segment_cols + categorical_segment_cols,
            )
            - y
        )
        ** 2
    ).mean() > 1e-2, "The model shouldn't have fit well"

    # Assert proper usage gives good results
    h.fit(
        df_train,
        uncalibrated_col,
        truth_col,
        categorical_feature_column_names=categorical_segment_cols,
        numerical_feature_column_names=numerical_segment_cols,
    )

    assert (
        (
            h.predict(
                df_train,
                uncalibrated_col,
                categorical_segment_cols,
                numerical_segment_cols,
            )
            - y
        )
        ** 2
    ).mean() < 1e-5, "The model did not fit perfectly"


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
@pytest.mark.parametrize("patience", [(4), (9)])
def test_patience_in_mcgrad(patience: int, calibrator_class, rng):
    # Check if the patience is correctly set in the MCGrad object: we use the dummy function that always increases the score
    # and check if the early stopping with a given patience stops at the correct round.

    num_rounds = 20  # This needs to be > than 1 + patience
    df_train = pd.DataFrame(
        {
            "feature1": rng.rand(50),
            "prediction": rng.rand(50),
            "label": rng.randint(0, 2, 50),
        }
    )

    score_increments = np.linspace(0, 0.01, 100)
    score_index = [0]

    def dummy_score_func(
        y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        score = score_increments[score_index[0]]
        score_index[0] += 1
        return score

    mcgrad = calibrator_class(
        num_rounds=num_rounds,
        early_stopping=True,
        early_stopping_score_func=wrap_sklearn_metric_func(dummy_score_func),
        early_stopping_minimize_score=True,
        patience=patience,
        lightgbm_params={
            "num_leaves": 2,
            "n_estimators": 1,
            "max_depth": 2,
        },
    )
    mcgrad.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=[],
        numerical_feature_column_names=["feature1"],
    )
    effective_num_rounds = len(mcgrad.mr)
    extra_evaluation_due_to_early_stopping = (
        1 if (mcgrad.early_stopping and effective_num_rounds < mcgrad.num_rounds) else 0
    )

    assert (
        len(
            mcgrad._performance_metrics[
                f"avg_valid_performance_{mcgrad.early_stopping_score_func.name}"
            ]
        )
        == 1 + patience + len(mcgrad.mr) + extra_evaluation_due_to_early_stopping
    ), "Patience not used correctly."


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_predict_with_num_rounds_0(calibrator_class, rng):
    # Check if the predictions are the same as the original prediction column when num_rounds=0
    df_train = pd.DataFrame(
        {
            "feature1": rng.randint(0, 3, 50),
            "feature2": rng.rand(50),
            "prediction": rng.rand(50),
            "label": rng.randint(0, 2, 50),
        }
    )
    mcgrad = calibrator_class(
        num_rounds=0,  # this has to be 0
        early_stopping=True,
        save_training_performance=True,
        lightgbm_params={
            "num_leaves": 2,
            "n_estimators": 1,
            "max_depth": 2,
        },
    ).fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["feature1"],
        numerical_feature_column_names=["feature2"],
    )
    predictions_all_rounds = mcgrad.predict(
        df_train, "prediction", return_all_rounds=True
    )
    predictions_final_round = mcgrad.predict(
        df_train, "prediction", return_all_rounds=False
    )

    # Check that the predictions are the same as the original prediction column in both cases
    assert np.allclose(predictions_all_rounds, df_train["prediction"]), (
        "Predictions with 'return_all_rounds' = True do not match the original prediction column."
    )
    assert np.allclose(predictions_final_round, df_train["prediction"]), (
        "Predictions with 'return_all_rounds' = False do not match the original prediction column."
    )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_number_rounds_after_fitting_with_0_rounds(calibrator_class, rng):
    # Check if the number of rounds is 0 after fitting with 0 rounds by checking the length of the mr attribute
    df_train = pd.DataFrame(
        {
            "feature1": rng.randint(0, 3, 50),
            "feature2": rng.rand(50),
            "prediction": rng.rand(50),
            "label": rng.randint(0, 2, 50),
        }
    )

    mcgrad = calibrator_class(
        num_rounds=0,
        early_stopping=True,
        save_training_performance=True,
        lightgbm_params={
            "num_leaves": 2,
            "n_estimators": 1,
            "max_depth": 2,
        },
    ).fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["feature1"],
        numerical_feature_column_names=["feature2"],
    )

    assert len(mcgrad.mr) == 0, (
        "MCGrad's number of rounds should be 0 after setting num_rounds to 0."
    )


@pytest.mark.parametrize("calibrator_class", [methods.MCGrad, methods.RegressionMCGrad])
def test_that_default_early_stopping_score_func_minimization_adheres_to_scikitlearn_convention(
    calibrator_class,
):
    mcb = calibrator_class()
    if mcb.early_stopping_score_func.name.endswith(
        "_loss"
    ) or mcb.early_stopping_score_func.name.endswith("_error"):
        assert mcb.early_stopping_minimize_score, (
            'For loss functions that end with "_loss" or "_error" the minimization should be True. If you are sure this is correct, please change the test.'
        )
    elif mcb.early_stopping_score_func.name.endswith("_score"):
        assert not mcb.early_stopping_minimize_score, (
            'For score functions that end with "_score" the minimization should be False. If you are sure this is correct, please change the test.'
        )
    else:
        raise ValueError(
            f"Default early stopping score function {mcb.early_stopping_score_func.name} does not adhere to scikit learn naming convention (suffix '_loss' -> lower is better, suffix '_score' -> higher is better). Make sure to set early_stopping_minimize_score correctly and adapt this test."
        )


def test_mce_below_initial_and_mce_below_strong_evidence_threshold_are_false_when_mce_is_greater_than_THR(
    rng,
):
    df_train = pd.DataFrame(
        {
            "feature1": rng.randint(0, 3, 50),
            "feature2": rng.rand(50),
            "prediction": rng.rand(50),
            "label": rng.randint(0, 2, 50),
        }
    )

    def mce_sigma_scale_mock():
        class WrappedFuncMockMCE:
            name = "Multicalibration Error<br>(mce_sigma_scale)"

            def __call__(
                self,
                df: pd.DataFrame,
                label_column: str,
                score_column: str,
                weight_column: str | None,
            ) -> float:
                return methods.MCGrad().MCE_STRONG_EVIDENCE_THRESHOLD + 0.1

        return WrappedFuncMockMCE()

    mcgrad = methods.MCGrad(
        num_rounds=5,
        early_stopping=True,
        early_stopping_score_func=mce_sigma_scale_mock(),
        early_stopping_minimize_score=True,
        early_stopping_use_crossvalidation=False,
        lightgbm_params={
            "num_leaves": 2,
            "n_estimators": 1,
            "max_depth": 2,
        },
    ).fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["feature1"],
        numerical_feature_column_names=["feature2"],
    )
    assert not mcgrad.mce_below_initial and mcgrad.mce_below_initial is not None, (
        "MCE is equal to the initial value. Thus, mce_below_initial must be False."
    )
    assert (
        not mcgrad.mce_below_strong_evidence_threshold
        and mcgrad.mce_below_strong_evidence_threshold is not None
    ), (
        f"MCE is greater than {mcgrad.MCE_STRONG_EVIDENCE_THRESHOLD}. Thus, mce_below_strong_evidence_threshold must be False."
    )
    assert (
        not mcgrad._mce_is_satisfactory and mcgrad._mce_is_satisfactory is not None
    ), (
        "MCE is neither below the initial value nor below strong evidence threshold. Thus, mce_is_satisfactory must be False."
    )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_extract_features_categorical_features_overflow(calibrator_class):
    mcgrad = calibrator_class(encode_categorical_variables=False)
    x_cat = np.array([np.iinfo(np.int32).max + 1, np.nan, 0])
    with pytest.raises(ValueError) as exc_info:
        mcgrad._extract_features(
            df=pd.DataFrame({"cat_feature": x_cat}),
            categorical_feature_column_names=["cat_feature"],
            numerical_feature_column_names=None,
        )
    assert "categorical feature values" in str(exc_info.value)
    assert "integer overflow" in str(exc_info.value)


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_extract_features_categorical_features_negative(calibrator_class):
    mcgrad = calibrator_class(encode_categorical_variables=False)
    x_cat = np.array([-1, np.nan, 0])
    with pytest.raises(ValueError) as exc_info:
        mcgrad._extract_features(
            df=pd.DataFrame({"cat_feature": x_cat}),
            categorical_feature_column_names=["cat_feature"],
            numerical_feature_column_names=None,
        )
    assert "categorical feature values" in str(exc_info.value)
    assert "missing" in str(exc_info.value)


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_extract_features_categorical_features_valid(calibrator_class):
    mcgrad = calibrator_class(encode_categorical_variables=False)
    x_cat = np.array([1, np.nan, 0])
    mcgrad._extract_features(
        df=pd.DataFrame({"cat_feature": x_cat}),
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=None,
    )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_extract_features_numerical_features_valid(calibrator_class):
    mcgrad = calibrator_class(encode_categorical_variables=False)
    x_num = np.array([1.0, np.nan, 0])
    mcgrad._extract_features(
        df=pd.DataFrame({"num_feature": x_num}),
        categorical_feature_column_names=None,
        numerical_feature_column_names=["num_feature"],
    )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
@pytest.mark.parametrize("best_num_rounds", [(0), (1), (2), (3)])
def test_mcgrad_early_stopping_returns_correct_number_of_rounds(
    calibrator_class, best_num_rounds: int
):
    data_len = 10

    x = np.arange(data_len)

    # Arbitrary data
    df_train = pd.DataFrame(
        {
            "feature1": x,
            "prediction": np.linspace(0.1, 0.9, data_len),
            "label": [0] * (data_len // 2) + [1] * (data_len - (data_len // 2)),
        }
    )

    n_folds = 3
    num_rounds = 3

    # We create a dummy score function that increases for best_num_rounds rounds, then decreases to 0
    score_increments = []

    for i in range(best_num_rounds):
        for _ in range(n_folds):
            score_increments.append(i + 1)

    # First n_folds are for estimating the performance before training
    score_increments = (
        ([0] * n_folds)
        + score_increments
        + ([-1] * (num_rounds - best_num_rounds) * n_folds)
    )

    score_index = [0]  # Use a list to allow modification within the closure

    def dummy_score_func(
        y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        score = score_increments[score_index[0]]
        score_index[0] += 1
        return score

    mcgrad = calibrator_class(
        num_rounds=num_rounds,
        n_folds=n_folds,
        early_stopping=True,
        early_stopping_use_crossvalidation=True,
        early_stopping_score_func=wrap_sklearn_metric_func(dummy_score_func),
        early_stopping_minimize_score=False,
        lightgbm_params={
            "num_leaves": 2,
            "n_estimators": 1,
            "max_depth": 2,
        },
    )
    mcgrad.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=[],
        numerical_feature_column_names=["feature1"],
    )

    assert len(mcgrad.mr) == best_num_rounds


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_cross_val_timeout_in_mcgrad(calibrator_class, rng):
    # Check if the cross validation timeout is correctly set in the MCGrad object: we use the dummy function that always increases the score

    data_len = 10

    df_train = pd.DataFrame(
        {
            "feature1": np.arange(data_len),
            "prediction": np.linspace(0.1, 0.9, data_len),
            "label": rng.randint(0, 2, data_len),
        }
    )

    # We create a dummy score function that always returns increasing values
    score_increments = np.linspace(0, 0.01, 100)  # 100 steps of tiny increments
    score_index = [0]  # Use a list to allow modification within the closure

    # This keeps increasing the score until the timeout is reached
    def dummy_score_func(
        y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        score = score_increments[score_index[0]]
        score_index[0] += 1
        return score

    # Timeout after one round
    num_rounds = 10
    early_stopping_timeout = 20

    mcgrad = calibrator_class(
        num_rounds=num_rounds,
        early_stopping=True,
        early_stopping_use_crossvalidation=True,
        n_folds=3,
        early_stopping_score_func=wrap_sklearn_metric_func(dummy_score_func),
        early_stopping_minimize_score=False,
        early_stopping_timeout=early_stopping_timeout,
        lightgbm_params={
            "min_child_samples": 1,
            "num_leaves": 2,
            "n_estimators": 1,
            "max_depth": 2,
        },
    )

    returned_times = [
        1,  # Round 0: metrics first evaluation
        2,
        early_stopping_timeout - 1,  # Round 2: last valid round
        early_stopping_timeout + 1,
        early_stopping_timeout + 2,
    ]

    # Mock the elapsed time check
    mcgrad._get_elapsed_time = Mock(side_effect=returned_times)

    mcgrad.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=[],
        numerical_feature_column_names=["feature1"],
    )
    assert len(mcgrad.mr) == 2

    del mcgrad

    # No timeout (timeout not reached or set to None)
    num_rounds = 3
    for early_stopping_timeout in [max(returned_times) + 1, None]:
        mcgrad = calibrator_class(
            num_rounds=num_rounds,
            early_stopping=True,
            early_stopping_use_crossvalidation=True,
            n_folds=3,  # Few folds to make sure the timeout is not reached
            early_stopping_score_func=wrap_sklearn_metric_func(dummy_score_func),
            early_stopping_minimize_score=False,
            early_stopping_timeout=early_stopping_timeout,
            lightgbm_params={
                "num_leaves": 2,
                "n_estimators": 1,
                "max_depth": 2,
            },
        )
        mcgrad._get_elapsed_time = Mock(side_effect=returned_times)
        mcgrad.fit(
            df_train=df_train,
            prediction_column_name="prediction",
            label_column_name="label",
            categorical_feature_column_names=[],
            numerical_feature_column_names=["feature1"],
        )
        assert len(mcgrad.mr) == num_rounds


@pytest.mark.parametrize(
    "calibrator_class, labels,is_valid",
    [
        # MCGrad
        ## Valid cases
        (methods.MCGrad, [0, 1], True),
        (methods.MCGrad, [True, False], True),
        ## Invalid cases
        ### Only one unique label
        (methods.MCGrad, [0, 0], False),
        (methods.MCGrad, [1, 1], False),
        (methods.MCGrad, [True, True], False),
        (methods.MCGrad, [False, False], False),
        ### Value > 1
        (methods.MCGrad, [0, 1.1], False),
        ### Invalid type
        (methods.MCGrad, ["a", "b"], False),
        ### Missing values in label
        (methods.MCGrad, [0, None], False),
        (methods.MCGrad, [0, np.nan], False),
        (methods.MCGrad, [0, np.inf], False),
        # RegressionMCGrad
        ## Valid cases
        (methods.RegressionMCGrad, [1.0, 101.0], True),
        ## Invalid cases
        ### Missing values
        (methods.RegressionMCGrad, [np.nan, 101.0], False),
        ### Infinite values
        (methods.RegressionMCGrad, [float("inf"), 101.0], False),
        ### String
        (methods.RegressionMCGrad, ["a", "b"], False),
        ### No variance
        (methods.RegressionMCGrad, [1.0, 1.0], False),
    ],
)
def test_mcgrad__check_labels_fails_when_expected(calibrator_class, labels, is_valid):
    df = pd.DataFrame({"label": labels})
    calibrator = calibrator_class()
    if is_valid:
        calibrator._check_labels(df, "label")
    else:
        with pytest.raises(ValueError):
            calibrator._check_labels(df, "label")


@pytest.mark.parametrize(
    "calibrator_class,scores,is_valid",
    [
        # MCGrad
        ## Valid cases
        (methods.MCGrad, [0.1, 0.2], True),
        ## Invalid cases
        ### Missing values
        (methods.MCGrad, [0.1, None], False),
        (methods.MCGrad, [0.1, np.nan], False),
        ### Out of bounds
        (methods.MCGrad, [0.1, 1.1], False),
        (methods.MCGrad, [0.1, -0.1], False),
        (methods.MCGrad, [-0.1, 0.1], False),
        ### Infinite values
        (methods.MCGrad, [0.1, np.inf], False),
        # RegressionMCGrad
        ## Valid cases
        (methods.RegressionMCGrad, [1.0, 101.0], True),
        (methods.RegressionMCGrad, [-50.0, 50.0], True),
        ## Invalid cases
        ### Missing values
        (methods.RegressionMCGrad, [1.0, None], False),
        (methods.RegressionMCGrad, [1.0, np.nan], False),
        ### Infinite values
        (methods.RegressionMCGrad, [1.0, float("inf")], False),
        (methods.RegressionMCGrad, [float("-inf"), 1.0], False),
    ],
)
def test_mcgrad__check_predictions_fails_when_expected(
    calibrator_class, scores, is_valid
):
    df = pd.DataFrame({"score": scores})
    calibrator = calibrator_class()
    if is_valid:
        calibrator._check_predictions(df, "score")
    else:
        with pytest.raises(ValueError):
            calibrator._check_predictions(df, "score")


def test_basemcgrad_implementations_transform_inverse_transform_invariance():
    # Find all subclasses of _BaseMCGrad. This only works for classes that are imported in this file
    # so we're operating on the assumption that there's at least on other relevant test for any MCGrad implementation.
    def get_all_subclasses(cls):
        all_subclasses = []
        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(get_all_subclasses(subclass))
        return all_subclasses

    subclasses = get_all_subclasses(methods._BaseMCGrad)
    assert len(subclasses) > 0, "Expected at least one subclass of _BaseMCGrad"

    predictions = np.array([0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999])

    for subclass in subclasses:
        transformed = subclass._transform_predictions(predictions)
        inverse_transformed = subclass._inverse_transform_predictions(transformed)
        np.testing.assert_allclose(
            inverse_transformed,
            predictions,
            rtol=1e-10,
            err_msg=f"_inverse_transform_predictions(_transform_predictions(predictions)) != predictions for {subclass.__name__}",
        )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
@pytest.mark.parametrize(
    "cat_data,num_data,allow_missing,should_raise",
    [
        ([[0, 1]], [[0.1, 0.2]], True, False),
        ([[0, None]], [[0.1, 0.2]], True, False),
        ([[0, 1]], [[0.1, None]], True, False),
        ([[0, 1]], [[0.1, 0.2]], False, False),
        ([[0, None]], [[0.1, 0.2]], False, True),
        ([[0, 1]], [[0.1, None]], False, True),
        ([[]], [[0.1, None]], False, True),
    ],
)
def test_mcgrad__check_segment_features_fails_when_expected(
    calibrator_class, cat_data, num_data, allow_missing, should_raise
):
    df = pd.DataFrame(cat_data + num_data)
    cat_colnames = [str(i) for i in range(len(cat_data))]
    num_colnames = [str(i) for i in range(len(cat_data), len(cat_data) + len(num_data))]
    df.columns = cat_colnames + num_colnames
    mcb = calibrator_class(allow_missing_segment_feature_values=allow_missing)
    if should_raise:
        with pytest.raises(ValueError):
            mcb._check_segment_features(df, cat_colnames, num_colnames)
    else:
        mcb._check_segment_features(df, cat_colnames, num_colnames)


@pytest.mark.parametrize(
    "early_stopping_use_crossvalidation,weights,scoring_function,expected_method",
    [
        (False, [1, 1, 1], None, methods._EstimationMethod.HOLDOUT),
        (True, [1, 1, 1, 1, 1], None, methods._EstimationMethod.CROSS_VALIDATION),
        (None, [1, 1, 1], None, methods._EstimationMethod.CROSS_VALIDATION),
        (None, [1, 1, 1, 1, 1], None, methods._EstimationMethod.HOLDOUT),
        (
            None,
            [100, 1, 1, 1, 1, 1],
            None,
            methods._EstimationMethod.CROSS_VALIDATION,
        ),
        (
            None,
            [100, 100, 100, 100, 1, 1],
            None,
            methods._EstimationMethod.HOLDOUT,
        ),
        (
            None,
            [1, 1, 1, 1, 1],
            skmetrics.average_precision_score,
            methods._EstimationMethod.CROSS_VALIDATION,
        ),
    ],
)
def test_mcgrad__determine_estimation_method(
    early_stopping_use_crossvalidation,
    weights,
    scoring_function,
    expected_method,
):
    # For regression MCGrad we don't have a ESS threshold so we just run it for binary for now
    mcb = methods.MCGrad(
        early_stopping=True,
        early_stopping_use_crossvalidation=early_stopping_use_crossvalidation,
        early_stopping_score_func=wrap_sklearn_metric_func(scoring_function)
        if scoring_function is not None
        else None,
        early_stopping_minimize_score=True if scoring_function is not None else None,
    )

    mcb.ESS_THRESHOLD_FOR_CROSS_VALIDATION = 4

    assert expected_method == mcb._determine_estimation_method(np.array(weights))


@pytest.mark.parametrize(
    "calibrator_class,scores,segment,expected_mask,allow_missing_segment_features",
    [
        # MCGrad tests
        (methods.MCGrad, [0.1, 0.2], [0, 1], [True, True], True),
        (methods.MCGrad, [0.1, None], [0, 1], [True, False], True),
        (methods.MCGrad, [0.1, 1.1], [0, 1], [True, False], True),
        (methods.MCGrad, [0.1, -0.1], [0, 1], [True, False], True),
        (methods.MCGrad, [0.1, np.nan], [0, 1], [True, False], True),
        (methods.MCGrad, [0.1, np.inf], [0, 1], [True, False], True),
        (methods.MCGrad, [-0.1, 0.1], [0, 1], [False, True], True),
        (methods.MCGrad, [None, 0.1], [0, None], [False, True], True),
        (methods.MCGrad, [None, 0.1], [0, None], [False, False], False),
        # RegressionMCGrad tests - no out-of-bounds checking
        (methods.RegressionMCGrad, [1.0, 101.0], [0, 1], [True, True], True),
        (methods.RegressionMCGrad, [1.0, None], [0, 1], [True, False], True),
        (methods.RegressionMCGrad, [-50.0, 50.0], [0, 1], [True, True], True),
        (methods.RegressionMCGrad, [1.0, np.nan], [0, 1], [True, False], True),
        (methods.RegressionMCGrad, [1.0, np.inf], [0, 1], [True, False], True),
        (methods.RegressionMCGrad, [float("-inf"), 1.0], [0, 1], [False, True], True),
        (methods.RegressionMCGrad, [None, 1.0], [0, None], [False, True], True),
        (methods.RegressionMCGrad, [None, 1.0], [0, None], [False, False], False),
    ],
)
def test_mcgrad__get_output_presence_mask_works_correctly(
    calibrator_class,
    scores,
    segment,
    expected_mask,
    allow_missing_segment_features,
):
    df = pd.DataFrame({"score": scores, "segment": segment})
    mcb = calibrator_class(
        allow_missing_segment_feature_values=allow_missing_segment_features
    )
    mask = mcb._get_output_presence_mask(
        df,
        prediction_column_name="score",
        categorical_feature_column_names=["segment"],
        numerical_feature_column_names=[],
    )
    assert np.array_equal(mask, expected_mask)


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_mcgrad_internal_state_reset_when_fitting_again(calibrator_class, rng):
    df_train = pd.DataFrame(
        {
            "feature1": rng.rand(50),
            "prediction": rng.rand(50),
            "label": rng.randint(0, 2, 50),
        }
    )

    score_increments = np.linspace(0, 0.01, 20)  # 100 steps of tiny increments
    score_index = [0]  # Use a list to allow modification within the closure

    def dummy_score_func(
        y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        score = score_increments[score_index[0]]
        score_index[0] += 1
        return score

    mcgrad = calibrator_class(
        num_rounds=1,
        early_stopping=True,
        early_stopping_score_func=wrap_sklearn_metric_func(dummy_score_func),
        early_stopping_minimize_score=False,
        save_training_performance=True,
        n_folds=2,
        lightgbm_params={
            "num_leaves": 2,
            "n_estimators": 1,
            "max_depth": 2,
        },
    )
    # Fitting MCGrad for the first time
    mcgrad.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=[],
        numerical_feature_column_names=["feature1"],
    )
    # Fitting MCGrad again should reset the training parameters
    mcgrad.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=[],
        numerical_feature_column_names=["feature1"],
    )
    # Number of rounds should be 1, not 2 as well as number of evaluations
    assert (
        1
        == len(mcgrad.mr)
        == len(mcgrad._performance_metrics["avg_valid_performance_dummy_score_func"])
        - 1
        == len(mcgrad._performance_metrics["avg_train_performance_dummy_score_func"])
        - 1
    ), (
        "The internal state - including number of Boosters & evaluations of MCGrad - should be reset when fitting MCGrad multiple times."
    )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_prepare_mcgrad_processed_data_matches_individual_operations(calibrator_class):
    df = generate_test_data(5)
    model = calibrator_class(early_stopping=False, num_rounds=1)

    cat_features = ["City", "Gender"]
    num_features = None

    internal_data = model._preprocess_input_data(
        df=df,
        prediction_column_name="Prediction",
        label_column_name="Label",
        weight_column_name=None,
        categorical_feature_column_names=cat_features,
        numerical_feature_column_names=num_features,
        is_fit_phase=True,
    )

    x_direct = model._extract_features(
        df=df,
        categorical_feature_column_names=cat_features,
        numerical_feature_column_names=num_features,
        is_fit_phase=False,
    )

    predictions_direct = model._transform_predictions(df["Prediction"].values)
    y_direct = df["Label"].values.astype(float)
    presence_mask_direct = model._get_output_presence_mask(
        df, "Prediction", cat_features, num_features or []
    )

    np.testing.assert_array_equal(internal_data.features, x_direct)
    np.testing.assert_array_equal(internal_data.predictions, predictions_direct)
    np.testing.assert_array_equal(internal_data.labels, y_direct)
    np.testing.assert_array_equal(
        internal_data.output_presence_mask, presence_mask_direct
    )
    np.testing.assert_array_equal(internal_data.weights, np.ones(len(df)))


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_prepare_mcgrad_processed_data_with_weights(calibrator_class):
    df = generate_test_data(5)
    df["Weight"] = np.array([1.0, 2.0, 1.0, 3.0, 1.0])

    model = calibrator_class(early_stopping=False, num_rounds=1)

    internal_data = model._preprocess_input_data(
        df=df,
        prediction_column_name="Prediction",
        label_column_name="Label",
        weight_column_name="Weight",
        categorical_feature_column_names=["City", "Gender"],
        numerical_feature_column_names=None,
        is_fit_phase=True,
    )

    expected_weights = df["Weight"].values.astype(float)
    np.testing.assert_array_equal(internal_data.weights, expected_weights)


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_prepare_mcgrad_processed_data_presence_mask_with_nan_predictions(
    calibrator_class,
):
    df = generate_test_data(5)
    df.loc[2, "Prediction"] = np.nan

    model = calibrator_class(early_stopping=False, num_rounds=1)

    internal_data = model._preprocess_input_data(
        df=df,
        prediction_column_name="Prediction",
        label_column_name="Label",
        weight_column_name=None,
        categorical_feature_column_names=["City", "Gender"],
        numerical_feature_column_names=None,
        is_fit_phase=True,
    )

    assert not internal_data.output_presence_mask[2]
    assert internal_data.output_presence_mask[[0, 1, 3, 4]].all()


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
@pytest.mark.parametrize(
    "estimation_method,has_custom_validation_set,expected_splitter_type",
    [
        (methods._EstimationMethod.CROSS_VALIDATION, False, "cv"),
        (methods._EstimationMethod.HOLDOUT, False, "holdout"),
        (
            methods._EstimationMethod.HOLDOUT,
            True,
            "noop splitter",
        ),
    ],
)
def test_determine_train_test_splitter_returns_correct_splitter(
    calibrator_class,
    estimation_method,
    has_custom_validation_set,
    expected_splitter_type,
):
    # Setup: Create model instance
    model = calibrator_class(
        early_stopping=False,
        num_rounds=1,
    )

    # Execute: Call the method to determine the splitter
    splitter = model._determine_train_test_splitter(
        estimation_method=estimation_method,
        has_custom_validation_set=has_custom_validation_set,
    )

    # Assert: Verify the correct splitter type is returned
    if expected_splitter_type == "cv":
        # For cross-validation, it should be either KFold or StratifiedKFold
        assert isinstance(splitter, (StratifiedKFold, KFold)), (
            "Expected cross-validation splitter"
        )
    elif expected_splitter_type == "holdout":
        # For holdout, it should be TrainTestSplitWrapper with positive test_size
        assert isinstance(splitter, utils.TrainTestSplitWrapper), (
            "Expected TrainTestSplitWrapper"
        )
        assert splitter.test_size > 0.0, (
            "Expected positive test_size for regular holdout"
        )
    elif expected_splitter_type == "noop splitter":
        # For provided holdout, it should be TrainTestSplitWrapper with zero test_size
        assert isinstance(splitter, utils.NoopSplitterWrapper), (
            "Expected NoopSplitterWrapper"
        )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_determine_train_test_splitter_raises_error_for_cv_with_custom_validation_set(
    calibrator_class,
):
    # Setup: Create model instance
    model = calibrator_class(
        early_stopping=False,
        num_rounds=1,
    )

    # Execute & Assert: Verify ValueError is raised when cross validation is used with custom validation set
    with pytest.raises(
        ValueError,
        match="Custom validation set was provided while cross validation was enabled for early stopping. Please set early_stopping_use_crossvalidation to False or remove df_val",
    ):
        model._determine_train_test_splitter(
            estimation_method=methods._EstimationMethod.CROSS_VALIDATION,
            has_custom_validation_set=True,
        )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_determine_train_test_splitter_noop_splitter_returned(
    calibrator_class,
):
    # Setup: Create model instance
    model = calibrator_class(
        early_stopping=False,
        num_rounds=1,
    )

    # Execute: Get the provided holdout splitter for custom validation set
    splitter = model._determine_train_test_splitter(
        estimation_method=methods._EstimationMethod.HOLDOUT,
        has_custom_validation_set=True,
    )

    assert isinstance(splitter, utils.NoopSplitterWrapper), (
        "Expected NoopSplitterWrapper"
    )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
@pytest.mark.parametrize(
    "estimation_method,expected_n_folds",
    [
        (
            methods._EstimationMethod.CROSS_VALIDATION,
            5,
        ),  # Assuming N_FOLDS default is 5
        (methods._EstimationMethod.HOLDOUT, 1),
        (
            methods._EstimationMethod.AUTO,
            1,
        ),  # AUTO should also return 1 in default case
    ],
)
def test_determine_n_folds_returns_correct_value(
    calibrator_class,
    estimation_method,
    expected_n_folds,
):
    # Setup: Create model instance
    model = calibrator_class(
        early_stopping=False,
        num_rounds=1,
    )

    # Execute: Determine n_folds
    n_folds = model._determine_n_folds(estimation_method=estimation_method)

    # Assert: Verify correct n_folds is returned
    # Special handling for CROSS_VALIDATION since N_FOLDS may be set differently
    if estimation_method == methods._EstimationMethod.CROSS_VALIDATION:
        assert n_folds == model.n_folds
    else:
        assert n_folds == expected_n_folds


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
    ],
)
def test_prepare_mcgrad_processed_data_presence_mask_with_out_of_bounds_predictions(
    calibrator_class,
):
    df = generate_test_data(5)
    df.loc[1, "Prediction"] = -0.1
    df.loc[3, "Prediction"] = 1.5

    model = calibrator_class(early_stopping=False, num_rounds=1)

    internal_data = model._preprocess_input_data(
        df=df,
        prediction_column_name="Prediction",
        label_column_name="Label",
        weight_column_name=None,
        categorical_feature_column_names=["City", "Gender"],
        numerical_feature_column_names=None,
        is_fit_phase=True,
    )

    assert not internal_data.output_presence_mask[1]
    assert not internal_data.output_presence_mask[3]
    assert internal_data.output_presence_mask[[0, 2, 4]].all()


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
    ],
)
def test_prepare_mcgrad_processed_data_presence_mask_with_missing_segment_features(
    calibrator_class,
):
    df = generate_test_data(5)
    df.loc[2, "City"] = None

    model = calibrator_class(
        early_stopping=False,
        num_rounds=1,
        allow_missing_segment_feature_values=False,
    )

    internal_data = model._preprocess_input_data(
        df=df,
        prediction_column_name="Prediction",
        label_column_name="Label",
        weight_column_name=None,
        categorical_feature_column_names=["City", "Gender"],
        numerical_feature_column_names=None,
        is_fit_phase=True,
    )

    assert not internal_data.output_presence_mask[2]
    assert internal_data.output_presence_mask[[0, 1, 3, 4]].all()


def test_platt_scaling_with_features_categorical_features(rng):
    df = pd.DataFrame(
        {
            "prediction": rng.uniform(0.2, 0.8, 100),
            "label": rng.randint(0, 2, 100),
            "cat_feature": rng.choice(["A", "B", "C"], 100),
        }
    )

    calibrator = methods.PlattScalingWithFeatures()
    calibrator.fit(
        df_train=df,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["cat_feature"],
    )

    predictions = calibrator.predict(
        df=df,
        prediction_column_name="prediction",
        categorical_feature_column_names=["cat_feature"],
    )

    assert len(predictions) == len(df)
    assert calibrator.ohe is not None
    assert calibrator.ohe_columns is not None
    assert len(calibrator.ohe_columns) > 0


def test_platt_scaling_with_features_numerical_features(rng):
    df = pd.DataFrame(
        {
            "prediction": rng.uniform(0.2, 0.8, 100),
            "label": rng.randint(0, 2, 100),
            "num_feature": rng.uniform(0, 100, 100),
        }
    )

    calibrator = methods.PlattScalingWithFeatures()
    calibrator.fit(
        df_train=df,
        prediction_column_name="prediction",
        label_column_name="label",
        numerical_feature_column_names=["num_feature"],
    )

    predictions = calibrator.predict(
        df=df,
        prediction_column_name="prediction",
        numerical_feature_column_names=["num_feature"],
    )

    assert len(predictions) == len(df)
    assert calibrator.kbd is not None
    assert calibrator.kbd_columns is not None
    assert len(calibrator.kbd_columns) > 0


def test_platt_scaling_with_features_both_categorical_and_numerical(rng):
    df = pd.DataFrame(
        {
            "prediction": rng.uniform(0.2, 0.8, 100),
            "label": rng.randint(0, 2, 100),
            "cat_feature": rng.choice(["X", "Y", "Z"], 100),
            "num_feature": rng.uniform(0, 50, 100),
        }
    )

    calibrator = methods.PlattScalingWithFeatures()
    calibrator.fit(
        df_train=df,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
    )

    predictions = calibrator.predict(
        df=df,
        prediction_column_name="prediction",
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
    )

    assert len(predictions) == len(df)
    assert calibrator.ohe is not None
    assert calibrator.kbd is not None
    assert calibrator.ohe_columns is not None
    assert calibrator.kbd_columns is not None


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_predict_does_not_modify_input_predictions_array(calibrator_class, rng):
    """
    Test that _predict does not modify the input predictions array in-place.

    This is a regression test for a bug where _predict would alias the input
    array instead of copying it, causing in-place modifications via += and *= operators.
    """
    n_samples = 30
    df = pd.DataFrame(
        {
            "prediction": rng.uniform(0.3, 0.7, n_samples),
            "label": rng.randint(0, 2, n_samples),
            "feature1": rng.choice(["A", "B"], n_samples),
        }
    )

    mcgrad = calibrator_class(
        early_stopping=False,
        num_rounds=1,
        lightgbm_params={
            "num_leaves": 2,
            "n_estimators": 1,
            "max_depth": 2,
        },
    )

    mcgrad.fit(
        df_train=df,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["feature1"],
    )

    test_df = df.head(10)
    preprocessed_data = mcgrad._preprocess_input_data(
        df=test_df,
        prediction_column_name="prediction",
        label_column_name=None,
        weight_column_name=None,
        categorical_feature_column_names=["feature1"],
        numerical_feature_column_names=[],
        is_fit_phase=False,
    )

    original_predictions = preprocessed_data.predictions.copy()

    _ = mcgrad._predict(
        x=preprocessed_data.features,
        transformed_predictions=preprocessed_data.predictions,
        return_all_rounds=False,
    )

    np.testing.assert_array_equal(
        preprocessed_data.predictions,
        original_predictions,
    )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_early_stopping_produces_same_model_as_manual_num_rounds(calibrator_class, rng):
    """
    Test that early stopping with N rounds produces the same model as manually setting num_rounds=N.

    This is a regression test for a bug where:
    1. Early stopping determines N rounds is optimal
    2. During early stopping, _predict modifies preprocessed_data.predictions in-place
    3. Final model training uses the modified predictions instead of original ones
    4. Result: Different model than manually setting num_rounds=N
    """
    n_samples = 100
    df = pd.DataFrame(
        {
            "prediction": rng.uniform(0.2, 0.8, n_samples),
            "label": rng.randint(0, 2, n_samples),
            "feature1": rng.choice(["A", "B", "C"], n_samples),
            "feature2": rng.randn(n_samples),
        }
    )

    mcgrad_with_es = calibrator_class(
        early_stopping=True,
        num_rounds=5,
        save_training_performance=True,
        patience=0,
        random_state=42,
        lightgbm_params={
            "num_leaves": 4,
            "n_estimators": 10,
            "max_depth": 3,
        },
    )

    mcgrad_with_es.fit(
        df_train=df,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["feature1"],
        numerical_feature_column_names=["feature2"],
    )

    num_rounds_determined = len(mcgrad_with_es.mr)

    mcgrad_manual = calibrator_class(
        early_stopping=False,
        num_rounds=num_rounds_determined,
        random_state=42,
        lightgbm_params={
            "num_leaves": 4,
            "n_estimators": 10,
            "max_depth": 3,
        },
    )

    mcgrad_manual.fit(
        df_train=df,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["feature1"],
        numerical_feature_column_names=["feature2"],
    )

    test_df = df.sample(50, random_state=999)

    predictions_with_es = mcgrad_with_es.predict(
        df=test_df,
        prediction_column_name="prediction",
        categorical_feature_column_names=["feature1"],
        numerical_feature_column_names=["feature2"],
    )

    predictions_manual = mcgrad_manual.predict(
        df=test_df,
        prediction_column_name="prediction",
        categorical_feature_column_names=["feature1"],
        numerical_feature_column_names=["feature2"],
    )

    np.testing.assert_allclose(
        predictions_with_es,
        predictions_manual,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        methods.MCGrad,
        methods.RegressionMCGrad,
    ],
)
def test_multiple_predict_calls_produce_consistent_results(calibrator_class, rng):
    """
    Test that calling predict multiple times on the same data produces identical results.

    This verifies that predict does not have side effects that alter subsequent predictions.
    """
    n_samples = 50
    df = pd.DataFrame(
        {
            "prediction": rng.uniform(0.2, 0.8, n_samples),
            "label": rng.randint(0, 2, n_samples),
            "feature1": rng.choice(["P", "Q", "R"], n_samples),
        }
    )

    mcgrad = calibrator_class(
        early_stopping=False,
        num_rounds=1,
        random_state=42,
        lightgbm_params={
            "num_leaves": 2,
            "n_estimators": 1,
            "max_depth": 2,
        },
    )

    mcgrad.fit(
        df_train=df,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["feature1"],
    )

    test_df = df.sample(20, random_state=123)

    predictions_first = mcgrad.predict(
        df=test_df,
        prediction_column_name="prediction",
        categorical_feature_column_names=["feature1"],
    )

    predictions_second = mcgrad.predict(
        df=test_df,
        prediction_column_name="prediction",
        categorical_feature_column_names=["feature1"],
    )

    predictions_third = mcgrad.predict(
        df=test_df,
        prediction_column_name="prediction",
        categorical_feature_column_names=["feature1"],
    )

    np.testing.assert_array_equal(predictions_first, predictions_second)
    np.testing.assert_array_equal(predictions_second, predictions_third)


def test_segmentwise_calibrator_with_no_categorical_features_equivalent_to_underlying_calibrator(
    rng,
):
    df = pd.DataFrame(
        {
            "prediction": rng.uniform(0.2, 0.8, 100),
            "label": rng.randint(0, 2, 100),
        }
    )

    segmentwise_calibrator = methods.SegmentwiseCalibrator(methods.PlattScaling)
    segmentwise_calibrator.fit(
        df_train=df,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=None,
    )

    direct_calibrator = methods.PlattScaling()
    direct_calibrator.fit(
        df_train=df,
        prediction_column_name="prediction",
        label_column_name="label",
    )

    segmentwise_predictions = segmentwise_calibrator.predict(
        df=df,
        prediction_column_name="prediction",
        categorical_feature_column_names=None,
    )

    direct_predictions = direct_calibrator.predict(
        df=df,
        prediction_column_name="prediction",
    )

    assert len(segmentwise_predictions) == len(df)
    assert isinstance(
        segmentwise_calibrator.calibrator_per_segment["()"], methods.PlattScaling
    )
    np.testing.assert_allclose(segmentwise_predictions, direct_predictions)


def test_segmentwise_calibrator_falls_back_to_identity_mapping_for_single_class_segment(
    rng,
):
    df = pd.DataFrame(
        {
            "prediction": rng.uniform(0.2, 0.8, 100),
            # Segment A: all 0s (single class), Segment B: mix of 0s and 1s (two classes)
            "label": [0] * 50 + [0] * 25 + [1] * 25,
            "segment": ["A"] * 50 + ["B"] * 50,
        }
    )

    calibrator = methods.SegmentwiseCalibrator(methods.PlattScaling)
    calibrator.fit(
        df_train=df,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["segment"],
    )

    assert "('A',)" in calibrator.calibrator_per_segment
    assert "('B',)" in calibrator.calibrator_per_segment
    # Segment A has only one class (all 0s), so it falls back to IdentityCalibrator
    assert isinstance(
        calibrator.calibrator_per_segment["('A',)"], methods.IdentityCalibrator
    )
    # Segment B has both classes, so PlattScaling can be fit
    assert isinstance(calibrator.calibrator_per_segment["('B',)"], methods.PlattScaling)


def test_segmentwise_calibrator_falls_back_to_identity_mapping_for_unseen_segment(rng):
    df_train = pd.DataFrame(
        {
            "prediction": rng.uniform(0.2, 0.8, 100),
            "label": rng.randint(0, 2, 100),
            "segment": rng.choice(["A", "B"], 100),
        }
    )

    df_test = pd.DataFrame(
        {
            "prediction": rng.uniform(0.2, 0.8, 20),
            "segment": ["C"] * 20,
        }
    )

    calibrator = methods.SegmentwiseCalibrator(methods.PlattScaling)
    calibrator.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["segment"],
    )

    predictions = calibrator.predict(
        df=df_test,
        prediction_column_name="prediction",
        categorical_feature_column_names=["segment"],
    )

    assert len(predictions) == len(df_test)
    assert "('C',)" in calibrator.calibrator_per_segment
    assert isinstance(
        calibrator.calibrator_per_segment["('C',)"], methods.IdentityCalibrator
    )
    np.testing.assert_array_equal(predictions, df_test["prediction"].values)


@pytest.mark.parametrize(
    "calibrator_class,calibrator_kwargs",
    [
        (
            methods.MCGrad,
            {
                "num_rounds": 2,
                "early_stopping": False,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
        (
            methods.RegressionMCGrad,
            {
                "num_rounds": 2,
                "early_stopping": False,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
    ],
)
def test_mcgrad_fit_does_not_modify_input_dataframe(
    calibrator_class, calibrator_kwargs
):
    df_train = pd.DataFrame(
        {
            "prediction": [0.1, 0.2, 0.3, 0.4, 0.5],
            "label": [0, 1, 0, 1, 0],
            "cat_feature": ["A", "B", "A", "B", "A"],
            "num_feature": [1.0, 2.0, 3.0, 4.0, 5.0],
            "weight": [1.0, 1.0, 2.0, 2.0, 1.0],
        }
    )

    df_train_original = df_train.copy()

    calibrator = calibrator_class(**calibrator_kwargs)
    calibrator.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        weight_column_name="weight",
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
    )

    pd.testing.assert_frame_equal(df_train, df_train_original)


@pytest.mark.parametrize(
    "calibrator_class,calibrator_kwargs",
    [
        (
            methods.MCGrad,
            {
                "num_rounds": 2,
                "early_stopping": False,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
        (
            methods.RegressionMCGrad,
            {
                "num_rounds": 2,
                "early_stopping": False,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
    ],
)
def test_mcgrad_predict_does_not_modify_input_dataframe(
    calibrator_class, calibrator_kwargs
):
    df_train = pd.DataFrame(
        {
            "prediction": [0.1, 0.2, 0.3, 0.4, 0.5],
            "label": [0, 1, 0, 1, 0],
            "cat_feature": ["A", "B", "A", "B", "A"],
            "num_feature": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    df_test = pd.DataFrame(
        {
            "prediction": [0.15, 0.25, 0.35],
            "cat_feature": ["A", "B", "A"],
            "num_feature": [1.5, 2.5, 3.5],
        }
    )

    df_test_original = df_test.copy()

    calibrator = calibrator_class(**calibrator_kwargs)
    calibrator.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
    )

    _ = calibrator.predict(
        df=df_test,
        prediction_column_name="prediction",
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
    )

    pd.testing.assert_frame_equal(df_test, df_test_original)


@pytest.mark.parametrize(
    "calibrator_class,calibrator_kwargs",
    [
        (methods.PlattScaling, {}),
        (methods.IsotonicRegression, {}),
        (methods.MultiplicativeAdjustment, {}),
        (methods.AdditiveAdjustment, {}),
        (methods.IdentityCalibrator, {}),
    ],
)
def test_simple_calibrator_fit_does_not_modify_input_dataframe(
    calibrator_class, calibrator_kwargs, rng
):
    df_train = pd.DataFrame(
        {
            "prediction": rng.uniform(0.1, 0.9, 50),
            "label": rng.randint(0, 2, 50),
            "weight": rng.uniform(0.5, 2.0, 50),
        }
    )

    df_train_original = df_train.copy()

    calibrator = calibrator_class(**calibrator_kwargs)
    calibrator.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        weight_column_name="weight",
    )

    pd.testing.assert_frame_equal(df_train, df_train_original)


@pytest.mark.parametrize(
    "calibrator_class,calibrator_kwargs",
    [
        (methods.PlattScaling, {}),
        (methods.IsotonicRegression, {}),
        (methods.MultiplicativeAdjustment, {}),
        (methods.AdditiveAdjustment, {}),
        (methods.IdentityCalibrator, {}),
    ],
)
def test_simple_calibrator_predict_does_not_modify_input_dataframe(
    calibrator_class, calibrator_kwargs, rng
):
    df_train = pd.DataFrame(
        {
            "prediction": rng.uniform(0.1, 0.9, 50),
            "label": rng.randint(0, 2, 50),
        }
    )

    df_test = pd.DataFrame(
        {
            "prediction": rng.uniform(0.1, 0.9, 20),
        }
    )

    df_test_original = df_test.copy()

    calibrator = calibrator_class(**calibrator_kwargs)
    calibrator.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
    )

    _ = calibrator.predict(
        df=df_test,
        prediction_column_name="prediction",
    )

    pd.testing.assert_frame_equal(df_test, df_test_original)


def test_platt_scaling_with_features_fit_does_not_modify_input_dataframe(rng):
    df_train = pd.DataFrame(
        {
            "prediction": rng.uniform(0.1, 0.9, 100),
            "label": rng.randint(0, 2, 100),
            "cat_feature": rng.choice(["A", "B", "C"], 100),
            "num_feature": rng.uniform(0, 100, 100),
            "weight": rng.uniform(0.5, 2.0, 100),
        }
    )

    df_train_original = df_train.copy()

    calibrator = methods.PlattScalingWithFeatures()
    calibrator.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        weight_column_name="weight",
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
    )

    pd.testing.assert_frame_equal(df_train, df_train_original)


def test_platt_scaling_with_features_predict_does_not_modify_input_dataframe(rng):
    df_train = pd.DataFrame(
        {
            "prediction": rng.uniform(0.1, 0.9, 100),
            "label": rng.randint(0, 2, 100),
            "cat_feature": rng.choice(["A", "B", "C"], 100),
            "num_feature": rng.uniform(0, 100, 100),
        }
    )

    df_test = pd.DataFrame(
        {
            "prediction": rng.uniform(0.1, 0.9, 20),
            "cat_feature": rng.choice(["A", "B", "C"], 20),
            "num_feature": rng.uniform(0, 100, 20),
        }
    )

    df_test_original = df_test.copy()

    calibrator = methods.PlattScalingWithFeatures()
    calibrator.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
    )

    _ = calibrator.predict(
        df=df_test,
        prediction_column_name="prediction",
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
    )

    pd.testing.assert_frame_equal(df_test, df_test_original)


def test_segmentwise_calibrator_fit_does_not_modify_input_dataframe(rng):
    df_train = pd.DataFrame(
        {
            "prediction": rng.uniform(0.1, 0.9, 100),
            "label": rng.randint(0, 2, 100),
            "segment": rng.choice(["A", "B"], 100),
            "weight": rng.uniform(0.5, 2.0, 100),
        }
    )

    df_train_original = df_train.copy()

    calibrator = methods.SegmentwiseCalibrator(methods.PlattScaling)
    calibrator.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        weight_column_name="weight",
        categorical_feature_column_names=["segment"],
    )

    pd.testing.assert_frame_equal(df_train, df_train_original)


def test_segmentwise_calibrator_predict_does_not_modify_input_dataframe(rng):
    df_train = pd.DataFrame(
        {
            "prediction": rng.uniform(0.1, 0.9, 100),
            "label": rng.randint(0, 2, 100),
            "segment": rng.choice(["A", "B"], 100),
        }
    )

    df_test = pd.DataFrame(
        {
            "prediction": rng.uniform(0.1, 0.9, 20),
            "segment": rng.choice(["A", "B"], 20),
        }
    )

    df_test_original = df_test.copy()

    calibrator = methods.SegmentwiseCalibrator(methods.PlattScaling)
    calibrator.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["segment"],
    )

    _ = calibrator.predict(
        df=df_test,
        prediction_column_name="prediction",
        categorical_feature_column_names=["segment"],
    )

    pd.testing.assert_frame_equal(df_test, df_test_original)


@pytest.mark.parametrize(
    "calibrator_class,calibrator_kwargs",
    [
        (
            methods.MCGrad,
            {
                "num_rounds": 2,
                "early_stopping": True,
                "early_stopping_use_crossvalidation": True,
                "n_folds": 2,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
        (
            methods.RegressionMCGrad,
            {
                "num_rounds": 2,
                "early_stopping": True,
                "early_stopping_use_crossvalidation": True,
                "n_folds": 2,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
    ],
)
def test_mcgrad_early_stopping_crossvalidation_does_not_modify_input_dataframe(
    calibrator_class, calibrator_kwargs
):
    df_train = pd.DataFrame(
        {
            "prediction": [0.1, 0.2, 0.3, 0.4, 0.5],
            "label": [0, 1, 0, 1, 0],
            "cat_feature": ["A", "B", "A", "B", "A"],
            "num_feature": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    df_train_original = df_train.copy()

    calibrator = calibrator_class(**calibrator_kwargs)
    calibrator.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
    )

    pd.testing.assert_frame_equal(df_train, df_train_original)


@pytest.mark.parametrize(
    "calibrator_class,calibrator_kwargs",
    [
        (
            methods.MCGrad,
            {
                "num_rounds": 2,
                "early_stopping": True,
                "early_stopping_use_crossvalidation": False,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
        (
            methods.RegressionMCGrad,
            {
                "num_rounds": 2,
                "early_stopping": True,
                "early_stopping_use_crossvalidation": False,
                "lightgbm_params": {"max_depth": 2, "n_estimators": 2},
            },
        ),
    ],
)
def test_mcgrad_early_stopping_holdout_does_not_modify_input_dataframe(
    calibrator_class, calibrator_kwargs, rng
):
    df_train = pd.DataFrame(
        {
            "prediction": rng.uniform(0.1, 0.9, 50),
            "label": rng.randint(0, 2, 50),
            "cat_feature": rng.choice(["A", "B"], 50),
            "num_feature": rng.uniform(0, 10, 50),
        }
    )

    df_val = pd.DataFrame(
        {
            "prediction": rng.uniform(0.1, 0.9, 30),
            "label": rng.randint(0, 2, 30),
            "cat_feature": rng.choice(["A", "B"], 30),
            "num_feature": rng.uniform(0, 10, 30),
        }
    )

    df_train_original = df_train.copy()
    df_val_original = df_val.copy()

    calibrator = calibrator_class(**calibrator_kwargs)
    calibrator.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["cat_feature"],
        numerical_feature_column_names=["num_feature"],
        df_val=df_val,
    )

    pd.testing.assert_frame_equal(df_train, df_train_original)
    pd.testing.assert_frame_equal(df_val, df_val_original)


def test_segmentwise_calibrator_ambiguous_segment_keys():
    """When multiple categorical features are joined with '_', ambiguous keys can occur.
    For example, features ["A_B", "C"] and ["A", "B_C"] both produce "A_B_C".
    This test verifies that such cases are handled correctly.
    """
    df_train = pd.DataFrame(
        {
            "prediction": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "label": [0, 0, 1, 1, 0, 1],
            "feature1": ["A_B", "A_B", "A_B", "A", "A", "A"],
            "feature2": ["C", "C", "C", "B_C", "B_C", "B_C"],
        }
    )

    calibrator = methods.SegmentwiseCalibrator(methods.PlattScaling)
    calibrator.fit(
        df_train=df_train,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["feature1", "feature2"],
    )

    num_segments = len(calibrator.calibrator_per_segment)
    assert num_segments == 2, (
        f"Expected 2 distinct segments for [('A_B', 'C'), ('A', 'B_C')], "
        f"but got {num_segments} segment(s): {list(calibrator.calibrator_per_segment.keys())}. "
        "This suggests segment keys are being incorrectly merged due to string concatenation."
    )


def test_additive_adjustment_fit_does_not_crash_on_zero_sum_weights():
    df = pd.DataFrame({"prediction": [0.1], "label": [0], "weight": [0]})
    cal = methods.AdditiveAdjustment()
    cal.fit(df, "prediction", "label", "weight")
    assert cal.offset == 0.0


def test_platt_scaling_with_features_fit_does_not_crash_on_single_class():
    df = pd.DataFrame(
        {"prediction": [0.1, 0.2, 0.3], "label": [0, 0, 0], "cat": ["a", "b", "a"]}
    )
    cal = methods.PlattScalingWithFeatures()
    cal.fit(df, "prediction", "label", categorical_feature_column_names=["cat"])
    # Should behave like identity or reasonable fallback
    preds = cal.predict(df, "prediction", categorical_feature_column_names=["cat"])
    # Since log_reg should be None (identity), preds should equal input predictions
    np.testing.assert_allclose(preds, df["prediction"].values)


def test_segmentwise_calibrator_fit_does_not_crash_on_empty_dataframe():
    # Bug 3: SegmentwiseCalibrator.predict crashes on empty DataFrame
    cal = methods.SegmentwiseCalibrator(methods.IdentityCalibrator)
    df_train = pd.DataFrame({"p": [0.1], "y": [0], "s": ["a"]})
    cal.fit(df_train, "p", "y", categorical_feature_column_names=["s"])

    df_empty = pd.DataFrame({"p": [], "s": []})
    preds = cal.predict(df_empty, "p", categorical_feature_column_names=["s"])
    assert len(preds) == 0


def test_mcgrad_subclass_defaults_missing_lightgbm_params():
    class SubMCGrad(methods._BaseMCGrad):
        DEFAULT_HYPERPARAMS = {
            "monotone_t": False,
            "early_stopping": True,
            "patience": 0,
            "n_folds": 5,
        }

        def _objective(self):
            return "binary"

        @property
        def _default_early_stopping_metric(self):
            m = Mock(spec=_ScoreFunctionInterface)
            m.name = "mock_metric"
            return m, True

        def _transform_predictions(self, p):
            return p

        def _inverse_transform_predictions(self, p):
            return p

        def _compute_unshrink_factor(self, y, p, w):
            return 1.0

        def _check_predictions(self, df, col):
            pass

        def _check_labels(self, df, col):
            pass

        def _predictions_out_of_bounds(self, p):
            return np.zeros_like(p, dtype=bool)

        @property
        def _cv_splitter(self):
            return Mock()

        @property
        def _holdout_splitter(self):
            return Mock()

        @property
        def _noop_splitter(self):
            return Mock()

    # This should not raise KeyError
    model = SubMCGrad()
    assert isinstance(model.lightgbm_params, dict)


def test_mcgrad_default_minimization_behavior():
    class AUCCalibrator(methods._BaseMCGrad):
        DEFAULT_HYPERPARAMS = {
            "monotone_t": False,
            "early_stopping": True,
            "patience": 0,
            "n_folds": 5,
            "lightgbm_params": {},
        }

        def _objective(self):
            return "binary"

        @property
        def _default_early_stopping_metric(self):
            m = Mock(spec=_ScoreFunctionInterface)
            m.name = "auc"
            # Return tuple with minimize=False since AUC should be maximized
            return m, False

        def _transform_predictions(self, p):
            return p

        def _inverse_transform_predictions(self, p):
            return p

        def _compute_unshrink_factor(self, y, p, w):
            return 1.0

        def _check_predictions(self, df, col):
            pass

        def _check_labels(self, df, col):
            pass

        def _predictions_out_of_bounds(self, p):
            return np.zeros_like(p, dtype=bool)

        @property
        def _cv_splitter(self):
            return Mock()

        @property
        def _holdout_splitter(self):
            return Mock()

        @property
        def _noop_splitter(self):
            return Mock()

    # The tuple return type ensures minimize_score is set correctly from the metric
    cal = AUCCalibrator(early_stopping=True)
    assert cal.early_stopping_minimize_score is False
