# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict

import warnings

import numpy as np
import pandas as pd
import pytest
from multicalibration import methods
from multicalibration._compat import DeprecatedAlias, DeprecatedAttributesMixin


def test_monotone_t_get_emits_deprecation_warning() -> None:
    model = methods.MCBoost()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = model.MONOTONE_T
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "MONOTONE_T is deprecated" in str(w[0].message)
        assert "monotone_t" in str(w[0].message)


def test_monotone_t_set_emits_deprecation_warning() -> None:
    model = methods.MCBoost()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.MONOTONE_T = True
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "MONOTONE_T is deprecated" in str(w[0].message)


def test_monotone_t_get_returns_correct_value() -> None:
    model = methods.MCBoost(monotone_t=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert model.MONOTONE_T is True

    model2 = methods.MCBoost(monotone_t=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert model2.MONOTONE_T is False


def test_monotone_t_set_updates_new_attribute() -> None:
    model = methods.MCBoost(monotone_t=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.MONOTONE_T = True
    assert model.monotone_t is True


def test_deprecated_attributes_mixin_has_expected_aliases() -> None:
    assert isinstance(DeprecatedAttributesMixin.MONOTONE_T, DeprecatedAlias)
    assert isinstance(DeprecatedAttributesMixin.EARLY_STOPPING, DeprecatedAlias)
    assert isinstance(
        DeprecatedAttributesMixin.EARLY_STOPPING_ESTIMATION_METHOD, DeprecatedAlias
    )
    assert isinstance(DeprecatedAttributesMixin.EARLY_STOPPING_TIMEOUT, DeprecatedAlias)
    assert isinstance(DeprecatedAttributesMixin.N_FOLDS, DeprecatedAlias)
    assert isinstance(DeprecatedAttributesMixin.NUM_ROUNDS, DeprecatedAlias)
    assert isinstance(DeprecatedAttributesMixin.PATIENCE, DeprecatedAlias)


@pytest.mark.parametrize(
    "new_class,deprecated_class",
    [
        (methods.MCGrad, methods.MCBoost),
        (methods.RegressionMCGrad, methods.RegressionMCBoost),
    ],
)
def test_mcgrad_and_mcboost_produce_equivalent_predictions(
    # pyre-ignore[2]
    new_class,
    # pyre-ignore[2]
    deprecated_class,
) -> None:
    rng = np.random.RandomState(42)
    n_samples = 100
    df = pd.DataFrame(
        {
            "prediction": rng.uniform(0.2, 0.8, n_samples),
            "label": rng.randint(0, 2, n_samples),
            "feature1": rng.choice(["A", "B", "C"], n_samples),
        }
    )

    mcgrad = new_class(
        early_stopping=False,
        num_rounds=2,
        random_state=42,
        lightgbm_params={"max_depth": 2, "n_estimators": 2},
    )
    mcboost = deprecated_class(
        early_stopping=False,
        num_rounds=2,
        random_state=42,
        lightgbm_params={"max_depth": 2, "n_estimators": 2},
    )

    mcgrad.fit(
        df_train=df,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["feature1"],
    )
    mcboost.fit(
        df_train=df,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=["feature1"],
    )

    test_df = df.sample(20, random_state=123)

    predictions_mcgrad = mcgrad.predict(
        df=test_df,
        prediction_column_name="prediction",
        categorical_feature_column_names=["feature1"],
    )
    predictions_mcboost = mcboost.predict(
        df=test_df,
        prediction_column_name="prediction",
        categorical_feature_column_names=["feature1"],
    )

    np.testing.assert_array_equal(predictions_mcgrad, predictions_mcboost)
