# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict

from typing import Any, Callable

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

# pandas >= 2.2.0 deprecated the default include_groups=True behavior
_PANDAS_GROUPBY_INCLUDE_GROUPS: bool
try:
    _pandas_version: tuple[int, ...] = tuple(
        int(x)
        for x in pd.__version__.split(".")[:2]  # pyre-ignore[16]
    )
    _PANDAS_GROUPBY_INCLUDE_GROUPS = _pandas_version >= (2, 2)
except (ValueError, AttributeError):
    _PANDAS_GROUPBY_INCLUDE_GROUPS = False


def groupby_apply(
    grouped: pd.core.groupby.DataFrameGroupBy,
    func: Callable[..., Any],
    **kwargs: Any,
) -> pd.DataFrame | pd.Series:
    """
    Wrapper for DataFrame.groupby().apply() that handles the include_groups
    deprecation across pandas versions.
    """
    if _PANDAS_GROUPBY_INCLUDE_GROUPS:
        return grouped.apply(func, include_groups=False, **kwargs)
    else:
        return grouped.apply(func, **kwargs)


def create_kbins_discretizer(**kwargs: Any) -> KBinsDiscretizer:
    """
    Factory for KBinsDiscretizer.
    Enforces 'linear' method on newer sklearn versions to maintain
    mathematical consistency with older versions and silence warnings.
    """
    kwargs = kwargs.copy()

    # Attempt 1: Try to explicitly set 'linear'.
    # This silences the FutureWarning in newer versions (>=1.6) because we
    # are no longer relying on the default.
    kwargs.setdefault("quantile_method", "linear")

    try:
        return KBinsDiscretizer(**kwargs)
    except TypeError:
        # Attempt 2: Fallback for older sklearn (<1.6).
        # These versions don't have 'quantile_method' but default to 'linear' anyway.
        kwargs.pop("quantile_method", None)
        return KBinsDiscretizer(**kwargs)
