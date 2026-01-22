# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

import pandas as pd

from .._compat import groupby_apply


def test_groupby_apply_basic_functionality():
    df = pd.DataFrame({"group": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})
    grouped = df.groupby("group")

    result = groupby_apply(grouped, lambda x: x["value"].sum())

    assert result["A"] == 3
    assert result["B"] == 7
