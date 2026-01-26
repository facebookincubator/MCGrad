# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# @noautodeps - This directory is OSS-only (synced to GitHub via ShipIt)

"""MCGrad: Production-ready multicalibration for machine learning."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("MCGrad")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
