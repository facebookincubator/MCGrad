# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Pytest configuration for MCGrad test suite.

This file configures platform-specific test skipping to handle known
compatibility issues with certain dependencies on Apple Silicon (ARM64).
"""

import platform
import sys

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "arm64_incompatible: mark test as incompatible with Apple Silicon (ARM64)",
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip tests marked as arm64_incompatible on Apple Silicon.

    This allows tests to run on all platforms (Linux x86_64, Intel Mac, Windows)
    while gracefully skipping known problematic tests on ARM64 Macs.
    """
    # Check if running on Apple Silicon (ARM64 Mac)
    is_arm64_mac = sys.platform == "darwin" and platform.machine() == "arm64"

    if is_arm64_mac:
        skip_arm64 = pytest.mark.skip(
            reason="Test incompatible with Apple Silicon due to PyTorch/Ax-Platform ARM64 issues. "
            "See: https://github.com/facebook/Ax/issues/2537"
        )
        for item in items:
            if "arm64_incompatible" in item.keywords:
                item.add_marker(skip_arm64)
