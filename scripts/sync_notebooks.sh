#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Script to sync tutorial notebooks from .py sources to .ipynb files.
# This is run automatically by pre-commit hooks, but can also be run manually.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIALS_DIR="$SCRIPT_DIR/../tutorials"

# Check if jupytext is installed
if ! command -v jupytext &> /dev/null; then
    echo "jupytext not found. Installing..."
    pip install jupytext
fi

# Convert all .py tutorials to .ipynb
for tutorial in "$TUTORIALS_DIR"/*.py; do
    if [ -f "$tutorial" ]; then
        basename="${tutorial%.py}"
        ipynb_file="${basename}.ipynb"

        echo "Converting: $(basename "$tutorial") -> $(basename "$ipynb_file")"
        jupytext --to notebook "$tutorial" -o "$ipynb_file"
    fi
done

echo "✓ All notebooks synced"
