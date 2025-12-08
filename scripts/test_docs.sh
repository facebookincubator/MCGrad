#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Script to test both Sphinx and Docusaurus documentation builds

set -e  # Exit on error

echo "🧪 Testing Documentation Builds"
echo "================================"
echo ""

# Test 1: Build Sphinx documentation
echo "📚 Step 1: Building Sphinx documentation..."
echo "-------------------------------------------"
cd sphinx

# Check if sphinx-build is available
if ! command -v sphinx-build &> /dev/null; then
    echo "❌ sphinx-build not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Build the docs
echo "Building HTML docs..."
make clean
make html

if [ $? -eq 0 ]; then
    echo "✅ Sphinx build successful!"
    echo "   Output: sphinx/build/html/index.html"
else
    echo "❌ Sphinx build failed!"
    exit 1
fi

cd ..
echo ""

# Test 2: Build Docusaurus documentation
echo "🌐 Step 2: Building Docusaurus website..."
echo "-------------------------------------------"
cd website

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "❌ npm not found. Please install Node.js"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

# Build the docs
echo "Building Docusaurus site..."
npm run build

if [ $? -eq 0 ]; then
    echo "✅ Docusaurus build successful!"
    echo "   Output: website/build/index.html"
else
    echo "❌ Docusaurus build failed!"
    exit 1
fi

cd ..
echo ""

# Summary
echo "🎉 All documentation builds successful!"
echo "========================================"
echo ""
echo "To view the documentation:"
echo ""
echo "Sphinx (API Reference):"
echo "  cd sphinx/build/html && python3 -m http.server 8000"
echo "  Then open: http://localhost:8000"
echo ""
echo "Docusaurus (User Guide):"
echo "  cd website && npm run serve"
echo "  Then open: http://localhost:3000"
echo ""
echo "Note: The 'API Reference' link in Docusaurus points to"
echo "https://mcgrad.readthedocs.io/ which won't work until"
echo "you set up the project on ReadTheDocs."
