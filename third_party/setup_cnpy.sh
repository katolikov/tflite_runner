#!/bin/bash

# Script to download and setup cnpy library for NPY file I/O

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CNPY_DIR="$SCRIPT_DIR/cnpy"

echo "Setting up cnpy library for NPY file support..."

if [ -d "$CNPY_DIR" ]; then
    echo "cnpy directory already exists, removing..."
    rm -rf "$CNPY_DIR"
fi

echo "Cloning cnpy repository..."
git clone https://github.com/rogersce/cnpy.git "$CNPY_DIR"

echo ""
echo "cnpy library downloaded successfully to:"
echo "  $CNPY_DIR"
echo ""
echo "Files available:"
echo "  - cnpy.h (header file)"
echo "  - cnpy.cpp (implementation)"
