#!/bin/bash

# Master setup script to download all third-party dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "TensorFlow Lite Runner - Dependency Setup"
echo "=========================================="
echo ""

# Make scripts executable
chmod +x "$SCRIPT_DIR/setup_tflite.sh"
chmod +x "$SCRIPT_DIR/setup_headers.sh"
chmod +x "$SCRIPT_DIR/setup_cnpy.sh"
chmod +x "$SCRIPT_DIR/setup_stb.sh"

# Step 1: Download TensorFlow Lite libraries
echo "Step 1: Setting up TensorFlow Lite libraries..."
"$SCRIPT_DIR/setup_tflite.sh"

# Step 2: Download TensorFlow Lite headers
echo ""
echo "Step 2: Setting up TensorFlow Lite headers..."
"$SCRIPT_DIR/setup_headers.sh"

# Step 3: Setup cnpy
echo ""
echo "Step 3: Setting up cnpy library..."
"$SCRIPT_DIR/setup_cnpy.sh"

# Step 4: Setup stb_image_write
echo ""
echo "Step 4: Setting up stb_image_write..."
"$SCRIPT_DIR/setup_stb.sh"

echo ""
echo "=========================================="
echo "All dependencies setup completed!"
echo "=========================================="
echo ""
echo "Directory structure:"
echo "  third_party/"
echo "  ├── tensorflow/          (TensorFlow Lite headers)"
echo "  ├── libs/                (Prebuilt TFLite libraries)"
echo "  │   ├── arm64-v8a/"
echo "  │   └── armeabi-v7a/"
echo "  ├── cnpy/                (NPY file I/O library)"
echo "  └── stb/                 (STB image library)"
echo ""
echo "You can now run ./build.sh to compile the project!"
