#!/bin/bash

# Alternative script to download TensorFlow Lite headers directly from GitHub

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFLITE_VERSION="v2.14.0"
DOWNLOAD_DIR="$SCRIPT_DIR/tensorflow"

echo "Downloading TensorFlow Lite headers from GitHub..."
echo "Version: $TFLITE_VERSION"

# Create directory structure
mkdir -p "$DOWNLOAD_DIR/lite"

# Clone TensorFlow Lite headers (sparse checkout)
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "Cloning TensorFlow repository (lite directory only)..."
git clone --depth 1 --filter=blob:none --sparse https://github.com/tensorflow/tensorflow.git -b "$TFLITE_VERSION"
cd tensorflow
git sparse-checkout set tensorflow/lite

echo "Copying TensorFlow Lite headers..."
cp -r tensorflow/lite "$DOWNLOAD_DIR/"

# Also get flatbuffers
echo "Getting FlatBuffers..."
mkdir -p "$SCRIPT_DIR/tensorflow/lite/tools/make/downloads"
cd "$SCRIPT_DIR/tensorflow/lite/tools/make/downloads"
git clone --depth 1 https://github.com/google/flatbuffers.git

# Clean up
echo "Cleaning up..."
rm -rf "$TEMP_DIR"

echo ""
echo "TensorFlow Lite headers downloaded successfully to:"
echo "  $DOWNLOAD_DIR"
