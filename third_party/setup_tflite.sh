#!/bin/bash

# Script to download TensorFlow Lite prebuilt libraries for Android NDK

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFLITE_VERSION="2.14.0"
DOWNLOAD_DIR="$SCRIPT_DIR"

echo "Downloading TensorFlow Lite libraries for Android NDK..."
echo "Version: $TFLITE_VERSION"
echo "Target directory: $DOWNLOAD_DIR"

# Create directories
mkdir -p "$DOWNLOAD_DIR/libs/arm64-v8a"
mkdir -p "$DOWNLOAD_DIR/libs/armeabi-v7a"

# Download prebuilt libraries from TensorFlow Lite releases
echo ""
echo "Downloading TensorFlow Lite prebuilt libraries..."

# For arm64-v8a
echo "Downloading arm64-v8a libraries..."
TFLITE_ARM64_URL="https://github.com/tensorflow/tensorflow/releases/download/v${TFLITE_VERSION}/libtensorflowlite_c.so"

# Note: TensorFlow doesn't provide official prebuilt .a libraries for Android
# We'll need to provide instructions for building from source or using AAR
echo ""
echo "WARNING: TensorFlow Lite prebuilt static libraries (.a) are not officially provided."
echo "You have two options:"
echo ""
echo "Option 1: Extract from AAR (shared libraries only):"

# Download AAR files which contain .so files
echo "Downloading TensorFlow Lite AAR packages..."
TFLITE_AAR_URL="https://repo1.maven.org/maven2/org/tensorflow/tensorflow-lite/${TFLITE_VERSION}/tensorflow-lite-${TFLITE_VERSION}.aar"
TFLITE_GPU_AAR_URL="https://repo1.maven.org/maven2/org/tensorflow/tensorflow-lite-gpu/${TFLITE_VERSION}/tensorflow-lite-gpu-${TFLITE_VERSION}.aar"

# Check if wget or curl is available
if command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget -O"
elif command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl -L -o"
else
    echo "Error: Neither wget nor curl found. Please install one of them."
    exit 1
fi

$DOWNLOAD_CMD "$DOWNLOAD_DIR/tensorflow-lite.aar" "$TFLITE_AAR_URL"
$DOWNLOAD_CMD "$DOWNLOAD_DIR/tensorflow-lite-gpu.aar" "$TFLITE_GPU_AAR_URL"

# Extract AAR files (they are ZIP archives)
echo ""
echo "Extracting AAR packages..."
cd "$DOWNLOAD_DIR"
unzip -q -o tensorflow-lite.aar -d tensorflow-lite-extracted
unzip -q -o tensorflow-lite-gpu.aar -d tensorflow-lite-gpu-extracted

# Copy shared libraries (.so files)
echo ""
echo "Copying shared libraries..."
if [ -d "tensorflow-lite-extracted/jni" ]; then
    for arch in arm64-v8a armeabi-v7a; do
        if [ -d "tensorflow-lite-extracted/jni/$arch" ]; then
            mkdir -p "$DOWNLOAD_DIR/libs/$arch"
            cp tensorflow-lite-extracted/jni/$arch/*.so "$DOWNLOAD_DIR/libs/$arch/" 2>/dev/null || true
        fi
    done
fi

if [ -d "tensorflow-lite-gpu-extracted/jni" ]; then
    for arch in arm64-v8a armeabi-v7a; do
        if [ -d "tensorflow-lite-gpu-extracted/jni/$arch" ]; then
            mkdir -p "$DOWNLOAD_DIR/libs/$arch"
            cp tensorflow-lite-gpu-extracted/jni/$arch/*.so "$DOWNLOAD_DIR/libs/$arch/" 2>/dev/null || true
        fi
    done
fi

# Create dummy libtensorflowlite.a for linking (will use .so instead)
echo ""
echo "Creating library stubs..."
for arch in arm64-v8a armeabi-v7a; do
    if [ ! -f "$DOWNLOAD_DIR/libs/$arch/libtensorflowlite.a" ]; then
        # Create empty archive as placeholder
        # The actual linking will use .so files
        touch "$DOWNLOAD_DIR/libs/$arch/libtensorflowlite_stub.a"
    fi
done

# Clean up
echo ""
echo "Cleaning up temporary files..."
rm -rf tensorflow-lite-extracted tensorflow-lite-gpu-extracted
rm -f tensorflow-lite.aar tensorflow-lite-gpu.aar

# List what was downloaded
echo ""
echo "==================================="
echo "TensorFlow Lite setup completed!"
echo "==================================="
echo ""
echo "Shared libraries (.so) installed in:"
for arch in arm64-v8a armeabi-v7a; do
    if [ -d "$DOWNLOAD_DIR/libs/$arch" ]; then
        echo "  $DOWNLOAD_DIR/libs/$arch/:"
        ls -lh "$DOWNLOAD_DIR/libs/$arch/" | grep "\.so" || echo "    (no .so files found)"
    fi
done

echo ""
echo "IMPORTANT: TensorFlow Lite AAR contains shared libraries (.so) only."
echo "For static linking (.a), you need to build TensorFlow Lite from source."
echo ""
echo "For this project, we'll use shared libraries with dynamic linking."
echo "The CMakeLists.txt will be configured to link against .so files."
echo "  $DOWNLOAD_DIR/tensorflow/"
echo ""
echo "Note: If headers are not found, you may need to manually download"
echo "the TensorFlow Lite source headers from:"
echo "https://github.com/tensorflow/tensorflow/tree/v${TFLITE_VERSION}/tensorflow/lite"
