#!/bin/bash

# Script to deploy and run TensorFlow Lite Runner on Android device

set -e

# Configuration
ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
BUILD_DIR="build-${ANDROID_ABI}"
DEVICE_DIR="/data/local/tmp/tflite_runner"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "TensorFlow Lite Runner - Android Deploy"
echo "=========================================="
echo ""

# Check if adb is available
if ! command -v adb &> /dev/null; then
    echo "Error: adb not found. Please install Android SDK Platform Tools."
    exit 1
fi

# Check if device is connected
if ! adb devices | grep -q "device$"; then
    echo "Error: No Android device connected or device unauthorized."
    echo "Please connect a device and ensure USB debugging is enabled."
    exit 1
fi

DEVICE_NAME=$(adb shell getprop ro.product.model)
echo -e "${GREEN}Connected device: $DEVICE_NAME${NC}"

# Check if executable exists
if [ ! -f "$BUILD_DIR/tflite_runner" ]; then
    echo -e "${YELLOW}Executable not found. Building first...${NC}"
    ./build.sh
fi

echo ""
echo "Deploying to device..."

# Create directory on device
adb shell "mkdir -p $DEVICE_DIR"

# Push executable
echo "Pushing executable..."
adb push "$BUILD_DIR/tflite_runner" "$DEVICE_DIR/"

# Push required shared libraries
echo "Pushing shared libraries..."
if [ -f "third_party/libs/$ANDROID_ABI/libtensorflowlite_jni.so" ]; then
    adb push "third_party/libs/$ANDROID_ABI/libtensorflowlite_jni.so" "$DEVICE_DIR/"
fi

if [ -f "third_party/libs/$ANDROID_ABI/libtensorflowlite_gpu_jni.so" ]; then
    adb push "third_party/libs/$ANDROID_ABI/libtensorflowlite_gpu_jni.so" "$DEVICE_DIR/"
fi

if [ -f "$BUILD_DIR/libc++_shared.so" ]; then
    adb push "$BUILD_DIR/libc++_shared.so" "$DEVICE_DIR/"
fi

# Make executable
adb shell "chmod +x $DEVICE_DIR/tflite_runner"

echo ""
echo -e "${GREEN}=========================================="
echo "Deployment completed successfully!"
echo "==========================================${NC}"
echo ""
echo "Executable location on device:"
echo "  $DEVICE_DIR/tflite_runner"
echo ""
echo "To run on device:"
echo "  1. Push your model and input files:"
echo "     adb push model.tflite $DEVICE_DIR/"
echo "     adb push input.npy $DEVICE_DIR/"
echo ""
echo "  2. Run the executable:"
echo "     adb shell \"cd $DEVICE_DIR && LD_LIBRARY_PATH=. ./tflite_runner --model model.tflite --input input.npy --output output.npy\""
echo ""
echo "  3. Pull results:"
echo "     adb pull $DEVICE_DIR/output.npy ."
echo "     adb pull $DEVICE_DIR/output.png ."
