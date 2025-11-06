#!/bin/bash

# Build script for TensorFlow Lite Runner on Android

set -e

# Configuration
BUILD_TYPE="${BUILD_TYPE:-Release}"
ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
ANDROID_PLATFORM="${ANDROID_PLATFORM:-android-29}"
BUILD_DIR="build-${ANDROID_ABI}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "TensorFlow Lite Runner - Android Build"
echo "=========================================="
echo ""

# Check for Android NDK
if [ -z "$ANDROID_NDK_HOME" ] && [ -z "$ANDROID_NDK" ]; then
    echo -e "${RED}Error: ANDROID_NDK_HOME or ANDROID_NDK environment variable not set${NC}"
    echo ""
    echo "Please set the ANDROID_NDK_HOME environment variable to your Android NDK path:"
    echo "  export ANDROID_NDK_HOME=/path/to/android-ndk"
    echo ""
    echo "You can download the Android NDK from:"
    echo "  https://developer.android.com/ndk/downloads"
    exit 1
fi

NDK_PATH="${ANDROID_NDK_HOME:-$ANDROID_NDK}"
echo -e "${GREEN}Android NDK found: $NDK_PATH${NC}"

# Check if third_party dependencies are set up
if [ ! -d "third_party/tensorflow" ] || [ ! -d "third_party/cnpy" ] || [ ! -d "third_party/stb" ]; then
    echo -e "${YELLOW}Warning: Third-party dependencies not found${NC}"
    echo "Running dependency setup..."
    cd third_party
    chmod +x setup_all.sh
    ./setup_all.sh
    cd ..
fi

echo ""
echo "Build Configuration:"
echo "  Build Type: $BUILD_TYPE"
echo "  Android ABI: $ANDROID_ABI"
echo "  Android Platform: $ANDROID_PLATFORM"
echo "  Build Directory: $BUILD_DIR"
echo ""

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring CMake..."
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="$NDK_PATH/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$ANDROID_ABI" \
    -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DANDROID_STL=c++_shared

echo ""
echo "Building..."
cmake --build . --config "$BUILD_TYPE" -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

cd ..

echo ""
echo -e "${GREEN}=========================================="
echo "Build completed successfully!"
echo "==========================================${NC}"
echo ""
echo "Executable location:"
echo "  $BUILD_DIR/tflite_runner"
echo ""
echo "To build for different architectures, use:"
echo "  ANDROID_ABI=arm64-v8a ./build.sh"
echo "  ANDROID_ABI=armeabi-v7a ./build.sh"
echo ""
echo "To push to Android device:"
echo "  adb push $BUILD_DIR/tflite_runner /data/local/tmp/"
echo "  adb shell chmod +x /data/local/tmp/tflite_runner"
