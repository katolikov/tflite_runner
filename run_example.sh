#!/bin/bash

# Example script showing full workflow: build, deploy, and run on Android device

set -e

# Configuration
MODEL_PATH="${1:-model.tflite}"
INPUT_NPY="${2:-input.npy}"
OUTPUT_NPY="${3:-output.npy}"
OUTPUT_PNG="${4:-output.png}"

ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
DEVICE_DIR="/data/local/tmp/tflite_runner"

echo "=========================================="
echo "TensorFlow Lite Runner - Full Example"
echo "=========================================="
echo ""
echo "Model: $MODEL_PATH"
echo "Input: $INPUT_NPY"
echo "Output NPY: $OUTPUT_NPY"
echo "Output PNG: $OUTPUT_PNG"
echo ""

# Step 1: Build
echo "Step 1: Building for Android ($ANDROID_ABI)..."
./build.sh

# Step 2: Deploy
echo ""
echo "Step 2: Deploying to Android device..."
./deploy.sh

# Step 3: Push model and input data
echo ""
echo "Step 3: Pushing model and input data..."

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    echo ""
    echo "Usage: $0 <model.tflite> <input.npy> [output.npy] [output.png]"
    exit 1
fi

if [ ! -f "$INPUT_NPY" ]; then
    echo "Error: Input NPY file not found: $INPUT_NPY"
    echo ""
    echo "Usage: $0 <model.tflite> <input.npy> [output.npy] [output.png]"
    exit 1
fi

adb push "$MODEL_PATH" "$DEVICE_DIR/"
adb push "$INPUT_NPY" "$DEVICE_DIR/"

MODEL_BASENAME=$(basename "$MODEL_PATH")
INPUT_BASENAME=$(basename "$INPUT_NPY")

# Step 4: Run inference
echo ""
echo "Step 4: Running inference on device..."
echo ""

adb shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=. ./tflite_runner \
    --model $MODEL_BASENAME \
    --input $INPUT_BASENAME \
    --output $OUTPUT_NPY \
    --output-png $OUTPUT_PNG"

# Step 5: Pull results
echo ""
echo "Step 5: Retrieving results..."
adb pull "$DEVICE_DIR/$OUTPUT_NPY" .
adb pull "$DEVICE_DIR/$OUTPUT_PNG" . 2>/dev/null || echo "Note: PNG output not available"

echo ""
echo "=========================================="
echo "Example completed successfully!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  ./$OUTPUT_NPY"
if [ -f "$OUTPUT_PNG" ]; then
    echo "  ./$OUTPUT_PNG"
fi
