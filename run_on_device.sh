#!/bin/bash

# Script to list available Android devices and run inference on a specific device

set -e

# Configuration
ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
BUILD_DIR="build-${ANDROID_ABI}"
DEVICE_DIR="/data/local/tmp/tflite_runner"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

function show_usage() {
    echo "Usage: $0 [OPTIONS] <model.tflite> <input.npy> <output.npy> [output.png]"
    echo ""
    echo "Options:"
    echo "  -d, --device SERIAL    Target specific device by serial number"
    echo "  -l, --list             List available devices"
    echo "  --no-gpu              Disable GPU acceleration"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --list"
    echo "  $0 -d R5CR30KPVEZ model.tflite input.npy output.npy"
    echo "  $0 model.tflite input.npy output.npy output.png"
}

function list_devices() {
    echo -e "${BLUE}=== Available Android Devices ===${NC}"
    echo ""
    
    devices=$(adb devices | tail -n +2 | grep -v "^$")
    
    if [ -z "$devices" ]; then
        echo -e "${RED}No devices found!${NC}"
        echo "Please connect a device and enable USB debugging."
        exit 1
    fi
    
    echo "$devices" | while read -r line; do
        serial=$(echo "$line" | awk '{print $1}')
        status=$(echo "$line" | awk '{print $2}')
        
        if [ "$status" = "device" ]; then
            # Get device info
            model=$(adb -s "$serial" shell getprop ro.product.model 2>/dev/null | tr -d '\r')
            manufacturer=$(adb -s "$serial" shell getprop ro.product.manufacturer 2>/dev/null | tr -d '\r')
            android_version=$(adb -s "$serial" shell getprop ro.build.version.release 2>/dev/null | tr -d '\r')
            cpu_abi=$(adb -s "$serial" shell getprop ro.product.cpu.abi 2>/dev/null | tr -d '\r')
            
            echo -e "${GREEN}Serial:       ${NC}$serial"
            echo -e "${GREEN}Model:        ${NC}$manufacturer $model"
            echo -e "${GREEN}Android:      ${NC}$android_version"
            echo -e "${GREEN}Architecture: ${NC}$cpu_abi"
            echo ""
        else
            echo -e "${YELLOW}Serial:       ${NC}$serial"
            echo -e "${YELLOW}Status:       ${NC}$status (not available)"
            echo ""
        fi
    done
}

function check_device() {
    local device_serial="$1"
    
    if [ -z "$device_serial" ]; then
        # No specific device, check if exactly one is connected
        device_count=$(adb devices | tail -n +2 | grep "device$" | wc -l)
        if [ "$device_count" -eq 0 ]; then
            echo -e "${RED}Error: No devices connected${NC}"
            list_devices
            exit 1
        elif [ "$device_count" -gt 1 ]; then
            echo -e "${RED}Error: Multiple devices connected${NC}"
            echo "Please specify which device to use with -d option:"
            echo ""
            list_devices
            exit 1
        fi
    else
        # Check if specified device exists
        if ! adb devices | grep -q "$device_serial"; then
            echo -e "${RED}Error: Device $device_serial not found${NC}"
            list_devices
            exit 1
        fi
    fi
}

function get_adb_cmd() {
    local device_serial="$1"
    if [ -z "$device_serial" ]; then
        echo "adb"
    else
        echo "adb -s $device_serial"
    fi
}

# Parse arguments
DEVICE_SERIAL=""
LIST_ONLY=false
USE_GPU=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--device)
            DEVICE_SERIAL="$2"
            shift 2
            ;;
        -l|--list)
            LIST_ONLY=true
            shift
            ;;
        --no-gpu)
            USE_GPU=false
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

# Handle list-only mode
if [ "$LIST_ONLY" = true ]; then
    list_devices
    exit 0
fi

# Validate arguments
if [ $# -lt 3 ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo ""
    show_usage
    exit 1
fi

MODEL_PATH="$1"
INPUT_NPY="$2"
OUTPUT_NPY="$3"
OUTPUT_PNG="${4:-}"

# Check if files exist
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model file not found: $MODEL_PATH${NC}"
    exit 1
fi

if [ ! -f "$INPUT_NPY" ]; then
    echo -e "${RED}Error: Input NPY file not found: $INPUT_NPY${NC}"
    exit 1
fi

# Check device
check_device "$DEVICE_SERIAL"
ADB=$(get_adb_cmd "$DEVICE_SERIAL")

# Get device info
DEVICE_MODEL=$($ADB shell getprop ro.product.model 2>/dev/null | tr -d '\r')
DEVICE_CPU_ABI=$($ADB shell getprop ro.product.cpu.abi 2>/dev/null | tr -d '\r')

echo -e "${BLUE}=========================================="
echo "TensorFlow Lite Runner - Device Specific"
echo -e "==========================================${NC}"
echo ""
if [ -n "$DEVICE_SERIAL" ]; then
    echo -e "${GREEN}Target Device Serial:${NC} $DEVICE_SERIAL"
fi
echo -e "${GREEN}Device Model:        ${NC} $DEVICE_MODEL"
echo -e "${GREEN}Device Architecture: ${NC} $DEVICE_CPU_ABI"
echo -e "${GREEN}Model:               ${NC} $MODEL_PATH"
echo -e "${GREEN}Input:               ${NC} $INPUT_NPY"
echo -e "${GREEN}Output NPY:          ${NC} $OUTPUT_NPY"
if [ -n "$OUTPUT_PNG" ]; then
    echo -e "${GREEN}Output PNG:          ${NC} $OUTPUT_PNG"
fi
echo -e "${GREEN}GPU Acceleration:    ${NC} $([ "$USE_GPU" = true ] && echo "Enabled" || echo "Disabled")"
echo ""

# Check if architecture matches
if [[ "$DEVICE_CPU_ABI" == "arm64"* ]] && [ "$ANDROID_ABI" != "arm64-v8a" ]; then
    echo -e "${YELLOW}Warning: Device is 64-bit but binary is built for $ANDROID_ABI${NC}"
    echo "Consider rebuilding with: ANDROID_ABI=arm64-v8a ./build.sh"
    echo ""
elif [[ "$DEVICE_CPU_ABI" == "armeabi"* ]] && [ "$ANDROID_ABI" != "armeabi-v7a" ]; then
    echo -e "${YELLOW}Warning: Device is 32-bit but binary is built for $ANDROID_ABI${NC}"
    echo "Consider rebuilding with: ANDROID_ABI=armeabi-v7a ./build.sh"
    echo ""
fi

# Build if needed
if [ ! -f "$BUILD_DIR/tflite_runner" ]; then
    echo -e "${YELLOW}Executable not found. Building first...${NC}"
    ./build.sh
fi

# Deploy
echo "Step 1: Deploying to device..."
$ADB shell "mkdir -p $DEVICE_DIR"
$ADB push "$BUILD_DIR/tflite_runner" "$DEVICE_DIR/" > /dev/null

# Push shared libraries
if [ -f "third_party/libs/$ANDROID_ABI/libtensorflowlite_jni.so" ]; then
    $ADB push "third_party/libs/$ANDROID_ABI/libtensorflowlite_jni.so" "$DEVICE_DIR/" > /dev/null
fi
if [ -f "third_party/libs/$ANDROID_ABI/libtensorflowlite_gpu_jni.so" ]; then
    $ADB push "third_party/libs/$ANDROID_ABI/libtensorflowlite_gpu_jni.so" "$DEVICE_DIR/" > /dev/null
fi

$ADB shell "chmod +x $DEVICE_DIR/tflite_runner"

# Push model and input
echo "Step 2: Pushing model and input data..."
MODEL_BASENAME=$(basename "$MODEL_PATH")
INPUT_BASENAME=$(basename "$INPUT_NPY")

$ADB push "$MODEL_PATH" "$DEVICE_DIR/" > /dev/null
$ADB push "$INPUT_NPY" "$DEVICE_DIR/" > /dev/null

# Construct command
CMD="cd $DEVICE_DIR && LD_LIBRARY_PATH=. ./tflite_runner --model $MODEL_BASENAME --input $INPUT_BASENAME --output $OUTPUT_NPY"

if [ -n "$OUTPUT_PNG" ]; then
    CMD="$CMD --output-png $OUTPUT_PNG"
fi

if [ "$USE_GPU" = false ]; then
    CMD="$CMD --no-gpu"
fi

# Run inference
echo ""
echo "Step 3: Running inference on device..."
echo -e "${BLUE}============================================${NC}"
$ADB shell "$CMD"
echo -e "${BLUE}============================================${NC}"

# Pull results
echo ""
echo "Step 4: Retrieving results..."
$ADB pull "$DEVICE_DIR/$OUTPUT_NPY" . > /dev/null
echo "  ✓ $OUTPUT_NPY"

if [ -n "$OUTPUT_PNG" ]; then
    if $ADB shell "test -f $DEVICE_DIR/$OUTPUT_PNG" 2>/dev/null; then
        $ADB pull "$DEVICE_DIR/$OUTPUT_PNG" . > /dev/null
        echo "  ✓ $OUTPUT_PNG"
    fi
fi

echo ""
echo -e "${GREEN}=========================================="
echo "Execution completed successfully!"
echo -e "==========================================${NC}"
