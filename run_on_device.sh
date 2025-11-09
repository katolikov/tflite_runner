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

function detect_libcxx_path() {
    local ndk_path="$1"
    local abi="$2"
    local triple=""

    case "$abi" in
        arm64-v8a) triple="aarch64-linux-android" ;;
        armeabi-v7a) triple="arm-linux-androideabi" ;;
        x86) triple="i686-linux-android" ;;
        x86_64) triple="x86_64-linux-android" ;;
        *)
            return 1
            ;;
    esac

    local prebuilt_dir="$ndk_path/toolchains/llvm/prebuilt"
    if [ ! -d "$prebuilt_dir" ]; then
        return 1
    fi

    for host_dir in "$prebuilt_dir"/*; do
        if [ -d "$host_dir/sysroot/usr/lib/$triple" ]; then
            local candidate="$host_dir/sysroot/usr/lib/$triple/libc++_shared.so"
            if [ -f "$candidate" ]; then
                echo "$candidate"
                return 0
            fi
        fi
    done

    return 1
}

function derive_png_filename() {
    local npy="$1"
    if [[ "$npy" == *.npy ]]; then
        echo "${npy%.npy}.png"
    else
        echo "${npy}.png"
    fi
}

function show_usage() {
    echo "Usage: $0 [OPTIONS] <model.tflite> <output.npy> [output.png]"
    echo "       Legacy positional input: $0 [OPTIONS] <model.tflite> <input.npy> <output.npy> [output.png]"
    echo ""
    echo "Options:"
    echo "  -d, --device SERIAL    Target specific device by serial number"
    echo "  -l, --list             List available devices"
    echo "  --input PATH          Add an input .npy file (repeat for multiple inputs)"
    echo "  --inputs CSV          Comma-separated list of input .npy files"
    echo "  --output PATH         Add an output .npy path (repeat). If omitted, files will be auto-named."
    echo "  --output-dir PATH     Host directory for auto outputs (default: ./outputs)"
    echo "  --no-gpu              Disable GPU acceleration"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --list"
    echo "  $0 -d SERIAL model.tflite output.npy --input alpha.npy --input img1.npy --input img2.npy"
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
INPUT_FILES=()
OUTPUT_FILES=()
HOST_OUTPUT_DIR="outputs"
POSITIONAL=()
OUTPUT_PNG=""
CUSTOM_PNG_FOR_FIRST=false

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
        --input)
            if [ $# -lt 2 ]; then
                echo -e "${RED}Error: --input requires a path${NC}"
                exit 1
            fi
            INPUT_FILES+=("$2")
            shift 2
            ;;
        --inputs)
            if [ $# -lt 2 ]; then
                echo -e "${RED}Error: --inputs requires a comma-separated list${NC}"
                exit 1
            fi
            IFS=',' read -ra EXTRA_INPUTS <<< "$2"
            for item in "${EXTRA_INPUTS[@]}"; do
                trimmed="$(echo "$item" | xargs)"
                if [ -n "$trimmed" ]; then
                    INPUT_FILES+=("$trimmed")
                fi
            done
            shift 2
            ;;
        --output)
            if [ $# -lt 2 ]; then
                echo -e "${RED}Error: --output requires a path${NC}"
                exit 1
            fi
            OUTPUT_FILES+=("$2")
            shift 2
            ;;
        --output-dir)
            if [ $# -lt 2 ]; then
                echo -e "${RED}Error: --output-dir requires a path${NC}"
                exit 1
            fi
            HOST_OUTPUT_DIR="$2"
            shift 2
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
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL[@]}"

# Handle list-only mode
if [ "$LIST_ONLY" = true ]; then
    list_devices
    exit 0
fi

NDK_PATH="${ANDROID_NDK_HOME:-$ANDROID_NDK}"
if [ -z "$NDK_PATH" ]; then
    echo -e "${RED}Error: ANDROID_NDK_HOME or ANDROID_NDK must be set to locate libc++_shared.so${NC}"
    exit 1
fi

# Validate arguments / legacy positional handling
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: Missing model path${NC}"
    echo ""
    show_usage
    exit 1
fi

MODEL_PATH="$1"
shift

if [ ${#INPUT_FILES[@]} -eq 0 ]; then
    if [ $# -lt 1 ]; then
        echo -e "${RED}Error: Missing input .npy (use --input)${NC}"
        exit 1
    fi
    INPUT_FILES+=("$1")
    shift
fi

if [ ${#OUTPUT_FILES[@]} -eq 0 ] && [ $# -ge 1 ]; then
    OUTPUT_FILES+=("$1")
    shift
fi

if [ $# -ge 1 ]; then
    OUTPUT_PNG="$1"
    CUSTOM_PNG_FOR_FIRST=true
fi

# Check if files exist
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model file not found: $MODEL_PATH${NC}"
    exit 1
fi

for input_file in "${INPUT_FILES[@]}"; do
    if [ ! -f "$input_file" ]; then
        echo -e "${RED}Error: Input NPY file not found: $input_file${NC}"
        exit 1
    fi
done

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
echo -e "${GREEN}Inputs:${NC}"
for idx in "${!INPUT_FILES[@]}"; do
    echo "  [$idx] ${INPUT_FILES[$idx]}"
done
if [ ${#OUTPUT_FILES[@]} -gt 0 ]; then
    echo -e "${GREEN}Output files:${NC}"
    for idx in "${!OUTPUT_FILES[@]}"; do
        echo "  [$idx] ${OUTPUT_FILES[$idx]}"
    done
else
    echo -e "${GREEN}Outputs:${NC} auto-named in \"$HOST_OUTPUT_DIR\" (will mirror device output directory)"
fi
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

MODEL_BASENAME=$(basename "$MODEL_PATH")
INPUT_BASENAMES=()
for input_file in "${INPUT_FILES[@]}"; do
    INPUT_BASENAMES+=("$(basename "$input_file")")
done
OUTPUT_BASENAMES=()
for output_file in "${OUTPUT_FILES[@]}"; do
    OUTPUT_BASENAMES+=("$(basename "$output_file")")
done
USE_AUTO_OUTPUT=true
DEVICE_OUTPUT_DIR="outputs_device"
if [ ${#OUTPUT_FILES[@]} -gt 0 ]; then
    USE_AUTO_OUTPUT=false
fi
OUTPUT_PNG_BASENAME=""
if [ -n "$OUTPUT_PNG" ]; then
    OUTPUT_PNG_BASENAME=$(basename "$OUTPUT_PNG")
fi

# Build if needed
if [ ! -f "$BUILD_DIR/tflite_runner" ]; then
    echo -e "${YELLOW}Executable not found. Building first...${NC}"
    ./build.sh
fi

# Deploy
echo "Step 1: Deploying to device..."
$ADB shell "mkdir -p $DEVICE_DIR"
$ADB shell "rm -rf $DEVICE_DIR/$DEVICE_OUTPUT_DIR" > /dev/null
if [ "$USE_AUTO_OUTPUT" = true ]; then
    $ADB shell "mkdir -p $DEVICE_DIR/$DEVICE_OUTPUT_DIR" > /dev/null
fi
$ADB push "$BUILD_DIR/tflite_runner" "$DEVICE_DIR/" > /dev/null

# Push shared libraries
if [ -f "third_party/libs/$ANDROID_ABI/libtensorflowlite_jni.so" ]; then
    $ADB push "third_party/libs/$ANDROID_ABI/libtensorflowlite_jni.so" "$DEVICE_DIR/" > /dev/null
fi
if [ -f "third_party/libs/$ANDROID_ABI/libtensorflowlite_gpu_jni.so" ]; then
    $ADB push "third_party/libs/$ANDROID_ABI/libtensorflowlite_gpu_jni.so" "$DEVICE_DIR/" > /dev/null
fi

LIBCXX_PATH=$(detect_libcxx_path "$NDK_PATH" "$ANDROID_ABI")
if [ -n "$LIBCXX_PATH" ]; then
    $ADB push "$LIBCXX_PATH" "$DEVICE_DIR/" > /dev/null
else
    echo -e "${YELLOW}Warning: Could not locate libc++_shared.so for ABI $ANDROID_ABI. The binary may fail to run if it is missing on the device.${NC}"
fi

$ADB shell "chmod +x $DEVICE_DIR/tflite_runner"

# Push model and input
echo "Step 2: Pushing model and input data..."
$ADB push "$MODEL_PATH" "$DEVICE_DIR/$MODEL_BASENAME" > /dev/null
for idx in "${!INPUT_FILES[@]}"; do
    $ADB push "${INPUT_FILES[$idx]}" "$DEVICE_DIR/${INPUT_BASENAMES[$idx]}" > /dev/null
done

# Construct command
CMD="cd $DEVICE_DIR && LD_LIBRARY_PATH=. ./tflite_runner --model $MODEL_BASENAME"
for base in "${INPUT_BASENAMES[@]}"; do
    CMD="$CMD --input $base"
done
if [ "$USE_AUTO_OUTPUT" = true ]; then
    CMD="$CMD --output-dir $DEVICE_OUTPUT_DIR"
else
    for base in "${OUTPUT_BASENAMES[@]}"; do
        CMD="$CMD --output $base"
    done
fi

if [ -n "$OUTPUT_PNG_BASENAME" ]; then
    CMD="$CMD --output-png $OUTPUT_PNG_BASENAME"
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
if [ "$USE_AUTO_OUTPUT" = true ]; then
    mkdir -p "$HOST_OUTPUT_DIR"
    $ADB pull "$DEVICE_DIR/$DEVICE_OUTPUT_DIR/." "$HOST_OUTPUT_DIR" > /dev/null
    echo "  ✓ Auto outputs copied to $HOST_OUTPUT_DIR"
else
    for idx in "${!OUTPUT_FILES[@]}"; do
        dest="${OUTPUT_FILES[$idx]}"
        mkdir -p "$(dirname "$dest")" >/dev/null 2>&1
        $ADB pull "$DEVICE_DIR/${OUTPUT_BASENAMES[$idx]}" "$dest" > /dev/null
        echo "  ✓ $dest"

        if [ "$CUSTOM_PNG_FOR_FIRST" != "true" ] || [ "$idx" -ne 0 ]; then
            device_png=$(derive_png_filename "${OUTPUT_BASENAMES[$idx]}")
            host_png=$(derive_png_filename "$dest")
            if $ADB shell "test -f $DEVICE_DIR/$device_png" 2>/dev/null; then
                mkdir -p "$(dirname "$host_png")" >/dev/null 2>&1
                $ADB pull "$DEVICE_DIR/$device_png" "$host_png" > /dev/null
                echo "    ✓ $host_png"
            fi
        fi
    done
fi

if [ -n "$OUTPUT_PNG" ]; then
    if $ADB shell "test -f $DEVICE_DIR/$OUTPUT_PNG_BASENAME" 2>/dev/null; then
        mkdir -p "$(dirname "$OUTPUT_PNG")" >/dev/null 2>&1
        $ADB pull "$DEVICE_DIR/$OUTPUT_PNG_BASENAME" "$OUTPUT_PNG" > /dev/null
        echo "  ✓ $OUTPUT_PNG"
    fi
fi

echo ""
echo -e "${GREEN}=========================================="
echo "Execution completed successfully!"
echo -e "==========================================${NC}"
