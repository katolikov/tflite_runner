# TensorFlow Lite Runner - AI Agent Instructions

## Project Overview
A production-ready TensorFlow Lite inference runner for Android devices with GPU delegate support, optimized for Exynos GPUs. Enables running TFLite models with NPY input/output and PNG visualization.

## Technology Stack
- **Primary Language**: C++17
- **Framework**: TensorFlow Lite C++ API v2.14.0
- **GPU Acceleration**: TFLite GPU Delegate (OpenCL/OpenGL ES)
- **Build System**: CMake 3.18+ with Android NDK toolchain
- **Target Platform**: Android (arm64-v8a, armeabi-v7a), API level 29+
- **Dependencies**: cnpy (NPY I/O), stb_image_write (PNG output)

## Project Structure
```
├── CMakeLists.txt          # Android NDK build config
├── build.sh                # Main build script
├── deploy.sh               # Deploy to device script
├── run_example.sh          # End-to-end workflow example
├── include/
│   ├── tflite_runner.h    # Core runner with GPU delegate
│   ├── npy_io.h           # NPY format reader/writer
│   └── image_utils.h      # PNG conversion utilities
├── src/
│   ├── main.cpp           # CLI with argument parsing
│   ├── tflite_runner.cpp  # TFLite inference + GPU init
│   ├── npy_io.cpp         # NPY file I/O via cnpy
│   └── image_utils.cpp    # Image normalization + PNG export
└── third_party/
    ├── setup_all.sh       # Download all dependencies
    └── README.md          # Dependency documentation
```

## Critical Developer Workflows

### 1. Initial Setup
```bash
# Setup dependencies (only once)
cd third_party && ./setup_all.sh

# Build for 64-bit ARM (default)
./build.sh

# Deploy to connected Android device
./deploy.sh
```

### 2. Device-Specific Deployment
```bash
# List available devices
./run_on_device.sh --list

# Run on specific device
./run_on_device.sh -d R5CR30KPVEZ model.tflite input.npy output.npy

# Multiple device testing
./run_on_device.sh -d DEVICE1 model.tflite input.npy out1.npy
./run_on_device.sh -d DEVICE2 model.tflite input.npy out2.npy
```

### 3. Build Variations
```bash
# 32-bit ARM
ANDROID_ABI=armeabi-v7a ./build.sh

# Debug build with symbols
BUILD_TYPE=Debug ./build.sh

# Different API level
ANDROID_PLATFORM=android-30 ./build.sh
```

### 3. Running Inference
```bash
# Full workflow (build + deploy + run)
./run_example.sh model.tflite input.npy output.npy output.png

# Device-specific execution with profiling
./run_on_device.sh -d SERIAL model.tflite input.npy output.npy

# Manual execution on device
adb shell "cd /data/local/tmp/tflite_runner && \
  LD_LIBRARY_PATH=. ./tflite_runner \
  --model model.tflite --input input.npy --output output.npy"
```

### 4. Performance Profiling
```bash
# Automatic profiling (enabled by default)
./run_on_device.sh model.tflite input.npy output.npy
# Shows: timing breakdown, GPU/CPU operation placement

# CPU-only comparison
./run_on_device.sh --no-gpu model.tflite input.npy output_cpu.npy

# View detailed logs
adb logcat -s TFLiteRunner:V NPY_IO:V ImageUtils:V
```

## Code Conventions

### GPU Delegate Initialization Pattern
```cpp
// From tflite_runner.cpp - ALWAYS use these settings for Exynos
TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
```

### Inference Pipeline (src/tflite_runner.cpp)
1. `LoadModel()` - FlatBufferModel from file
2. `InitGPUDelegate()` - Configure and attach GPU delegate
3. `AllocateTensors()` - MUST call before inference
4. `RunInference()` - Copy input → Invoke → Copy output
5. Handle multiple tensor types: float32, uint8, int8

### NPY I/O Convention (src/npy_io.cpp)
- Uses `cnpy` library (third_party/cnpy/)
- Supports float, int8, uint8 tensors
- Shape stored as `std::vector<size_t>`
- ALWAYS check word_size before copying data

### PNG Output (src/image_utils.cpp)
- Auto-normalizes float data to [0, 255]
- Supports 1, 3, 4 channel images
- Uses `stb_image_write.h` (header-only)
- Image shape inference: NHWC format (batch, height, width, channels)

## Android-Specific Patterns

### Logging Convention
```cpp
#include <android/log.h>
#define LOG_TAG "ComponentName"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
```

### Shared Library Loading
ALWAYS set `LD_LIBRARY_PATH=.` when running on device:
```bash
adb shell "cd /data/local/tmp/tflite_runner && LD_LIBRARY_PATH=. ./tflite_runner ..."
```

### Device Testing Workflow
```bash
# Monitor logs in real-time
adb logcat -s TFLiteRunner:* NPY_IO:* ImageUtils:*

# Check architecture match
adb shell getprop ro.product.cpu.abi  # Should match ANDROID_ABI

# Profile performance
adb shell "cd /data/local/tmp/tflite_runner && time LD_LIBRARY_PATH=. ..."
```

## Common Pitfalls & Solutions

### 1. Tensor Allocation
❌ **Wrong**: Invoke before AllocateTensors
✅ **Correct**: Call `AllocateTensors()` in `InitGPUDelegate()` after delegate attachment

### 2. Input Shape Mismatch
- Model expects specific shape (e.g., [1, 224, 224, 3])
- NPY file must match exactly (including batch dimension)
- Check with `GetInputShape()` before inference

### 3. GPU Delegate Failures
- Silent failures are common - ALWAYS check return codes
- Falls back to CPU automatically if GPU init fails
- Exynos devices: Requires OpenCL or OpenGL ES 3.1+
- Use `--no-gpu` flag to force CPU mode for debugging

### 4. Library Path Issues
- `libtensorflowlite_gpu_delegate.so` MUST be in same directory as executable
- Set `LD_LIBRARY_PATH=.` in shell command
- Also push `libc++_shared.so` if using shared STL

### 5. NPY Type Conversion
```cpp
// From npy_io.cpp - Handle different word sizes
if (arr.word_size == sizeof(float)) {
    // Direct copy
} else if (arr.word_size == sizeof(double)) {
    // Cast double → float
}
```

## Third-Party Dependencies

### Automatic Setup
```bash
cd third_party && ./setup_all.sh
```

Downloads:
1. **TensorFlow Lite** (setup_tflite.sh)
   - Prebuilt libraries from Maven Central
   - arm64-v8a and armeabi-v7a variants
   - libtensorflowlite.a + libtensorflowlite_gpu_delegate.so

2. **TFLite Headers** (setup_headers.sh)
   - Sparse clone from tensorflow/tensorflow repo
   - Tag: v2.14.0
   - Includes flatbuffers dependency

3. **cnpy** (setup_cnpy.sh)
   - Clone from rogersce/cnpy
   - cnpy.h + cnpy.cpp (compiled into binary)

4. **stb_image_write** (setup_stb.sh)
   - Single-header library from nothings/stb
   - PNG encoding only

### Manual Header Installation (if setup_headers.sh fails)
Download from: https://github.com/tensorflow/tensorflow/tree/v2.14.0/tensorflow/lite
Extract to: `third_party/tensorflow/lite/`

## Integration Points

### Model Requirements
- Format: `.tflite` (FlatBuffer)
- Input: Any shape, float32/uint8/int8
- Output: Any shape, auto-detected type
- Quantization: Fully supported (INT8, UINT8)

### Input Data Format (NPY)
```python
# Python preparation
import numpy as np
data = np.array([[...]], dtype=np.float32)  # Include batch dim
np.save('input.npy', data)
```

### Output Processing
```python
# Load results
output = np.load('output.npy')
# Shape matches model output tensor shape
```

## Performance Optimization

### Model-Level
- Use INT8 quantization (4x smaller, faster)
- Optimize for mobile (MobileNet, EfficientNet architectures)
- Reduce input resolution if possible

### Runtime-Level
- GPU delegate enabled by default (Exynos optimized)
- Single-batch inference (batch=1)
- Thread count optimized by TFLite automatically

### Benchmarking Commands
```bash
# Time on device
adb shell "cd /data/local/tmp/tflite_runner && \
  time LD_LIBRARY_PATH=. ./tflite_runner --model model.tflite --input input.npy --output out.npy"

# CPU usage monitoring
adb shell top -m 10 -s 1
```

## Debugging

### Enable Verbose Logging
```bash
# Watch all component logs
adb logcat -s TFLiteRunner:V NPY_IO:V ImageUtils:V Main:V
```

### Common Error Messages
1. "Failed to load model" → Check .tflite file path and permissions
2. "Input data size mismatch" → Verify NPY shape matches model input
3. "Failed to invoke interpreter" → Check GPU delegate or try --no-gpu
4. "Unsupported tensor type" → Model uses uncommon type (not float32/uint8/int8)

### Validation Strategy
1. Test model on desktop TFLite first (Python)
2. Verify NPY file shape and dtype
3. Deploy and run with --no-gpu to isolate GPU issues
4. Enable GPU and compare outputs

## Key Files Reference

- **CMakeLists.txt**: Android NDK toolchain config, links libtensorflowlite.a and GPU delegate .so
- **src/tflite_runner.cpp**: GPU delegate initialization, profiling with std::chrono, multi-input/output support
- **src/npy_io.cpp**: cnpy wrapper with type conversion (float/double/int8/uint8)
- **src/main.cpp**: CLI argument parsing, profiling output formatting
- **build.sh**: Detects ANDROID_NDK_HOME, invokes cmake with android.toolchain.cmake
- **deploy.sh**: Pushes executable + shared libs, sets execute permissions
- **run_on_device.sh**: Device selection, architecture checking, auto-deploy and execute

## Getting Started for AI Agents

1. **Verify Prerequisites**: `ANDROID_NDK_HOME` set, device connected (`adb devices`)
2. **Setup Dependencies**: `cd third_party && ./setup_all.sh`
3. **Build**: `./build.sh`
4. **Test Workflow**: `./run_example.sh model.tflite input.npy out.npy`
5. **Modify Code**: Edit src/ files, rebuild with `./build.sh`

## References
- TensorFlow Lite C++ Guide: https://www.tensorflow.org/lite/guide/inference
- GPU Delegate: https://www.tensorflow.org/lite/performance/gpu
- Android NDK: https://developer.android.com/ndk/guides
- Project Documentation: README.md, USAGE.md
