# TensorFlow Lite Runner for Android with GPU Support

A high-performance TensorFlow Lite inference runner for Android devices with GPU delegate support, specifically optimized for Exynos devices. This tool allows you to run TFLite models on Android with NPY file I/O and PNG image output.

## Features

- ✅ **TensorFlow Lite GPU Delegate** - Hardware acceleration for Exynos GPUs
- ✅ **NPY File Support** - Read input and write output in NumPy format
- ✅ **PNG Image Output** - Visualize model outputs as images
- ✅ **Cross-Architecture** - Support for arm64-v8a and armeabi-v7a
- ✅ **Command-Line Interface** - Easy to use CLI for batch processing
- ✅ **Multiple Data Types** - Support for float32, int8, and uint8 models
- ✅ **Performance Profiling** - Detailed timing for each pipeline stage
- ✅ **Operation Tracking** - Monitor which ops run on GPU vs CPU
- ✅ **Multi-Input/Output** - Support for models with multiple tensors
- ✅ **Device Selection** - Run on specific Android device by serial number

## Quick Start

### Prerequisites

1. **Android NDK** (r21 or later)
   ```bash
   # Download from: https://developer.android.com/ndk/downloads
   export ANDROID_NDK_HOME=/path/to/android-ndk
   ```

2. **Android Device** with USB debugging enabled
   ```bash
   # Enable USB debugging in Developer Options
   # Connect device via USB
   adb devices  # Verify connection
   ```

3. **Build Tools**
   - CMake 3.18+
   - wget
   - git
   - adb (Android Debug Bridge)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/katolikov/tflite_runner.git
   cd tflite_runner
   ```

2. **Setup dependencies**
   ```bash
   cd third_party
   chmod +x setup_all.sh
   ./setup_all.sh
   cd ..
   ```

3. **Build for Android**
   ```bash
   chmod +x build.sh
   ./build.sh
   
   # Or for 32-bit ARM:
   # ANDROID_ABI=armeabi-v7a ./build.sh
   ```

4. **Deploy to device**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

## Usage

### Device Selection

List available devices:
```bash
./run_on_device.sh --list
```

Run on specific device:
```bash
./run_on_device.sh -d R5CR30KPVEZ model.tflite input.npy output.npy
```

### Basic Usage

```bash
# On your computer, prepare files
adb push model.tflite /data/local/tmp/tflite_runner/
adb push input.npy /data/local/tmp/tflite_runner/

# Run on device
adb shell "cd /data/local/tmp/tflite_runner && \
  LD_LIBRARY_PATH=. ./tflite_runner \
  --model model.tflite \
  --input input.npy \
  --output output.npy \
  --output-png output.png"

# Retrieve results
adb pull /data/local/tmp/tflite_runner/output.npy .
adb pull /data/local/tmp/tflite_runner/output.png .
```

### Using the Example Script

```bash
chmod +x run_example.sh
./run_example.sh model.tflite input.npy output.npy output.png
```

### Command-Line Options

```
TensorFlow Lite Runner for Android with GPU Support

Required options:
  --model <path>       Path to .tflite model file
  --input <path>       Path to input .npy file
  --output <path>      Path to output .npy file

Optional options:
  --output-png <path>  Path to output .png file (for image outputs)
  --no-gpu            Disable GPU delegate (use CPU only)
  --help              Show help message
```

### Performance Profiling

The runner automatically provides detailed performance metrics:

```
=== Performance Profiling ===
Model Load:         45.23 ms
Delegate Init:      123.45 ms
Tensor Allocation:  12.34 ms
Input Copy:         1.23 ms
Inference:          8.45 ms
Output Copy:        0.89 ms
Total Runtime:      191.59 ms

=== Operation Placement ===
Total Operations:   42
GPU Operations:     38 (90.5%)
CPU Operations:     4 (9.5%)

Operations running on CPU:
  - RESHAPE
  - SOFTMAX
```

## Preparing Input Data

Create NPY files from Python:

```python
import numpy as np

# Example: Image input (224x224x3)
input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
np.save('input.npy', input_data)

# Example: Quantized input (uint8)
input_data = np.random.randint(0, 256, (1, 224, 224, 3), dtype=np.uint8)
np.save('input_quantized.npy', input_data)
```

## Reading Output Data

```python
import numpy as np

# Load output
output = np.load('output.npy')
print(f"Output shape: {output.shape}")
print(f"Output data type: {output.dtype}")
print(f"Output values: {output}")

# For classification models
class_id = np.argmax(output)
confidence = output[0, class_id]
print(f"Predicted class: {class_id}, Confidence: {confidence:.4f}")
```

## Project Structure

```
tflite_runner/
├── CMakeLists.txt              # CMake build configuration
├── build.sh                    # Android build script
├── deploy.sh                   # Deploy to device script
├── run_example.sh              # Full workflow example
├── include/
│   ├── tflite_runner.h        # Main TFLite runner interface
│   ├── npy_io.h               # NPY file I/O
│   └── image_utils.h          # PNG image utilities
├── src/
│   ├── main.cpp               # CLI entry point
│   ├── tflite_runner.cpp      # TFLite GPU delegate implementation
│   ├── npy_io.cpp             # NPY reader/writer
│   └── image_utils.cpp        # Image conversion and saving
└── third_party/
    ├── setup_all.sh           # Download all dependencies
    ├── tensorflow/            # TensorFlow Lite headers & libraries
    ├── cnpy/                  # NPY format library
    └── stb/                   # STB image library
```

## GPU Delegate Configuration

The runner is optimized for Exynos GPUs with the following settings:

- **Inference Priority**: Minimum latency
- **Inference Preference**: Fast single answer
- **Experimental Flags**: Quantization support enabled

To disable GPU and use CPU only:
```bash
./tflite_runner --model model.tflite --input input.npy --output output.npy --no-gpu
```

## Supported Model Types

- **Float32 models** - Standard floating-point models
- **Quantized INT8 models** - 8-bit integer quantized models
- **Quantized UINT8 models** - Unsigned 8-bit quantized models
- **Dynamic range quantization**
- **Full integer quantization**

## Performance Tips

1. **Use GPU delegate** for best performance on Exynos devices
2. **Quantize your models** - INT8 models are faster and smaller
3. **Optimize input size** - Smaller inputs run faster
4. **Batch size = 1** - Mobile inference typically uses single samples
5. **Profile on device** - Use Android Profiler for bottleneck analysis

## Troubleshooting

### GPU Delegate Fails

```
Failed to initialize GPU delegate, falling back to CPU
```

**Solutions:**
- Ensure device supports OpenCL or OpenGL ES 3.1+
- Update device drivers
- Try with `--no-gpu` flag
- Check `adb logcat` for detailed error messages

### Library Not Found

```
error: library "libtensorflowlite_gpu_delegate.so" not found
```

**Solution:**
```bash
# Ensure LD_LIBRARY_PATH is set correctly
adb shell "cd /data/local/tmp/tflite_runner && \
  LD_LIBRARY_PATH=. ./tflite_runner ..."
```

### Input Shape Mismatch

```
Error: Input data size mismatch
```

**Solution:**
- Check model input shape: The tool will print expected shape
- Verify NPY file shape matches model requirements
- Ensure data type matches (float32 vs uint8)

### Permission Denied

```
Permission denied
```

**Solution:**
```bash
adb shell chmod +x /data/local/tmp/tflite_runner/tflite_runner
```

## Building from Source

### Debug Build

```bash
BUILD_TYPE=Debug ./build.sh
```

### 32-bit ARM Build

```bash
ANDROID_ABI=armeabi-v7a ./build.sh
```

### Custom Android Platform

```bash
ANDROID_PLATFORM=android-30 ./build.sh
```

## Testing on Device

### Check Device Architecture

```bash
adb shell getprop ro.product.cpu.abi
# Output: arm64-v8a or armeabi-v7a
```

### Monitor Logs

```bash
# In separate terminal
adb logcat -s TFLiteRunner:* NPY_IO:* ImageUtils:*
```

### Check GPU Support

```bash
adb shell dumpsys | grep -i gpu
adb shell getprop | grep gpu
```

## Examples

### Image Classification

```bash
# MobileNet v2 example
./run_example.sh mobilenet_v2.tflite cat_224x224.npy classification.npy
```

### Object Detection

```bash
# SSD MobileNet example
./run_example.sh ssd_mobilenet.tflite image_300x300.npy detections.npy
```

### Image-to-Image Models

```bash
# Style transfer example
./run_example.sh style_transfer.tflite input_256x256.npy styled.npy styled.png
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting section

## Acknowledgments

- TensorFlow Lite team for the excellent mobile ML framework
- cnpy library for NPY file support
- stb libraries for image I/O