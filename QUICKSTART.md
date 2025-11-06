# Quick Start Guide

Get up and running with TensorFlow Lite Runner in 5 minutes.

## Prerequisites Check

```bash
# 1. Check Android NDK is installed
echo $ANDROID_NDK_HOME
# Should print path like: /Users/username/Library/Android/sdk/ndk/25.1.8937393

# 2. Check device connection
adb devices
# Should show your device (not "unauthorized")

# 3. Check device architecture
adb shell getprop ro.product.cpu.abi
# Should show: arm64-v8a or armeabi-v7a
```

## Installation (First Time Only)

```bash
# 1. Clone repository
git clone https://github.com/katolikov/tflite_runner.git
cd tflite_runner

# 2. Setup dependencies (~5 minutes)
cd third_party
./setup_all.sh
cd ..

# 3. Build for Android (~2 minutes)
./build.sh

# 4. Deploy to device (~30 seconds)
./deploy.sh
```

## Running Your First Model

### Option A: Quick Test (No Model Required)

```bash
# Download a test model
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip
unzip mobilenet_v1_1.0_224_quant_and_labels.zip

# Create test input
python3 -c "import numpy as np; np.save('test_input.npy', np.random.randint(0, 256, (1, 224, 224, 3), dtype=np.uint8))"

# Run inference
./run_example.sh mobilenet_v1_1.0_224_quant.tflite test_input.npy output.npy output.png

# View results
python3 -c "import numpy as np; print(np.load('output.npy'))"
```

### Option B: With Your Own Model

```bash
# 1. Prepare your image
python tools/prepare_input.py my_image.jpg --output input.npy --size 224

# 2. Run inference
./run_example.sh my_model.tflite input.npy output.npy output.png

# 3. Analyze results
python tools/analyze_output.py output.npy --model my_model.tflite
```

## Common Commands

```bash
# List available devices
./run_on_device.sh --list

# Run on specific device
./run_on_device.sh -d SERIAL model.tflite input.npy output.npy

# Build for different architectures
ANDROID_ABI=arm64-v8a ./build.sh    # 64-bit ARM (default)
ANDROID_ABI=armeabi-v7a ./build.sh  # 32-bit ARM

# Debug mode
BUILD_TYPE=Debug ./build.sh

# Run without GPU (CPU only)
./run_on_device.sh --no-gpu model.tflite input.npy output.npy

# Monitor logs
adb logcat -s TFLiteRunner:* NPY_IO:* ImageUtils:*

# Clean build
rm -rf build-* && ./build.sh
```

## Understanding Performance Output

Every inference automatically shows:

```
=== Performance Profiling ===
Model Load:         45.23 ms     # Model loading time
Delegate Init:      123.45 ms    # GPU setup (one-time)
Inference:          8.45 ms      # Actual inference (key metric)
Total Runtime:      191.59 ms    # End-to-end time

=== Operation Placement ===
GPU Operations:     38 (90.5%)   # Higher is better
CPU Operations:     4 (9.5%)     # Lower is better
```

**Target**: <20ms inference time for real-time applications

## Troubleshooting

### "ANDROID_NDK_HOME not set"

```bash
# Find your NDK installation
ls ~/Library/Android/sdk/ndk/  # macOS
ls ~/Android/Sdk/ndk/          # Linux

# Set environment variable
export ANDROID_NDK_HOME=~/Library/Android/sdk/ndk/25.1.8937393
```

### "No device connected"

```bash
# Check USB debugging is enabled
# Settings > Developer Options > USB Debugging

# Try different USB cable/port
adb kill-server
adb start-server
adb devices
```

### "GPU delegate failed"

```bash
# Use CPU mode
./run_example.sh model.tflite input.npy output.npy --no-gpu

# Or manually:
adb shell "cd /data/local/tmp/tflite_runner && \
  LD_LIBRARY_PATH=. ./tflite_runner \
  --model model.tflite --input input.npy --output output.npy --no-gpu"
```

### "Input shape mismatch"

```python
# Check your model's expected input shape
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
print(interpreter.get_input_details()[0]['shape'])

# Then prepare input with correct shape
# E.g., for [1, 224, 224, 3]:
python tools/prepare_input.py image.jpg --output input.npy --size 224
```

## Next Steps

- Read [PROFILING.md](PROFILING.md) for performance optimization guide
- Read [USAGE.md](USAGE.md) for detailed usage instructions
- Check [README.md](README.md) for full documentation
- Explore [tools/README.md](tools/README.md) for helper scripts
- Review [.github/copilot-instructions.md](.github/copilot-instructions.md) for development guide

## Performance Tips

1. **Use quantized models** - 4x faster, smaller
2. **Enable GPU** - Default, works on most Exynos devices
3. **Reduce input size** - 224x224 is often sufficient
4. **Profile on device** - Use `time` command to measure

## Example Workflows

### Image Classification

```bash
# MobileNet v2
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v2_1.0_224.tflite
python tools/prepare_input.py cat.jpg --output input.npy --size 224
./run_example.sh mobilenet_v2_1.0_224.tflite input.npy output.npy
python tools/analyze_output.py output.npy
```

### Object Detection

```bash
# SSD MobileNet (300x300 input)
python tools/prepare_input.py street.jpg --output input.npy --size 300 --quantize
./run_example.sh detect.tflite input.npy detections.npy
python tools/analyze_output.py detections.npy --model-type detection
```

### Style Transfer

```bash
# Style transfer (typically 256x256)
python tools/prepare_input.py photo.jpg --output input.npy --size 256
./run_example.sh style_transfer.tflite input.npy styled.npy styled.png
# View styled.png
```

## Support

- **Issues**: Open an issue on GitHub
- **Documentation**: Check README.md and USAGE.md
- **Examples**: See examples in tools/README.md

Happy inferencing! 🚀
