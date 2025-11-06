# TensorFlow Lite Runner - Usage Guide

## Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Building the Project](#building-the-project)
3. [Preparing Models and Data](#preparing-models-and-data)
4. [Running Inference](#running-inference)
5. [Advanced Configuration](#advanced-configuration)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

## Setup and Installation

### 1. Install Android NDK

Download and install the Android NDK:

```bash
# Download from https://developer.android.com/ndk/downloads
# Extract and set environment variable
export ANDROID_NDK_HOME=/path/to/android-ndk

# Add to ~/.bashrc or ~/.zshrc to make permanent
echo 'export ANDROID_NDK_HOME=/path/to/android-ndk' >> ~/.bashrc
```

### 2. Install Required Tools

**macOS:**
```bash
brew install cmake wget git android-platform-tools
```

**Linux:**
```bash
sudo apt-get install cmake wget git adb
```

### 3. Connect Android Device

```bash
# Enable Developer Options and USB Debugging on your device
# Settings > About Phone > Tap "Build Number" 7 times
# Settings > Developer Options > Enable USB Debugging

# Connect via USB and verify
adb devices
# Should show your device
```

### 4. Clone and Setup

```bash
git clone https://github.com/katolikov/tflite_runner.git
cd tflite_runner

# Setup all dependencies
cd third_party
chmod +x setup_all.sh
./setup_all.sh
cd ..
```

## Building the Project

### Standard Build (64-bit ARM)

```bash
chmod +x build.sh
./build.sh
```

### 32-bit ARM Build

```bash
ANDROID_ABI=armeabi-v7a ./build.sh
```

### Debug Build

```bash
BUILD_TYPE=Debug ./build.sh
```

### Clean Build

```bash
rm -rf build-*
./build.sh
```

## Preparing Models and Data

### Convert TensorFlow Model to TFLite

```python
import tensorflow as tf

# Method 1: From SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_dir')
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Method 2: From Keras model
model = tf.keras.models.load_model('model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Method 3: With quantization (recommended for mobile)
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_dir')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_quant_model)
```

### Create Input NPY Files

```python
import numpy as np
from PIL import Image

# Example 1: Image Classification (MobileNet input)
image = Image.open('input.jpg').resize((224, 224))
image_array = np.array(image).astype(np.float32)
# Normalize to [-1, 1] or [0, 1] depending on model
image_array = (image_array / 127.5) - 1.0  # For MobileNet
image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
np.save('input.npy', image_array)

# Example 2: Random Input for Testing
random_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
np.save('random_input.npy', random_input)

# Example 3: Quantized Input (uint8)
image_uint8 = np.array(image).astype(np.uint8)
image_uint8 = np.expand_dims(image_uint8, axis=0)
np.save('input_quantized.npy', image_uint8)

# Example 4: Object Detection (SSD MobileNet)
image = Image.open('input.jpg').resize((300, 300))
image_array = np.array(image).astype(np.uint8)
image_array = np.expand_dims(image_array, axis=0)
np.save('od_input.npy', image_array)
```

### Inspect Model Details

```python
import tensorflow as tf

# Load and inspect model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
print("Input shape:", input_details[0]['shape'])
print("Input type:", input_details[0]['dtype'])

# Get output details
output_details = interpreter.get_output_details()
print("Output shape:", output_details[0]['shape'])
print("Output type:", output_details[0]['dtype'])
```

## Running Inference

### Method 1: Manual Steps

```bash
# 1. Build the project
./build.sh

# 2. Deploy to device
./deploy.sh

# 3. Push model and input
adb push model.tflite /data/local/tmp/tflite_runner/
adb push input.npy /data/local/tmp/tflite_runner/

# 4. Run inference
adb shell "cd /data/local/tmp/tflite_runner && \
  LD_LIBRARY_PATH=. ./tflite_runner \
  --model model.tflite \
  --input input.npy \
  --output output.npy \
  --output-png output.png"

# 5. Retrieve results
adb pull /data/local/tmp/tflite_runner/output.npy .
adb pull /data/local/tmp/tflite_runner/output.png .
```

### Method 2: Using Example Script

```bash
chmod +x run_example.sh
./run_example.sh model.tflite input.npy output.npy output.png
```

### Method 3: CPU-Only Mode

```bash
adb shell "cd /data/local/tmp/tflite_runner && \
  LD_LIBRARY_PATH=. ./tflite_runner \
  --model model.tflite \
  --input input.npy \
  --output output.npy \
  --no-gpu"
```

## Advanced Configuration

### Check Device Info

```bash
# Check architecture
adb shell getprop ro.product.cpu.abi

# Check Android version
adb shell getprop ro.build.version.release

# Check GPU info
adb shell dumpsys | grep -i gpu

# Check available memory
adb shell cat /proc/meminfo
```

### Monitor Performance

```bash
# Watch CPU usage
adb shell top -m 10

# Monitor logcat
adb logcat -s TFLiteRunner:* NPY_IO:* ImageUtils:*

# Profile with systrace (advanced)
python $ANDROID_SDK/platform-tools/systrace/systrace.py \
  --time=10 -o trace.html sched freq idle
```

### Batch Processing

```bash
# Process multiple files
for input in inputs/*.npy; do
  basename=$(basename $input .npy)
  ./run_example.sh model.tflite $input outputs/${basename}_out.npy
done
```

## Performance Optimization

### Model Optimization

1. **Quantization** - Convert to INT8:
   ```python
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   converter.target_spec.supported_types = [tf.int8]
   ```

2. **Pruning** - Remove unnecessary weights
3. **Architecture Search** - Use efficient architectures (MobileNet, EfficientNet)

### Runtime Optimization

1. **Use GPU Delegate** - Enabled by default
2. **Reduce Input Size** - Smaller inputs = faster inference
3. **Batch Size = 1** - Optimize for single-sample inference
4. **Use Quantized Models** - INT8 models are 4x smaller and faster

### Benchmarking

```bash
# Time inference on device
adb shell "cd /data/local/tmp/tflite_runner && \
  time LD_LIBRARY_PATH=. ./tflite_runner \
  --model model.tflite \
  --input input.npy \
  --output output.npy"
```

## Troubleshooting

### Issue: GPU Delegate Initialization Fails

**Symptoms:**
```
Failed to initialize GPU delegate, falling back to CPU
```

**Solutions:**
1. Check device GPU support:
   ```bash
   adb shell dumpsys | grep -i opengl
   ```
2. Try CPU-only mode: `--no-gpu`
3. Update device firmware
4. Check logcat for details: `adb logcat | grep GPU`

### Issue: Library Not Found

**Symptoms:**
```
error while loading shared libraries: libtensorflowlite_gpu_delegate.so
```

**Solution:**
Ensure `LD_LIBRARY_PATH` is set:
```bash
adb shell "cd /data/local/tmp/tflite_runner && \
  LD_LIBRARY_PATH=. ./tflite_runner ..."
```

### Issue: Input Shape Mismatch

**Symptoms:**
```
Input data size mismatch: expected 150528, got 602112
```

**Solutions:**
1. Check model input shape:
   ```python
   import tensorflow as tf
   interpreter = tf.lite.Interpreter(model_path='model.tflite')
   interpreter.allocate_tensors()
   print(interpreter.get_input_details()[0]['shape'])
   ```
2. Reshape your input data accordingly
3. Ensure batch dimension is included

### Issue: Out of Memory

**Symptoms:**
```
Failed to allocate tensors
```

**Solutions:**
1. Use quantized models (INT8)
2. Reduce model size
3. Check available memory:
   ```bash
   adb shell cat /proc/meminfo | grep MemAvailable
   ```

### Issue: Permission Denied

**Symptoms:**
```
/data/local/tmp/tflite_runner: Permission denied
```

**Solution:**
```bash
adb shell chmod +x /data/local/tmp/tflite_runner/tflite_runner
```

### Issue: adb: device unauthorized

**Solution:**
1. Check device screen for authorization popup
2. Revoke and re-enable USB debugging
3. Try different USB cable/port

## Best Practices

1. **Test on Desktop First** - Validate model behavior before deploying
2. **Use Quantized Models** - Better performance and smaller size
3. **Profile on Target Device** - Performance varies by device
4. **Handle Errors Gracefully** - Check return codes
5. **Version Control Models** - Track model versions with code
6. **Monitor Memory Usage** - Prevent OOM on resource-constrained devices
7. **Validate Outputs** - Compare with reference implementation

## Additional Resources

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [Model Optimization](https://www.tensorflow.org/lite/performance/model_optimization)
- [Android NDK Documentation](https://developer.android.com/ndk/guides)
- [GPU Delegate Guide](https://www.tensorflow.org/lite/performance/gpu)
