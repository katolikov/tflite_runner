# Performance Profiling and Multi-Device Guide

## Performance Profiling

The TFLite Runner automatically tracks timing for each stage of the inference pipeline and reports operation placement (GPU vs CPU).

### Profiling Output

Every inference run includes detailed timing information:

```
=== Performance Profiling ===
Model Load:         45.23 ms      # Time to load .tflite file
Delegate Init:      123.45 ms     # GPU delegate initialization
Tensor Allocation:  12.34 ms      # Memory allocation for tensors
Input Copy:         1.23 ms       # Copy input data to device
Inference:          8.45 ms       # Actual inference time
Output Copy:        0.89 ms       # Copy output data from device
Total Runtime:      191.59 ms     # End-to-end execution time

=== Operation Placement ===
Total Operations:   42
GPU Operations:     38 (90.5%)    # Operations accelerated by GPU
CPU Operations:     4 (9.5%)      # Operations fallback to CPU

Operations running on CPU:
  - RESHAPE                        # Typically not GPU-accelerated
  - SOFTMAX                        # May run on CPU depending on shape
```

### Understanding the Metrics

**Model Load**: Time to parse the FlatBuffer model file
- First run is slower due to file I/O
- Subsequent runs may benefit from OS caching

**Delegate Init**: GPU delegate setup time
- One-time cost per session
- Includes shader compilation
- Typically 50-200ms on Exynos devices

**Tensor Allocation**: Memory buffer setup
- Allocated once, reused for multiple inferences
- Size depends on model complexity

**Inference**: Pure computation time
- The most critical metric for performance
- Target: <20ms for real-time applications

**Operation Placement**: Shows hardware utilization
- Higher GPU % = better acceleration
- CPU operations are bottlenecks
- Some ops (Reshape, Transpose) typically run on CPU

### Optimizing Based on Profiling

#### High GPU Operations Count (>80%)
✅ Good! Model is well-optimized for mobile GPU
- Focus on model quantization for further speedup
- Consider reducing input resolution

#### Low GPU Operations Count (<50%)
⚠️ Many operations running on CPU
- Check which ops are on CPU
- Some ops may not have GPU kernels
- Consider model architecture changes

#### Long Inference Time
1. **Use INT8 quantization** - 4x speedup
2. **Reduce input size** - Linear speedup with smaller inputs
3. **Optimize model architecture** - Use mobile-optimized models

#### Long Delegate Init
- Normal for first run (shader compilation)
- Persistent across inferences in same session
- Cannot be optimized without changing delegate

## Multi-Input/Output Models

The runner supports models with multiple input and output tensors.

### Automatic Detection

The runner automatically detects model structure:

```
Input tensor count: 2
Input[0]: name=input_image, type=float32, dims=[1, 224, 224, 3]
Input[1]: name=input_mask, type=uint8, dims=[1, 224, 224, 1]

Output tensor count: 3
Output[0]: name=classification, type=float32, dims=[1, 1000]
Output[1]: name=boxes, type=float32, dims=[1, 100, 4]
Output[2]: name=scores, type=float32, dims=[1, 100]
```

### Input Preparation

For multi-input models, concatenate inputs in NPY file or prepare separately:

**Option 1: Prepare each input separately**
```python
import numpy as np

# First input (image)
image = np.random.rand(1, 224, 224, 3).astype(np.float32)
np.save('input_0.npy', image)

# Second input (mask)
mask = np.random.randint(0, 2, (1, 224, 224, 1), dtype=np.uint8)
np.save('input_1.npy', mask)
```

**Note**: Currently the CLI supports single input file. For multi-input models, you'll need to modify the code or concatenate inputs in preprocessing.

### Output Handling

For multi-output models, each output is saved with suffix:

```python
# Load all outputs
output_0 = np.load('output_0.npy')  # First output tensor
output_1 = np.load('output_1.npy')  # Second output tensor
output_2 = np.load('output_2.npy')  # Third output tensor
```

## Device-Specific Deployment

Use `run_on_device.sh` to target specific devices when multiple are connected.

### List Available Devices

```bash
./run_on_device.sh --list
```

Output:
```
=== Available Android Devices ===

Serial:       R5CR30KPVEZ
Model:        Samsung Galaxy S22
Android:      13
Architecture: arm64-v8a

Serial:       emulator-5554
Model:        Android SDK Emulator
Android:      12
Architecture: arm64-v8a
```

### Run on Specific Device

```bash
# Using full command
./run_on_device.sh -d R5CR30KPVEZ model.tflite input.npy output.npy

# With PNG output
./run_on_device.sh -d R5CR30KPVEZ model.tflite input.npy output.npy output.png

# Without GPU
./run_on_device.sh -d R5CR30KPVEZ --no-gpu model.tflite input.npy output.npy
```

### Script Features

1. **Architecture Detection**: Warns if binary doesn't match device
2. **Auto-build**: Builds if executable not found
3. **Device Info**: Shows model and Android version
4. **Error Handling**: Clear error messages for common issues

### Multiple Device Workflow

```bash
# Test same model on different devices
./run_on_device.sh -d DEVICE1_SERIAL model.tflite input.npy out1.npy
./run_on_device.sh -d DEVICE2_SERIAL model.tflite input.npy out2.npy

# Compare performance
python tools/analyze_output.py out1.npy
python tools/analyze_output.py out2.npy
```

## Benchmarking

### Single Inference Timing

```bash
# CPU only
./run_on_device.sh --no-gpu model.tflite input.npy output.npy

# GPU accelerated
./run_on_device.sh model.tflite input.npy output.npy
```

Compare inference times from profiling output.

### Multiple Runs

For accurate benchmarking, run multiple inferences:

```bash
# Modify source to loop inference
# Or use shell loop
for i in {1..10}; do
    ./run_on_device.sh model.tflite input.npy output_$i.npy
done
```

### Profiling Tips

1. **Warm-up**: First run includes shader compilation
2. **Device Load**: Close other apps before benchmarking
3. **Thermal**: Watch for thermal throttling on long runs
4. **Power**: Keep device plugged in for consistent performance
5. **Background**: Disable auto-sync and notifications

## Common Performance Patterns

### Mobile-Optimized Models (MobileNet, EfficientNet)

```
GPU Operations: 90-95%
Inference: 5-15ms
CPU Ops: Reshape, Softmax at end
```

### Detection Models (SSD, YOLO)

```
GPU Operations: 85-90%
Inference: 20-50ms
CPU Ops: NMS, Reshape, Concat
```

### Segmentation Models

```
GPU Operations: 95-98%
Inference: 30-100ms
CPU Ops: Minimal, mostly Reshape
```

## Troubleshooting Performance

### High CPU Operation Count

**Problem**: >20% operations on CPU
**Solutions**:
1. Check model architecture
2. Some ops don't have GPU kernels
3. Consider model conversion options
4. Profile with TensorFlow Lite benchmark tool

### Slow Inference Despite GPU

**Problem**: GPU ops high but inference slow
**Solutions**:
1. Check input size - larger inputs = slower
2. Quantize model to INT8
3. Verify GPU delegate is actually being used
4. Check for memory transfer bottlenecks

### Inconsistent Timing

**Problem**: Timing varies significantly between runs
**Solutions**:
1. Run warmup inference first
2. Check for thermal throttling
3. Close background apps
4. Lock CPU/GPU frequencies (requires root)

### Long First Inference

**Problem**: First inference much slower
**Cause**: Shader compilation on first run
**Solution**: Normal behavior, measure subsequent runs

## Advanced Profiling

### Using Android Profiler

```bash
# Record trace while running
adb shell am profile start <package> /data/local/tmp/trace.trace

# Run inference
./run_on_device.sh model.tflite input.npy output.npy

# Stop profiling
adb shell am profile stop <package>

# Pull trace
adb pull /data/local/tmp/trace.trace
```

### Memory Profiling

Check memory usage:
```bash
# Before inference
adb shell dumpsys meminfo | grep tflite_runner

# Monitor during inference
adb shell top -m 10 | grep tflite
```

### GPU Profiling (Samsung devices)

```bash
# GPU utilization
adb shell cat /sys/class/kgsl/kgsl-3d0/gpubusy

# GPU frequency
adb shell cat /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq
```

## Best Practices

1. **Always warm up**: Run inference once before timing
2. **Compare apples to apples**: Same device, same conditions
3. **Watch temperature**: Throttling affects results
4. **Profile on target device**: Emulator ≠ real device
5. **Test both GPU and CPU**: Some models run better on CPU
6. **Monitor operation placement**: Aim for >80% GPU ops
7. **Use multiple runs**: Average over 10+ inferences
8. **Document conditions**: Note device model, Android version, temperature
