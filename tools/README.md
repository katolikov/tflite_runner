# TensorFlow Lite Runner - Tools

Helper Python scripts for preparing input data and analyzing model outputs.

## Installation

Install required Python packages:

```bash
pip install numpy pillow tensorflow
```

## prepare_input.py

Prepare image input data for TensorFlow Lite models.

### Usage

```bash
# Basic usage (MobileNet normalization)
python tools/prepare_input.py image.jpg --output input.npy

# Custom image size
python tools/prepare_input.py image.jpg --output input.npy --size 299

# Different normalization methods
python tools/prepare_input.py image.jpg --output input.npy --normalize imagenet
python tools/prepare_input.py image.jpg --output input.npy --normalize zero_one

# Quantized input (uint8)
python tools/prepare_input.py image.jpg --output input.npy --quantize

# Channels-first format (NCHW)
python tools/prepare_input.py image.jpg --output input.npy --channels-first
```

### Normalization Methods

- **mobilenet**: [0, 255] → [-1, 1] (default)
- **imagenet**: ImageNet mean/std normalization
- **inception**: [0, 255] → [-1, 1]
- **zero_one**: [0, 255] → [0, 1]
- **none**: No normalization

## analyze_output.py

Analyze TensorFlow Lite model output.

### Usage

```bash
# Auto-detect model type
python tools/analyze_output.py output.npy

# Specify model type
python tools/analyze_output.py output.npy --model-type classification

# With labels file
python tools/analyze_output.py output.npy --labels labels.txt --top-k 10

# With model inspection
python tools/analyze_output.py output.npy --model model.tflite
```

### Model Types

- **classification**: Image classification models
- **detection**: Object detection models
- **segmentation**: Semantic segmentation models
- **auto**: Auto-detect based on output shape (default)

## benchmark.py

Performance benchmarking tool with statistical analysis.

### Usage

```bash
# Basic benchmark (10 runs)
python tools/benchmark.py model.tflite input.npy

# Custom number of runs
python tools/benchmark.py model.tflite input.npy --runs 50

# Compare GPU vs CPU
python tools/benchmark.py model.tflite input.npy --compare-cpu

# Target specific device
python tools/benchmark.py model.tflite input.npy --device R5CR30KPVEZ --runs 20
```

### Output

```
=== BENCHMARK RESULTS ===

GPU Mode:
  Mean:     8.45 ms
  Median:   8.42 ms
  Std Dev:  0.23 ms
  Min:      8.12 ms
  Max:      8.89 ms

  GPU Operations: 38 (90.5%)
  CPU Operations: 4

CPU Mode:
  Mean:     34.23 ms
  Median:   34.15 ms
  Std Dev:  1.12 ms
  Min:      32.45 ms
  Max:      36.78 ms

  GPU Speedup: 4.05x faster than CPU

Performance Classification:
  ✓ Excellent - Suitable for real-time applications

Recommendations:
  • Excellent GPU utilization!
```

## Examples

### Example 1: Image Classification

```bash
# Prepare input
python tools/prepare_input.py cat.jpg --output input.npy --size 224

# Run on device
./run_example.sh mobilenet_v2.tflite input.npy output.npy

# Analyze results
python tools/analyze_output.py output.npy --labels imagenet_labels.txt --top-k 5
```

### Example 2: Object Detection

```bash
# Prepare input (SSD models typically use 300x300)
python tools/prepare_input.py street.jpg --output input.npy --size 300 --quantize

# Run on device
./run_example.sh ssd_mobilenet.tflite input.npy output.npy

# Analyze results
python tools/analyze_output.py output.npy --model-type detection
```

### Example 3: Segmentation

```bash
# Prepare input
python tools/prepare_input.py scene.jpg --output input.npy --size 513

# Run on device
./run_example.sh deeplabv3.tflite input.npy output.npy output.png

# Analyze results
python tools/analyze_output.py output.npy --model-type segmentation
```

## Getting ImageNet Labels

Download ImageNet labels for classification models:

```bash
wget https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt
```

## Tips

1. **Check model documentation** for required input size and normalization
2. **Use --quantize** for quantized (uint8) models
3. **Visualize outputs** with --output-png for image-like outputs
4. **Profile on device** to measure actual inference time
