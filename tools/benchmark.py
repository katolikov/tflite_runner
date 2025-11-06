#!/usr/bin/env python3
"""
Benchmark script to compare GPU vs CPU performance and analyze profiling data

Usage:
    python benchmark.py model.tflite input.npy --runs 10
"""

import argparse
import subprocess
import re
import statistics
import sys


def run_inference(device, model, input_file, use_gpu=True):
    """Run inference and extract profiling data"""
    cmd = ["./run_on_device.sh"]
    
    if device:
        cmd.extend(["-d", device])
    
    if not use_gpu:
        cmd.append("--no-gpu")
    
    cmd.extend([model, input_file, "output.npy"])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Parse timing data
        timings = {}
        for line in output.split('\n'):
            if 'Model Load:' in line:
                timings['model_load'] = float(re.search(r'([\d.]+) ms', line).group(1))
            elif 'Delegate Init:' in line:
                timings['delegate_init'] = float(re.search(r'([\d.]+) ms', line).group(1))
            elif 'Inference:' in line and 'Tensor' not in line:
                timings['inference'] = float(re.search(r'([\d.]+) ms', line).group(1))
            elif 'Total Runtime:' in line:
                timings['total'] = float(re.search(r'([\d.]+) ms', line).group(1))
            elif 'GPU Operations:' in line:
                match = re.search(r'(\d+) \(([\d.]+)%\)', line)
                if match:
                    timings['gpu_ops'] = int(match.group(1))
                    timings['gpu_percent'] = float(match.group(2))
            elif 'CPU Operations:' in line:
                match = re.search(r'(\d+) \(([\d.]+)%\)', line)
                if match:
                    timings['cpu_ops'] = int(match.group(1))
        
        return timings
    except subprocess.CalledProcessError as e:
        print(f"Error running inference: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Benchmark TFLite model performance')
    parser.add_argument('model', help='Path to .tflite model')
    parser.add_argument('input', help='Path to input .npy file')
    parser.add_argument('--runs', '-r', type=int, default=10,
                       help='Number of benchmark runs (default: 10)')
    parser.add_argument('--device', '-d', help='Target device serial number')
    parser.add_argument('--compare-cpu', action='store_true',
                       help='Also benchmark CPU-only mode')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TensorFlow Lite Performance Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Runs: {args.runs}")
    if args.device:
        print(f"Device: {args.device}")
    print()
    
    # GPU benchmark
    print("Running GPU benchmark...")
    gpu_times = []
    gpu_stats = None
    
    for i in range(args.runs):
        print(f"  Run {i+1}/{args.runs}...", end='', flush=True)
        timings = run_inference(args.device, args.model, args.input, use_gpu=True)
        if timings:
            gpu_times.append(timings['inference'])
            if gpu_stats is None:
                gpu_stats = timings
            print(f" {timings['inference']:.2f} ms")
        else:
            print(" FAILED")
            sys.exit(1)
    
    # CPU benchmark if requested
    cpu_times = []
    cpu_stats = None
    
    if args.compare_cpu:
        print("\nRunning CPU benchmark...")
        for i in range(args.runs):
            print(f"  Run {i+1}/{args.runs}...", end='', flush=True)
            timings = run_inference(args.device, args.model, args.input, use_gpu=False)
            if timings:
                cpu_times.append(timings['inference'])
                if cpu_stats is None:
                    cpu_stats = timings
                print(f" {timings['inference']:.2f} ms")
            else:
                print(" FAILED")
    
    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    print("\nGPU Mode:")
    print(f"  Mean:     {statistics.mean(gpu_times):.2f} ms")
    print(f"  Median:   {statistics.median(gpu_times):.2f} ms")
    print(f"  Std Dev:  {statistics.stdev(gpu_times):.2f} ms")
    print(f"  Min:      {min(gpu_times):.2f} ms")
    print(f"  Max:      {max(gpu_times):.2f} ms")
    
    if gpu_stats and 'gpu_ops' in gpu_stats:
        print(f"\n  GPU Operations: {gpu_stats['gpu_ops']} ({gpu_stats['gpu_percent']:.1f}%)")
        print(f"  CPU Operations: {gpu_stats['cpu_ops']}")
    
    if cpu_times:
        print("\nCPU Mode:")
        print(f"  Mean:     {statistics.mean(cpu_times):.2f} ms")
        print(f"  Median:   {statistics.median(cpu_times):.2f} ms")
        print(f"  Std Dev:  {statistics.stdev(cpu_times):.2f} ms")
        print(f"  Min:      {min(cpu_times):.2f} ms")
        print(f"  Max:      {max(cpu_times):.2f} ms")
        
        speedup = statistics.mean(cpu_times) / statistics.mean(gpu_times)
        print(f"\n  GPU Speedup: {speedup:.2f}x faster than CPU")
    
    # Performance classification
    mean_gpu = statistics.mean(gpu_times)
    print("\nPerformance Classification:")
    if mean_gpu < 10:
        print("  ✓ Excellent - Suitable for real-time applications")
    elif mean_gpu < 20:
        print("  ✓ Good - Suitable for most interactive use cases")
    elif mean_gpu < 50:
        print("  ~ Fair - May struggle with real-time constraints")
    else:
        print("  ✗ Poor - Consider model optimization")
    
    print("\nRecommendations:")
    if gpu_stats and 'gpu_percent' in gpu_stats:
        if gpu_stats['gpu_percent'] < 70:
            print("  • Low GPU utilization - consider model architecture changes")
        elif gpu_stats['gpu_percent'] > 90:
            print("  • Excellent GPU utilization!")
    
    if mean_gpu > 20:
        print("  • Consider INT8 quantization for 4x speedup")
        print("  • Reduce input resolution if possible")
        print("  • Try mobile-optimized architectures (MobileNet, EfficientNet)")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
