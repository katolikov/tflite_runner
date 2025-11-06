#!/usr/bin/env python3
"""
Helper script to analyze TensorFlow Lite model and output

Usage:
    python analyze_output.py output.npy --model model.tflite
"""

import argparse
import numpy as np


def load_labels(labels_path):
    """Load class labels from text file"""
    with open(labels_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def analyze_classification(output, labels=None, top_k=5):
    """Analyze classification model output"""
    print("\n=== Classification Results ===")
    
    # Flatten output if needed
    if output.ndim > 1:
        output = output.flatten()
    
    # Get top K predictions
    top_indices = np.argsort(output)[-top_k:][::-1]
    
    print(f"\nTop {top_k} predictions:")
    for i, idx in enumerate(top_indices, 1):
        confidence = output[idx]
        label = labels[idx] if labels and idx < len(labels) else f"Class {idx}"
        print(f"  {i}. {label}: {confidence:.4f} ({confidence*100:.2f}%)")


def analyze_detection(output):
    """Analyze object detection model output"""
    print("\n=== Object Detection Results ===")
    print(f"Output shape: {output.shape}")
    
    # Common detection output formats:
    # SSD: [1, num_detections], [1, num_detections, 4], [1, num_detections], [1]
    # YOLO: [1, grid_h, grid_w, num_anchors * (5 + num_classes)]
    
    if output.ndim == 2:
        print(f"Number of detections: {output.shape[1]}")
    elif output.ndim == 3:
        print(f"Grid size: {output.shape[1]}x{output.shape[2]}")
    
    print("\nNote: Detection output format varies by model.")
    print("Please refer to your model's documentation for interpretation.")


def analyze_segmentation(output):
    """Analyze segmentation model output"""
    print("\n=== Segmentation Results ===")
    print(f"Output shape: {output.shape}")
    
    if output.ndim == 4:
        batch, height, width, classes = output.shape
        print(f"Batch size: {batch}")
        print(f"Segmentation map size: {height}x{width}")
        print(f"Number of classes: {classes}")
        
        # Get dominant class per pixel
        segmentation_map = np.argmax(output[0], axis=-1)
        unique_classes = np.unique(segmentation_map)
        print(f"\nClasses present in image: {unique_classes.tolist()}")
        
        for cls in unique_classes:
            pixel_count = np.sum(segmentation_map == cls)
            percentage = (pixel_count / (height * width)) * 100
            print(f"  Class {cls}: {pixel_count} pixels ({percentage:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='Analyze TensorFlow Lite model output')
    parser.add_argument('output', help='Output NPY file path')
    parser.add_argument('--model-type', '-t', 
                       choices=['classification', 'detection', 'segmentation', 'auto'],
                       default='auto',
                       help='Model type (default: auto-detect)')
    parser.add_argument('--labels', '-l', help='Path to labels file (one label per line)')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                       help='Number of top predictions to show (classification)')
    parser.add_argument('--model', '-m', help='Path to TFLite model (for inspection)')
    
    args = parser.parse_args()
    
    # Load output
    print(f"Loading output: {args.output}")
    output = np.load(args.output)
    
    print(f"\n=== Output Information ===")
    print(f"Shape: {output.shape}")
    print(f"Data type: {output.dtype}")
    print(f"Value range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"Mean: {output.mean():.4f}")
    print(f"Std: {output.std():.4f}")
    
    # Load labels if provided
    labels = None
    if args.labels:
        labels = load_labels(args.labels)
        print(f"\nLoaded {len(labels)} labels")
    
    # Inspect model if provided
    if args.model:
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=args.model)
            interpreter.allocate_tensors()
            
            output_details = interpreter.get_output_details()
            print(f"\n=== Model Output Details ===")
            print(f"Name: {output_details[0]['name']}")
            print(f"Shape: {output_details[0]['shape']}")
            print(f"Type: {output_details[0]['dtype']}")
        except ImportError:
            print("\nNote: Install tensorflow to inspect model details")
        except Exception as e:
            print(f"\nWarning: Could not inspect model: {e}")
    
    # Auto-detect model type if needed
    model_type = args.model_type
    if model_type == 'auto':
        if output.ndim <= 2 or (output.ndim == 3 and output.shape[0] == 1):
            model_type = 'classification'
        elif output.ndim == 4:
            if output.shape[-1] > 100:  # Likely segmentation
                model_type = 'segmentation'
            else:
                model_type = 'detection'
        else:
            model_type = 'classification'
        
        print(f"\nAuto-detected model type: {model_type}")
    
    # Analyze based on model type
    if model_type == 'classification':
        analyze_classification(output, labels, args.top_k)
    elif model_type == 'detection':
        analyze_detection(output)
    elif model_type == 'segmentation':
        analyze_segmentation(output)


if __name__ == '__main__':
    main()
