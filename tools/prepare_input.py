#!/usr/bin/env python3
"""
Helper script to prepare input data for TensorFlow Lite Runner

Usage:
    python prepare_input.py image.jpg --output input.npy --size 224 --normalize mobilenet
"""

import argparse
import numpy as np
from PIL import Image


def load_image(image_path, target_size):
    """Load and resize image"""
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((target_size, target_size))
    return np.array(img)


def normalize_input(image_array, method='imagenet'):
    """Normalize image according to different methods"""
    image_array = image_array.astype(np.float32)
    
    if method == 'imagenet':
        # ImageNet normalization: mean=[123.68, 116.78, 103.94], std=[58.393, 57.12, 57.375]
        mean = np.array([123.68, 116.78, 103.94])
        std = np.array([58.393, 57.12, 57.375])
        image_array = (image_array - mean) / std
    elif method == 'mobilenet':
        # MobileNet normalization: [0, 255] -> [-1, 1]
        image_array = (image_array / 127.5) - 1.0
    elif method == 'inception':
        # Inception normalization: [0, 255] -> [-1, 1]
        image_array = (image_array / 127.5) - 1.0
    elif method == 'zero_one':
        # Simple [0, 255] -> [0, 1]
        image_array = image_array / 255.0
    elif method == 'none':
        # No normalization
        pass
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return image_array


def main():
    parser = argparse.ArgumentParser(description='Prepare input data for TensorFlow Lite Runner')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('--output', '-o', default='input.npy', help='Output NPY file path')
    parser.add_argument('--size', '-s', type=int, default=224, help='Target image size (default: 224)')
    parser.add_argument('--normalize', '-n', default='mobilenet',
                       choices=['imagenet', 'mobilenet', 'inception', 'zero_one', 'none'],
                       help='Normalization method')
    parser.add_argument('--quantize', '-q', action='store_true',
                       help='Output as uint8 (quantized) instead of float32')
    parser.add_argument('--channels-first', action='store_true',
                       help='Output in NCHW format instead of NHWC')
    
    args = parser.parse_args()
    
    print(f"Loading image: {args.image}")
    image_array = load_image(args.image, args.size)
    print(f"Image shape: {image_array.shape}")
    
    if not args.quantize:
        print(f"Normalizing with method: {args.normalize}")
        image_array = normalize_input(image_array, args.normalize)
    else:
        print("Using uint8 quantization (no normalization)")
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    # Convert to channels-first if requested
    if args.channels_first:
        print("Converting to NCHW format")
        image_array = np.transpose(image_array, (0, 3, 1, 2))
    
    print(f"Final shape: {image_array.shape}")
    print(f"Data type: {image_array.dtype}")
    print(f"Value range: [{image_array.min():.4f}, {image_array.max():.4f}]")
    
    # Save to NPY
    np.save(args.output, image_array)
    print(f"\nSaved to: {args.output}")
    print(f"File size: {np.load(args.output).nbytes / 1024:.2f} KB")


if __name__ == '__main__':
    main()
