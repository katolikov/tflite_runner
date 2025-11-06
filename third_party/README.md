# Third-Party Dependencies

This directory contains third-party libraries required by the TensorFlow Lite Runner.

## Setup Instructions

Run the setup script to download all dependencies:

```bash
cd third_party
chmod +x setup_all.sh
./setup_all.sh
```

This will download:
- TensorFlow Lite libraries (arm64-v8a, armeabi-v7a)
- TensorFlow Lite GPU delegate
- TensorFlow Lite headers
- cnpy (NumPy file format library)
- stb_image_write (PNG output library)

## Individual Setup Scripts

You can also run individual setup scripts:

```bash
./setup_tflite.sh    # Download TFLite prebuilt libraries
./setup_headers.sh   # Download TFLite headers from GitHub
./setup_cnpy.sh      # Clone cnpy library
./setup_stb.sh       # Download stb_image_write.h
```

## Directory Structure After Setup

```
third_party/
├── tensorflow/
│   └── lite/                  # TensorFlow Lite headers
├── libs/
│   ├── arm64-v8a/            # 64-bit ARM libraries
│   │   ├── libtensorflowlite.a
│   │   └── libtensorflowlite_gpu_delegate.so
│   └── armeabi-v7a/          # 32-bit ARM libraries
│       ├── libtensorflowlite.a
│       └── libtensorflowlite_gpu_delegate.so
├── cnpy/
│   ├── cnpy.h
│   └── cnpy.cpp
└── stb/
    └── stb_image_write.h
```

## Notes

- The setup scripts require `wget` and `git` to be installed
- Internet connection is required to download dependencies
- Total download size is approximately 50-100 MB
- TensorFlow Lite version used: 2.14.0
