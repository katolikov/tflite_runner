cmake_minimum_required(VERSION 3.18)
project(tflite_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Android-specific settings
if(ANDROID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
endif()

# TensorFlow Lite paths
set(TFLITE_DIR "${CMAKE_SOURCE_DIR}/third_party/tensorflow/lite")
# Shared libraries extracted from AARs are stored under third_party/libs/<abi>
set(TFLITE_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/libs/${ANDROID_ABI}")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/tensorflow/lite/tools/make/downloads/flatbuffers/include
    ${CMAKE_SOURCE_DIR}/third_party/stb
    ${CMAKE_SOURCE_DIR}/third_party/cnpy
)

# Source files
set(SOURCES
    src/main.cpp
    src/tflite_runner.cpp
    src/npy_io.cpp
    src/image_utils.cpp
    third_party/cnpy/cnpy.cpp
)

# Create executable
add_executable(tflite_runner ${SOURCES})

# Link TensorFlow Lite libraries
# Note: Using shared libraries from AAR (JNI .so names)
target_link_libraries(tflite_runner
    ${TFLITE_LIB_DIR}/libtensorflowlite_jni.so
    ${TFLITE_LIB_DIR}/libtensorflowlite_gpu_jni.so
    android
    log
    z
)

# Additional compile definitions
target_compile_definitions(tflite_runner PRIVATE
    TFLITE_GPU_BINARY_RELEASE
)
