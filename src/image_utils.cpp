#include "image_utils.h"
#include <android/log.h>
#include <algorithm>
#include <cmath>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define LOG_TAG "ImageUtils"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace tflite_runner {

std::vector<uint8_t> ImageUtils::NormalizeToUInt8(const std::vector<float>& data) {
    std::vector<uint8_t> result(data.size());
    
    if (data.empty()) {
        return result;
    }
    
    // Find min and max values
    float min_val = *std::min_element(data.begin(), data.end());
    float max_val = *std::max_element(data.begin(), data.end());
    
    LOGI("Normalizing data: min=%.6f, max=%.6f", min_val, max_val);
    
    // Check if data is already in [0, 1] or [0, 255] range
    if (min_val >= 0.0f && max_val <= 1.0f) {
        // Data in [0, 1], scale to [0, 255]
        for (size_t i = 0; i < data.size(); i++) {
            result[i] = static_cast<uint8_t>(data[i] * 255.0f);
        }
    } else if (min_val >= 0.0f && max_val <= 255.0f) {
        // Data already in [0, 255]
        for (size_t i = 0; i < data.size(); i++) {
            result[i] = static_cast<uint8_t>(std::round(data[i]));
        }
    } else {
        // Normalize to [0, 255] from arbitrary range
        float range = max_val - min_val;
        if (range < 1e-6f) {
            // All values are the same
            std::fill(result.begin(), result.end(), 128);
        } else {
            for (size_t i = 0; i < data.size(); i++) {
                float normalized = (data[i] - min_val) / range;
                result[i] = static_cast<uint8_t>(normalized * 255.0f);
            }
        }
    }
    
    return result;
}

bool ImageUtils::SaveAsPNG(const std::string& filename,
                          const std::vector<float>& data,
                          int width, int height, int channels) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        LOGE("Invalid dimensions: width=%d, height=%d, channels=%d", 
             width, height, channels);
        return false;
    }
    
    if (data.size() != static_cast<size_t>(width * height * channels)) {
        LOGE("Data size mismatch: expected %d, got %zu", 
             width * height * channels, data.size());
        return false;
    }
    
    // Normalize to uint8
    std::vector<uint8_t> uint8_data = NormalizeToUInt8(data);
    
    // Save using stb_image_write
    int result = stbi_write_png(filename.c_str(), width, height, channels,
                                uint8_data.data(), width * channels);
    
    if (result == 0) {
        LOGE("Failed to save PNG: %s", filename.c_str());
        return false;
    }
    
    LOGI("Saved PNG: %s (%dx%d, %d channels)", 
         filename.c_str(), width, height, channels);
    return true;
}

bool ImageUtils::SaveAsPNGUInt8(const std::string& filename,
                               const std::vector<uint8_t>& data,
                               int width, int height, int channels) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        LOGE("Invalid dimensions: width=%d, height=%d, channels=%d", 
             width, height, channels);
        return false;
    }
    
    if (data.size() != static_cast<size_t>(width * height * channels)) {
        LOGE("Data size mismatch: expected %d, got %zu", 
             width * height * channels, data.size());
        return false;
    }
    
    // Save using stb_image_write
    int result = stbi_write_png(filename.c_str(), width, height, channels,
                                data.data(), width * channels);
    
    if (result == 0) {
        LOGE("Failed to save PNG: %s", filename.c_str());
        return false;
    }
    
    LOGI("Saved PNG: %s (%dx%d, %d channels)", 
         filename.c_str(), width, height, channels);
    return true;
}

} // namespace tflite_runner
