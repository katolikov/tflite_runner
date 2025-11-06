#include "npy_io.h"
#include "cnpy.h"
#include <android/log.h>

#define LOG_TAG "NPY_IO"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace tflite_runner {

bool NPYReader::LoadNPY(const std::string& filename, 
                        std::vector<float>& data,
                        std::vector<size_t>& shape) {
    try {
        cnpy::NpyArray arr = cnpy::npy_load(filename);
        
        shape = arr.shape;
        
        size_t total_size = 1;
        for (size_t dim : shape) {
            total_size *= dim;
        }
        
        data.resize(total_size);
        
        // Handle different data types
        if (arr.word_size == sizeof(float)) {
            float* arr_data = arr.data<float>();
            std::copy(arr_data, arr_data + total_size, data.begin());
        } else if (arr.word_size == sizeof(double)) {
            double* arr_data = arr.data<double>();
            for (size_t i = 0; i < total_size; i++) {
                data[i] = static_cast<float>(arr_data[i]);
            }
        } else {
            LOGE("Unsupported NPY word size: %zu", arr.word_size);
            return false;
        }
        
        LOGI("Loaded NPY file: %s, shape: [", filename.c_str());
        for (size_t dim : shape) {
            LOGI("%zu, ", dim);
        }
        LOGI("]");
        
        return true;
    } catch (const std::exception& e) {
        LOGE("Failed to load NPY file %s: %s", filename.c_str(), e.what());
        return false;
    }
}

bool NPYReader::LoadNPYInt8(const std::string& filename,
                           std::vector<int8_t>& data,
                           std::vector<size_t>& shape) {
    try {
        cnpy::NpyArray arr = cnpy::npy_load(filename);
        
        shape = arr.shape;
        
        size_t total_size = 1;
        for (size_t dim : shape) {
            total_size *= dim;
        }
        
        data.resize(total_size);
        
        if (arr.word_size == sizeof(int8_t)) {
            int8_t* arr_data = arr.data<int8_t>();
            std::copy(arr_data, arr_data + total_size, data.begin());
        } else {
            LOGE("Unsupported NPY word size for int8: %zu", arr.word_size);
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        LOGE("Failed to load NPY int8 file %s: %s", filename.c_str(), e.what());
        return false;
    }
}

bool NPYReader::LoadNPYUInt8(const std::string& filename,
                            std::vector<uint8_t>& data,
                            std::vector<size_t>& shape) {
    try {
        cnpy::NpyArray arr = cnpy::npy_load(filename);
        
        shape = arr.shape;
        
        size_t total_size = 1;
        for (size_t dim : shape) {
            total_size *= dim;
        }
        
        data.resize(total_size);
        
        if (arr.word_size == sizeof(uint8_t)) {
            uint8_t* arr_data = arr.data<uint8_t>();
            std::copy(arr_data, arr_data + total_size, data.begin());
        } else {
            LOGE("Unsupported NPY word size for uint8: %zu", arr.word_size);
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        LOGE("Failed to load NPY uint8 file %s: %s", filename.c_str(), e.what());
        return false;
    }
}

bool NPYWriter::SaveNPY(const std::string& filename,
                       const std::vector<float>& data,
                       const std::vector<size_t>& shape) {
    try {
        cnpy::npy_save(filename, data.data(), shape, "w");
        LOGI("Saved NPY file: %s", filename.c_str());
        return true;
    } catch (const std::exception& e) {
        LOGE("Failed to save NPY file %s: %s", filename.c_str(), e.what());
        return false;
    }
}

bool NPYWriter::SaveNPYInt8(const std::string& filename,
                           const std::vector<int8_t>& data,
                           const std::vector<size_t>& shape) {
    try {
        cnpy::npy_save(filename, data.data(), shape, "w");
        LOGI("Saved NPY int8 file: %s", filename.c_str());
        return true;
    } catch (const std::exception& e) {
        LOGE("Failed to save NPY int8 file %s: %s", filename.c_str(), e.what());
        return false;
    }
}

bool NPYWriter::SaveNPYUInt8(const std::string& filename,
                            const std::vector<uint8_t>& data,
                            const std::vector<size_t>& shape) {
    try {
        cnpy::npy_save(filename, data.data(), shape, "w");
        LOGI("Saved NPY uint8 file: %s", filename.c_str());
        return true;
    } catch (const std::exception& e) {
        LOGE("Failed to save NPY uint8 file %s: %s", filename.c_str(), e.what());
        return false;
    }
}

} // namespace tflite_runner
