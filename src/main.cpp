#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <iomanip>
#include <android/log.h>
#include "tflite_runner.h"
#include "npy_io.h"
#include "image_utils.h"

#define LOG_TAG "Main"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

void print_usage(const char* program_name) {
    std::cout << "TensorFlow Lite Runner for Android with GPU Support\n";
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "\nRequired options:\n";
    std::cout << "  --model <path>       Path to .tflite model file\n";
    std::cout << "  --input <path>       Path to input .npy file\n";
    std::cout << "  --output <path>      Path to output .npy file\n";
    std::cout << "\nOptional options:\n";
    std::cout << "  --output-png <path>  Path to output .png file (for image outputs)\n";
    std::cout << "  --no-gpu            Disable GPU delegate (use CPU only)\n";
    std::cout << "  --help              Show this help message\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << program_name << " --model model.tflite --input input.npy --output output.npy --output-png output.png\n";
}

struct Config {
    std::string model_path;
    std::string input_path;
    std::string output_path;
    std::string output_png_path;
    bool use_gpu = true;
};

bool parse_arguments(int argc, char* argv[], Config& config) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        } else if (arg == "--model" && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            config.input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_path = argv[++i];
        } else if (arg == "--output-png" && i + 1 < argc) {
            config.output_png_path = argv[++i];
        } else if (arg == "--no-gpu") {
            config.use_gpu = false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }
    
    // Validate required arguments
    if (config.model_path.empty()) {
        std::cerr << "Error: --model is required\n";
        return false;
    }
    if (config.input_path.empty()) {
        std::cerr << "Error: --input is required\n";
        return false;
    }
    if (config.output_path.empty()) {
        std::cerr << "Error: --output is required\n";
        return false;
    }
    
    return true;
}

int main(int argc, char* argv[]) {
    LOGI("TensorFlow Lite Runner starting...");
    
    // Parse command line arguments
    Config config;
    if (!parse_arguments(argc, argv, config)) {
        if (argc > 1) {
            print_usage(argv[0]);
        }
        return 1;
    }
    
    std::cout << "=== TensorFlow Lite Runner ===\n";
    std::cout << "Model: " << config.model_path << "\n";
    std::cout << "Input: " << config.input_path << "\n";
    std::cout << "Output NPY: " << config.output_path << "\n";
    if (!config.output_png_path.empty()) {
        std::cout << "Output PNG: " << config.output_png_path << "\n";
    }
    std::cout << "GPU: " << (config.use_gpu ? "Enabled" : "Disabled") << "\n";
    std::cout << "==============================\n\n";
    
    // Create TFLite runner
    tflite_runner::TFLiteRunner runner;
    
    // Load model
    std::cout << "Loading model...\n";
    if (!runner.LoadModel(config.model_path)) {
        std::cerr << "Failed to load model\n";
        return 1;
    }
    std::cout << "Model loaded successfully\n";
    
    // Initialize GPU delegate if enabled
    if (config.use_gpu) {
        std::cout << "Initializing GPU delegate for Exynos...\n";
        if (!runner.InitGPUDelegate()) {
            std::cerr << "Failed to initialize GPU delegate, falling back to CPU\n";
            // Continue with CPU
        } else {
            std::cout << "GPU delegate initialized successfully\n";
        }
    }
    
    // Load input data
    std::cout << "Loading input data...\n";
    std::vector<float> input_data;
    std::vector<size_t> input_shape;
    
    if (!tflite_runner::NPYReader::LoadNPY(config.input_path, input_data, input_shape)) {
        std::cerr << "Failed to load input NPY file\n";
        return 1;
    }
    
    std::cout << "Input loaded: shape = [";
    for (size_t i = 0; i < input_shape.size(); i++) {
        std::cout << input_shape[i];
        if (i < input_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "], size = " << input_data.size() << "\n";
    
    // Verify input shape matches model
    std::vector<int> model_input_shape = runner.GetInputShape();
    std::cout << "Model expects input shape: [";
    for (size_t i = 0; i < model_input_shape.size(); i++) {
        std::cout << model_input_shape[i];
        if (i < model_input_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    // Run inference
    std::cout << "\nRunning inference...\n";
    std::vector<float> output_data;
    
    if (!runner.RunInference(input_data, output_data)) {
        std::cerr << "Inference failed\n";
        return 1;
    }
    
    std::cout << "Inference completed successfully\n";
    
    // Get output shape
    std::vector<int> output_shape = runner.GetOutputShape();
    std::cout << "Output shape: [";
    for (size_t i = 0; i < output_shape.size(); i++) {
        std::cout << output_shape[i];
        if (i < output_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "], size = " << output_data.size() << "\n";
    
    // Save output as NPY
    std::cout << "\nSaving output NPY...\n";
    std::vector<size_t> output_shape_size_t(output_shape.begin(), output_shape.end());
    
    if (!tflite_runner::NPYWriter::SaveNPY(config.output_path, output_data, output_shape_size_t)) {
        std::cerr << "Failed to save output NPY file\n";
        return 1;
    }
    std::cout << "Output saved to: " << config.output_path << "\n";
    
    // Save output as PNG if requested and if it's image-like data
    if (!config.output_png_path.empty()) {
        std::cout << "\nSaving output PNG...\n";
        
        // Try to interpret output as image
        // Common formats: [batch, height, width, channels] or [batch, channels, height, width]
        int width = 0, height = 0, channels = 1;
        
        if (output_shape.size() == 4) {
            // NHWC format (batch, height, width, channels)
            if (output_shape[0] == 1) {
                height = output_shape[1];
                width = output_shape[2];
                channels = output_shape[3];
            }
        } else if (output_shape.size() == 3) {
            // HWC format (height, width, channels)
            height = output_shape[0];
            width = output_shape[1];
            channels = output_shape[2];
        } else if (output_shape.size() == 2) {
            // HW format (height, width) - grayscale
            height = output_shape[0];
            width = output_shape[1];
            channels = 1;
        }
        
        if (width > 0 && height > 0 && (channels == 1 || channels == 3 || channels == 4)) {
            if (tflite_runner::ImageUtils::SaveAsPNG(config.output_png_path, output_data,
                                                     width, height, channels)) {
                std::cout << "PNG saved to: " << config.output_png_path << "\n";
            } else {
                std::cerr << "Warning: Failed to save PNG\n";
            }
        } else {
            std::cerr << "Warning: Output shape not suitable for PNG conversion\n";
            std::cerr << "Expected image-like dimensions, got: ";
            for (size_t i = 0; i < output_shape.size(); i++) {
                std::cerr << output_shape[i];
                if (i < output_shape.size() - 1) std::cerr << "x";
            }
            std::cerr << "\n";
        }
    }
    
    std::cout << "\n=== Execution completed successfully ===\n";
    
    // Print profiling information
    std::cout << "\n=== Performance Profiling ===\n";
    const auto& stats = runner.GetTimingStats();
    std::cout << "Model Load:         " << std::fixed << std::setprecision(2) 
              << stats.model_load_ms << " ms\n";
    std::cout << "Delegate Init:      " << stats.delegate_init_ms << " ms\n";
    std::cout << "Tensor Allocation:  " << stats.tensor_allocation_ms << " ms\n";
    std::cout << "Input Copy:         " << stats.input_copy_ms << " ms\n";
    std::cout << "Inference:          " << stats.inference_ms << " ms\n";
    std::cout << "Output Copy:        " << stats.output_copy_ms << " ms\n";
    std::cout << "Total Runtime:      " << stats.total_ms << " ms\n";
    
    // Print operation placement stats
    auto op_stats = runner.GetOpPlacementStats();
    std::cout << "\n=== Operation Placement ===\n";
    std::cout << "Total Operations:   " << op_stats.total_ops << "\n";
    std::cout << "GPU Operations:     " << op_stats.gpu_ops << " ("
              << std::fixed << std::setprecision(1)
              << (op_stats.total_ops > 0 ? (100.0 * op_stats.gpu_ops / op_stats.total_ops) : 0.0)
              << "%)\n";
    std::cout << "CPU Operations:     " << op_stats.cpu_ops << " ("
              << (op_stats.total_ops > 0 ? (100.0 * op_stats.cpu_ops / op_stats.total_ops) : 0.0)
              << "%)\n";
    
    if (!op_stats.cpu_op_names.empty()) {
        std::cout << "\nOperations running on CPU:\n";
        for (const auto& op_name : op_stats.cpu_op_names) {
            std::cout << "  - " << op_name << "\n";
        }
    }
    
    std::cout << "\n====================================\n";
    
    LOGI("TensorFlow Lite Runner finished successfully");
    
    return 0;
}
