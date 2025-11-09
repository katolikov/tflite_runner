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
    std::cout << "  --input <path>       Path to input .npy file (repeatable)\n";
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
    std::vector<std::string> input_paths;
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
            config.input_paths.push_back(argv[++i]);
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
    if (config.input_paths.empty()) {
        std::cerr << "Error: At least one --input is required\n";
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
    std::cout << "Inputs (" << config.input_paths.size() << "):\n";
    for (size_t i = 0; i < config.input_paths.size(); ++i) {
        std::cout << "  [" << i << "] " << config.input_paths[i] << "\n";
    }
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
    std::vector<std::vector<float>> inputs_data;
    inputs_data.reserve(config.input_paths.size());
    
    for (size_t idx = 0; idx < config.input_paths.size(); ++idx) {
        const auto& input_path = config.input_paths[idx];
        std::vector<float> input_data;
        std::vector<size_t> input_shape;
        
        if (!tflite_runner::NPYReader::LoadNPY(input_path, input_data, input_shape)) {
            std::cerr << "Failed to load input NPY file: " << input_path << "\n";
            return 1;
        }
        
        std::cout << "Input[" << idx << "] loaded: shape = [";
        for (size_t i = 0; i < input_shape.size(); i++) {
            std::cout << input_shape[i];
            if (i < input_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "], size = " << input_data.size() << "\n";
        
        inputs_data.push_back(std::move(input_data));
    }
    
    int model_input_count = runner.GetInputTensorCount();
    std::cout << "Model expects " << model_input_count << " input(s)\n";
    for (int i = 0; i < model_input_count; ++i) {
        std::vector<int> model_input_shape = runner.GetInputShape(i);
        std::cout << "  Model Input[" << i << "] shape: [";
        for (size_t j = 0; j < model_input_shape.size(); j++) {
            std::cout << model_input_shape[j];
            if (j < model_input_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    if (model_input_count > 0 &&
        model_input_count != static_cast<int>(inputs_data.size())) {
        std::cerr << "Warning: Model expects " << model_input_count
                  << " inputs but " << inputs_data.size()
                  << " were provided.\n";
    }
    
    // Run inference
    std::cout << "\nRunning inference...\n";
    std::vector<std::vector<float>> outputs;
    if (!runner.RunInferenceMulti(inputs_data, outputs)) {
        std::cerr << "Inference failed\n";
        return 1;
    }
    
    std::vector<float> output_data;
    if (!outputs.empty()) {
        output_data = outputs[0];
    }
    
    std::cout << "Inference completed successfully\n";
    runner.PrintProfilingInfo();
    
    const auto& timing = runner.GetTimingStats();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nTiming profile (ms):\n";
    std::cout << "  Model load:        " << timing.model_load_ms << "\n";
    std::cout << "  Delegate init:     " << timing.delegate_init_ms << "\n";
    std::cout << "  Tensor allocation: " << timing.tensor_allocation_ms << "\n";
    std::cout << "  Input copy:        " << timing.input_copy_ms << "\n";
    std::cout << "  Inference:         " << timing.inference_ms << "\n";
    std::cout << "  Output copy:       " << timing.output_copy_ms << "\n";
    std::cout << "  Total:             " << timing.total_ms << "\n";
    std::cout << std::defaultfloat;
    
    const auto& mem = runner.GetMemoryAfterInference();
    if (mem.rss_kb > 0 || mem.vm_kb > 0) {
        std::cout << "\nMemory after inference (kB):\n";
        std::cout << "  RSS: " << mem.rss_kb << "\n";
        std::cout << "  VM:  " << mem.vm_kb << "\n";
    }

    const auto& gpu_mem = runner.GetGpuMemoryAfterInference();
    if (gpu_mem.available) {
        std::cout << "\nGPU memory snapshot (" << gpu_mem.source_path << "):\n";
        std::cout << gpu_mem.raw_report << "\n";
    } else {
        std::cout << "\nGPU memory snapshot not available on this device (kgsl/mali stats not exposed).\n";
    }
    
    tflite_runner::OpPlacementStats op_stats = runner.GetOpPlacementStats();
    std::cout << "\nDelegate placement:\n";
    std::cout << "  GPU ops: " << op_stats.gpu_ops << " / " << op_stats.total_ops << "\n";
    std::cout << "  CPU ops: " << op_stats.cpu_ops << " / " << op_stats.total_ops << "\n";
    if (config.use_gpu) {
        if (op_stats.cpu_ops == 0 && op_stats.total_ops > 0) {
            std::cout << "  All operations ran on the GPU delegate.\n";
        } else if (op_stats.cpu_ops > 0) {
            std::cout << "  WARNING: Some ops fell back to CPU; adjust the model/delegate for full GPU coverage.\n";
        }
    }
    
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
    
    if (!op_stats.cpu_op_names.empty()) {
        std::cout << "  Ops executed on CPU:\n";
        for (const auto& op_name : op_stats.cpu_op_names) {
            std::cout << "    - " << op_name << "\n";
        }
    }
    
    std::cout << "\n=== Execution completed successfully ===\n";
    LOGI("TensorFlow Lite Runner finished successfully");
    
    return 0;
}
