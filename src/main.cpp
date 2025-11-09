#include <cctype>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <system_error>
#include <string>
#include <vector>
#include <android/log.h>
#include "tflite_runner.h"
#include "npy_io.h"
#include "image_utils.h"

#define LOG_TAG "Main"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

std::string JoinPath(const std::string& dir, const std::string& filename) {
    if (dir.empty() || dir == "." || dir == "./") {
        return filename;
    }
    if (dir.back() == '/') {
        return dir + filename;
    }
    return dir + "/" + filename;
}

std::string SanitizeFilename(const std::string& name) {
    if (name.empty()) {
        return "";
    }
    std::string sanitized;
    sanitized.reserve(name.size());
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.') {
            sanitized.push_back(c);
        } else {
            sanitized.push_back('_');
        }
    }
    // Remove leading underscores
    while (!sanitized.empty() && sanitized.front() == '_') {
        sanitized.erase(sanitized.begin());
    }
    if (sanitized.empty()) {
        sanitized = "output";
    }
    return sanitized;
}

std::vector<std::string> GenerateAutoOutputNames(const tflite_runner::TFLiteRunner& runner,
                                                 size_t output_count) {
    std::vector<std::string> names;
    names.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i) {
        std::string tensor_name = runner.GetOutputTensorName(static_cast<int>(i));
        std::string base = SanitizeFilename(tensor_name);
        if (base.empty()) {
            base = "output_" + std::to_string(i);
        }
        names.push_back(base + ".npy");
    }
    return names;
}

void EnsureDirectoryExists(const std::string& path) {
    if (path.empty()) return;
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
}

void EnsureParentDirectory(const std::string& file_path) {
    if (file_path.empty()) return;
    std::filesystem::path p(file_path);
    auto parent = p.parent_path();
    if (!parent.empty()) {
        EnsureDirectoryExists(parent.string());
    }
}

}  // namespace

void print_usage(const char* program_name) {
    std::cout << "TensorFlow Lite Runner for Android with GPU Support\n";
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "\nRequired options:\n";
    std::cout << "  --model <path>       Path to .tflite model file\n";
    std::cout << "  --input <path>       Path to input .npy file (repeatable)\n";
    std::cout << "  --output <path>      Path to output .npy file (repeatable, optional)\n";
    std::cout << "\nOptional options:\n";
    std::cout << "  --output-dir <path>  Directory for auto-generated outputs (default: ./outputs)\n";
    std::cout << "  --output-png <path>  Path to output .png file (for image outputs)\n";
    std::cout << "  --no-gpu            Disable GPU delegate (use CPU only)\n";
    std::cout << "  --help              Show this help message\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << program_name << " --model model.tflite --input input.npy --output output.npy --output-png output.png\n";
}

struct Config {
    std::string model_path;
    std::vector<std::string> input_paths;
    std::vector<std::string> output_paths;
    std::string output_dir = "outputs";
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
            config.output_paths.push_back(argv[++i]);
        } else if (arg == "--output-dir" && i + 1 < argc) {
            config.output_dir = argv[++i];
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
    if (!config.output_paths.empty()) {
        std::cout << "Outputs (" << config.output_paths.size() << "):\n";
        for (size_t i = 0; i < config.output_paths.size(); ++i) {
            std::cout << "  [" << i << "] " << config.output_paths[i] << "\n";
        }
    } else {
        std::cout << "Outputs: auto-named in directory \"" << config.output_dir << "\"\n";
    }
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
    std::vector<std::vector<size_t>> input_shapes;
    inputs_data.reserve(config.input_paths.size());
    input_shapes.reserve(config.input_paths.size());
    
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
        input_shapes.push_back(input_shape);
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

    if (!input_shapes.empty()) {
        std::vector<std::vector<int>> desired_shapes;
        const size_t limit = std::min(input_shapes.size(), static_cast<size_t>(model_input_count));
        desired_shapes.reserve(limit);
        for (size_t i = 0; i < limit; ++i) {
            std::vector<int> dims;
            dims.reserve(input_shapes[i].size());
            for (size_t dim : input_shapes[i]) {
                dims.push_back(static_cast<int>(dim));
            }
            desired_shapes.push_back(std::move(dims));
        }
        if (!desired_shapes.empty()) {
            if (!runner.ApplyInputShapes(desired_shapes)) {
                std::cerr << "Failed to apply input shapes from NPY metadata\n";
                return 1;
            }
        }
    }
    
    // Run inference
    std::cout << "\nRunning inference...\n";
    std::vector<std::vector<float>> outputs;
    if (!runner.RunInferenceMulti(inputs_data, outputs)) {
        std::cerr << "Inference failed\n";
        return 1;
    }
    
    if (outputs.empty()) {
        std::cerr << "Model produced no outputs\n";
        return 1;
    }
    
    const size_t output_count = outputs.size();
    std::vector<std::vector<size_t>> output_shapes;
    output_shapes.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i) {
        std::vector<int> shape_int = runner.GetOutputShape(static_cast<int>(i));
        std::vector<size_t> shape_size_t(shape_int.begin(), shape_int.end());
        output_shapes.push_back(shape_size_t);
    }
    
    std::vector<std::string> resolved_output_paths;
    if (!config.output_paths.empty()) {
        if (config.output_paths.size() != output_count) {
            if (!(config.output_paths.size() == 1 && output_count == 1)) {
                std::cerr << "Error: Provided " << config.output_paths.size()
                          << " output path(s) but model produced " << output_count << "\n";
                return 1;
            }
        }
        resolved_output_paths = config.output_paths;
        for (const auto& path : resolved_output_paths) {
            EnsureParentDirectory(path);
        }
    } else {
        EnsureDirectoryExists(config.output_dir);
        auto auto_names = GenerateAutoOutputNames(runner, output_count);
        for (size_t i = 0; i < auto_names.size(); ++i) {
            resolved_output_paths.push_back(JoinPath(config.output_dir, auto_names[i]));
        }
    }
    
    std::cout << "Inference completed successfully\n";
    
    std::cout << "\nOutputs (" << output_count << " tensors):\n";
    for (size_t i = 0; i < output_count; ++i) {
        std::cout << "  [" << i << "] shape = [";
        for (size_t j = 0; j < output_shapes[i].size(); ++j) {
            std::cout << output_shapes[i][j];
            if (j + 1 < output_shapes[i].size()) std::cout << ", ";
        }
        std::cout << "], file = " << resolved_output_paths[i] << "\n";
    }
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
    
    // Save output as NPY
    std::cout << "\nSaving output NPY files...\n";
    for (size_t i = 0; i < output_count; ++i) {
        if (!tflite_runner::NPYWriter::SaveNPY(resolved_output_paths[i], outputs[i], output_shapes[i])) {
            std::cerr << "Failed to save output tensor " << i << " to " << resolved_output_paths[i] << "\n";
            return 1;
        }
        std::cout << "  ✓ Tensor[" << i << "] -> " << resolved_output_paths[i] << "\n";
    }
    
    // Save output as PNG if requested and if it's image-like data (first output)
    if (!config.output_png_path.empty()) {
        std::cout << "\nSaving output PNG...\n";
        const auto& output_shape = output_shapes[0];
        const auto& output_data = outputs[0];
        
        // Try to interpret output as image
        int width = 0, height = 0, channels = 1;
        
        if (output_shape.size() == 4) {
            // NHWC format (batch, height, width, channels)
            if (output_shape[0] == 1) {
                height = static_cast<int>(output_shape[1]);
                width = static_cast<int>(output_shape[2]);
                channels = static_cast<int>(output_shape[3]);
            }
        } else if (output_shape.size() == 3) {
            // HWC format (height, width, channels)
            height = static_cast<int>(output_shape[0]);
            width = static_cast<int>(output_shape[1]);
            channels = static_cast<int>(output_shape[2]);
        } else if (output_shape.size() == 2) {
            // HW format (height, width) - grayscale
            height = static_cast<int>(output_shape[0]);
            width = static_cast<int>(output_shape[1]);
            channels = 1;
        }
        
        if (width > 0 && height > 0 && (channels == 1 || channels == 3 || channels == 4)) {
            EnsureParentDirectory(config.output_png_path);
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
