#include "tflite_runner.h"

#include <android/log.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define LOG_TAG "TFLiteRunner"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace tflite_runner {
namespace {

size_t ParseProcStatusValue(const std::string& line) {
    std::istringstream iss(line);
    std::string key;
    size_t value = 0;
    std::string unit;
    iss >> key >> value >> unit;
    return value;
}

MemoryStats ReadProcStatusMemory() {
    MemoryStats stats;
    std::ifstream status("/proc/self/status");
    if (!status.is_open()) {
        return stats;
    }

    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            stats.rss_kb = ParseProcStatusValue(line);
        } else if (line.rfind("VmSize:", 0) == 0) {
            stats.vm_kb = ParseProcStatusValue(line);
        }
    }
    return stats;
}

GpuMemorySnapshot ReadGpuMemorySnapshot() {
    static const std::vector<std::string> kGpuMemPaths = {
        "/sys/kernel/debug/kgsl/kgsl-3d0/memstat",
        "/d/kgsl/kgsl-3d0/memstat",
        "/sys/devices/virtual/kgsl/kgsl-3d0/memstat",
        "/proc/mali/meminfo",
        "/sys/devices/platform/mali/meminfo"
    };

    GpuMemorySnapshot snapshot;
    for (const auto& path : kGpuMemPaths) {
        std::ifstream file(path);
        if (!file.is_open()) {
            continue;
        }
        std::ostringstream buffer;
        buffer << file.rdbuf();
        snapshot.available = true;
        snapshot.source_path = path;
        snapshot.raw_report = buffer.str();
        break;
    }
    return snapshot;
}

size_t GetTensorElementCount(const TfLiteTensor* tensor) {
    if (!tensor) {
        return 0;
    }
    size_t count = 1;
    const int dims = TfLiteTensorNumDims(tensor);
    if (dims == 0) {
        return 1;
    }
    for (int i = 0; i < dims; ++i) {
        count *= static_cast<size_t>(TfLiteTensorDim(tensor, i));
    }
    return count;
}

void LogTensorInfo(const char* label, int index, const TfLiteTensor* tensor) {
    if (!tensor) {
        return;
    }
    LOGI("%s[%d]: name=%s, type=%d, dims=%d", label, index,
         tensor->name ? tensor->name : "", tensor->type,
         TfLiteTensorNumDims(tensor));
    for (int j = 0; j < TfLiteTensorNumDims(tensor); ++j) {
        LOGI("  dim[%d]: %d", j, TfLiteTensorDim(tensor, j));
    }
}

}  // namespace

TFLiteRunner::TFLiteRunner() = default;

TFLiteRunner::~TFLiteRunner() {
    Cleanup();
}

void TFLiteRunner::Cleanup() {
    if (gpu_delegate_) {
        TfLiteGpuDelegateV2Delete(gpu_delegate_);
        gpu_delegate_ = nullptr;
    }
    if (interpreter_) {
        TfLiteInterpreterDelete(interpreter_);
        interpreter_ = nullptr;
    }
    if (options_) {
        TfLiteInterpreterOptionsDelete(options_);
        options_ = nullptr;
    }
    if (model_) {
        TfLiteModelDelete(model_);
        model_ = nullptr;
    }
    tensors_allocated_ = false;
    timing_stats_ = TimingStats{};
    current_memory_ = MemoryStats{};
    memory_after_model_load_ = MemoryStats{};
    memory_after_delegate_init_ = MemoryStats{};
    memory_after_tensor_allocation_ = MemoryStats{};
    memory_after_inference_ = MemoryStats{};
}

bool TFLiteRunner::LoadModel(const std::string& model_path) {
    LOGI("Loading model from: %s", model_path.c_str());
    Cleanup();

    bool success = true;
    timing_stats_.model_load_ms = TimeFunction([&]() {
        model_ = TfLiteModelCreateFromFile(model_path.c_str());
        if (!model_) {
            success = false;
            return;
        }

        options_ = TfLiteInterpreterOptionsCreate();
        if (!options_) {
            success = false;
            return;
        }

        interpreter_ = TfLiteInterpreterCreate(model_, options_);
        if (!interpreter_) {
            success = false;
        }
    });

    if (!success || !model_ || !interpreter_) {
        LOGE("Failed to load model from: %s", model_path.c_str());
        return false;
    }

    LOGI("Model loaded successfully in %.2f ms", timing_stats_.model_load_ms);
    RecordMemorySnapshot(memory_after_model_load_);
    tensors_allocated_ = false;
    return true;
}

bool TFLiteRunner::InitGPUDelegate() {
    if (!interpreter_) {
        LOGE("Interpreter is not initialized");
        return false;
    }

    if (gpu_delegate_) {
        LOGI("GPU delegate already initialized");
        return true;
    }

    bool success = true;
    timing_stats_.delegate_init_ms = TimeFunction([&]() {
        TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
        options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
        options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        options.inference_preference =
            TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
        options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;

        gpu_delegate_ = TfLiteGpuDelegateV2Create(&options);
        if (!gpu_delegate_) {
            success = false;
            return;
        }

        if (TfLiteInterpreterModifyGraphWithDelegate(interpreter_,
                                                     gpu_delegate_) != kTfLiteOk) {
            LOGE("Failed to modify graph with GPU delegate");
            TfLiteGpuDelegateV2Delete(gpu_delegate_);
            gpu_delegate_ = nullptr;
            success = false;
        }
    });

    if (!success || !gpu_delegate_) {
        LOGE("GPU delegate initialization failed");
        return false;
    }

    RecordMemorySnapshot(memory_after_delegate_init_);
    RecordGpuMemorySnapshot(gpu_memory_after_delegate_init_);
    bool allocation_success = false;
    timing_stats_.tensor_allocation_ms = TimeFunction([&]() {
        allocation_success = AllocateTensors();
    });

    return allocation_success;
}

bool TFLiteRunner::AllocateTensors() {
    if (!interpreter_) {
        LOGE("Interpreter is not initialized");
        return false;
    }

    if (TfLiteInterpreterAllocateTensors(interpreter_) != kTfLiteOk) {
        LOGE("Failed to allocate tensors");
        tensors_allocated_ = false;
        return false;
    }

    tensors_allocated_ = true;

    const int input_count = TfLiteInterpreterGetInputTensorCount(interpreter_);
    LOGI("Input tensor count: %d", input_count);
    for (int i = 0; i < input_count; ++i) {
        LogTensorInfo("Input", i,
                      TfLiteInterpreterGetInputTensor(interpreter_, i));
    }

    const int output_count = TfLiteInterpreterGetOutputTensorCount(interpreter_);
    LOGI("Output tensor count: %d", output_count);
    for (int i = 0; i < output_count; ++i) {
        LogTensorInfo("Output", i,
                      TfLiteInterpreterGetOutputTensor(interpreter_, i));
    }

    RecordMemorySnapshot(memory_after_tensor_allocation_);
    return true;
}

MemoryStats TFLiteRunner::CaptureMemoryStats() const {
    return ReadProcStatusMemory();
}

void TFLiteRunner::RecordMemorySnapshot(MemoryStats& slot) {
    if (!profiling_enabled_) {
        return;
    }
    slot = CaptureMemoryStats();
    current_memory_ = slot;
}

GpuMemorySnapshot TFLiteRunner::CaptureGpuMemoryStats() const {
    if (!profiling_enabled_) {
        return {};
    }
    return ReadGpuMemorySnapshot();
}

void TFLiteRunner::RecordGpuMemorySnapshot(GpuMemorySnapshot& slot) {
    if (!profiling_enabled_) {
        return;
    }
    slot = CaptureGpuMemoryStats();
}

bool TFLiteRunner::RunInference(const std::vector<float>& input_data,
                                std::vector<float>& output_data) {
    std::vector<std::vector<float>> inputs = {input_data};
    std::vector<std::vector<float>> outputs;

    if (!RunInferenceMulti(inputs, outputs)) {
        return false;
    }

    if (!outputs.empty()) {
        output_data = outputs[0];
    }

    return true;
}

bool TFLiteRunner::RunInferenceMulti(
    const std::vector<std::vector<float>>& inputs,
    std::vector<std::vector<float>>& outputs) {
    if (!interpreter_) {
        LOGE("Interpreter is not initialized");
        return false;
    }

    if (!tensors_allocated_) {
        bool allocation_success = false;
        timing_stats_.tensor_allocation_ms = TimeFunction([&]() {
            allocation_success = AllocateTensors();
        });
        if (!allocation_success) {
            return false;
        }
    }

    LOGI("Running inference with %zu inputs", inputs.size());
    auto total_start = std::chrono::high_resolution_clock::now();

    const int expected_input_count =
        TfLiteInterpreterGetInputTensorCount(interpreter_);
    if (inputs.size() != static_cast<size_t>(expected_input_count)) {
        LOGE("Input count mismatch: expected %d, got %zu", expected_input_count,
             inputs.size());
        return false;
    }

    bool input_success = true;
    timing_stats_.input_copy_ms = TimeFunction([&]() {
        for (size_t i = 0; i < inputs.size(); ++i) {
            TfLiteTensor* input_tensor =
                TfLiteInterpreterGetInputTensor(interpreter_,
                                                 static_cast<int>(i));
            if (!input_tensor) {
                LOGE("Failed to get input tensor %zu", i);
                input_success = false;
                return;
            }

            const size_t input_size = GetTensorElementCount(input_tensor);
            if (inputs[i].size() != input_size) {
                LOGE("Input[%zu] data size mismatch: expected %zu, got %zu", i,
                     input_size, inputs[i].size());
                input_success = false;
                return;
            }

            TfLiteStatus status = kTfLiteError;
            switch (input_tensor->type) {
                case kTfLiteFloat32:
                    status = TfLiteTensorCopyFromBuffer(
                        input_tensor, inputs[i].data(),
                        inputs[i].size() * sizeof(float));
                    break;
                case kTfLiteUInt8: {
                    std::vector<uint8_t> quantized(inputs[i].size());
                    for (size_t j = 0; j < inputs[i].size(); ++j) {
                        quantized[j] = static_cast<uint8_t>(inputs[i][j]);
                    }
                    status = TfLiteTensorCopyFromBuffer(
                        input_tensor, quantized.data(), quantized.size());
                    break;
                }
                case kTfLiteInt8: {
                    std::vector<int8_t> quantized(inputs[i].size());
                    for (size_t j = 0; j < inputs[i].size(); ++j) {
                        quantized[j] = static_cast<int8_t>(inputs[i][j]);
                    }
                    status = TfLiteTensorCopyFromBuffer(
                        input_tensor, quantized.data(), quantized.size());
                    break;
                }
                default:
                    LOGE("Unsupported input tensor type: %d", input_tensor->type);
                    input_success = false;
                    return;
            }

            if (status != kTfLiteOk) {
                LOGE("Failed to copy data into input tensor %zu", i);
                input_success = false;
                return;
            }
        }
    });

    if (!input_success) {
        return false;
    }

    bool invoke_success = true;
    timing_stats_.inference_ms = TimeFunction([&]() {
        if (TfLiteInterpreterInvoke(interpreter_) != kTfLiteOk) {
            LOGE("Failed to invoke interpreter");
            invoke_success = false;
        }
    });

    if (!invoke_success) {
        return false;
    }

    bool output_success = true;
    timing_stats_.output_copy_ms = TimeFunction([&]() {
        const int output_count =
            TfLiteInterpreterGetOutputTensorCount(interpreter_);
        outputs.resize(output_count);

        for (int i = 0; i < output_count; ++i) {
            const TfLiteTensor* output_tensor =
                TfLiteInterpreterGetOutputTensor(interpreter_, i);
            if (!output_tensor) {
                LOGE("Failed to get output tensor %d", i);
                output_success = false;
                return;
            }

            const size_t output_size = GetTensorElementCount(output_tensor);
            outputs[i].resize(output_size);

            TfLiteStatus status = kTfLiteError;
            switch (output_tensor->type) {
                case kTfLiteFloat32:
                    status = TfLiteTensorCopyToBuffer(
                        output_tensor, outputs[i].data(),
                        output_size * sizeof(float));
                    break;
                case kTfLiteUInt8: {
                    std::vector<uint8_t> buffer(output_size);
                    status = TfLiteTensorCopyToBuffer(output_tensor,
                                                       buffer.data(),
                                                       buffer.size());
                    if (status == kTfLiteOk) {
                        for (size_t j = 0; j < output_size; ++j) {
                            outputs[i][j] = static_cast<float>(buffer[j]);
                        }
                    }
                    break;
                }
                case kTfLiteInt8: {
                    std::vector<int8_t> buffer(output_size);
                    status = TfLiteTensorCopyToBuffer(output_tensor,
                                                       buffer.data(),
                                                       buffer.size());
                    if (status == kTfLiteOk) {
                        for (size_t j = 0; j < output_size; ++j) {
                            outputs[i][j] = static_cast<float>(buffer[j]);
                        }
                    }
                    break;
                }
                default:
                    LOGE("Unsupported output tensor type: %d",
                         output_tensor->type);
                    output_success = false;
                    return;
            }

            if (status != kTfLiteOk) {
                LOGE("Failed to copy data from output tensor %d", i);
                output_success = false;
                return;
            }
        }
    });

    if (!output_success) {
        return false;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    timing_stats_.total_ms = std::chrono::duration<double, std::milli>(
        total_end - total_start)
                                 .count();

    RecordMemorySnapshot(memory_after_inference_);
    RecordGpuMemorySnapshot(gpu_memory_after_inference_);
    LOGI("Inference completed in %.2f ms", timing_stats_.inference_ms);
    return true;
}

std::vector<int> TFLiteRunner::GetInputShape(int index) const {
    std::vector<int> shape;
    if (!interpreter_) {
        return shape;
    }

    const int input_count = TfLiteInterpreterGetInputTensorCount(interpreter_);
    if (index < 0 || index >= input_count) {
        return shape;
    }

    const TfLiteTensor* tensor =
        TfLiteInterpreterGetInputTensor(interpreter_, index);
    if (!tensor) {
        return shape;
    }

    for (int i = 0; i < TfLiteTensorNumDims(tensor); ++i) {
        shape.push_back(TfLiteTensorDim(tensor, i));
    }
    return shape;
}

std::vector<int> TFLiteRunner::GetOutputShape(int index) const {
    std::vector<int> shape;
    if (!interpreter_) {
        return shape;
    }

    const int output_count = TfLiteInterpreterGetOutputTensorCount(interpreter_);
    if (index < 0 || index >= output_count) {
        return shape;
    }

    const TfLiteTensor* tensor =
        TfLiteInterpreterGetOutputTensor(interpreter_, index);
    if (!tensor) {
        return shape;
    }

    for (int i = 0; i < TfLiteTensorNumDims(tensor); ++i) {
        shape.push_back(TfLiteTensorDim(tensor, i));
    }
    return shape;
}

int TFLiteRunner::GetInputTensorCount() const {
    return interpreter_ ? TfLiteInterpreterGetInputTensorCount(interpreter_) : 0;
}

int TFLiteRunner::GetOutputTensorCount() const {
    return interpreter_ ? TfLiteInterpreterGetOutputTensorCount(interpreter_) : 0;
}

std::string TFLiteRunner::GetOutputTensorName(int index) const {
    if (!interpreter_) {
        return "";
    }
    const int count = TfLiteInterpreterGetOutputTensorCount(interpreter_);
    if (index < 0 || index >= count) {
        return "";
    }
    const TfLiteTensor* tensor = TfLiteInterpreterGetOutputTensor(interpreter_, index);
    if (!tensor || !tensor->name) {
        return "";
    }
    return tensor->name;
}

OpPlacementStats TFLiteRunner::GetOpPlacementStats() const {
    OpPlacementStats stats;
    if (!interpreter_ || !interpreter_->impl) {
        return stats;
    }

    const auto& execution_plan = interpreter_->impl->execution_plan();
    stats.total_ops = static_cast<int>(execution_plan.size());

    for (int node_index : execution_plan) {
        const auto* node_and_registration =
            interpreter_->impl->node_and_registration(node_index);
        if (!node_and_registration) {
            continue;
        }

        const TfLiteNode& node = node_and_registration->first;
        const TfLiteRegistration& registration = node_and_registration->second;

        if (node.delegate) {
            stats.gpu_ops++;
        } else {
            stats.cpu_ops++;
            const char* op_name = registration.custom_name
                                      ? registration.custom_name
                                      : tflite::EnumNameBuiltinOperator(
                                            static_cast<tflite::BuiltinOperator>(
                                                registration.builtin_code));
            if (op_name) {
                stats.cpu_op_names.emplace_back(op_name);
            }
        }
    }

    return stats;
}

void TFLiteRunner::PrintProfilingInfo() const {
    LOGI("=== Profiling Information ===");
    LOGI("Model Load:         %.2f ms", timing_stats_.model_load_ms);
    LOGI("Delegate Init:      %.2f ms", timing_stats_.delegate_init_ms);
    LOGI("Tensor Allocation:  %.2f ms", timing_stats_.tensor_allocation_ms);
    LOGI("Input Copy:         %.2f ms", timing_stats_.input_copy_ms);
    LOGI("Inference:          %.2f ms", timing_stats_.inference_ms);
    LOGI("Output Copy:        %.2f ms", timing_stats_.output_copy_ms);
    LOGI("Total Runtime:      %.2f ms", timing_stats_.total_ms);

    if (profiling_enabled_) {
        LOGI("=== Memory Snapshots (kB) ===");
        LOGI("After Model Load:   RSS=%zu, VM=%zu", memory_after_model_load_.rss_kb,
             memory_after_model_load_.vm_kb);
        LOGI("After Delegate:     RSS=%zu, VM=%zu",
             memory_after_delegate_init_.rss_kb,
             memory_after_delegate_init_.vm_kb);
        LOGI("After Allocation:   RSS=%zu, VM=%zu",
             memory_after_tensor_allocation_.rss_kb,
             memory_after_tensor_allocation_.vm_kb);
        LOGI("After Inference:    RSS=%zu, VM=%zu",
             memory_after_inference_.rss_kb, memory_after_inference_.vm_kb);

        LOGI("=== GPU Memory Snapshots ===");
        if (gpu_memory_after_delegate_init_.available) {
            LOGI("After Delegate Init (source: %s):",
                 gpu_memory_after_delegate_init_.source_path.c_str());
            LOGI("%s", gpu_memory_after_delegate_init_.raw_report.c_str());
        } else {
            LOGI("After Delegate Init: GPU mem stats unavailable on this device");
        }
        if (gpu_memory_after_inference_.available) {
            LOGI("After Inference (source: %s):",
                 gpu_memory_after_inference_.source_path.c_str());
            LOGI("%s", gpu_memory_after_inference_.raw_report.c_str());
        } else {
            LOGI("After Inference: GPU mem stats unavailable on this device");
        }
    } else {
        LOGI("Memory profiling disabled.");
    }

    OpPlacementStats op_stats = GetOpPlacementStats();
    LOGI("=== Operation Placement ===");
    LOGI("Total Operations:   %d", op_stats.total_ops);
    LOGI("GPU Operations:     %d (%.1f%%)", op_stats.gpu_ops,
         op_stats.total_ops > 0
             ? (100.0 * op_stats.gpu_ops / op_stats.total_ops)
             : 0.0);
    LOGI("CPU Operations:     %d (%.1f%%)", op_stats.cpu_ops,
         op_stats.total_ops > 0
             ? (100.0 * op_stats.cpu_ops / op_stats.total_ops)
             : 0.0);
    if (op_stats.cpu_ops == 0 && op_stats.total_ops > 0) {
        LOGI("GPU delegation: All ops executed on GPU.");
    } else if (op_stats.cpu_ops > 0) {
        LOGW("GPU delegation: %d ops executed on CPU fallback.", op_stats.cpu_ops);
    }

    if (!op_stats.cpu_op_names.empty()) {
        LOGI("CPU Operations:");
        for (const auto& op_name : op_stats.cpu_op_names) {
            LOGI("  - %s", op_name.c_str());
        }
    }
}

}  // namespace tflite_runner
