#include "tflite_runner.h"
#include <android/log.h>
#include <cstring>
#include <iomanip>
#include <sstream>

#define LOG_TAG "TFLiteRunner"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace tflite_runner {

TFLiteRunner::TFLiteRunner() {}

TFLiteRunner::~TFLiteRunner() {
    if (gpu_delegate_) {
        TfLiteGpuDelegateV2Delete(gpu_delegate_);
        gpu_delegate_ = nullptr;
    }
}

bool TFLiteRunner::LoadModel(const std::string& model_path) {
    LOGI("Loading model from: %s", model_path.c_str());
    
    timing_stats_.model_load_ms = TimeFunction([&]() {
        model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    });
    
    if (!model_) {
        LOGE("Failed to load model from: %s", model_path.c_str());
        return false;
    }
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    builder(&interpreter_);
    
    if (!interpreter_) {
        LOGE("Failed to create interpreter");
        return false;
    }
    
    LOGI("Model loaded successfully in %.2f ms", timing_stats_.model_load_ms);
    return true;
}

bool TFLiteRunner::InitGPUDelegate() {
    LOGI("Initializing GPU delegate for Exynos device");
    
    timing_stats_.delegate_init_ms = TimeFunction([&]() {
        // Configure GPU delegate options for optimal performance on Exynos
        TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
        options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
        options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
        
        // Enable GL sharing for better performance
        options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
        
        gpu_delegate_ = TfLiteGpuDelegateV2Create(&options);
        if (gpu_delegate_) {
            if (interpreter_->ModifyGraphWithDelegate(gpu_delegate_) != kTfLiteOk) {
                LOGE("Failed to modify graph with GPU delegate");
                TfLiteGpuDelegateV2Delete(gpu_delegate_);
                gpu_delegate_ = nullptr;
            }
        }
    });
    
    if (!gpu_delegate_) {
        LOGE("GPU delegate initialization failed");
        return false;
    }
    
    LOGI("GPU delegate initialized successfully in %.2f ms", timing_stats_.delegate_init_ms);
    
    timing_stats_.tensor_allocation_ms = TimeFunction([&]() {
        AllocateTensors();
    });
    
    return true;
}

bool TFLiteRunner::AllocateTensors() {
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        LOGE("Failed to allocate tensors");
        return false;
    }
    
    LOGI("Tensors allocated successfully in %.2f ms", timing_stats_.tensor_allocation_ms);
    
    // Log input tensor info
    int input_count = interpreter_->inputs().size();
    LOGI("Input tensor count: %d", input_count);
    
    for (int i = 0; i < input_count; i++) {
        TfLiteTensor* tensor = interpreter_->input_tensor(i);
        LOGI("Input[%d]: name=%s, type=%d, dims=%d", 
             i, tensor->name, tensor->type, tensor->dims->size);
        for (int j = 0; j < tensor->dims->size; j++) {
            LOGI("  dim[%d]: %d", j, tensor->dims->data[j]);
        }
    }
    
    // Log output tensor info
    int output_count = interpreter_->outputs().size();
    LOGI("Output tensor count: %d", output_count);
    
    for (int i = 0; i < output_count; i++) {
        TfLiteTensor* tensor = interpreter_->output_tensor(i);
        LOGI("Output[%d]: name=%s, type=%d, dims=%d", 
             i, tensor->name, tensor->type, tensor->dims->size);
        for (int j = 0; j < tensor->dims->size; j++) {
            LOGI("  dim[%d]: %d", j, tensor->dims->data[j]);
        }
    }
    
    return true;
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

bool TFLiteRunner::RunInferenceMulti(const std::vector<std::vector<float>>& inputs,
                                     std::vector<std::vector<float>>& outputs) {
    LOGI("Running inference with %zu inputs", inputs.size());
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Validate input count
    int expected_input_count = interpreter_->inputs().size();
    if (inputs.size() != static_cast<size_t>(expected_input_count)) {
        LOGE("Input count mismatch: expected %d, got %zu", 
             expected_input_count, inputs.size());
        return false;
    }
    
    // Copy input data
    timing_stats_.input_copy_ms = TimeFunction([&]() {
        for (size_t i = 0; i < inputs.size(); i++) {
            TfLiteTensor* input_tensor = interpreter_->input_tensor(i);
            if (!input_tensor) {
                LOGE("Failed to get input tensor %zu", i);
                return;
            }
            
            // Calculate expected input size
            size_t input_size = 1;
            for (int j = 0; j < input_tensor->dims->size; j++) {
                input_size *= input_tensor->dims->data[j];
            }
            
            if (inputs[i].size() != input_size) {
                LOGE("Input[%zu] data size mismatch: expected %zu, got %zu", 
                     i, input_size, inputs[i].size());
                return;
            }
            
            // Copy input data based on type
            if (input_tensor->type == kTfLiteFloat32) {
                float* input_ptr = interpreter_->typed_input_tensor<float>(i);
                std::memcpy(input_ptr, inputs[i].data(), inputs[i].size() * sizeof(float));
            } else if (input_tensor->type == kTfLiteUInt8) {
                uint8_t* input_ptr = interpreter_->typed_input_tensor<uint8_t>(i);
                for (size_t j = 0; j < inputs[i].size(); j++) {
                    input_ptr[j] = static_cast<uint8_t>(inputs[i][j]);
                }
            } else if (input_tensor->type == kTfLiteInt8) {
                int8_t* input_ptr = interpreter_->typed_input_tensor<int8_t>(i);
                for (size_t j = 0; j < inputs[i].size(); j++) {
                    input_ptr[j] = static_cast<int8_t>(inputs[i][j]);
                }
            } else {
                LOGE("Unsupported input tensor type: %d", input_tensor->type);
            }
        }
    });
    
    // Run inference
    timing_stats_.inference_ms = TimeFunction([&]() {
        if (interpreter_->Invoke() != kTfLiteOk) {
            LOGE("Failed to invoke interpreter");
        }
    });
    
    LOGI("Inference completed in %.2f ms", timing_stats_.inference_ms);
    
    // Copy output data
    timing_stats_.output_copy_ms = TimeFunction([&]() {
        int output_count = interpreter_->outputs().size();
        outputs.resize(output_count);
        
        for (int i = 0; i < output_count; i++) {
            TfLiteTensor* output_tensor = interpreter_->output_tensor(i);
            if (!output_tensor) {
                LOGE("Failed to get output tensor %d", i);
                return;
            }
            
            // Calculate output size
            size_t output_size = 1;
            for (int j = 0; j < output_tensor->dims->size; j++) {
                output_size *= output_tensor->dims->data[j];
            }
            
            outputs[i].resize(output_size);
            
            // Copy output data based on type
            if (output_tensor->type == kTfLiteFloat32) {
                const float* output_ptr = interpreter_->typed_output_tensor<float>(i);
                std::memcpy(outputs[i].data(), output_ptr, output_size * sizeof(float));
            } else if (output_tensor->type == kTfLiteUInt8) {
                const uint8_t* output_ptr = interpreter_->typed_output_tensor<uint8_t>(i);
                for (size_t j = 0; j < output_size; j++) {
                    outputs[i][j] = static_cast<float>(output_ptr[j]);
                }
            } else if (output_tensor->type == kTfLiteInt8) {
                const int8_t* output_ptr = interpreter_->typed_output_tensor<int8_t>(i);
                for (size_t j = 0; j < output_size; j++) {
                    outputs[i][j] = static_cast<float>(output_ptr[j]);
                }
            } else {
                LOGE("Unsupported output tensor type: %d", output_tensor->type);
            }
        }
    });
    
    auto total_end = std::chrono::high_resolution_clock::now();
    timing_stats_.total_ms = std::chrono::duration<double, std::milli>(
        total_end - total_start).count();
    
    return true;
}

OpPlacementStats TFLiteRunner::GetOpPlacementStats() const {
    OpPlacementStats stats;
    
    if (!interpreter_) {
        return stats;
    }
    
    // Get execution plan
    const auto& execution_plan = interpreter_->execution_plan();
    stats.total_ops = execution_plan.size();
    
    // Count GPU vs CPU ops
    for (int node_index : execution_plan) {
        const TfLiteNode& node = interpreter_->node_and_registration(node_index)->first;
        const TfLiteRegistration& registration = 
            interpreter_->node_and_registration(node_index)->second;
        
        // Check if operation is delegated (GPU) or not (CPU)
        if (node.delegate) {
            stats.gpu_ops++;
        } else {
            stats.cpu_ops++;
            
            // Get operation name
            const char* op_name = registration.custom_name 
                ? registration.custom_name 
                : tflite::EnumNameBuiltinOperator(
                    static_cast<tflite::BuiltinOperator>(registration.builtin_code));
            
            if (op_name) {
                stats.cpu_op_names.push_back(op_name);
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
    
    // Operation placement stats
    OpPlacementStats op_stats = GetOpPlacementStats();
    LOGI("=== Operation Placement ===");
    LOGI("Total Operations:   %d", op_stats.total_ops);
    LOGI("GPU Operations:     %d (%.1f%%)", 
         op_stats.gpu_ops, 
         op_stats.total_ops > 0 ? (100.0 * op_stats.gpu_ops / op_stats.total_ops) : 0.0);
    LOGI("CPU Operations:     %d (%.1f%%)", 
         op_stats.cpu_ops,
         op_stats.total_ops > 0 ? (100.0 * op_stats.cpu_ops / op_stats.total_ops) : 0.0);
    
    if (!op_stats.cpu_op_names.empty()) {
        LOGI("CPU Operations:");
        for (const auto& op_name : op_stats.cpu_op_names) {
            LOGI("  - %s", op_name.c_str());
        }
    }
}

std::vector<int> TFLiteRunner::GetInputShape(int index) const {
    std::vector<int> shape;
    if (interpreter_ && index < interpreter_->inputs().size()) {
        TfLiteTensor* tensor = interpreter_->input_tensor(index);
        for (int i = 0; i < tensor->dims->size; i++) {
            shape.push_back(tensor->dims->data[i]);
        }
    }
    return shape;
}

std::vector<int> TFLiteRunner::GetOutputShape(int index) const {
    std::vector<int> shape;
    if (interpreter_ && index < interpreter_->outputs().size()) {
        TfLiteTensor* tensor = interpreter_->output_tensor(index);
        for (int i = 0; i < tensor->dims->size; i++) {
            shape.push_back(tensor->dims->data[i]);
        }
    }
    return shape;
}

int TFLiteRunner::GetInputTensorCount() const {
    return interpreter_ ? interpreter_->inputs().size() : 0;
}

int TFLiteRunner::GetOutputTensorCount() const {
    return interpreter_ ? interpreter_->outputs().size() : 0;
}

} // namespace tflite_runner
