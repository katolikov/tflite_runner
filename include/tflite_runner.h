#ifndef TFLITE_RUNNER_H
#define TFLITE_RUNNER_H

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

namespace tflite_runner {

// Timing statistics
struct TimingStats {
    double model_load_ms = 0.0;
    double delegate_init_ms = 0.0;
    double tensor_allocation_ms = 0.0;
    double input_copy_ms = 0.0;
    double inference_ms = 0.0;
    double output_copy_ms = 0.0;
    double total_ms = 0.0;
};

// Operation placement statistics
struct OpPlacementStats {
    int total_ops = 0;
    int gpu_ops = 0;
    int cpu_ops = 0;
    std::vector<std::string> cpu_op_names;
};

class TFLiteRunner {
public:
    TFLiteRunner();
    ~TFLiteRunner();

    // Load model from file
    bool LoadModel(const std::string& model_path);

    // Initialize GPU delegate for Exynos devices
    bool InitGPUDelegate();

    // Run inference with single input/output
    bool RunInference(const std::vector<float>& input_data, 
                     std::vector<float>& output_data);

    // Run inference with multiple inputs/outputs
    bool RunInferenceMulti(const std::vector<std::vector<float>>& inputs,
                          std::vector<std::vector<float>>& outputs);

    // Get input/output tensor shapes
    std::vector<int> GetInputShape(int index = 0) const;
    std::vector<int> GetOutputShape(int index = 0) const;

    // Get tensor information
    int GetInputTensorCount() const;
    int GetOutputTensorCount() const;

    // Get timing statistics
    const TimingStats& GetTimingStats() const { return timing_stats_; }

    // Get operation placement statistics
    OpPlacementStats GetOpPlacementStats() const;

    // Print detailed profiling info
    void PrintProfilingInfo() const;

    // Enable/disable profiling
    void SetProfilingEnabled(bool enabled) { profiling_enabled_ = enabled; }

private:
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    TfLiteDelegate* gpu_delegate_ = nullptr;
    
    TimingStats timing_stats_;
    bool profiling_enabled_ = true;
    
    bool AllocateTensors();
    
    // Helper for timing
    template<typename Func>
    double TimeFunction(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

} // namespace tflite_runner

#endif // TFLITE_RUNNER_H
