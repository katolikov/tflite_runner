#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <string>
#include <vector>

namespace tflite_runner {

class ImageUtils {
public:
    // Save float tensor as PNG (normalizes to 0-255 range)
    static bool SaveAsPNG(const std::string& filename,
                         const std::vector<float>& data,
                         int width, int height, int channels);
    
    // Save uint8 tensor as PNG directly
    static bool SaveAsPNGUInt8(const std::string& filename,
                              const std::vector<uint8_t>& data,
                              int width, int height, int channels);
    
    // Normalize float data to 0-255 range
    static std::vector<uint8_t> NormalizeToUInt8(const std::vector<float>& data);
};

} // namespace tflite_runner

#endif // IMAGE_UTILS_H
