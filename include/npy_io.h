#ifndef NPY_IO_H
#define NPY_IO_H

#include <string>
#include <vector>

namespace tflite_runner {

class NPYReader {
public:
    static bool LoadNPY(const std::string& filename, 
                       std::vector<float>& data,
                       std::vector<size_t>& shape);
    
    static bool LoadNPYInt8(const std::string& filename,
                           std::vector<int8_t>& data,
                           std::vector<size_t>& shape);
    
    static bool LoadNPYUInt8(const std::string& filename,
                            std::vector<uint8_t>& data,
                            std::vector<size_t>& shape);
};

class NPYWriter {
public:
    static bool SaveNPY(const std::string& filename,
                       const std::vector<float>& data,
                       const std::vector<size_t>& shape);
    
    static bool SaveNPYInt8(const std::string& filename,
                           const std::vector<int8_t>& data,
                           const std::vector<size_t>& shape);
    
    static bool SaveNPYUInt8(const std::string& filename,
                            const std::vector<uint8_t>& data,
                            const std::vector<size_t>& shape);
};

} // namespace tflite_runner

#endif // NPY_IO_H
