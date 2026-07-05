#include "hermes/utils.h"
#include "hermes/Tensor.h"

#include <vector>
#include <fstream>
#include <iostream>

namespace hermes {
    void incrementOdometer(std::vector<size_t>& indices, const std::vector<size_t>& shape, size_t ndim) {
        ++indices[ndim - 1];
        size_t cur = ndim - 1;
        while (indices[cur] == shape[cur]) {
            indices[cur] = 0;
            cur--; 
            ++indices[cur];
        }
    }

    Tensor loadTensor(const std::string& path) {
        std::ifstream reader(path, std::ios_base::binary);
        if (!reader) throw std::runtime_error("cannot open " + path);
        int32_t ndims;
        reader.read(reinterpret_cast<char*>(&ndims), sizeof(int32_t));
        if (!reader) throw std::runtime_error("truncated read: " + path);
        std::vector<int32_t> shape(ndims);
        reader.read(reinterpret_cast<char*>(shape.data()), ndims * sizeof(int32_t));
        if (!reader) throw std::runtime_error("truncated read: " + path);
        std::vector<size_t> shapeSizeT(shape.begin(), shape.end());
        
        Tensor result{shapeSizeT};
        reader.read(reinterpret_cast<char*>(result.dataPtr()), hermes::Tensor::numel(shapeSizeT) * sizeof(float));
        if (!reader) throw std::runtime_error("truncated read: " + path);
        return result;
    }
}