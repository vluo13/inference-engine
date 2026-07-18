#ifndef HERMES_TEST_UTILS_H
#define HERMES_TEST_UTILS_H

#include <vector>
#include <fstream>

#include <gtest/gtest.h>
#include "hermes/Tensor.h"

inline void expectTensorEq(const hermes::Tensor& expected, const hermes::Tensor& actual, float tol = 1e-5f) {
    ASSERT_EQ(expected.shape(), actual.shape());
    
    size_t ndim = expected.shape().size();
    size_t count = hermes::Tensor::numel(expected.shape());
    std::vector<size_t> indices(ndim);
    
    for (size_t i = 0; i < count; i++) {
        EXPECT_NEAR(expected.at(indices), actual.at(indices), tol);
        if (i + 1 == count) break;
        
        indices[ndim - 1]++;
        size_t cur = ndim - 1;
        while (indices[cur] == expected.shape()[cur]) {
            indices[cur] = 0;
            cur--;
            indices[cur]++;
        }
    }
}
 
/**
 * single tensor 
 * format is ndims, shape, data
 */
inline hermes::Tensor loadTensor(const std::string& path) {
    std::ifstream reader(path, std::ios_base::binary);
    if (!reader) throw std::runtime_error("cannot open " + path);
    int32_t ndims;
    reader.read(reinterpret_cast<char*>(&ndims), sizeof(int32_t));
    if (!reader) throw std::runtime_error("truncated read: " + path);
    std::vector<int32_t> shape(ndims);
    reader.read(reinterpret_cast<char*>(shape.data()), ndims * sizeof(int32_t));
    if (!reader) throw std::runtime_error("truncated read: " + path);
    std::vector<size_t> shapeSizeT(shape.begin(), shape.end());
    
    hermes::Tensor result{shapeSizeT};
    reader.read(reinterpret_cast<char*>(result.dataPtr()), hermes::Tensor::numel(shapeSizeT) * sizeof(float));
    if (!reader) throw std::runtime_error("truncated read: " + path);
    return result;
}

#endif