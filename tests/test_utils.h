#ifndef HERMES_TEST_UTILS_H
#define HERMES_TEST_UTILS_H

#include <gtest/gtest.h>
#include "hermes/Tensor.h"

#include <vector>

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

#endif