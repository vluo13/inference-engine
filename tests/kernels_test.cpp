#include <gtest/gtest.h>
#include "hermes/Tensor.h"
#include "hermes/utils.h"
#include "hermes/kernels.h"

void expectTensorEq(const hermes::Tensor& expected, const hermes::Tensor& actual, float tol = 1e-5f) {
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

TEST(Ops, Gelu) {
  hermes::Tensor input = hermes::loadTensor("/Users/vincentluo/projects/inference-engine/tests/tensors/tensor3d.bin");
  hermes::Tensor expected = hermes::loadTensor("/Users/vincentluo/projects/inference-engine/tests/tensors/tensor3d_gelu.bin");
  expectTensorEq(expected, hermes::gelu(input), 1e-5f);
}

TEST(Ops, Zero) {
  hermes::Tensor input = hermes::loadTensor("/Users/vincentluo/projects/inference-engine/tests/tensors/gelu_zero_in.bin");
  hermes::Tensor expected = hermes::loadTensor("/Users/vincentluo/projects/inference-engine/tests/tensors/gelu_zero_out.bin");
  expectTensorEq(expected, hermes::gelu(input), 1e-5f);
}

TEST(Ops, Neg) {
  hermes::Tensor input = hermes::loadTensor("/Users/vincentluo/projects/inference-engine/tests/tensors/gelu_neg_in.bin");
  hermes::Tensor expected = hermes::loadTensor("/Users/vincentluo/projects/inference-engine/tests/tensors/gelu_neg_out.bin");
  expectTensorEq(expected, hermes::gelu(input), 1e-5f);
}

TEST(Ops, Large) {
  hermes::Tensor input = hermes::loadTensor("/Users/vincentluo/projects/inference-engine/tests/tensors/gelu_large_in.bin");
  hermes::Tensor expected = hermes::loadTensor("/Users/vincentluo/projects/inference-engine/tests/tensors/gelu_large_out.bin");
  expectTensorEq(expected, hermes::gelu(input), 1e-5f);
}