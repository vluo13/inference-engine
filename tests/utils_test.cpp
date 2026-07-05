#include <gtest/gtest.h>
#include "test_utils.h"
#include "hermes/utils.h"
#include "hermes/Tensor.h"

TEST(Utils, LoadSimple) {
    hermes::Tensor a({3});
    a.at({0}) = 1;
    a.at({1}) = 2;
    a.at({2}) = 3;
    expectTensorEq(hermes::loadTensor("/Users/vincentluo/projects/inference-engine/tests/data/tensors/tensor.bin"), a);
}

TEST(Utils, Load3D) {
    hermes::Tensor a({2, 2, 3});
    for (size_t i = 0; i < 12; i++) a.dataPtr()[i] = i + 1;
    expectTensorEq(hermes::loadTensor("/Users/vincentluo/projects/inference-engine/tests/data/tensors/tensor3d.bin"), a);
}