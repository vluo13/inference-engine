#include <gtest/gtest.h>
#include "hermes/Tensor.h"
#include "hermes/ops.h"
#include "test_utils.h"

TEST(Ops, SimpleAdd) {
  hermes::Tensor a = hermes::Tensor::ones({1, 3});
  hermes::Tensor b({1, 3});
  b.at({0, 0}) = 2;
  b.at({0, 1}) = 3;
  b.at({0, 2}) = 4;
  hermes::Tensor c({1, 3});
  c.at({0, 0}) = 3;
  c.at({0, 1}) = 4;
  c.at({0, 2}) = 5;
  expectTensorEq(c, hermes::add(a, b));
}

TEST(Ops, ScalarAdd) {
  hermes::Tensor a = hermes::Tensor::ones({1, 3});
  float b = 2;
  hermes::Tensor c({1, 3});
  c.at({0, 0}) = 3;
  c.at({0, 1}) = 3;
  c.at({0, 2}) = 3;
  expectTensorEq(c, hermes::add(a, b));
}

TEST(Ops, AddSameShape) {
    hermes::Tensor a = hermes::Tensor::ones({2, 3});
    hermes::Tensor b = hermes::Tensor::ones({2, 3});
    hermes::Tensor expected({2, 3});
    for (size_t i = 0; i < 6; i++) expected.dataPtr()[i] = 2.0f;
    expectTensorEq(expected, hermes::add(a, b));
}

TEST(Ops, AddBroadcastRowVector) {
    // (1, 3) + (2, 3) → (2, 3)
    hermes::Tensor a({1, 3});
    a.dataPtr()[0] = 10; a.dataPtr()[1] = 20; a.dataPtr()[2] = 30;
    hermes::Tensor b({2, 3});
    for (size_t i = 0; i < 6; i++) b.dataPtr()[i] = i + 1;
    hermes::Tensor expected({2, 3});
    expected.dataPtr()[0] = 11; expected.dataPtr()[1] = 22; expected.dataPtr()[2] = 33;
    expected.dataPtr()[3] = 14; expected.dataPtr()[4] = 25; expected.dataPtr()[5] = 36;
    expectTensorEq(expected, hermes::add(a, b));
}

TEST(Ops, AddBroadcastDifferentRanks) {
    // (3,) + (2, 3) → (2, 3)
    hermes::Tensor a({3});
    a.dataPtr()[0] = 1; a.dataPtr()[1] = 2; a.dataPtr()[2] = 3;
    hermes::Tensor b({2, 3});
    for (size_t i = 0; i < 6; i++) b.dataPtr()[i] = 10;
    // expected: [[11, 12, 13], [11, 12, 13]]
    hermes::Tensor expected({2, 3});
    expected.dataPtr()[0] = 11; expected.dataPtr()[1] = 12; expected.dataPtr()[2] = 13;
    expected.dataPtr()[3] = 11; expected.dataPtr()[4] = 12; expected.dataPtr()[5] = 13;
    expectTensorEq(expected, hermes::add(a, b));
}

TEST(Ops, AddBroadcastBothDirections) {
    // (3, 1) + (1, 4) → (3, 4)
    hermes::Tensor a({3, 1});
    a.dataPtr()[0] = 1; a.dataPtr()[1] = 2; a.dataPtr()[2] = 3;
    hermes::Tensor b({1, 4});
    b.dataPtr()[0] = 10; b.dataPtr()[1] = 20; b.dataPtr()[2] = 30; b.dataPtr()[3] = 40;
    // expected:
    // [[11, 21, 31, 41],
    //  [12, 22, 32, 42],
    //  [13, 23, 33, 43]]
    hermes::Tensor expected({3, 4});
    expected.dataPtr()[0] = 11; expected.dataPtr()[1] = 21; expected.dataPtr()[2] = 31; expected.dataPtr()[3] = 41;
    expected.dataPtr()[4] = 12; expected.dataPtr()[5] = 22; expected.dataPtr()[6] = 32; expected.dataPtr()[7] = 42;
    expected.dataPtr()[8] = 13; expected.dataPtr()[9] = 23; expected.dataPtr()[10] = 33; expected.dataPtr()[11] = 43;
    expectTensorEq(expected, hermes::add(a, b));
}

TEST(Ops, AddIncompatibleShapes) {
    hermes::Tensor a = hermes::Tensor::ones({2, 3});
    hermes::Tensor b = hermes::Tensor::ones({4, 5});
    EXPECT_THROW(hermes::add(a, b), std::runtime_error);
}

TEST(Ops, SubSameShape) {
    hermes::Tensor a({2, 2});
    a.dataPtr()[0] = 5; a.dataPtr()[1] = 7; a.dataPtr()[2] = 9; a.dataPtr()[3] = 11;
    hermes::Tensor b({2, 2});
    b.dataPtr()[0] = 1; b.dataPtr()[1] = 2; b.dataPtr()[2] = 3; b.dataPtr()[3] = 4;
    hermes::Tensor expected({2, 2});
    expected.dataPtr()[0] = 4; expected.dataPtr()[1] = 5; expected.dataPtr()[2] = 6; expected.dataPtr()[3] = 7;
    expectTensorEq(expected, hermes::sub(a, b));
}

TEST(Ops, SubScalarReverse) {
    // 10 - tensor
    hermes::Tensor a({3});
    a.dataPtr()[0] = 1; a.dataPtr()[1] = 2; a.dataPtr()[2] = 3;
    hermes::Tensor expected({3});
    expected.dataPtr()[0] = 9; expected.dataPtr()[1] = 8; expected.dataPtr()[2] = 7;
    expectTensorEq(expected, hermes::sub(10.0f, a));
}

TEST(Ops, MulSameShape) {
    hermes::Tensor a({2, 2});
    a.dataPtr()[0] = 1; a.dataPtr()[1] = 2; a.dataPtr()[2] = 3; a.dataPtr()[3] = 4;
    hermes::Tensor b({2, 2});
    b.dataPtr()[0] = 5; b.dataPtr()[1] = 6; b.dataPtr()[2] = 7; b.dataPtr()[3] = 8;
    hermes::Tensor expected({2, 2});
    expected.dataPtr()[0] = 5; expected.dataPtr()[1] = 12; expected.dataPtr()[2] = 21; expected.dataPtr()[3] = 32;
    expectTensorEq(expected, hermes::mul(a, b));
}

TEST(Ops, MulScalar) {
    hermes::Tensor a = hermes::Tensor::ones({2, 2});
    hermes::Tensor expected({2, 2});
    for (size_t i = 0; i < 4; i++) expected.dataPtr()[i] = 3.0f;
    expectTensorEq(expected, hermes::mul(a, 3.0f));
}

TEST(Ops, DivSameShape) {
    hermes::Tensor a({2});
    a.dataPtr()[0] = 10; a.dataPtr()[1] = 20;
    hermes::Tensor b({2});
    b.dataPtr()[0] = 2; b.dataPtr()[1] = 4;
    hermes::Tensor expected({2});
    expected.dataPtr()[0] = 5; expected.dataPtr()[1] = 5;
    expectTensorEq(expected, hermes::div(a, b));
}

TEST(Ops, DivScalarReverse) {
    hermes::Tensor a({2});
    a.dataPtr()[0] = 2; a.dataPtr()[1] = 4;
    hermes::Tensor expected({2});
    expected.dataPtr()[0] = 6; expected.dataPtr()[1] = 3;
    expectTensorEq(expected, hermes::div(12.0f, a));
}