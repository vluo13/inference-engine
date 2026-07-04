#include <gtest/gtest.h>
#include "hermes/Tensor.h"
#include "hermes/ops.h"

TEST(Tensor, ConstructionShape) {
  hermes::Tensor t({2, 3, 4});
  EXPECT_EQ(t.shape(), (std::vector<size_t>{2, 3, 4}));
  EXPECT_EQ(t.strides(), (std::vector<size_t>{12, 4, 1}));
  EXPECT_EQ(hermes::Tensor::numel(t.shape()), 24);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(t.dataPtr()) % 128, 0);
}

TEST(Tensor, Zeros) {
  hermes::Tensor t = hermes::Tensor::zeros({2, 2});
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ(t.dataPtr()[i], 0);
  }
}

TEST(Tensor, Ones) {
  hermes::Tensor t = hermes::Tensor::ones({2, 2});
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ(t.dataPtr()[i], 1);
  }
}

TEST(Tensor, Indexing) {
    hermes::Tensor t = hermes::Tensor::zeros({2, 3});
    t.at({0, 0}) = 1;
    t.at({0, 2}) = 3;
    t.at({1, 0}) = 4;
    t.at({1, 2}) = 6;
    // verify positions via dataPtr (linear order: row-major)
    EXPECT_EQ(t.dataPtr()[0], 1);  // [0,0]
    EXPECT_EQ(t.dataPtr()[2], 3);  // [0,2]
    EXPECT_EQ(t.dataPtr()[3], 4);  // [1,0]
    EXPECT_EQ(t.dataPtr()[5], 6);  // [1,2]
}

TEST(Tensor, numel) {
  EXPECT_EQ(hermes::Tensor::numel({2, 3, 4}), 24);
}

TEST(Tensor, computeStrides) {
  std::vector<size_t> strides = {12, 4, 1};
  EXPECT_EQ(hermes::Tensor::computeStrides({2, 3, 4}), strides);
}

TEST(Tensor, computeBroadcastShapeInvalid) {
  EXPECT_THROW(hermes::Tensor::computeBroadcastShape({3, 1}, {4, 1}), std::runtime_error);
}

TEST(Tensor, computeBroadcastShape) {
  EXPECT_EQ(hermes::Tensor::computeBroadcastShape({3, 1}, {1, 4}), (std::vector<size_t>{3, 4}));
}

TEST(Tensor, Transpose) {
  hermes::Tensor t({2, 3});
  for (size_t i = 0; i < 6; i++) t.dataPtr()[i] = i;
  // t is [[0, 1, 2], [3, 4, 5]]
  
  auto tt = t.transpose(0, 1);
  EXPECT_EQ(tt.shape(), (std::vector<size_t>{3, 2}));
  EXPECT_EQ(tt.dataPtr(), t.dataPtr());  // shared storage
  
  // values at swapped indices match
  for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++) {
          EXPECT_EQ(tt.at({j, i}), t.at({i, j}));
      }
  }
}

TEST(Tensor, ReshapeContiguous) {
  hermes::Tensor t({2, 6});
  for (size_t i = 0; i < 12; i++) t.dataPtr()[i] = i;
  
  auto r = t.reshape({3, 4});
  EXPECT_EQ(r.shape(), (std::vector<size_t>{3, 4}));
  EXPECT_EQ(r.dataPtr(), t.dataPtr());  // view, same storage
  
  // values preserved at corresponding linear positions
  for (size_t i = 0; i < 12; i++) {
      EXPECT_EQ(r.dataPtr()[i], t.dataPtr()[i]);
  }
}

TEST(Tensor, ReshapeNonContiguous) {
  hermes::Tensor t({2, 3});
  for (size_t i = 0; i < 6; i++) t.dataPtr()[i] = i;
  // t: [[0, 1, 2], [3, 4, 5]]
  
  auto tt = t.transpose(0, 1);
  // tt logical: [[0, 3], [1, 4], [2, 5]]
  
  auto r = tt.reshape({6});
  EXPECT_NE(r.dataPtr(), t.dataPtr());  // copy happened
  
  // logical order of tt is [0, 3, 1, 4, 2, 5]
  std::vector<float> expected = {0, 3, 1, 4, 2, 5};
  for (size_t i = 0; i < 6; i++) {
      EXPECT_EQ(r.dataPtr()[i], expected[i]);
  }
}

TEST(Tensor, SimpleSlice) {
  hermes::Tensor a({2, 3});
  for (size_t i = 0; i < 6; i++) a.dataPtr()[i] = i;
  hermes::Tensor b = a.slice(0, 0, 1);
  EXPECT_EQ(b.shape(), (std::vector<size_t>{1, 3}));
  EXPECT_EQ(b.at({0,0}), 0);
  EXPECT_EQ(b.at({0,1}), 1);
  EXPECT_EQ(b.at({0,2}), 2);
  EXPECT_THROW(b.at({1, 0}), std::runtime_error);
}

TEST(Tensor, MiddleSlice) {
  hermes::Tensor a({2, 3});
  for (size_t i = 0; i < 6; i++) a.dataPtr()[i] = i;
  hermes::Tensor b = a.slice(1, 1, 2);
  EXPECT_EQ(b.shape(), (std::vector<size_t>{2, 1}));
  EXPECT_EQ(b.at({0,0}), 1);
  EXPECT_EQ(b.at({1,0}), 4);
  EXPECT_THROW(b.at({0, 1}), std::runtime_error);
}

TEST(Tensor, InvlalidIndex) {
  hermes::Tensor a({2, 3});
  for (size_t i = 0; i < 6; i++) a.dataPtr()[i] = i;
  EXPECT_THROW(a.slice(10, 0, 0), std::runtime_error);
  EXPECT_THROW(a.slice(0, 0, 10), std::runtime_error);
  EXPECT_THROW(a.slice(0, 2, 1), std::runtime_error);
}

TEST(Tensor, SlicingSlice) {
  hermes::Tensor a({5, 2});
  // 0 1 
  // 2 3
  // 4 5
  // 6 7
  // 8 9
  for (size_t i = 0; i < 10; i++) a.dataPtr()[i] = i;
  hermes::Tensor b = a.slice(1, 0, 1);
  EXPECT_EQ(b.shape(), (std::vector<size_t>{5, 1}));
  for (size_t i = 0; i < 5; i++) EXPECT_EQ(b.at({i, 0}), i * 2);

  hermes::Tensor c = b.slice(0, 1, 4);
  EXPECT_EQ(c.shape(), (std::vector<size_t>{3, 1}));
  EXPECT_EQ(c.at({0, 0}), 2);
  EXPECT_EQ(c.at({1, 0}), 4);
  EXPECT_EQ(c.at({2, 0}), 6);
}

TEST(Tensor, ShallowCopy) {
  hermes::Tensor a({2, 3});
  a.at({0, 0}) = 42;
  
  hermes::Tensor b = a;
  EXPECT_EQ(b.dataPtr(), a.dataPtr());
  EXPECT_EQ(b.at({0, 0}), 42);
  
  b.at({1, 1}) = 99;
  EXPECT_EQ(a.at({1, 1}), 99);  // mutation visible through a
}