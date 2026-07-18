#include <string>
#include <vector>
#include <stdexcept>

#include <gtest/gtest.h>
#include "test_utils.h"
#include "hermes/Tensor.h"
#include "hermes/utils.h"
#include "hermes/model.h"

namespace {

// Load weights once — 500MB per test would be absurd.
hermes::GPT2& model() {
    static hermes::GPT2 m = hermes::GPT2::load(
        "/Users/admin/Desktop/inference-engine/weights/weights.bin");
    return m;
} 

hermes::Tensor toTensor(const std::vector<int>& tokens) {
    hermes::Tensor t{std::vector<size_t>{tokens.size()}};
    for (size_t i = 0; i < tokens.size(); ++i)
        t.dataPtr()[i] = static_cast<float>(tokens[i]);
    return t;
}

// Deterministic in-range token ids. Content doesn't affect control flow (no
// data-dependent branching), so only T matters for coverage.
hermes::Tensor makeSeq(size_t T) {
    std::vector<int> t(T);
    for (size_t i = 0; i < T; ++i)
        t[i] = static_cast<int>((i * 7919) % 50257);
    return toTensor(t);
}

}

TEST(GPT2Forward, Prompt8) {
    std::vector<int> tokens = {15496, 11, 314, 1101, 257, 3303, 2746, 11};
    hermes::Tensor expected = loadTensor(
        TENSOR_DIR"logits_prompt8.bin");
    expectTensorEq(expected, model().forward(toTensor(tokens)), 1e-3f);
}

TEST(GPT2Forward, SingleToken) {
    // T=1: degenerate causal mask (attends only to itself), first wpe row only.
    std::vector<int> tokens = {15496};
    hermes::Tensor expected = loadTensor(
        TENSOR_DIR"logits_single.bin");
    expectTensorEq(expected, model().forward(toTensor(tokens)), 1e-3f);
}

TEST(GPT2Forward, Long300) {
    // Large T×T mask, deep into wpe. Bugs invisible at T=8 surface here.
    hermes::Tensor expected = loadTensor(
        TENSOR_DIR"logits_long300.bin");
    expectTensorEq(expected, model().forward(makeSeq(300)), 1e-3f);
}

TEST(GPT2Forward, MaxBlockSize) {
    // T=1024 exactly: wpe has 1024 rows, so this is the off-by-one boundary.
    hermes::Tensor expected = loadTensor(
        TENSOR_DIR"logits_max1024.bin");
    expectTensorEq(expected, model().forward(makeSeq(1024)), 1e-3f);
}

TEST(GPT2Forward, ThrowsBeyondBlockSize) {
    // wpe only has block_size rows; longer sequences must be rejected, not
    // silently read out of bounds.
    EXPECT_THROW(model().forward(makeSeq(1025)), std::runtime_error);
}