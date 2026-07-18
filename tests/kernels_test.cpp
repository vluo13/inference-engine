#include <gtest/gtest.h>
#include "test_utils.h"
#include "hermes/Tensor.h"
#include "hermes/kernels.h"

#include <cmath>
#include <limits>
#include <vector>
#include <stdexcept>

namespace {
hermes::Tensor makeTensor(std::vector<size_t> shape, std::vector<float> values) {
    hermes::Tensor t{std::move(shape)};
    EXPECT_EQ(hermes::Tensor::numel(t.shape()), values.size());
    for (size_t i = 0; i < values.size(); ++i) t.dataPtr()[i] = values[i];
    return t;
}
} 

TEST(Gelu, Zero) {
	hermes::Tensor input = loadTensor(TENSOR_DIR"gelu_zero_in.bin");
	hermes::Tensor expected = loadTensor(TENSOR_DIR"gelu_zero_out.bin");
	expectTensorEq(expected, hermes::gelu(input), 1e-5f);
}

TEST(Gelu, Neg) {
	hermes::Tensor input = loadTensor(TENSOR_DIR"gelu_neg_in.bin");
	hermes::Tensor expected = loadTensor(TENSOR_DIR"gelu_neg_out.bin");
	expectTensorEq(expected, gelu(input), 1e-5f);
}

TEST(Gelu, Large) {
	hermes::Tensor input = loadTensor(TENSOR_DIR"gelu_large_in.bin");
	hermes::Tensor expected = loadTensor(TENSOR_DIR"gelu_large_out.bin");
	expectTensorEq(expected, gelu(input), 1e-5f);
}

TEST(GeLURand, MatchesPyTorch) {
    for (int i = 0; i < 4; ++i) {
        SCOPED_TRACE("gelu_rand case " + std::to_string(i));
        std::string base =  TENSOR_DIR"gelu_rand_" + std::to_string(i);
        hermes::Tensor in  = loadTensor(base + "_in.bin");
        hermes::Tensor ref = loadTensor(base + "_out.bin");
        expectTensorEq(ref, gelu(in), 1e-4f);
    }
}

/**
 * LayerNorm
 */
TEST(LayerNorm, UsesBiasedVariance) {
    // Row [1,2,3,4]: mean 2.5, sum of squared dev = 5.
    //   biased var  = 5/4 = 1.25   (correct)
    //   unbiased var= 5/3 = 1.6667 (wrong)
    // With weight=1, bias=0 the normalized output distinguishes the two.
    hermes::Tensor x = makeTensor({1, 4}, {1, 2, 3, 4});
    hermes::Tensor weight = hermes::Tensor::ones({4});
    hermes::Tensor bias = hermes::Tensor::zeros({4});

    hermes::Tensor got = hermes::layerNorm(x, weight, bias, 1e-5f);

    double eps = 1e-5;
    double biasedStd = std::sqrt(1.25 + eps);
    double unbiasedStd = std::sqrt(5.0 / 3.0 + eps);
    // First element: (1 - 2.5)/std.
    EXPECT_NEAR(got.dataPtr()[0], -1.5 / biasedStd, 1e-5);
    // Confirm it is NOT the unbiased result.
    EXPECT_GT(std::abs(got.dataPtr()[0] - (-1.5 / unbiasedStd)), 1e-3);
}

TEST(LayerNorm, NormalizesToZeroMeanUnitVarianceProperty) {
    // With weight=1, bias=0 each row should have ~0 mean and ~1 (biased) var.
    hermes::Tensor x = makeTensor({2, 5},
        {10, 12, 9, 11, 13, -3, -1, -2, 0, 6});
    hermes::Tensor weight = hermes::Tensor::ones({5});
    hermes::Tensor bias = hermes::Tensor::zeros({5});
    hermes::Tensor out = hermes::layerNorm(x, weight, bias, 1e-5f);

    for (size_t r = 0; r < 2; ++r) {
        double mean = 0;
        for (size_t j = 0; j < 5; ++j) mean += out.at({r, j});
        mean /= 5;
        EXPECT_NEAR(mean, 0.0, 1e-4) << "row " << r;

        double var = 0;
        for (size_t j = 0; j < 5; ++j) {
            double d = out.at({r, j}) - mean;
            var += d * d;
        }
        var /= 5;
        EXPECT_NEAR(var, 1.0, 1e-3) << "row " << r;
    }
}

TEST(LayerNorm, HandlesTransposedView) {
    // base (2,4) -> transpose -> view (4,2). View row r = {base[0,r], base[1,r]}.
    // For a 2-element row {a,b}: mean=(a+b)/2, biased var=((a-mean)^2+(b-mean)^2)/2.
    // With weight=1 bias=0, normalized pair is {-1,+1}/sqrt(1+eps/var)... simpler:
    // the two outputs are exact negatives, and (a-mean)/sqrt(var+eps).
    hermes::Tensor base = makeTensor({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    hermes::Tensor view = base.transpose(0, 1);  // (4,2), non-contiguous
    ASSERT_FALSE(view.isContiguous());
    ASSERT_EQ(view.shape(), (std::vector<size_t>{4, 2}));

    hermes::Tensor weight = hermes::Tensor::ones({2});
    hermes::Tensor bias = hermes::Tensor::zeros({2});
    hermes::Tensor got = hermes::layerNorm(view, weight, bias, 1e-5f);

    double eps = 1e-5;
    for (size_t r = 0; r < 4; ++r) {
        double a = base.at({0, r}), b = base.at({1, r});  // the view's row r
        double mean = (a + b) / 2.0;
        double var = ((a - mean) * (a - mean) + (b - mean) * (b - mean)) / 2.0;
        double rstd = std::sqrt(var + eps);
        EXPECT_NEAR(got.at({r, 0}), (a - mean) / rstd, 1e-5) << "row " << r;
        EXPECT_NEAR(got.at({r, 1}), (b - mean) / rstd, 1e-5) << "row " << r;
    }
}

TEST(LayerNormRand, MatchesPyTorch) {
    for (int i = 0; i < 4; ++i) {
        SCOPED_TRACE("layernorm_rand case " + std::to_string(i));
        std::string base = TENSOR_DIR"layernorm_rand_" + std::to_string(i);
        hermes::Tensor in  = loadTensor(base + "_in.bin");
        hermes::Tensor w   = loadTensor(base + "_w.bin");
        hermes::Tensor b   = loadTensor(base + "_b.bin");
        hermes::Tensor ref = loadTensor(base + "_out.bin");
        expectTensorEq(ref, hermes::layerNorm(in, w, b, 1e-5f), 1e-4f);
    }
}

/**
 * Softmax
 */
TEST(Softmax, BasicRow) {
    hermes::Tensor in = makeTensor({3}, {1, 2, 3});
    hermes::Tensor out = hermes::softmax(in);

    double e0 = std::exp(1.0 - 3.0), e1 = std::exp(2.0 - 3.0), e2 = std::exp(0.0);
    double s = e0 + e1 + e2;
    EXPECT_NEAR(out.dataPtr()[0], e0 / s, 1e-6);
    EXPECT_NEAR(out.dataPtr()[1], e1 / s, 1e-6);
    EXPECT_NEAR(out.dataPtr()[2], e2 / s, 1e-6);
    EXPECT_NEAR(out.dataPtr()[0] + out.dataPtr()[1] + out.dataPtr()[2], 1.0f, 1e-6);
}

TEST(Softmax, LargeValuesNoNaNAndSumsToOne) {
    // Without subtract-max these overflow to inf/NaN.
    hermes::Tensor in = makeTensor({3}, {100.0f, 101.0f, 102.0f});
    hermes::Tensor out = hermes::softmax(in);

    float sum = 0;
    for (size_t i = 0; i < 3; ++i) {
		EXPECT_TRUE(std::isfinite(out.dataPtr()[i]));
		sum += out.dataPtr()[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5);

    // Translation invariance: softmax([100,101,102]) == softmax([0,1,2]).
    hermes::Tensor shifted = hermes::softmax(makeTensor({3}, {0.0f, 1.0f, 2.0f}));
    for (size_t i = 0; i < 3; ++i)
    	EXPECT_NEAR(out.dataPtr()[i], shifted.dataPtr()[i], 1e-6);
}

TEST(Softmax, NegInfBecomesZero) {
    // Attention-mask case: -inf positions must come out exactly 0, no NaN.
    float ninf = -std::numeric_limits<float>::infinity();
    hermes::Tensor in = makeTensor({3}, {1.0f, ninf, 2.0f});
    hermes::Tensor out = hermes::softmax(in);

    EXPECT_EQ(out.dataPtr()[1], 0.0f);
    EXPECT_TRUE(std::isfinite(out.dataPtr()[0]));
    EXPECT_TRUE(std::isfinite(out.dataPtr()[2]));
    EXPECT_NEAR(out.dataPtr()[0] + out.dataPtr()[1] + out.dataPtr()[2], 1.0f, 1e-6);

    double e0 = std::exp(1.0 - 2.0), e2 = std::exp(0.0);
    double s = e0 + e2;
    EXPECT_NEAR(out.dataPtr()[0], e0 / s, 1e-6);
    EXPECT_NEAR(out.dataPtr()[2], e2 / s, 1e-6);
}

TEST(Softmax, RowsSumToOneProperty) {
    hermes::Tensor in = makeTensor({2, 2, 4},
        {3, 1, 4, 1, 5, 9, 2, 6, -2, -1, 0, 1, 10, 10, 10, 10});
    hermes::Tensor out = hermes::softmax(in);
    for (size_t a = 0; a < 2; ++a)
        for (size_t b = 0; b < 2; ++b) {
            float s = 0;
            for (size_t c = 0; c < 4; ++c) s += out.at({a, b, c});
            EXPECT_NEAR(s, 1.0f, 1e-6);
        }
}

// Softmax must handle non-contiguous input: in the model it receives views.
// A transposed tensor shares the row-major buffer but presents swapped strides,
// so a kernel walking dataPtr() linearly reads the wrong elements unless it
// normalizes on entry.
TEST(Softmax, HandlesTransposedView) {
    hermes::Tensor base = makeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
    hermes::Tensor view = base.transpose(0, 1);  // (3,2), non-contiguous
    ASSERT_FALSE(view.isContiguous());
    ASSERT_EQ(view.shape(), (std::vector<size_t>{3, 2}));

    hermes::Tensor out = hermes::softmax(view);
    ASSERT_EQ(out.shape(), (std::vector<size_t>{3, 2}));

    // View row r = {base[0,r], base[1,r]} = {r+1, r+4} = {x, x+3}.
    // softmax over {x, x+3} is x-independent -> every row identical.
    double e0 = std::exp(-3.0), e1 = std::exp(0.0), s = e0 + e1;
    for (size_t r = 0; r < 3; ++r) {
        EXPECT_NEAR(out.at({r, 0}), e0 / s, 1e-6) << "row " << r;
        EXPECT_NEAR(out.at({r, 1}), e1 / s, 1e-6) << "row " << r;
    }
}

TEST(SoftmaxRand, MatchesPyTorch) {
    for (int i = 0; i < 5; ++i) {
        SCOPED_TRACE("softmax_rand case " + std::to_string(i));
        std::string base = TENSOR_DIR"softmax_rand_" + std::to_string(i);
        hermes::Tensor in  = loadTensor(base + "_in.bin");
        hermes::Tensor ref = loadTensor(base + "_out.bin");
        expectTensorEq(ref, hermes::softmax(in), 1e-5f);
    }
}

/**
 * Matmul tests
 */
TEST(Matmul, Square2x2) {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]] -> [[19,22],[43,50]]
    hermes::Tensor a = makeTensor({2, 2}, {1, 2, 3, 4});
    hermes::Tensor b = makeTensor({2, 2}, {5, 6, 7, 8});
    hermes::Tensor out = hermes::Tensor::zeros({2, 2});
    hermes::gemm(a.dataPtr(), b.dataPtr(), out.dataPtr(), 2, 2, 2);
    expectTensorEq(makeTensor({2, 2}, {19, 22, 43, 50}), out, 1e-5f);
}

TEST(Matmul, NonSquare2x3times3x4) {
    hermes::Tensor a = makeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
    hermes::Tensor b = makeTensor({3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    hermes::Tensor out = hermes::Tensor::zeros({2, 4});
    hermes::gemm(a.dataPtr(), b.dataPtr(), out.dataPtr(), 2, 3, 4);
    // Hand-computed A @ B.
    expectTensorEq(makeTensor({2, 4}, {38, 44, 50, 56, 83, 98, 113, 128}), out, 1e-5f);
}

// matmul (2D) must accept transposed operands: attention does Q @ K^T where K^T
// is a non-contiguous view. The wrapper must make operands contiguous before
// calling gemm (which walks float* linearly).
TEST(Matmul, HandlesBothOperandsTransposed) {
    // A: (3,2) -> transpose -> (2,3) = [[1,3,5],[2,4,6]]
    // B: (4,3) -> transpose -> (3,4) = [[1,4,7,10],[2,5,8,11],[3,6,9,12]]
    // A@B (2,4), hand-computed:
    //   row0 = [1,3,5]: [1*1+3*2+5*3, 1*4+3*5+5*6, 1*7+3*8+5*9, 1*10+3*11+5*12]
    //        = [22, 49, 76, 103]
    //   row1 = [2,4,6]: [2+8+18, 8+20+36, 14+32+54, 20+44+72]
    //        = [28, 64, 100, 136]
    hermes::Tensor abase = makeTensor({3, 2}, {1, 2, 3, 4, 5, 6});
    hermes::Tensor a = abase.transpose(0, 1);
    hermes::Tensor bbase = makeTensor({4, 3},
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    hermes::Tensor b = bbase.transpose(0, 1);
    ASSERT_FALSE(a.isContiguous());
    ASSERT_FALSE(b.isContiguous());

    ASSERT_EQ(a.shape(), (std::vector<size_t>{2,3}));
    ASSERT_EQ(b.shape(), (std::vector<size_t>{3,4}));

    hermes::Tensor got = hermes::matmul(a, b);
    expectTensorEq(makeTensor({2, 4}, {22, 49, 76, 103, 28, 64, 100, 136}),
                   got, 1e-5f);
}

TEST(MatmulRand, MatchesPyTorch) {
    for (int i = 0; i < 5; ++i) {
        SCOPED_TRACE("matmul_rand case " + std::to_string(i));
        std::string base = TENSOR_DIR"matmul_rand_" + std::to_string(i);
        hermes::Tensor a   = loadTensor(base + "_a.bin");
        hermes::Tensor b   = loadTensor(base + "_b.bin");
        hermes::Tensor ref = loadTensor(base + "_out.bin");
        expectTensorEq(ref, hermes::matmul(a, b), 1e-3f);
    }
}

/**
 * Batched matmul
 */
TEST(Bmm, Batched2x2x3times2x3x2) {
    // Batch 0: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
    // Batch 1: [[7,8,9],[10,11,12]] @ [[2,0],[1,3],[4,5]] = [[58,69],[79,93]]
    hermes::Tensor a = makeTensor({2, 2, 3},
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    hermes::Tensor b = makeTensor({2, 3, 2},
        {1, 2, 3, 4, 5, 6, 2, 0, 1, 3, 4, 5});
    hermes::Tensor expected = makeTensor({2, 2, 2},
        {22, 28, 49, 64, 58, 69, 79, 93});
    expectTensorEq(expected, hermes::bmm(a, b), 1e-5f);
}

TEST(Bmm, ThrowsOnRankMismatch) {
    hermes::Tensor a = makeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
    hermes::Tensor b = makeTensor({2, 3, 4}, std::vector<float>(24, 1.0f));
    EXPECT_THROW(hermes::bmm(a, b), std::runtime_error);
}

TEST(Bmm, ThrowsOnLeadingDimMismatch) {
    hermes::Tensor a = makeTensor({2, 3, 4}, std::vector<float>(24, 1.0f));
    hermes::Tensor b = makeTensor({5, 4, 2}, std::vector<float>(40, 1.0f));
    EXPECT_THROW(hermes::bmm(a, b), std::runtime_error);
}

TEST(Bmm, ThrowsOnInnerDimMismatch) {
    hermes::Tensor a = makeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
    hermes::Tensor b = makeTensor({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    EXPECT_THROW(hermes::bmm(a, b), std::runtime_error);
}

TEST(Bmm, HandlesTransposedOperand) {
    // Batch 0: A=[[1,2],[3,4]], Kbase=[[1,2],[3,4]] -> K^T=[[1,3],[2,4]]
    //   A@K^T = [[1+6,2+8],[3+12,6+16]] = [[7,10],[15,22]]
    // Batch 1: A=[[5,6],[7,8]], Kbase=[[5,6],[7,8]] -> K^T=[[5,7],[6,8]]
    //   A@K^T = [[25+36,35+48],[35+48,49+64]] = [[61,83],[83,113]]
    hermes::Tensor a = makeTensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    hermes::Tensor kbase = makeTensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    hermes::Tensor k = kbase.transpose(1, 2);  // (2,2,2), non-contiguous last two
    ASSERT_FALSE(k.isContiguous());

    hermes::Tensor got = hermes::bmm(a, k);
    expectTensorEq(makeTensor({2, 2, 2}, {5, 11, 11, 25, 61, 83, 83, 113}),
                   got, 1e-5f);
}

TEST(BmmRand, MatchesPyTorch) {
    for (int i = 0; i < 3; ++i) {
        SCOPED_TRACE("bmm_rand case " + std::to_string(i));
        std::string base = TENSOR_DIR"bmm_rand_" + std::to_string(i);
        hermes::Tensor a   = loadTensor(base + "_a.bin");
        hermes::Tensor b   = loadTensor(base + "_b.bin");
        hermes::Tensor ref = loadTensor(base + "_out.bin");
        expectTensorEq(ref, hermes::bmm(a, b), 1e-3f);
    } 
}