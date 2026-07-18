#include "hermes/kernels.h"

#include <cmath>
#include <cassert>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "hermes/Tensor.h"

namespace hermes {
    
Tensor gelu(const Tensor& a) {
    Tensor result {a.shape()};
    size_t n = Tensor::numel(a.shape());
    constexpr float twoOverPi = 0.7978845608028654f;
    for (size_t i = 0; i < n; ++i) {
        float x = a.dataPtr()[i];
        result.dataPtr()[i] = 0.5 * x * (1.0 + std::tanh(twoOverPi * (x + 0.044715 * (x * x * x)) ));
    }
    return result;
}

/**
 * Apparetly can do this in two passes though
 * See:  https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
 */
Tensor layerNorm(const Tensor& a, const Tensor& weight, const Tensor& bias, float eps) {
    if (!a.isContiguous()) {
        return layerNorm(a.contiguous(), weight, bias, eps);
    }
    
    size_t lastDim = a.shape().back();
    assert(lastDim == weight.shape()[0]);
    assert(weight.shape()[0] == bias.shape()[0]);
    
    size_t rows = Tensor::numel(a.shape()) / lastDim;
    Tensor result(a.shape());
    std::memcpy(result.dataPtr(), a.dataPtr(), Tensor::numel(a.shape()) * sizeof(float));
    for (size_t i = 0; i < rows; ++i) {
        float sum = 0;
        for (size_t j = i * lastDim; j < (i + 1) * lastDim; ++j) {
            sum += result.dataPtr()[j];
        }

        float mean = sum / lastDim;
        for (size_t j = i * lastDim; j < (i + 1) * lastDim; ++j) {
            result.dataPtr()[j] -= mean;
        }
        
        float var = 0;
        for (size_t j = i * lastDim; j < (i + 1) * lastDim; ++j) {
            var += result.dataPtr()[j] * result.dataPtr()[j];
        }
        var /= lastDim;

        float rstd = std::sqrt(var + eps);
        for (size_t j = i * lastDim; j < (i + 1) * lastDim; ++j) {
            result.dataPtr()[j] /= rstd;
        }
        
        size_t currentOffset = i * lastDim;
        for (size_t j = i * lastDim; j < (i + 1) * lastDim; ++j) {
            result.dataPtr()[j] *= weight.dataPtr()[j - currentOffset];
            result.dataPtr()[j] += bias.dataPtr()[j - currentOffset];
        }
    }

    return result;
}

Tensor softmax(const Tensor& a) {
    if (!a.isContiguous()) {
        return softmax(a.contiguous());
    }
    
    size_t lastDim = a.shape().back();
    
    size_t rows = Tensor::numel(a.shape()) / lastDim;
    Tensor result(a.shape());
    std::memcpy(result.dataPtr(), a.dataPtr(), Tensor::numel(a.shape()) * sizeof(float));
    for (size_t i = 0; i < rows; ++i) {
        float maxVal = -std::numeric_limits<float>::infinity();
        for (size_t j = i * lastDim; j < (i + 1) * lastDim; ++j) {
            maxVal = std::max(maxVal, result.dataPtr()[j]);
        }

        float sum = 0;
        for (size_t j = i * lastDim; j < (i + 1) * lastDim; ++j) {
            result.dataPtr()[j] = std::expf(result.dataPtr()[j] - maxVal);
            sum += result.dataPtr()[j];
        }

        for (size_t j = i * lastDim; j < (i + 1) * lastDim; ++j) {
            result.dataPtr()[j] /= sum;
        }
    }

    return result;
}

/**
 * Matmul along the last two dims
 */
void gemm(const float* a, const float* b, float* out, size_t M, size_t K, size_t N){
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < K; ++k) {
                out[i*N + j] += a[i*K + k] * b[k*N + j];
            }
        }
    }
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw std::runtime_error("Tensors are not 2D");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("Tensors are incompatible for matmul");
    }

    std::vector<size_t> resultShape = a.shape();
    resultShape.back() = b.shape().back();
    Tensor result = Tensor::zeros(resultShape);
    Tensor ac = a.contiguous();
    Tensor bc = b.contiguous();
    gemm(ac.dataPtr(), bc.dataPtr(), result.dataPtr(), resultShape[0], ac.shape()[1], resultShape[1]);
    return result;
}

/**
 * Matmul along the last two dims
 */
Tensor bmm(const Tensor& a, const Tensor& b) {
    size_t ndims = a.shape().size();
    if (ndims < 2) {
        throw std::runtime_error("Tensors do not have enough dimensions");
    }
    if (ndims != b.shape().size()) {
        throw std::runtime_error("Number of dimensions is not equal");
    }
    for (size_t i = 0; i < ndims-2; ++i) {
        if (a.shape()[i] != b.shape()[i]) {
            throw std::runtime_error("Leading dimensions are not equal");
        }
    }
    if (a.shape()[ndims-1] != b.shape()[ndims-2]) {
        throw std::runtime_error("Tensors are incompatible for matmul");
    }

    std::vector<size_t> resultShape = a.shape();
    resultShape.back() = b.shape().back();
    Tensor result = Tensor::zeros(resultShape);
    Tensor ac = a.contiguous();
    Tensor bc = b.contiguous();

    size_t M = ac.shape()[ndims-2];
    size_t K = ac.shape().back();
    size_t N = bc.shape().back();
    size_t aElems = M * K;
    size_t bElems = K * N;
    size_t outElems = M * N;
    size_t matrices = Tensor::numel(resultShape) / outElems;
    for (size_t i = 0; i < matrices; ++i) {
        size_t aOffset = i * aElems;
        size_t bOffset = i * bElems;
        size_t resultOffset = i * outElems;
        gemm(ac.dataPtr() + aOffset, bc.dataPtr() + bOffset, result.dataPtr() + resultOffset, M, K, N);
    }
    
    return result;
}

}