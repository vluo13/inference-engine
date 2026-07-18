#ifndef HERMES_KERNELS_H
#define HERMES_KERNELS_H

#include "hermes/Tensor.h"

namespace hermes {

Tensor gelu(const Tensor& a);
Tensor layerNorm(const Tensor& a, const Tensor& weight, const Tensor& bias, float eps=1e-5f); 
Tensor softmax(const Tensor& a);
void gemm(const float* a, const float* b, float* out, size_t M, size_t K, size_t N);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor bmm(const Tensor& a, const Tensor& b);

}

#endif