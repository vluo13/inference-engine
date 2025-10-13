#ifndef HERMES_KERNELS_H
#define HERMES_KERNELS_H

namespace hermes {

class Tensor;

// kernels
void add(const Tensor& a, const Tensor& b, Tensor& out);
void subtract(const Tensor& a, const Tensor& b, Tensor& out);
void scale(const Tensor& input, float factor, Tensor& out);
void multiply(const Tensor& a, const Tensor& b, Tensor& out);
void matmul_naive(const Tensor& a, const Tensor& b, Tensor& out);
void matmul(const Tensor& a, const Tensor& b, Tensor& out);

}

#endif 