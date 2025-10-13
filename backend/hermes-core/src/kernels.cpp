#include <stdexcept>
#include <kernels.h>
#include <Tensor.h>

void add(const hermes::Tensor& a, const hermes::Tensor& b, hermes::Tensor& out) {

}

void subtract(const hermes::Tensor& a, const hermes::Tensor& b, hermes::Tensor& out) {

}

void scale(const hermes::Tensor& input, float factor, hermes::Tensor& out) {
    if (input.shape() != out.shape()) {
        throw std::invalid_argument("Input dimensions do not match output dimensions.");
    }

    for (int i = 0; i < input.dim(); i++) {
        out.data()[i] *= factor;
    }
}

void multiply(const hermes::Tensor& a, const hermes::Tensor& b, hermes::Tensor& out) {

}

void multiply_naive(const hermes::Tensor& a, const hermes::Tensor& b, hermes::Tensor& out) {

}