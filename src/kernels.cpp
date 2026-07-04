#include "hermes/Tensor.h"

#include <cmath>


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
}