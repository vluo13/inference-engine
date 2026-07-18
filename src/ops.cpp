#include "hermes/ops.h"

#include "hermes/Tensor.h"
#include "hermes/utils.h"

namespace hermes {

namespace {

Tensor binaryOp(const Tensor& a, const Tensor& b, const auto& op) {
    std::vector<size_t> shape = Tensor::computeBroadcastShape(a.shape(), b.shape());
    std::vector<size_t> astrides = Tensor::computeBroadcastStrides(a.shape(), shape, a.strides());
    std::vector<size_t> bstrides = Tensor::computeBroadcastStrides(b.shape(), shape, b.strides());

    Tensor result {shape};
    size_t ndim = shape.size();
    size_t count = Tensor::numel(shape);
    std::vector<size_t> indices(ndim);
    for (size_t i = 0; i < count; ++i) {
        size_t aidx = Tensor::linearIndex(indices, astrides);
        size_t bidx = Tensor::linearIndex(indices, bstrides);
        result.dataPtr()[i] = op(a.dataPtr()[aidx], b.dataPtr()[bidx]);
        if (i + 1 == count) break; // guard against underflow on final interation

        incrementOdometer(indices, shape, ndim);
    }

    return result;
}

Tensor scalarOp(const Tensor& a, float b, const auto& op) {
    Tensor result {a.shape()};
    size_t ndim = a.shape().size();
    size_t count = Tensor::numel(a.shape());
    std::vector<size_t> indices(ndim);
    for (size_t i = 0; i < count; ++i) {
        result.dataPtr()[i] = op(a.at(indices), b);
        if (i + 1 == count) break; // guard against underflow on final interation

        incrementOdometer(indices, a.shape(), ndim);
    }

    return result;
}

Tensor scalarOpReverse(float a, const Tensor& b, const auto& op) {
    Tensor result {b.shape()};
    size_t ndim = b.shape().size();
    size_t count = Tensor::numel(b.shape());
    std::vector<size_t> indices(ndim);
    for (size_t i = 0; i < count; ++i) {
        result.dataPtr()[i] = op(a, b.at(indices));
        if (i + 1 == count) break; // guard against underflow on final interation

        incrementOdometer(indices, b.shape(), ndim);
    }

    return result;
}

}

Tensor add(const Tensor& a, const Tensor& b) {
    return binaryOp(a, b, [](float x, float y){return x + y;});
}

Tensor add(const Tensor& a, float b) {
    return scalarOp(a, b, [](float x, float y){return x + y;});
}

Tensor add(float a, const Tensor& b) {
    return add(b, a);
}

Tensor sub(const Tensor& a, const Tensor& b) {
    return binaryOp(a, b, [](float x, float y){return x - y;});
}

Tensor sub(const Tensor& a, float b) {
    return scalarOp(a, b, [](float x, float y){return x - y;});
}

Tensor sub(float a, const Tensor& b) {
    return scalarOpReverse(a, b, [](float x, float y){return x - y;});
}

Tensor mul(const Tensor& a, const Tensor& b) {
    return binaryOp(a, b, [](float x, float y){return x * y;});
}

Tensor mul(const Tensor& a, float b) {
    return scalarOp(a, b, [](float x, float y){return x * y;});
}

Tensor mul(float a, const Tensor& b) {
    return mul(b, a);
}

Tensor div(const Tensor& a, const Tensor& b) {
    return binaryOp(a, b, [](float x, float y){return x / y;});
}

Tensor div(const Tensor& a, float b) {
    return scalarOp(a, b, [](float x, float y){return x / y;});
}

Tensor div(float a, const Tensor& b) {
    return scalarOpReverse(a, b, [](float x, float y){return x / y;});
}

}