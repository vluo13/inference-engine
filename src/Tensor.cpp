#include "hermes/Tensor.h"

#include <cassert>
#include <memory>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#include "hermes/utils.h"

namespace {

std::shared_ptr<float> allocate(const std::vector<size_t>& shape) {
    size_t bytes = hermes::Tensor::numel(shape) * sizeof(float);
    size_t rounded = (bytes + 127) & ~static_cast<size_t>(127);
    void* ptr = std::aligned_alloc(128, rounded);
    if (!ptr) throw std::bad_alloc();
    std::shared_ptr<float> retval {static_cast<float*>(ptr), [](float* ptr){
        std::free(ptr);
    }};
    return retval;
}
}

namespace hermes {
Tensor::Tensor(const std::vector<size_t>& shape) 
    : data_{allocate(shape)}
    , shape_{shape}
    , strides_{computeStrides(shape_)} {}

Tensor::Tensor(std::vector<size_t>&& shape) 
    : data_{allocate(shape)}
    , shape_{std::move(shape)}
    , strides_{computeStrides(shape_)} {}

/**
 * @brief Generates a tensor of zeros with dimensions given by shape
 * 
 * @param shape 
 * @return Tensor 
 */
Tensor Tensor::zeros(const std::vector<size_t>& shape) {
    Tensor result {shape};
    size_t count = numel(shape);
    
    std::fill_n(result.dataPtr(), count, 0.0f);

    return result;
}

/**
 * @brief Generates a tensor of ones with dimensions given by shape
 * 
 * @param shape 
 * @return Tensor 
 */
Tensor Tensor::ones(const std::vector<size_t>& shape) {
    Tensor result {shape};
    size_t count = numel(shape);
    
    std::fill_n(result.dataPtr(), count, 1.0f);

    return result;
}

/**
 * @brief Returns a tensor with dimension newShape
 * 
 * @param shape 
 * @return Tensor 
 */
Tensor Tensor::reshape(std::vector<size_t> newShape) const {
    assert(numel(shape_) == numel(newShape));

    if (isContiguous()) {
        std::vector<size_t> newStrides = computeStrides(newShape);
        return Tensor {data_, std::move(newShape), std::move(newStrides), offset_};
    }
    else {
        return contiguous().reshape(std::move(newShape));
    }
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
    std::vector<size_t> newShape = shape_;
    std::swap(newShape[dim0], newShape[dim1]);
    std::vector<size_t> newStrides = strides_;
    std::swap(newStrides[dim0], newStrides[dim1]);
    
    return Tensor {data_, std::move(newShape), std::move(newStrides), offset_};
}

Tensor Tensor::slice(size_t dim, size_t start, size_t end) const {
    if (dim >= shape_.size()) {
        throw std::runtime_error("Slicing along invalid dimension");
    }
    if (start > end || end > shape_[dim]) {
        throw std::runtime_error("Slice bounds are invalid");
    }
    std::vector<size_t> newShape = shape_;
    newShape[dim] = end - start;
    size_t newOffset = offset_ + (strides_[dim] * start);
    return Tensor {data_, std::move(newShape), strides_, newOffset};
}

float& Tensor::at(const std::vector<size_t>& indices) {
    size_t ndim = shape_.size();
    assert(indices.size() == ndim);
    for (size_t i = 0; i < ndim; ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::runtime_error("Index out of bounds");
        }
    }
    return dataPtr()[linearIndex(indices, strides_)];
}

float Tensor::at(const std::vector<size_t>& indices) const {
    size_t ndim = shape_.size();
    assert(indices.size() == ndim);
    for (size_t i = 0; i < ndim; ++i) {
        assert(indices[i] < shape_[i]);
    }
    return dataPtr()[linearIndex(indices, strides_)];
}

float* Tensor::dataPtr() {
    return data_.get() + offset_;
}

const float* Tensor::dataPtr() const {
    return data_.get() + offset_;
}

const std::vector<size_t>& Tensor::shape() const { 
    return shape_; 
}

const std::vector<size_t>& Tensor::strides() const {
    return strides_;
}

/**
 * @brief Checks whether the data of a tensor is contiguous in memory
 * 
 * @return true 
 * @return false 
 */
bool Tensor::isContiguous() const{
    return computeStrides(shape_) == strides_;
}

/**
 * @brief Makes a tensor contiguous in memorhy
 */
Tensor Tensor::contiguous() const {
    if (isContiguous()) {
        return *this;
    }
    Tensor result {shape_};

    size_t ndim = shape_.size();
    size_t count = numel(shape_);
    std::vector<size_t> indices(ndim);
    for (size_t i = 0; i < count; ++i) {
        result.dataPtr()[i] = at(indices);
        if (i + 1 == count) break; // guard against underflow on final interation

        incrementOdometer(indices, shape_, ndim);
    }

    return result;
}

/**
 * @brief Returns the total number of elements in a tensor with dimensions given by shape
 * 
 * @param shape 
 * @return size_t 
 */
size_t Tensor::numel(const std::vector<size_t>& shape){
    size_t count = 1;
    for (size_t dim : shape) {
        assert(dim > 0);
        count *= dim;
    }

    return count;
}

/**
 * @brief Helper function to determine the offset of the value at the index given by indices. Does not account for offset.
 * 
 * @param indices 
 * @return size_t 
 */
size_t Tensor::linearIndex(const std::vector<size_t>& indices, const std::vector<size_t>& strides) {
    size_t idx = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        idx += indices[i] * strides[i];
    }
    return idx;
}

/**
 * @brief Computes the strides of a tensor given its dimensions
 * 
 * @param shape 
 * @return std::vector<size_t> 
 */
std::vector<size_t> Tensor::computeStrides(const std::vector<size_t>& shape) {
    size_t size = shape.size();
    assert(size > 0);
    std::vector<size_t> strides(size);
    strides[size-1] = 1;
    for (int i = static_cast<int>(size)-2; i >= 0; i--) {
        assert(shape[i+1] > 0);
        strides[i] = shape[i+1] * strides[i+1];
    }
    return strides;
}

/**
 * @brief Computes the strides of a tensor for broadcasting
 * 
 * @param shape 
 * @param targetShape 
 * @param strides 
 * @return std::vector<size_t> 
 */
std::vector<size_t> Tensor::computeBroadcastStrides(const std::vector<size_t>& shape, const std::vector<size_t>& targetShape, const std::vector<size_t>& strides) {
    size_t ndims = targetShape.size();
    size_t pad = ndims - shape.size();
    std::vector<size_t> broadcastStrides(ndims);
    for (size_t i = 0; i < ndims; ++i) {
        if (i < pad) {
            broadcastStrides[i] = 0;
        }
        else {
            broadcastStrides[i] = (shape[i - pad] == 1 && targetShape[i] > 1) ? 0 : strides[i - pad];
        }
    }
    return broadcastStrides;
}

std::vector<size_t> Tensor::computeBroadcastShape(const std::vector<size_t>& ashape, const std::vector<size_t>& bshape) {
    int asize = ashape.size();
    int bsize = bshape.size();
    if (asize > bsize) {
        return computeBroadcastShape(bshape, ashape);
    }

    std::vector<size_t> shape(bsize);
    for (int i = bsize - 1; i >= 0; i--) {
        if (i - bsize + asize >= 0) {
            if (ashape[i - bsize + asize] == bshape[i]) {
                shape[i] = bshape[i];
            }
            else if (ashape[i - bsize + asize] == 1) {
                shape[i] = bshape[i];
            }
            else if (bshape[i] == 1) {
                shape[i] = ashape[i - bsize + asize];
            }
            else {
                throw std::runtime_error("Tensors are not compatible for broadcasting");
            }
        }
        else {
            shape[i] = bshape[i];
        }
    }

    return shape;
}

/**
 * @brief Private constructor for Tensor class
 * 
 * @param data 
 * @param shape 
 * @param strides 
 * @param offset 
 */
Tensor::Tensor(std::shared_ptr<float> data, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset)
    : data_(data)
    , shape_(shape)
    , strides_(strides)
    , offset_(offset) {}
    
}