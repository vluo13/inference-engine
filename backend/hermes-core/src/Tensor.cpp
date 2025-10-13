#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <functional>
#include <array>
//#include "hermes/kernels.h"
#include "hermes/Tensor.h"

namespace hermes {

// shape constructor
Tensor::Tensor(const std::vector<int>& shape)
        : shape_(shape),
            strides_(computeStrides(shape)),
            offset_(0) {
    int size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    this->data_ = std::make_shared<std::vector<float>>(size);
}
// copy constructor
Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data)
        : shape_(shape),
            strides_(computeStrides(shape)),
            offset_(0) {
    int size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    if (data.size() != size) {
        throw std::invalid_argument("Data size does not match shape dimensions.");
    }
    data_ = std::make_shared<std::vector<float>>(data); // Makes a copy
}

// move
Tensor::Tensor(const std::vector<int>& shape, std::vector<float>&& data)
        : shape_(shape),
            strides_(computeStrides(shape)),
            offset_(0) {
    int size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    if (data.size() != size) {
        throw std::invalid_argument("Data size does not match shape dimensions.");
    }
    data_ = std::make_shared<std::vector<float>>(std::move(data)); // Makes a copy
}

// Private constructor
Tensor::Tensor(const std::vector<int>& shape, 
        const std::vector<int>& strides, 
        std::shared_ptr<std::vector<float>> data, 
        std::size_t offset)
    : shape_(shape), strides_(strides), data_(data), offset_(offset) {}

// Private helpers

std::vector<int> Tensor::computeStrides(const std::vector<int>& shape) const {
    if (shape.empty()) {
        return {};
    }
    std::vector<int> strides(shape.size());
    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

bool Tensor::isContiguous() const {
    return strides_ == computeStrides(shape_);
}

size_t Tensor::get_offset(const std::vector<int>& indices) const {
    if (indices.size() != this->strides_.size()) {
        throw std::invalid_argument("Number of indices must match number of dimensions.");
    }
    size_t offset = 0;
    for (int i = 0; i < indices.size(); i++) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        offset += indices[i] * this->strides_[i];
    }
    return offset;
}

// getters
const std::vector<int>& Tensor::shape() const { 
    return shape_; 
}
const std::vector<int>& Tensor::strides() const { 
    return strides_; 
}
int Tensor::dim() const { 
    return shape_.size(); 
}
float* Tensor::data() {
    return data_->data() + offset_;
}
const float* Tensor::data() const {
    return data_->data() + offset_;
}

// view opeartions
/**
 * @brief Returns the transpose of the current instance
 * 
 * @param dim0 index of dimension to be swapped
 * @param dim1 index of dimension to be swapped
 * @return transposed tensor
 */
Tensor Tensor::transpose(int dim0, int dim1) const {
    // Error checking for dimension indices
        if (dim0 < 0 || dim0 >= static_cast<int>(shape_.size())) {
        throw std::out_of_range("dim0 is out of range");
    }
        if (dim1 < 0 || dim1 >= static_cast<int>(shape_.size())) {
        throw std::out_of_range("dim1 is out of range");
    }
    
    std::vector<int> newShape(shape_);
    std::swap(newShape[dim0], newShape[dim1]);

    std::vector<int> newStrides(strides_);
    std::swap(newStrides[dim0], newStrides[dim1]);

    return Tensor(newShape, newStrides, data_, offset_);
}

/**
 * @brief slices the Tensor along dim
 * 
 * @param dim dimension to slice along
 * @param start start of the slice (inclusive)
 * @param end end of the slice (exclusive)
 * @return Tensor sliced tensor
 */
Tensor Tensor::slice(int dim, int start, int end) const {
    if (dim < 0 || dim >= static_cast<int>(shape_.size())) {
           throw std::out_of_range("Slice dimension is out of range.");
    }
    if (start < 0 || end > shape_[dim] || start > end) {
        throw std::out_of_range("Slice indices are out of range.");
    }
    std::vector<int> new_shape = shape_;
    new_shape[dim] = end - start;
    size_t new_offset = offset_ + start * strides_[dim];
    return Tensor(new_shape, strides_, data_, new_offset);
}

/**
 * @brief Reshapes the tensor to new dimensions
 * 
 * @param new_shape new dimensions of tensor
 * @return Tensor reshaped tensor
 */
Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    int old_size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    int new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
    if (old_size != new_size) {
        throw std::invalid_argument("New dimensions are invalid");
    }

    // check if our tensor is contiguous
    if (isContiguous()) {
        // Preserve current offset when creating a view with new shape/strides
        return Tensor(new_shape, computeStrides(new_shape), data_, offset_);
    }
    else {
        Tensor result(new_shape);
        float* dest_ptr = result.data();
    std::vector<int> current_indices(static_cast<int>(shape_.size()), 0);
        for (size_t i = 0; i < old_size; ++i) {
            // Calculate the 1D offset in the *source* (this) tensor using its strides
            size_t src_offset = 0;
                for(int d = 0; d < static_cast<int>(shape_.size()); d++) {
                src_offset += current_indices[d] * strides_[d];
            }
            // Copy the single element
            dest_ptr[i] = (*data_)[offset_ + src_offset];

            // Increment the N-dimensional index for the next iteration
        for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
                // increment then check if we rollover
                // if we rollover then increment the next
                if (++current_indices[d] < shape_[d]) {
                    break;
                }
                // since we rolled over, need to set to zero
                current_indices[d] = 0;
            }
        }
        return result;
    }
}

// Convenience Wrappers
// Tensor Tensor::add(const Tensor& other) const {
//     Tensor result(this->shape_);
//     hermes::add(*this, other, result);
//     return result;
// }
// Tensor Tensor::subtract(const Tensor& other) const {
//     Tensor result(this->shape_);
//     hermes::subtract(*this, other, result);
//     return result;
// }
// Tensor Tensor::scale(float factor) const {
//     Tensor result(this->shape_);
//     hermes::scale(*this, factor, result);
//     return result;
// }
// Tensor Tensor::multiply(const Tensor& other) const {
//     Tensor result(this->shape_);
//     hermes::multiply(*this, other, result);
//     return result;
// }
// Tensor Tensor::matmul(const Tensor& other) const{
//     std::vector<int> resultShape = {this->shape_[0], other.shape_[1]};
//     Tensor result(resultShape);
//     hermes::matmul(*this, other, result);
//     return result;
// }

}