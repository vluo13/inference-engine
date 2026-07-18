#ifndef HERMES_TENSOR_H
#define HERMES_TENSOR_H

#include <memory>
#include <vector>

namespace hermes {
    
class Tensor 
{
public: 
    // Constructors
    explicit Tensor(const std::vector<size_t>& shape);
    explicit Tensor(std::vector<size_t>&& shape);
    
    // Factory
    static Tensor zeros(const std::vector<size_t>& shape);
    static Tensor ones(const std::vector<size_t>& shape);

    // Tensor Operaetions
    Tensor reshape(std::vector<size_t> newShape) const;
    Tensor transpose(size_t dim0, size_t dim1) const;
    Tensor slice(size_t dim, size_t start, size_t end) const;
    float at(const std::vector<size_t>& indices) const;
    float& at(const std::vector<size_t>& indices);
    float* dataPtr();
    const float* dataPtr() const;
    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides() const;
    bool isContiguous() const;
    Tensor contiguous() const;

    static size_t numel(const std::vector<size_t>& shape);
    static size_t linearIndex(const std::vector<size_t>& indices, const std::vector<size_t>& strides);
    static std::vector<size_t> computeStrides(const std::vector<size_t>& shape);
    static std::vector<size_t> computeBroadcastStrides(const std::vector<size_t>& shape, const std::vector<size_t>& targetShape, const std::vector<size_t>& strides);
    static std::vector<size_t> computeBroadcastShape(const std::vector<size_t>& ashape, const std::vector<size_t>& bshape);

private:
    std::shared_ptr<float> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t offset_ {0};
    
    Tensor(std::shared_ptr<float> data, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset);
};

}

#endif