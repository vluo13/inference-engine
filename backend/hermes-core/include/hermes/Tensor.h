#ifndef HERMES_TENSOR_H
#define HERMES_TENSOR_H

#include <vector>
#include <memory>
#include <stdexcept>
#include <utility>
#include <numeric>
#include <array>
#include <type_traits>

namespace hermes {

class Tensor {
public:
    // constructors
    explicit Tensor(const std::vector<int>& shape); // need to do this because you could have implicit conversions with one parameter constructors
    Tensor(const std::vector<int>& shape, const std::vector<float>& data);
    Tensor(const std::vector<int>& shape, std::vector<float>&& data); // optimize when data is passed in as a rvalue

    // getters
    const std::vector<int>& shape() const;
    const std::vector<int>& strides() const;
    float* data();
    const float* data() const;

    // view opeartions
    Tensor transpose(int dim0, int dim1) const;
    Tensor slice(int dim, int start, int end) const;
    Tensor reshape(const std::vector<int>& new_shape) const;

    // For const Tensor
    template <typename... Args>
    const float& operator()(Args... indices) const {
        // Compile-time check for only integers only passes integers.
        static_assert((std::is_same_v<int, Args> && ...), "Indices must be integers");

        // Calculate offset
        size_t final_offset = get_offset({indices...});
        return (*data_)[offset_ + final_offset];
    }

    // For non const Tensor
    template <typename... Args>
    float& operator()(Args... indices) {
        // Calls the const version  then safely casts away 
        return const_cast<float&>(
            static_cast<const Tensor&>(*this).operator()(indices...)
        );
    }

    // convenience wrappers
    // Tensor add(const Tensor& other) const;
    // Tensor subtract(const Tensor& other) const;
    // Tensor scale(float factor) const;
    // Tensor multiply(const Tensor& other) const;
    // Tensor matmul(const Tensor& other) const;

    // helpers
    int dim() const;

private:
    std::shared_ptr<std::vector<float>> data_;
    std::vector<int> shape_;
    std::vector<int> strides_;
    std::size_t offset_;

    Tensor(const std::vector<int>& shape, 
        const std::vector<int>& strides, 
        std::shared_ptr<std::vector<float>> data, size_t offset);

    std::vector<int> computeStrides(const std::vector<int>& shape) const;
    bool isContiguous() const;
    size_t get_offset(const std::vector<int>& indices) const;
};


// operator overloads
// inline Tensor operator+(const Tensor& a, const Tensor& b) {
//     return a.add(b);
// }
// inline Tensor operator-(const Tensor& a, const Tensor& b) {
//     return a.subtract(b);
// }
// inline Tensor operator*(const Tensor& a, const Tensor& b) {
//     return a.multiply(b);
// }

} // namespace hermes

#endif // HERMES_TENSOR_H