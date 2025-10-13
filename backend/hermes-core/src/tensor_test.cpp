#include "hermes/Tensor.h"
#include <iostream>
#include <vector>
#include <cassert>

void testTensorOperations() {
    // Test 1: Reshape
    hermes::Tensor tensor({2, 3});
    auto reshaped = tensor.reshape({3, 2});
    assert(reshaped.shape() == std::vector<int>({3, 2}));

    // Test 2: Transpose
    auto transposed = tensor.transpose(0, 1);
    assert(transposed.shape() == std::vector<int>({3, 2}));

    // Test 3: Slice
    hermes::Tensor tensor2({4, 4});
    auto sliced = tensor2.slice(1, 1, 3);
    assert(sliced.shape() == std::vector<int>({4, 2}));

    // Test 4: Indexing
    hermes::Tensor tensor3({2, 2});
    tensor3.data()[0] = 1.0f;
    tensor3.data()[1] = 2.0f;
    tensor3.data()[2] = 3.0f;
    tensor3.data()[3] = 4.0f;
    assert(tensor3(0, 0) == 1.0f);
    assert(tensor3(1, 1) == 4.0f);

    // Test 5:
    // Let's create a 2x3 Tensor A
    hermes::Tensor A({2, 3});
    // Fill its memory: [1, 2, 3,  4, 5, 6]
    // Logical view of A:
    // 1  2  3
    // 4  5  6
    float* a_data = A.data();
    for(int i = 0; i < 6; ++i) { 
        a_data[i] = i + 1.0f; 
    }

    // Now, let's create B, the transpose of A
    hermes::Tensor B = A.transpose(0, 1);
    // Logical view of B (3x2):
    // 1  4
    // 2  5
    // 3  6

    // THE TEST:
    // Where is the value 5.0f in A? It's at A(1, 1).
    // Where should the value 5.0f be in B? It's at B(1, 1).
    // Where is the value 3.0f in A? It's at A(0, 2).
    // Where should the value 3.0f be in B? It's at B(2, 0).

    // So, your assert should look like this:
    assert( B(2, 0) == 3.0f ); // This is the critical test.

    std::cout << "All tests passed!" << std::endl;
}

int main() {
    testTensorOperations();
    return 0;
}

// g++ -std=c++20 -Iinclude -Lbuild -lhermes_core -o tensor_test src/tensor_test.cpp