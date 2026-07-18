#include "hermes/utils.h"

#include <vector>
#include <fstream>
#include <iostream>
#include <string>

#include "hermes/Tensor.h"

namespace hermes {

void incrementOdometer(std::vector<size_t>& indices, const std::vector<size_t>& shape, size_t ndim) {
    ++indices[ndim - 1];
    size_t cur = ndim - 1;
    while (indices[cur] == shape[cur]) {
        indices[cur] = 0;
        cur--; 
        ++indices[cur];
    }
}

/**
 * Assumes x is (T,)
 * table is (vocab_size, n_embed)
 */
Tensor embedding(const Tensor& x, const Tensor& table) { 
    size_t tokens = x.shape()[0];
    size_t vocab_size = table.shape()[0];
    size_t n_embed = table.shape()[1];
    Tensor out {std::vector<size_t>{tokens, n_embed}};
    for (size_t i = 0; i < tokens; ++i) {
        if (x.dataPtr()[i] >= vocab_size) {
            throw std::runtime_error("Token id is invalid: " + std::to_string(x.dataPtr()[i]));
        }
        memcpy(out.dataPtr() + (i * n_embed), table.dataPtr() + (static_cast<size_t>(x.dataPtr()[i]) * n_embed), sizeof(float) * n_embed);
    }
    return out;
}

}