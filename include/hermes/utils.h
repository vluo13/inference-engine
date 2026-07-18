#ifndef HERMES_UTILS_H
#define HERMES_UTILS_H

#include "hermes/Tensor.h"

#include <vector>

namespace hermes {

void incrementOdometer(std::vector<size_t>& indices, const std::vector<size_t>& shape, size_t ndim);
Tensor embedding(const Tensor& x, const Tensor& table);

}

#endif