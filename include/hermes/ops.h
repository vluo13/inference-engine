#ifndef HERMES_OPS_H
#define HERMES_OPS_H

#include "hermes/Tensor.h"

namespace hermes {

Tensor add(const Tensor& a, const Tensor& b);
Tensor add(const Tensor& a, float b);
Tensor add(float a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, float b);
Tensor sub(float a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, float b);
Tensor mul(float a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, float b);
Tensor div(float a, const Tensor& b);

}

#endif