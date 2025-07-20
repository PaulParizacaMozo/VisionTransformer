#pragma once

#include "core/Tensor.cuh"

class Loss
{
public:
    virtual ~Loss() = default;
    virtual float calculate(const Tensor &yPred, const Tensor &yTrue) = 0;
    virtual Tensor backward(const Tensor &yPred, const Tensor &yTrue) = 0;
};
