#pragma once
#include "core/Tensor.cuh"
#include <vector>

class Optimizer
{
protected:
    float learningRate;

public:
    explicit Optimizer(float learningRate) : learningRate(learningRate) {}
    virtual ~Optimizer() = default;
    virtual void update(std::vector<Tensor *> &parameters, const std::vector<Tensor *> &gradients) = 0;
};
