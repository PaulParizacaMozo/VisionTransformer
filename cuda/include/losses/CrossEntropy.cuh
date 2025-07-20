#pragma once

#include "losses/Loss.cuh"

Tensor softmax(const Tensor &logits);

class CrossEntropy : public Loss
{
private:
    Tensor softmaxOutput;

public:
    CrossEntropy() = default;
    float calculate(const Tensor &yPred, const Tensor &yTrue) override;
    Tensor backward(const Tensor &yPred, const Tensor &yTrue) override;
};
