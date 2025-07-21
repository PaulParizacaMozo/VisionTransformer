#pragma once

#include "layers/Layer.cuh"
#include "layers/Dense.cuh"
#include "activations/GELU.cuh"
#include <vector>
#include <memory>
#include <string>

class FeedForward : public Layer
{
private:
    Dense dense1;
    GELU activation;
    Dense dense2;

public:
    FeedForward(size_t embedding_dim, size_t hidden_dim);

    Tensor forward(const Tensor &input, bool isTraining) override;
    Tensor backward(const Tensor &outputGradient) override;

    std::vector<Tensor *> getParameters() override;
    std::vector<Tensor *> getGradients() override;
    std::string getName() const override { return "FeedForward"; }
};
