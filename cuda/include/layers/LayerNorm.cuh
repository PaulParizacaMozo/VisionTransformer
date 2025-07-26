#pragma once

#include "layers/Layer.cuh"
#include <vector>
#include <string>

class LayerNorm : public Layer
{
private:
    float epsilon;
    size_t featureSize;

    // Parámetros entrenables
    Tensor gamma;
    Tensor beta;

    // Gradientes
    Tensor gammaGradient;
    Tensor betaGradient;

    // Estado intermedio para backward
    Tensor inputTensor;
    Tensor mean;
    Tensor variance;
    Tensor normalizedInput;

public:
    LayerNorm(size_t featureSize, float epsilon = 1e-5f);

    Tensor forward(const Tensor &input, bool isTraining) override;
    Tensor backward(const Tensor &outputGradient) override;

    std::vector<Tensor *> getParameters() override;
    std::vector<Tensor *> getGradients() override;

    std::string getName() const override { return "LayerNorm"; }
};
