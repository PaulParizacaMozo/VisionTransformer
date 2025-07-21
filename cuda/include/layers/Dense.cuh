#pragma once
#include "layers/Layer.cuh"
#include <vector>
#include <string>
class Dense : public Layer
{
private:
    // Parámetros entrenables
    Tensor weights; ///< [input_size, output_size]
    Tensor bias;    ///< [1, output_size]

    // Gradientes asociados
    Tensor weightGradients; ///< dE/dW
    Tensor biasGradients;   ///< dE/db

    // Entrada cacheada (para backward)
    Tensor inputTensor;

public:
    // Constructor
    Dense(size_t inputSize, size_t outputSize);

    // Forward en GPU
    Tensor forward(const Tensor &input, bool isTraining) override;

    // Backward en GPU
    Tensor backward(const Tensor &outputGradient) override;

    // Acceso a parámetros entrenables
    std::vector<Tensor *> getParameters() override;

    // Acceso a gradientes
    std::vector<Tensor *> getGradients() override;

    // Nombre de la capa
    std::string getName() const override { return "Dense"; }
};
