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
    __host__ Dense(size_t inputSize, size_t outputSize);

    // Forward en GPU
    __host__ Tensor forward(const Tensor &input, bool isTraining) override;

    // Backward en GPU
    __host__ Tensor backward(const Tensor &outputGradient) override;

    // Acceso a parámetros entrenables
    __host__ std::vector<Tensor *> getParameters() override;

    // Acceso a gradientes
    __host__ std::vector<Tensor *> getGradients() override;

    // Nombre de la capa
    __host__ std::string getName() const override { return "Dense"; }
};
