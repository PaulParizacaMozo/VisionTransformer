#pragma once
#include "core/Tensor.cuh"
#include <string>
#include <vector>

class Layer
{
public:
    // Destructor virtual
    __host__ virtual ~Layer() = default;

    // Forward en GPU
    __host__ virtual Tensor forward(const Tensor &input, bool isTraining) = 0;

    // Backward en GPU
    __host__ virtual Tensor backward(const Tensor &outputGradient) = 0;

    // Parámetros entrenables
    __host__ virtual std::vector<Tensor *> getParameters() { return {}; }

    // Gradientes
    __host__ virtual std::vector<Tensor *> getGradients() { return {}; }

    // Nombre de la capa
    __host__ virtual std::string getName() const = 0;
};
