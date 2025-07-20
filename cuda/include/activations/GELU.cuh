#pragma once

#include "layers/Layer.cuh"
#include "core/Tensor.cuh"

/**
 * @class GELU
 * @brief Implementa la función de activación GELU (Gaussian Error Linear Unit) para GPU.
 *
 * Usa la aproximación rápida para GELU:
 * GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
 */
class GELU : public Layer
{
private:
    Tensor inputTensor; // Guardar input para usarlo en backward
public:
    GELU();
    Tensor forward(const Tensor &input, bool isTraining) override;
    Tensor backward(const Tensor &outputGradient) override;
    std::string getName() const override { return "GELU"; }
};
