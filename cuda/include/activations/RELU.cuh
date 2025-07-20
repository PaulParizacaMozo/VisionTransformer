#pragma once

#include "layers/Layer.cuh"
#include "core/Tensor.cuh"

/**
 * @class ReLU
 * @brief Implementa la función de activación ReLU (Rectified Linear Unit) en GPU.
 *
 * Aplica la función f(x) = max(0, x) elemento a elemento.
 * No tiene parámetros entrenables. Se implementa como una capa para integrarse fácilmente en modelos.
 */
class ReLU : public Layer
{
private:
    Tensor inputTensor; // Necesario para el backward (x > 0)

public:
    ReLU();
    Tensor forward(const Tensor &input, bool isTraining) override;
    Tensor backward(const Tensor &outputGradient) override;
    std::string getName() const override { return "ReLU"; }
};
