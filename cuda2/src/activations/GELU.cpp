#include "activations/GELU.hpp"
#include <cmath> // Para tanh, sqrt, pow

#ifdef _OPENMP
#include <omp.h>
#endif

// Constante para la aproximación de GELU: sqrt(2/pi)
const float SQRT_2_OVER_PI = 0.7978845608028654f;

GELU::GELU() {}

Tensor GELU::forward(const Tensor &input, bool isTraining)
{
  if (isTraining)
  {
    this->inputTensor = input;
  }
  return input.gelu_f();
  //   Tensor result(input.getShape());
  //   const auto &shape = input.getShape();

  //   // Implementación genérica que funciona para cualquier tensor contiguo.
  //   // Usar getData() es más rápido si sabemos que el tensor es contiguo.
  //   if (input.isContiguous() && result.isContiguous()) {
  //     const float *in_data = input.getData();
  //     float *out_data = result.getData();
  //     size_t size = input.getSize();

  // #pragma omp parallel for
  //     for (size_t i = 0; i < size; ++i) {
  //       float x = in_data[i];
  //       float x_cubed = x * x * x;
  //       float inner = SQRT_2_OVER_PI * (x + 0.044715f * x_cubed);
  //       out_data[i] = 0.5f * x * (1.0f + std::tanh(inner));
  //     }
  //   } else {
  //     // Fallback más lento para vistas no contiguas (usa operator())
  //     // (Por ahora, lanzamos un error o implementamos con bucles anidados si es necesario)
  //     throw std::runtime_error("GELU::forward solo implementado para tensores contiguos.");
  //   }

  // return result;
}

Tensor GELU::backward(const Tensor &outputGradient)
{
  Tensor inputGradient(inputTensor.getShape());
  const auto &shape = inputTensor.getShape();

  // La derivada de la aproximación de GELU es más compleja.
  // dGELU/dx = 0.5 * tanh(inner) + 0.5 * x * sech^2(inner) * d(inner)/dx
  // d(inner)/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
  // sech^2(z) = 1 - tanh^2(z)

  if (inputTensor.isContiguous() && outputGradient.isContiguous())
  {
    const float *in_data = inputTensor.getData();
    const float *grad_out_data = outputGradient.getData();
    float *grad_in_data = inputGradient.getData();
    size_t size = inputTensor.getSize();

#pragma omp parallel for
    for (size_t i = 0; i < size; ++i)
    {
      float x = in_data[i];
      float x_squared = x * x;
      float x_cubed = x_squared * x;

      float inner = SQRT_2_OVER_PI * (x + 0.044715f * x_cubed);
      float tanh_inner = std::tanh(inner);

      float d_inner_dx = SQRT_2_OVER_PI * (1.0f + 3.0f * 0.044715f * x_squared);

      float sech_squared = 1.0f - tanh_inner * tanh_inner;

      float dGELU_dx = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech_squared * d_inner_dx;

      grad_in_data[i] = dGELU_dx * grad_out_data[i];
    }
  }
  else
  {
    throw std::runtime_error("GELU::backward solo implementado para tensores contiguos.");
  }

  return inputGradient;
}
