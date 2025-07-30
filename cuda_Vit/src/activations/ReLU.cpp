#include "activations/ReLU.hpp"
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <iostream>

ReLU::ReLU() {}

/**
 * @brief Aplica la función de activación ReLU: f(x) = max(0, x).
 * @details Actualizado para soportar tensores 2D y 3D.
 */
Tensor ReLU::forward(const Tensor &input, bool isTraining)
{
  if (isTraining)
  {
    this->inputTensor = input;
  }

  Tensor result(input.getShape());
  const auto &shape = input.getShape();

  if (shape.size() == 2)
  {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < shape[0]; ++i)
    {
      for (size_t j = 0; j < shape[1]; ++j)
      {
        float val = input(i, j);
        result(i, j) = (val > 0) ? val : 0.0f;
      }
    }
  }
  else if (shape.size() == 3)
  { // --- AÑADIDO: Caso 3D ---
#pragma omp parallel for collapse(3)
    for (size_t i = 0; i < shape[0]; ++i)
    {
      for (size_t j = 0; j < shape[1]; ++j)
      {
        for (size_t k = 0; k < shape[2]; ++k)
        {
          float val = input(i, j, k);
          result(i, j, k) = (val > 0) ? val : 0.0f;
        }
      }
    }
  }
  else
  {
    // En lugar de lanzar un error, podríamos tener un fallback más lento pero genérico
    // si fuera necesario en el futuro. Por ahora, esto está bien.
    throw std::runtime_error("ReLU::forward solo soporta entradas 2D o 3D.");
  }

  return result;
}

/**
 * @brief Calcula el gradiente para la capa ReLU.
 * @details Actualizado para soportar tensores 2D y 3D.
 */
Tensor ReLU::backward(const Tensor &outputGradient)
{
  Tensor inputGradient(this->inputTensor.getShape());
  const auto &shape = this->inputTensor.getShape();

  if (shape.size() == 2)
  {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < shape[0]; ++i)
    {
      for (size_t j = 0; j < shape[1]; ++j)
      {
        inputGradient(i, j) = (this->inputTensor(i, j) > 0) ? outputGradient(i, j) : 0.0f;
      }
    }
  }
  else if (shape.size() == 3)
  { // --- AÑADIDO: Caso 3D ---
#pragma omp parallel for collapse(3)
    for (size_t i = 0; i < shape[0]; ++i)
    {
      for (size_t j = 0; j < shape[1]; ++j)
      {
        for (size_t k = 0; k < shape[2]; ++k)
        {
          inputGradient(i, j, k) = (this->inputTensor(i, j, k) > 0) ? outputGradient(i, j, k) : 0.0f;
        }
      }
    }
  }
  else
  {
    throw std::runtime_error("ReLU::backward solo soporta entradas 2D o 3D.");
  }

  return inputGradient;
}
