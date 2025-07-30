#include "layers/LayerNorm.cuh"
#include <cmath>
#include "utils/CudaUtils.cuh"
#include <numeric>
#include <iostream>

LayerNorm::LayerNorm(size_t featureSize, float epsilon) : featureSize(featureSize), epsilon(epsilon)
{
  // Inicializar los parámetros entrenables.
  // Gamma se inicializa a 1 para que al principio la capa no altere la escala.
  this->gamma = Tensor({1, featureSize});
  this->gamma.fill(1.0f);

  // Beta se inicializa a 0 para que al principio la capa no aplique desplazamiento.
  this->beta = Tensor({1, featureSize});
  this->beta.fill(0.0f);

  // Los gradientes se inicializan con las mismas formas, a cero.
  this->gammaGradient = Tensor({1, featureSize});
  this->betaGradient = Tensor({1, featureSize});
}

Tensor LayerNorm::forward(const Tensor &input, bool isTraining)
{
  auto cuda_result = layernorm_forward_cuda(input, this->gamma, this->beta, this->epsilon, true);
  if (isTraining)
  {
    this->inputTensor = cuda_result.input2D;
    this->mean = cuda_result.mean;
    this->variance = cuda_result.invStd; // Guardamos 1/sqrt(var + eps)
    this->normalizedInput = cuda_result.normalized;
  }
  return cuda_result.output;
}

Tensor LayerNorm::backward(const Tensor &outputGradient)
{
  const auto &gradShape = outputGradient.getShape();
  size_t batchSize = outputGradient.getSize() / this->featureSize;

  // Aplanamos el gradiente de salida a 2D.
  //
  Tensor grad2D = outputGradient.reshape({batchSize, this->featureSize});

  // Los gradientes de los parámetros se acumulan, así que los reseteamos.
  this->gammaGradient.fill(0.0f);
  this->betaGradient.fill(0.0f);
  Tensor inputGradient({batchSize, this->featureSize});

  // --- Derivadas ---
  // dL/dY es outputGradient (grad2D)
  // Y = gamma * X_hat + beta
  // X_hat = (X - mean) * inv_stddev
  // inv_stddev = 1 / sqrt(var + eps)

  // Iteramos sobre cada muestra del batch.
  // Lo hacemos secuencial para evitar race conditions, o usamos #pragma omp critical.
  // Por simplicidad, lo haremos secuencial aquí, la paralelización del bucle externo es más segura.
  for (size_t i = 0; i < batchSize; ++i)
  {
    float inv_stddev = this->variance(i, 0); // Reutilizamos el valor guardado

    // Acumuladores para derivadas intermedias
    // float dL_dgamma_sum = 0;
    // float dL_dbeta_sum = 0;

    float dL_dXhat_dot_Xhat_sum = 0;
    float dL_dXhat_sum = 0;

    // --- 1. Calcular gradientes de gamma y beta (y algunas sumas para después) ---
    // dL/dgamma = dL/dY * X_hat
    // dL/dbeta = dL/dY * 1
    // Estos se suman a lo largo del batch.
    for (size_t j = 0; j < this->featureSize; ++j)
    {
      float grad_y_ij = grad2D(i, j);
      float x_hat_ij = this->normalizedInput(i, j);

      // Acumulamos para el gradiente final de gamma/beta
      this->gammaGradient(0, j) += grad_y_ij * x_hat_ij;
      this->betaGradient(0, j) += grad_y_ij;

      // dL/dX_hat = dL/dY * gamma
      float dL_dXhat = grad_y_ij * this->gamma(0, j);

      // Sumas necesarias para el gradiente de la entrada
      dL_dXhat_sum += dL_dXhat;
      dL_dXhat_dot_Xhat_sum += dL_dXhat * x_hat_ij;
    }

    // --- 2. Calcular el gradiente de la entrada (dL/dX) ---
    // Este es el paso más complejo. Se deriva usando la regla de la cadena a través de la normalización.
    // La fórmula final del gradiente para un elemento X_ij es:
    // dL/dX_ij = (1/N) * gamma_j * inv_stddev * [ N*dL/dX_hat_ij - sum(dL/dX_hat) - X_hat_ij * sum(dL/dX_hat * X_hat) ]

    for (size_t j = 0; j < this->featureSize; ++j)
    {
      float dL_dXhat_ij = grad2D(i, j) * this->gamma(0, j);
      float x_hat_ij = this->normalizedInput(i, j);

      float term1 = this->featureSize * dL_dXhat_ij;
      float term2 = dL_dXhat_sum;
      float term3 = x_hat_ij * dL_dXhat_dot_Xhat_sum;

      inputGradient(i, j) = (1.0f / this->featureSize) * inv_stddev * (term1 - term2 - term3);
    }
  }

  // Devolvemos el gradiente con la forma original.
  return inputGradient.reshape(gradShape);
}

std::vector<Tensor *> LayerNorm::getParameters() { return {&this->gamma, &this->beta}; }

std::vector<Tensor *> LayerNorm::getGradients() { return {&this->gammaGradient, &this->betaGradient}; }
