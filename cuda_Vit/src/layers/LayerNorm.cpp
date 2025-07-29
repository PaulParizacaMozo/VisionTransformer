#include "layers/LayerNorm.hpp"
#include <cmath>
#include "utils/CudaUtils.hpp"
#include <numeric>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

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
  //   const auto &inputShape = input.getShape();
  //   if (inputShape.back() != this->featureSize)
  //   {
  //     throw std::runtime_error("La última dimensión de la entrada no coincide con featureSize de LayerNorm.");
  //   }

  //   // El número de muestras en el batch es el producto de todas las dimensiones excepto la última.
  //   size_t batchSize = input.getSize() / this->featureSize;

  //   // Aplanamos temporalmente la entrada a 2D {batchSize, featureSize} para facilitar los cálculos.
  //   Tensor input2D = input.reshape({batchSize, this->featureSize});

  //   // En modo entrenamiento, guardamos los valores necesarios para el backward pass.
  //   if (isTraining)
  //   {
  //     this->inputTensor = input2D;
  //     this->mean = Tensor({batchSize, 1});
  //     this->variance = Tensor({batchSize, 1});
  //   }

  //   Tensor output2D({batchSize, this->featureSize});
  //   this->normalizedInput = Tensor({batchSize, this->featureSize});

  // // Iteramos sobre cada muestra del batch.
  // #pragma omp parallel for
  //   for (size_t i = 0; i < batchSize; ++i)
  //   {
  //     // --- 1. Calcular la media ---
  //     float current_mean = 0.0f;
  //     for (size_t j = 0; j < this->featureSize; ++j)
  //     {
  //       current_mean += input2D(i, j);
  //     }
  //     current_mean /= this->featureSize;
  //     if (isTraining)
  //       this->mean(i, 0) = current_mean;

  //     // --- 2. Calcular la varianza ---
  //     float current_variance = 0.0f;
  //     for (size_t j = 0; j < this->featureSize; ++j)
  //     {
  //       float diff = input2D(i, j) - current_mean;
  //       current_variance += diff * diff;
  //     }
  //     current_variance /= this->featureSize;
  //     if (isTraining)
  //       this->variance(i, 0) = current_variance;

  //     float inv_stddev = 1.0f / std::sqrt(current_variance + this->epsilon);
  //     if (isTraining)
  //     {
  //       // Guardamos la varianza invertida para reutilizarla en el backward pass.
  //       // Para ser precisos, guardamos (1 / sqrt(var + eps)) en el tensor de varianza.
  //       this->variance(i, 0) = inv_stddev;
  //     }

  //     // --- 3. Normalizar, aplicar gamma y beta ---
  //     for (size_t j = 0; j < this->featureSize; ++j)
  //     {
  //       // Normalizar la entrada (x_hat)
  //       float x_hat = (input2D(i, j) - current_mean) * inv_stddev;
  //       if (isTraining)
  //         this->normalizedInput(i, j) = x_hat;

  //       // Aplicar escala (gamma) y desplazamiento (beta)
  //       output2D(i, j) = this->gamma(0, j) * x_hat + this->beta(0, j);
  //     }
  //   }
  // if (isTraining)
  // {
  //   if (verify(this->inputTensor, cuda_result.input2D, 1e-5f) == false)
  //   {
  //     std::cerr << "Error en la verificación de inputTensor\n";
  //   }

  //   if (verify(this->normalizedInput, cuda_result.normalized, 1e-5f) == false)
  //   {
  //     std::cerr << "Error en la verificación de normalizedInput\n";
  //   }
  //   if (verify(this->mean, cuda_result.mean, 1e-5f) == false)
  //   {
  //     std::cerr << "Error en la verificación de mean\n";
  //   }
  //   if (verify(this->variance, cuda_result.invStd, 1e-5f) == false)
  //   {
  //     std::cerr << "Error en la verificación de variance\n";
  //   }
  // }
  // if (verify(output2D.reshape(inputShape), cuda_result.output, 1e-5f) == false)
  // {
  //   std::cerr << "Error en la verificación de output2D\n";
  // }
  // Devolvemos el tensor con su forma original.
  // return output2D.reshape(inputShape);
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
  // OMP aquí puede ser problemático por la acumulación en gamma/beta gradients.
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
