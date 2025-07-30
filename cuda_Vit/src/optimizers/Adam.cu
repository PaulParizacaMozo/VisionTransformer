#include "optimizers/Adam.cuh"
#include <cmath>
#include "utils/CudaUtils.cuh"
#include <stdexcept>
#include <iostream>

Adam::Adam(float learningRate, float beta1, float beta2, float epsilon, float weight_decay)
    : Optimizer(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), weight_decay(weight_decay), t(0),
      initialized(false) {}

void Adam::update(std::vector<Tensor *> &parameters, const std::vector<Tensor *> &gradients)
{
  if (parameters.size() != gradients.size())
  {
    throw std::runtime_error("El numero de parametros y gradientes no coincide en Adam::update.");
  }

  // Inicializacion diferida: crea los tensores de momento m y v en la primera llamada.
  if (!initialized)
  {
    m.reserve(parameters.size());
    v.reserve(parameters.size());
    for (const auto &param : parameters)
    {
      m.emplace_back(param->getShape());
      v.emplace_back(param->getShape());
    }
    initialized = true;
  }

  t++; // Incrementa el contador de pasos.

  // Correccion de sesgo (bias correction) pre-calculada.
  const float beta1_t = std::pow(beta1, t);
  const float beta2_t = std::pow(beta2, t);

  // Itera sobre cada par de parametro/gradiente.
  for (size_t i = 0; i < parameters.size(); ++i)
  {

    adam_update_single_tensor_cuda(
        *parameters[i], *gradients[i], m[i], v[i],
        beta1, beta2,
        beta1_t, beta2_t,
        learningRate, epsilon,
        weight_decay);
  }
}
