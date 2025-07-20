#include "optimizers/Adam.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <iostream>

// Kernel para aplicar Adam en un solo tensor plano
__global__ void adam_update_kernel(float *param, const float *grad,
                                   float *m, float *v,
                                   float beta1, float beta2,
                                   float beta1_t, float beta2_t,
                                   float lr, float eps, float weight_decay,
                                   size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    float g = grad[idx];
    if (weight_decay > 0.0f)
    {
        g += weight_decay * param[idx];
    }

    // Actualización de momentos
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * (g * g);

    // Corrección de sesgo
    float m_hat = m[idx] / (1.0f - beta1_t);
    float v_hat = v[idx] / (1.0f - beta2_t);

    // Actualización del parámetro
    param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

Adam::Adam(float learningRate, float beta1, float beta2, float epsilon, float weight_decay)
    : Optimizer(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), weight_decay(weight_decay), t(0),
      initialized(false) {}

void Adam::update(std::vector<Tensor *> &parameters, const std::vector<Tensor *> &gradients)
{
    if (parameters.size() != gradients.size())
    {
        throw std::runtime_error("El numero de parametros y gradientes no coincide en Adam::update.");
    }
    if (!initialized)
    {
        m.reserve(parameters.size());
        v.reserve(parameters.size());
        for (const auto &param : parameters)
        {
            m.push_back(new Tensor(param->getShapeHost(), param->dims()));
            v.push_back(new Tensor(param->getShapeHost(), param->dims()));
        }
        initialized = true;
    }

    t++;
    float beta1_t = std::pow(beta1, t);
    float beta2_t = std::pow(beta2, t);
    for (size_t i = 0; i < parameters.size(); ++i)
    {
        Tensor *param = parameters[i];
        Tensor *grad = gradients[i];
        Tensor *m_i = m[i];
        Tensor *v_i = v[i];

        size_t size = param->size();

        dim3 blockSize(256);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x);

        adam_update_kernel<<<gridSize, blockSize>>>(
            param->getData(), grad->getData(),
            m_i->getData(), v_i->getData(),
            beta1, beta2, beta1_t, beta2_t,
            learningRate, epsilon, weight_decay,
            size);
    }
}