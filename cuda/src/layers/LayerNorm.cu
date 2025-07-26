#include "layers/LayerNorm.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

LayerNorm::LayerNorm(size_t featureSize, float epsilon)
    : featureSize(featureSize), epsilon(epsilon)
{

    gamma = Tensor({1, featureSize});
    gamma.fill(1.0f);
    beta = Tensor({1, featureSize});
    beta.fill(0.0f);

    gammaGradient = Tensor({1, featureSize});
    betaGradient = Tensor({1, featureSize});
}

__global__ void layernorm_forward_kernel(
    const float *input, float *output,
    float *mean, float *variance, float *normalized,
    const float *gamma, const float *beta,
    size_t batchSize, size_t featureSize,
    float epsilon, bool isTraining)
{

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batchSize)
        return;

    const float *inputRow = input + i * featureSize;
    float *outputRow = output + i * featureSize;
    float *normRow = normalized + i * featureSize;

    // --- 1. Calcular la media ---
    float mu = 0.0f;
    for (int j = 0; j < featureSize; ++j)
    {
        mu += inputRow[j];
    }
    mu /= featureSize;
    if (isTraining)
        mean[i] = mu;

    // --- 2. Calcular la varianza ---
    float var = 0.0f;
    for (int j = 0; j < featureSize; ++j)
    {
        float diff = inputRow[j] - mu;
        var += diff * diff;
    }
    var /= featureSize;

    // --- 3. Calcular la desviación estándar inversa ---
    float inv_stddev = rsqrtf(var + epsilon);

    if (isTraining)
        variance[i] = inv_stddev;

    // --- 4. Normalizar + aplicar gamma y beta ---
    for (int j = 0; j < featureSize; ++j)
    {
        float x_hat = (inputRow[j] - mu) * inv_stddev;
        if (isTraining)
            normRow[j] = x_hat;
        outputRow[j] = gamma[j] * x_hat + beta[j];
    }
}

Tensor LayerNorm::forward(const Tensor &input, bool isTraining)
{
    const auto *inputShape = input.getShapeHost();
    size_t ndim = input.dims();
    if (inputShape[ndim - 1] != featureSize)
        throw std::runtime_error("Input last dim must match featureSize in LayerNorm");

    size_t batchSize = input.size() / featureSize;

    size_t newShape[2] = {batchSize, featureSize};
    Tensor input2D = input.reshape(newShape, 2);

    if (isTraining)
    {
        inputTensor = input2D;
        mean = Tensor({batchSize, 1});
        variance = Tensor({batchSize, 1});
    }

    Tensor output2D({batchSize, featureSize});
    normalizedInput = Tensor({batchSize, featureSize});

    int threads = 256;
    int blocks = (batchSize + threads - 1) / threads;
    layernorm_forward_kernel<<<blocks, threads>>>(
        input2D.getData(), output2D.getData(),
        isTraining ? mean.getData() : nullptr,
        isTraining ? variance.getData() : nullptr,
        isTraining ? normalizedInput.getData() : nullptr,
        gamma.getData(), beta.getData(),
        batchSize, featureSize, epsilon, isTraining);

    return output2D.reshape(input.getShapeHost(), input.dims());
}

__global__ void layernorm_backward_kernel(
    const float *gradOut,    // dL/dY
    const float *normalized, // X_hat
    const float *variance,   // inv_stddev (ya precomputado en forward)
    const float *gamma,      // γ
    float *gradInput,        // dL/dX
    float *gradGamma,        // dL/dγ
    float *gradBeta,         // dL/dβ
    size_t batchSize,
    size_t featureSize)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batchSize)
        return;

    const float *gradOutRow = gradOut + i * featureSize;
    const float *normRow = normalized + i * featureSize;
    float *gradInRow = gradInput + i * featureSize;

    float inv_stddev = variance[i]; // Ya contiene 1 / sqrt(var + eps)

    float dL_dXhat_sum = 0.0f;
    float dL_dXhat_dot_Xhat_sum = 0.0f;

    // --- 1. Sumas necesarias para gradInput ---
    for (int j = 0; j < featureSize; ++j)
    {
        float dY = gradOutRow[j];
        float xhat = normRow[j];
        float dXhat = dY * gamma[j];

        dL_dXhat_sum += dXhat;
        dL_dXhat_dot_Xhat_sum += dXhat * xhat;

        // --- 2. Acumulamos gradientes de gamma y beta ---
        atomicAdd(&gradGamma[j], dY * xhat);
        atomicAdd(&gradBeta[j], dY);
    }

    // --- 3. Gradiente de la entrada ---
    for (int j = 0; j < featureSize; ++j)
    {
        float dY = gradOutRow[j];
        float xhat = normRow[j];
        float dXhat = dY * gamma[j];

        float term1 = featureSize * dXhat;
        float term2 = dL_dXhat_sum;
        float term3 = xhat * dL_dXhat_dot_Xhat_sum;

        gradInRow[j] = (1.0f / featureSize) * inv_stddev * (term1 - term2 - term3);
    }
}

Tensor LayerNorm::backward(const Tensor &outputGradient)
{
    const auto *gradShape = outputGradient.getShapeHost();
    size_t ndim = outputGradient.dims();
    size_t batchSize = outputGradient.size() / featureSize;

    size_t newShape[2] = {batchSize, featureSize};
    Tensor grad2D = outputGradient.reshape(newShape, 2);

    gammaGradient.fill(0.0f);
    betaGradient.fill(0.0f);
    Tensor inputGrad({batchSize, featureSize});

    int threads = 256;
    int blocks = (batchSize + threads - 1) / threads;

    layernorm_backward_kernel<<<blocks, threads>>>(
        grad2D.getData(),
        normalizedInput.getData(),
        variance.getData(),
        gamma.getData(),
        inputGrad.getData(),
        gammaGradient.getData(),
        betaGradient.getData(),
        batchSize,
        featureSize);

    return inputGrad.reshape(gradShape, ndim);
}

std::vector<Tensor *> LayerNorm::getParameters()
{
    return {&gamma, &beta};
}

std::vector<Tensor *> LayerNorm::getGradients()
{
    return {&gammaGradient, &betaGradient};
}
