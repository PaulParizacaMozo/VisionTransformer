#include "losses/CrossEntropy.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

// Kernel para aplicar softmax fila por fila
__global__ void softmax_kernel(const float *logits, float *output, size_t batchSize, size_t numClasses)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batchSize)
        return;

    const float *row = logits + i * numClasses;
    float *outRow = output + i * numClasses;

    float maxLogit = -INFINITY;
    for (size_t j = 0; j < numClasses; ++j)
        maxLogit = fmaxf(maxLogit, row[j]);

    float sumExp = 0.0f;
    for (size_t j = 0; j < numClasses; ++j)
    {
        outRow[j] = expf(row[j] - maxLogit);
        sumExp += outRow[j];
    }

    for (size_t j = 0; j < numClasses; ++j)
        outRow[j] /= sumExp;
}

Tensor softmax(const Tensor &logits)
{
    Tensor probabilities(logits.getShapeHost(), logits.dims());

    size_t batchSize = logits.dim(0);
    size_t numClasses = logits.dim(1);

    int threads = 128;
    int blocks = (batchSize + threads - 1) / threads;

    softmax_kernel<<<blocks, threads>>>(logits.getData(), probabilities.getData(), batchSize, numClasses);

    return probabilities;
}

// Kernel para calcular la entropía cruzada
__global__ void cross_entropy_kernel(const float *softmax, const float *yTrue, float *lossPerSample, size_t numClasses, float epsilon)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gridDim.x * blockDim.x)
        return;

    float loss = 0.0f;
    for (size_t j = 0; j < numClasses; ++j)
    {
        size_t idx = i * numClasses + j;
        if (yTrue[idx] == 1.0f)
        {
            loss -= logf(softmax[idx] + epsilon);
        }
    }
    lossPerSample[i] = loss;
}

float CrossEntropy::calculate(const Tensor &yPred, const Tensor &yTrue)
{
    if (yPred.size() != yTrue.size())
        throw std::runtime_error("Tamaños incompatibles entre yPred y yTrue");

    this->softmaxOutput = softmax(yPred);

    size_t batchSize = yPred.dim(0);
    size_t numClasses = yPred.dim(1);

    Tensor lossTensor({batchSize});

    float epsilon = 1e-12f;
    int threads = 128;
    int blocks = (batchSize + threads - 1) / threads;

    cross_entropy_kernel<<<blocks, threads>>>(
        softmaxOutput.getData(), yTrue.getData(), lossTensor.getData(), numClasses, epsilon);

    // Transferir la pérdida a CPU para sumar
    std::vector<float> hostLoss(batchSize);
    cudaMemcpy(hostLoss.data(), lossTensor.getData(), batchSize * sizeof(float), cudaMemcpyDeviceToHost);

    float totalLoss = 0.0f;
    for (float v : hostLoss)
        totalLoss += v;

    return totalLoss / batchSize;
}

__global__ void backward_crossentropy_kernel(float *grad, const float *yTrue, size_t totalSize, float invBatchSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalSize)
        grad[i] = (grad[i] - yTrue[i]) * invBatchSize;
}

Tensor CrossEntropy::backward(const Tensor & /*yPred*/, const Tensor &yTrue)
{
    Tensor grad = this->softmaxOutput;
    size_t totalSize = grad.size();
    float invBatchSize = 1.0f / yTrue.dim(0);

    int threads = 256;
    int blocks = (totalSize + threads - 1) / threads;
    backward_crossentropy_kernel<<<blocks, threads>>>(grad.getData(), yTrue.getData(), totalSize, invBatchSize);

    return grad;
}
