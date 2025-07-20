#include "activations/RELU.cuh"
#include <cuda_runtime.h>
#include <cmath>
ReLU::ReLU() {}

__global__ void relu_forward_kernel(const float *input, float *output, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    float x = input[idx];
    output[idx] = (x > 0.0f) ? x : 0.0f;
}

Tensor ReLU::forward(const Tensor &input, bool isTraining)
{
    if (isTraining)
        this->inputTensor = input;

    if (!input.isContiguous())
        throw std::runtime_error("ReLU::forward solo implementado para tensores contiguos.");

    size_t size = input.size();
    Tensor output(input.getShapeHost(), input.dims());

    dim3 blockDim(256);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);

    cudaGetLastError();
    relu_forward_kernel<<<gridDim, blockDim>>>(input.getData(), output.getData(), size);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel error (ReLU forward): ") + cudaGetErrorString(err));

    return output;
}

__global__ void relu_backward_kernel(const float *input, const float *grad_out, float *grad_in, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    float x = input[idx];
    grad_in[idx] = (x > 0.0f) ? grad_out[idx] : 0.0f;
}

Tensor ReLU::backward(const Tensor &outputGradient)
{
    if (!inputTensor.isContiguous() || !outputGradient.isContiguous())
        throw std::runtime_error("ReLU::backward solo implementado para tensores contiguos.");

    size_t size = inputTensor.size();
    Tensor gradInput(inputTensor.getShapeHost(), inputTensor.dims());

    dim3 blockDim(256);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);

    cudaGetLastError();
    relu_backward_kernel<<<gridDim, blockDim>>>(
        inputTensor.getData(), outputGradient.getData(), gradInput.getData(), size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel error (ReLU backward): ") + cudaGetErrorString(err));

    return gradInput;
}