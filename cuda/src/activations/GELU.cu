#include "activations/GELU.cuh"
#include <cuda_runtime.h>
#include <cmath>

#define SQRT_2_OVER_PI 0.7978845608f

GELU::GELU() {}

__global__ void gelu_forward_kernel(const float *input, float *output, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;
    float x = input[idx];
    float x_cubed = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + 0.044715f * x_cubed);
    float tanh_inner = tanhf(inner);
    float result = 0.5f * x * (1.0f + tanh_inner);
    output[idx] = result;
}

Tensor GELU::forward(const Tensor &input, bool isTraining)
{
    if (isTraining)
        this->inputTensor = input;

    if (!input.isContiguous())
        throw std::runtime_error("GELU::forward solo implementado para tensores contiguos.");

    size_t size = input.size();
    Tensor output(input.getShapeHost(), input.dims());
    dim3 blockDim(256);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    cudaGetLastError();
    gelu_forward_kernel<<<gridDim, blockDim>>>(input.getData(), output.getData(), size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel error (forward): ") + cudaGetErrorString(err));
    return output;
}

__global__ void gelu_backward_kernel(const float *input, const float *grad_out, float *grad_in, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    float x = input[idx];
    float x_squared = x * x;
    float x_cubed = x_squared * x;

    float inner = SQRT_2_OVER_PI * (x + 0.044715f * x_cubed);
    float tanh_inner = tanhf(inner);
    float sech_squared = 1.0f - tanh_inner * tanh_inner;
    float d_inner_dx = SQRT_2_OVER_PI * (1.0f + 3.0f * 0.044715f * x_squared);
    float dGELU_dx = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech_squared * d_inner_dx;

    grad_in[idx] = grad_out[idx] * dGELU_dx;
}

Tensor GELU::backward(const Tensor &outputGradient)
{
    if (!inputTensor.isContiguous() || !outputGradient.isContiguous())
        throw std::runtime_error("GELU::backward solo implementado para tensores contiguos.");

    size_t size = inputTensor.size();
    Tensor gradInput(inputTensor.getShapeHost(), inputTensor.dims());

    dim3 blockDim(256);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    cudaGetLastError();
    gelu_backward_kernel<<<gridDim, blockDim>>>(
        inputTensor.getData(), outputGradient.getData(), gradInput.getData(), size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel error (backward): ") + cudaGetErrorString(err));

    return gradInput;
}