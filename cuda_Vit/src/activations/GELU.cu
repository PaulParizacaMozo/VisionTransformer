#include "activations/GELU.cuh"
#include <cmath> // Para tanh, sqrt, pow
#include "utils/CudaUtils.cuh"

#include <iostream>

GELU::GELU() {}

Tensor GELU::forward(const Tensor &input, bool isTraining)
{
  if (isTraining)
  {
    this->inputTensor = input;
  }
  return gelu_forward_cuda(input);
}

Tensor GELU::backward(const Tensor &outputGradient)
{
  return gelu_backward_cuda(inputTensor, outputGradient);
}
