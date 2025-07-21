#include "layers/FeedForward.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

FeedForward::FeedForward(size_t embedding_dim, size_t hidden_dim)
    : dense1(embedding_dim, hidden_dim),
      activation(),
      dense2(hidden_dim, embedding_dim)
{
    // Inicialización completada en la lista inicializadora
}

Tensor FeedForward::forward(const Tensor &input, bool isTraining)
{
    Tensor hidden = dense1.forward(input, isTraining);
    Tensor activated = activation.forward(hidden, isTraining);
    Tensor output = dense2.forward(activated, isTraining);
    return output;
}

Tensor FeedForward::backward(const Tensor &outputGradient)
{
    Tensor grad = dense2.backward(outputGradient);
    grad = activation.backward(grad);
    grad = dense1.backward(grad);
    return grad;
}

std::vector<Tensor *> FeedForward::getParameters()
{
    auto params1 = dense1.getParameters();
    auto params2 = dense2.getParameters();
    params1.insert(params1.end(), params2.begin(), params2.end());
    return params1;
}

std::vector<Tensor *> FeedForward::getGradients()
{
    auto grads1 = dense1.getGradients();
    auto grads2 = dense2.getGradients();
    grads1.insert(grads1.end(), grads2.begin(), grads2.end());
    return grads1;
}
