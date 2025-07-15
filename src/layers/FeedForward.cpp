#include "layers/FeedForward.hpp"

/**
 * @brief Constructor que inicializa las sub-capas.
 */
FeedForward::FeedForward(size_t embedding_dim, size_t hidden_dim)
    : dense1(embedding_dim, hidden_dim), // Entrada: D, Salida: H
      activation(),                      // Sin parámetros
      dense2(hidden_dim, embedding_dim)  // Entrada: H, Salida: D
{
  // El cuerpo del constructor está vacío; la inicialización se hace en la lista de inicializadores.
}

/**
 * @brief Encadena el forward pass de las sub-capas.
 */
Tensor FeedForward::forward(const Tensor &input, bool isTraining) {
  Tensor x = dense1.forward(input, isTraining);
  x = activation.forward(x, isTraining);
  x = dense2.forward(x, isTraining);
  return x;
}

/**
 * @brief Encadena el backward pass de las sub-capas en orden inverso.
 */
Tensor FeedForward::backward(const Tensor &outputGradient) {
  Tensor grad = dense2.backward(outputGradient);
  grad = activation.backward(grad);
  grad = dense1.backward(grad);
  return grad;
}

/**
 * @brief Recolecta los parámetros de las dos capas Dense.
 */
std::vector<Tensor *> FeedForward::getParameters() {
  auto params1 = dense1.getParameters();
  auto params2 = dense2.getParameters();
  // Concatenamos los vectores de parámetros
  params1.insert(params1.end(), params2.begin(), params2.end());
  return params1;
}

/**
 * @brief Recolecta los gradientes de las dos capas Dense.
 */
std::vector<Tensor *> FeedForward::getGradients() {
  auto grads1 = dense1.getGradients();
  auto grads2 = dense2.getGradients();
  // Concatenamos los vectores de gradientes
  grads1.insert(grads1.end(), grads2.begin(), grads2.end());
  return grads1;
}
