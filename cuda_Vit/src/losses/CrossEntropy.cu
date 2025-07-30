#include "losses/CrossEntropy.cuh"
#include "utils/CudaUtils.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

/**
 * @brief Calcula la pérdida de entropía cruzada para un batch.
 */
float CrossEntropy::calculate(const Tensor &yPred, const Tensor &yTrue)
{
  if (yPred.getShape() != yTrue.getShape())
  {
    throw std::runtime_error("Las formas de predicción y etiquetas verdaderas no coinciden.");
  }

  // 1. Convertir los logits de salida de la red en probabilidades.
  //    Se guarda el resultado para reutilizarlo en el backward pass.
  // this->softmaxOutput = softmax(yPred);
  this->softmaxOutput = softmax_cuda(yPred);
  return cross_entropy_cuda(this->softmaxOutput, yTrue, this->class_weights);
}

/**
 * @brief Calcula el gradiente de (Softmax + CrossEntropy) respecto a los logits.
 */
Tensor CrossEntropy::backward(const Tensor & /*yPred*/, const Tensor &yTrue)
{
  return ce_backward_cuda(this->softmaxOutput, yTrue);
}
