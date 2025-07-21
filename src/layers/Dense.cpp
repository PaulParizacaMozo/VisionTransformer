#include "layers/Dense.hpp"
#include "core/Tensor.hpp" // Para matrixMultiply y otras operaciones
#include <stdexcept>

Dense::Dense(size_t inputSize, size_t outputSize) {
  // float stddev = std::sqrt(2.0f / static_cast<float>(inputSize));
  float stddev = 0.02f;
  this->weights = Tensor({inputSize, outputSize});
  this->weights.randomizeNormal(0.0f, stddev);

  // El bias sigue siendo un vector de {1, outputSize} para el broadcasting.
  // El bias de una capa densa no es 1D, es 2D {1, N} para broadcasting
  // en tensores de entrada {B, M}.
  this->bias = Tensor({1, outputSize});
  this->bias.fill(0.0f);

  this->weightGradients = Tensor({inputSize, outputSize});
  this->biasGradients = Tensor({1, outputSize});
}

/**
 * @brief Realiza el paso hacia adelante: Y = X * W + b.
 * @details Ahora soporta entradas 2D {batch, features_in} y 3D {batch, tokens, features_in}.
 */
Tensor Dense::forward(const Tensor &input, bool isTraining) {
  if (isTraining) {
    this->inputTensor = input; // Guardamos la entrada con su forma original
  }

  const auto &inputShape = input.getShape();
  size_t inputRank = inputShape.size();

  // Si la entrada es 3D, la aplanamos temporalmente a 2D para la multiplicación
  if (inputRank == 3) {
    size_t batchSize = inputShape[0];
    size_t numTokens = inputShape[1];
    size_t featuresIn = inputShape[2];
    Tensor input2D = input.reshape({batchSize * numTokens, featuresIn});

    // Y' = X_2D * W
    Tensor output2D = matrixMultiply(input2D, this->weights);
    // Y = Y' + b
    output2D.addBroadcast(this->bias);

    // Devolvemos la salida con su forma 3D original
    return output2D.reshape({batchSize, numTokens, this->bias.getShape()[1]});
  }

  // Si la entrada ya es 2D, procedemos como antes
  if (inputRank == 2) {
    Tensor output = matrixMultiply(input, this->weights);
    output.addBroadcast(this->bias);
    return output;
  }

  throw std::runtime_error("Dense::forward solo soporta entradas 2D o 3D.");
}

/**
 * @brief Realiza la retropropagación a través de la capa.
 */
Tensor Dense::backward(const Tensor &outputGradient) {
  const auto &inputShape = this->inputTensor.getShape();
  size_t inputRank = inputShape.size();

  Tensor grad_to_process = outputGradient;
  Tensor input_to_process = this->inputTensor;

  // Si la entrada original era 3D, aplanamos tanto la entrada guardada como el gradiente
  if (inputRank == 3) {
    size_t batchSize = inputShape[0];
    size_t numTokens = inputShape[1];
    size_t featuresIn = inputShape[2];
    size_t featuresOut = outputGradient.getShape()[2];

    // --- DEFENSA ---
    // Aseguramos que los tensores son contiguos antes de remodelar

    if (!grad_to_process.isContiguous()) {
      grad_to_process = grad_to_process.contiguous();
    }

    if (!input_to_process.isContiguous()) {
      input_to_process = input_to_process.contiguous();
    }

    grad_to_process = grad_to_process.reshape({batchSize * numTokens, featuresOut});
    input_to_process = input_to_process.reshape({batchSize * numTokens, featuresIn});
  }

  // --- Los cálculos de gradientes ahora se hacen siempre en 2D ---
  Tensor inputTransposed = input_to_process.transpose(0, 1);
  this->weightGradients = matrixMultiply(inputTransposed, grad_to_process);

  this->biasGradients = grad_to_process.sum(0);

  Tensor weightsTransposed = this->weights.transpose(0, 1);
  Tensor inputGradient2D = matrixMultiply(grad_to_process, weightsTransposed);

  // Si la entrada original era 3D, remodelamos el gradiente de salida a 3D
  if (inputRank == 3) {
    return inputGradient2D.reshape(inputShape);
  }

  return inputGradient2D;
}

std::vector<Tensor *> Dense::getParameters() { return {&this->weights, &this->bias}; }

std::vector<Tensor *> Dense::getGradients() { return {&this->weightGradients, &this->biasGradients}; }
