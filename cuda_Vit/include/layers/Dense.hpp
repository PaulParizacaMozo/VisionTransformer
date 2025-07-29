#ifndef DENSE_HPP
#define DENSE_HPP

#include "layers/Layer.hpp"
#include <cmath>

/**
 * @class Dense
 * @brief Una capa totalmente conectada (fully connected layer).
 *
 * Esta es una de las capas más comunes en las redes neuronales. Realiza una
 * transformación afín de la entrada, calculando la operación:
 *   `output = input * weights + bias`
 * donde `weights` y `bias` son los parámetros entrenables de la capa.
 *
 * La entrada debe ser un tensor 2D de forma {batch_size, input_size}, y la
 * salida será un tensor 2D de forma {batch_size, output_size}.
 */
class Dense : public Layer {
public:
  /**
   * @brief Constructor de la capa Dense.
   * @param inputSize El número de características de entrada (columnas del tensor de entrada).
   * @param outputSize El número de neuronas en la capa (columnas del tensor de salida).
   */
  Dense(size_t inputSize, size_t outputSize);

  /**
   * @brief Realiza el paso hacia adelante: `output = input * weights + bias`.
   * @param input Tensor de entrada de forma {batch_size, input_size}.
   * @param isTraining Si es `true`, almacena la entrada para el backward pass.
   * @return Tensor de salida de forma {batch_size, output_size}.
   * @override
   */
  Tensor forward(const Tensor &input, bool isTraining) override;

  /**
   * @brief Realiza el paso hacia atrás, calculando los gradientes.
   * @details Calcula tres gradientes:
   *          1. Gradiente de los pesos (dE/dW).
   *          2. Gradiente del bias (dE/db).
   *          3. Gradiente de la entrada (dE/dX), que se devuelve para la capa anterior.
   * @param outputGradient Gradiente de la pérdida respecto a la salida de esta capa (dE/dY).
   * @return Gradiente de la pérdida respecto a la entrada de esta capa (dE/dX).
   * @override
   */
  Tensor backward(const Tensor &outputGradient) override;

  /**
   * @brief Devuelve punteros a los parámetros entrenables (pesos y bias).
   * @return Un vector con punteros a `weights` y `bias`.
   * @override
   */
  std::vector<Tensor *> getParameters() override;

  /**
   * @brief Devuelve punteros a los gradientes de los parámetros.
   * @return Un vector con punteros a `weightGradients` y `biasGradients`.
   * @override
   */
  std::vector<Tensor *> getGradients() override;

  /**
   * @brief Devuelve el nombre de la capa.
   * @return El string "Dense".
   * @override
   */
  std::string getName() const override { return "Dense"; }

private:
  // Parámetros entrenables
  Tensor weights; ///< Matriz de pesos de la capa, de forma {input_size, output_size}.
  Tensor bias;    ///< Vector de bias de la capa, de forma {1, output_size}.

  // Gradientes correspondientes a los parámetros
  Tensor weightGradients; ///< Almacena dE/dW.
  Tensor biasGradients;   ///< Almacena dE/db.

  // Estado necesario para el backward pass
  Tensor inputTensor; ///< Copia de la entrada del forward pass, necesaria para calcular gradientes.
};

#endif // DENSE_HPP
