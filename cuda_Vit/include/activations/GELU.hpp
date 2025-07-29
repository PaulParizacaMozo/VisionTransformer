#ifndef GELU_HPP
#define GELU_HPP

#include "layers/Layer.hpp"

/**
 * @class GELU
 * @brief Implementa la función de activación GELU (Gaussian Error Linear Unit).
 *
 * Utiliza una aproximación rápida y precisa, popularizada por modelos como GPT y BERT.
 * GELU es una alternativa suave a ReLU y a menudo ofrece un mejor rendimiento en Transformers.
 */
class GELU : public Layer {
public:
  GELU();

  /**
   * @brief Aplica la función de activación GELU.
   * @param input Tensor de entrada (puede ser 2D o 3D).
   * @param isTraining Booleano que indica el modo de entrenamiento.
   * @return Tensor de salida con la misma forma que la entrada.
   * @override
   */
  Tensor forward(const Tensor &input, bool isTraining) override;

  /**
   * @brief Calcula el gradiente de la capa GELU.
   * @param outputGradient Gradiente que viene de la siguiente capa.
   * @return Gradiente con respecto a la entrada de esta capa.
   * @override
   */
  Tensor backward(const Tensor &outputGradient) override;

  // GELU no tiene parámetros, por lo que getParameters y getGradients devuelven vacío.

  std::string getName() const override { return "GELU"; }

private:
  // Guardamos la entrada para calcular el gradiente en el backward pass
  Tensor inputTensor;
};

#endif // GELU_HPP
