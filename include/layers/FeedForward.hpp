#ifndef FEEDFORWARD_HPP
#define FEEDFORWARD_HPP

#include "activations/GELU.hpp" // <-- Cambiar ReLU por GELU
#include "activations/ReLU.hpp" // O GELU si decides implementarla más adelante
#include "layers/Dense.hpp"
#include "layers/Layer.hpp"
#include <vector>

/**
 * @class FeedForward
 * @brief Implementa la red Feed-Forward (también conocida como MLP) dentro de un bloque Transformer.
 *
 * Consiste en dos capas lineales con una función de activación no lineal en medio.
 * La secuencia de operaciones es: Dense -> ReLU -> Dense.
 * Procesa cada token de la secuencia de forma independiente.
 */
class FeedForward : public Layer {
public:
  /**
   * @brief Constructor de la red Feed-Forward.
   * @param embedding_dim La dimensión de entrada y salida de la capa (ej. 128).
   * @param hidden_dim La dimensión de la capa oculta interna (típicamente 4 * embedding_dim).
   */
  FeedForward(size_t embedding_dim, size_t hidden_dim);

  /**
   * @brief Realiza el paso hacia adelante.
   * @param input Tensor de entrada de forma {batch, tokens, embedding_dim}.
   * @param isTraining Booleano que indica el modo de entrenamiento.
   * @return Tensor de salida con la misma forma que la entrada.
   * @override
   */
  Tensor forward(const Tensor &input, bool isTraining) override;

  /**
   * @brief Realiza el paso hacia atrás.
   * @param outputGradient Gradiente que viene de la siguiente operación.
   * @return Gradiente con respecto a la entrada de esta capa.
   * @override
   */
  Tensor backward(const Tensor &outputGradient) override;

  /**
   * @brief Recolecta y devuelve los parámetros de las capas Dense internas.
   * @override
   */
  std::vector<Tensor *> getParameters() override;

  /**
   * @brief Recolecta y devuelve los gradientes de las capas Dense internas.
   * @override
   */
  std::vector<Tensor *> getGradients() override;

  /**
   * @brief Devuelve el nombre de la capa.
   * @override
   */
  std::string getName() const override { return "FeedForward"; }

private:
  // Las capas que componen esta red.
  // No usamos punteros aquí porque las capas son parte integral de este objeto.
  Dense dense1;
  GELU activation;
  Dense dense2;
};

#endif // FEEDFORWARD_HPP
