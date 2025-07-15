#ifndef SGD_HPP
#define SGD_HPP

#include "optimizers/Optimizer.hpp"

/**
 * @class SGD
 * @brief Implementa el algoritmo de optimización de Descenso de Gradiente Estocástico (SGD).
 *
 * SGD es el algoritmo de optimización más fundamental. Actualiza cada parámetro
 * de la red moviéndolo una pequeña cantidad en la dirección opuesta a su
 * gradiente. La regla de actualización es:
 *   `param = param - learning_rate * gradiente`
 *
 * Aunque existen optimizadores más avanzados (como Adam), SGD (a menudo con
 * momento) sigue siendo una base sólida y muy utilizada.
 */
class SGD : public Optimizer {
public:
  /**
   * @brief Constructor para el optimizador SGD.
   * @param learningRate La tasa de aprendizaje (hiperparámetro). Un valor
   *        común para empezar es 0.01.
   */
  explicit SGD(float learningRate = 0.01f);

  /**
   * @brief Realiza un único paso de actualización de SGD.
   * @details Itera sobre todos los parámetros y sus gradientes correspondientes
   *          y aplica la regla de actualización de SGD.
   * @param parameters Vector de punteros a los parámetros entrenables del modelo.
   * @param gradients Vector de punteros a los gradientes correspondientes.
   * @override
   */
  void update(std::vector<Tensor *> &parameters, const std::vector<Tensor *> &gradients) override;
};

#endif // SGD_HPP
