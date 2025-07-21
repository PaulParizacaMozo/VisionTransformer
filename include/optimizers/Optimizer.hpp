#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "core/Tensor.hpp"
#include <vector>

/**
 * @class Optimizer
 * @brief Clase base abstracta para todos los algoritmos de optimización.
 *
 * Define la interfaz para los optimizadores como SGD (Descenso de Gradiente
 * Estocástico), Adam, etc. La principal responsabilidad de un optimizador es
 * tomar los gradientes calculados durante la retropropagación y utilizarlos
 * para actualizar los parámetros entrenables de la red (pesos y biases) con el
 * objetivo de minimizar la función de pérdida.
 */
class Optimizer {
public:
  /**
   * @brief Constructor que inicializa el optimizador con una tasa de aprendizaje.
   * @param learningRate La tasa de aprendizaje (learning rate), un hiperparámetro
   *        clave que controla el tamaño de los pasos de actualización.
   */
  explicit Optimizer(float learningRate) : learningRate(learningRate) {}

  /** @brief Destructor virtual para permitir la destrucción polimórfica. */
  virtual ~Optimizer() = default;

  /**
   * @brief Realiza un único paso de optimización para actualizar los parámetros.
   * @details Este es el método central de cualquier optimizador. Itera sobre los
   * parámetros y sus correspondientes gradientes para aplicar la regla de
   * actualización específica del algoritmo.
   * @param parameters Un vector de punteros a los tensores de los parámetros
   *        (ej. pesos y biases de todas las capas).
   * @param gradients Un vector de punteros a los tensores de los gradientes
   *        correspondientes a cada parámetro. El orden y tamaño deben coincidir
   *        con `parameters`.
   */
  virtual void update(std::vector<Tensor *> &parameters, const std::vector<Tensor *> &gradients) = 0;

  // ─── Nuevo setter ───
  void setLearningRate(float lr) { learningRate = lr; }
  float getLearningRate() const  { return learningRate; }

protected:
  /** @brief La tasa de aprendizaje (learning rate) para el algoritmo. */
  float learningRate;
};

#endif // OPTIMIZER_HPP
