#ifndef LOSS_HPP
#define LOSS_HPP

#include "core/Tensor.cuh"

/**
 * @class Loss
 * @brief Clase base abstracta para todas las funciones de pérdida (costo).
 *
 * Define la interfaz que debe implementar cualquier función de pérdida, como
 * la Entropía Cruzada Categórica (Categorical Cross-Entropy) o el Error
 * Cuadrático Medio (Mean Squared Error).
 *
 * Una función de pérdida tiene dos responsabilidades principales:
 * 1. Cuantificar qué tan "mal" están las predicciones del modelo en comparación
 *    con las etiquetas verdaderas.
 * 2. Calcular el gradiente inicial que se retropropagará a través de la red.
 */
class Loss
{
public:
  /** @brief Destructor virtual para permitir la destrucción polimórfica. */
  virtual ~Loss() = default;

  /**
   * @brief Calcula y devuelve el valor escalar de la pérdida.
   * @details Compara las predicciones del modelo con las etiquetas reales para
   * producir un único número que representa el error total del batch.
   * @param yPred El tensor de salida de la última capa del modelo (predicciones).
   *              Generalmente son los "logits" antes de la activación final (ej. Softmax).
   * @param yTrue El tensor con las etiquetas verdaderas (ground truth).
   *              A menudo en formato one-hot para problemas de clasificación.
   * @return Un valor flotante que representa la pérdida promedio del batch.
   */
  virtual float calculate(const Tensor &yPred, const Tensor &yTrue) = 0;

  /**
   * @brief Calcula el gradiente de la pérdida con respecto a las predicciones del modelo.
   * @details Este es el primer paso de la retropropagación. El tensor devuelto
   * (dE/dY_pred) se pasará como `outputGradient` al método `backward()` de la
   * última capa de la red.
   * @param yPred El tensor de predicciones del modelo.
   * @param yTrue El tensor de etiquetas verdaderas.
   * @return Un tensor con la misma forma que `yPred`, conteniendo el gradiente
   *         inicial de la retropropagación.
   */
  virtual Tensor backward(const Tensor &yPred, const Tensor &yTrue) = 0;
};

#endif // LOSS_HPP
