#ifndef CROSSENTROPY_HPP
#define CROSSENTROPY_HPP

#include "losses/Loss.hpp"

// La declaración de la función softmax
Tensor softmax(const Tensor &logits);

/**
 * @class CrossEntropy
 * @brief Implementa la función de pérdida de Entropía Cruzada Categórica.
 *
 * Esta clase combina la función de activación Softmax y la pérdida de Entropía
 * Cruzada en una sola operación. Este enfoque es numéricamente más estable y
 * computacionalmente más eficiente que tener una capa Softmax separada seguida
 * de una capa de pérdida de Entropía Cruzada.
 *
 * La magia de esta combinación reside en el cálculo del gradiente, que se
 * simplifica a una simple resta: `(softmax_output - true_labels)`.
 */
class CrossEntropy : public Loss
{
public:
  /** @brief Constructor por defecto. */
  CrossEntropy() = default;

  /**
   * @brief Calcula la pérdida de entropía cruzada.
   * @details Primero aplica la función Softmax a las predicciones (logits) para
   *          obtener probabilidades, y luego calcula la pérdida de entropía cruzada.
   * @param yPred Las predicciones del modelo (logits), de forma {batch, num_classes}.
   * @param yTrue Las etiquetas verdaderas (one-hot), de forma {batch, num_classes}.
   * @return El valor escalar de la pérdida promedio del batch.
   * @override
   */
  float calculate(const Tensor &yPred, const Tensor &yTrue) override;

  /**
   * @brief Calcula el gradiente inicial para la retropropagación.
   * @details Aprovecha la simplificación matemática de combinar Softmax y CrossEntropy.
   *          El gradiente es `(probabilidades - etiquetas_verdaderas)`.
   * @param yPred Ignorado, ya que las probabilidades se calcularon en `calculate()`.
   * @param yTrue Las etiquetas verdaderas (one-hot).
   * @return El tensor de gradiente, que se pasará a la última capa de la red.
   * @override
   */
  Tensor backward(const Tensor &yPred, const Tensor &yTrue) override;

private:
  /**
   * @brief Almacena las probabilidades calculadas por Softmax en `calculate()`.
   *        Se reutiliza en `backward()` para evitar recalcular Softmax.
   */
  Tensor softmaxOutput;
};

#endif // CROSSENTROPY_HPP
