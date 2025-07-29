#ifndef RELU_HPP
#define RELU_HPP

#include "layers/Layer.hpp"

/**
 * @class ReLU
 * @brief Implementa la función de activación Rectified Linear Unit (ReLU).
 *
 * ReLU es una de las funciones de activación más utilizadas. Realiza una
 * operación simple, no lineal y elemento a elemento:
 *   `f(x) = max(0, x)`
 *
 * No tiene parámetros entrenables. Como otras activaciones, se implementa como
 * una capa para que pueda ser integrada fácilmente en un modelo secuencial.
 */
class ReLU : public Layer {
public:
  /** @brief Constructor por defecto. */
  ReLU();

  /**
   * @brief Aplica la función ReLU elemento a elemento.
   * @details Calcula `max(0, x)` para cada elemento de la entrada. Si se está
   *          entrenando, almacena la entrada para el backward pass.
   * @param input El tensor de entrada.
   * @param isTraining Booleano que indica si se debe guardar la entrada.
   * @return Un nuevo tensor con la función ReLU aplicada.
   * @override
   */
  Tensor forward(const Tensor &input, bool isTraining) override;

  /**
   * @brief Realiza el paso hacia atrás para ReLU.
   * @details La derivada de ReLU es 1 si x > 0, y 0 si x <= 0. El gradiente
   *          de entrada se calcula multiplicando el gradiente de salida por
   *          esta derivada.
   * @param outputGradient El gradiente de la pérdida respecto a la salida de ReLU.
   * @return El gradiente de la pérdida respecto a la entrada de ReLU.
   * @override
   */
  Tensor backward(const Tensor &outputGradient) override;

  /**
   * @brief Devuelve el nombre de la capa.
   * @return El string "ReLU".
   * @override
   */
  std::string getName() const override { return "ReLU"; }

private:
  /**
   * @brief Almacena el tensor de entrada del forward pass.
   *        Es necesario para saber qué neuronas estaban activas (x > 0) y
   *        así calcular correctamente el gradiente en el backward pass.
   */
  Tensor inputTensor;
};

#endif // RELU_HPP
