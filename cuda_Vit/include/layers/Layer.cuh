#pragma once

#include "core/Tensor.hpp"
#include <string>
#include <vector>

/**
 * @class Layer
 * @brief Clase base abstracta para todas las capas de una red neuronal.
 *
 * Define la interfaz (el "contrato") que cada tipo de capa (ej. Dense, Conv2D,
 * ReLU) debe implementar. Esto permite que el modelo secuencial las trate de
 * manera polimórfica, sin necesidad de conocer su tipo concreto.
 */
class Layer
{
public:
  /** @brief Destructor virtual para permitir la destrucción polimórfica. */
  virtual ~Layer() = default;

  /**
   * @brief Realiza el paso hacia adelante (forward pass) de la capa.
   * @param input El tensor de entrada que proviene de la capa anterior o de los datos.
   * @param isTraining Booleano que indica si la red está en modo de entrenamiento.
   *        Esencial para capas como Dropout, que se comportan de manera diferente
   *        durante el entrenamiento y la inferencia.
   * @return Un tensor con el resultado de la operación de la capa.
   */
  virtual Tensor forward(const Tensor &input, bool isTraining) = 0;

  /**
   * @brief Realiza el paso hacia atrás (backward pass) o retropropagación.
   * @details Calcula el gradiente de la pérdida con respecto a la entrada de esta capa
   * (para pasarlo a la capa anterior) y calcula los gradientes de los parámetros
   * internos de la capa (si los tiene, ej. pesos y bias).
   * @param outputGradient El gradiente de la función de pérdida con respecto a la
   *        salida de esta capa (dE/dY).
   * @return El gradiente de la función de pérdida con respecto a la entrada de esta
   *         capa (dE/dX).
   */
  virtual Tensor backward(const Tensor &outputGradient) = 0;

  /**
   * @brief Devuelve un vector de punteros a los parámetros entrenables de la capa.
   * @details Las capas como Dense o Conv2D sobreescribirán este método para devolver
   * punteros a sus pesos y biases. Las capas sin parámetros (ej. Flatten, ReLU)
   * devuelven un vector vacío.
   * @return Un vector de punteros a los tensores de los parámetros.
   */
  virtual std::vector<Tensor *> getParameters() { return {}; }

  /**
   * @brief Devuelve un vector de punteros a los gradientes de los parámetros.
   * @details El orden y tamaño de este vector debe coincidir exactamente con el de
   * `getParameters()`. El optimizador usará estos gradientes para actualizar
   * los parámetros.
   * @return Un vector de punteros a los tensores de los gradientes.
   */
  virtual std::vector<Tensor *> getGradients() { return {}; }

  /**
   * @brief Devuelve el nombre de la capa.
   * @details Útil para imprimir resúmenes del modelo, depuración y serialización.
   * @return Un string con el nombre de la capa (ej. "Dense", "Conv2D").
   */
  virtual std::string getName() const = 0;
};