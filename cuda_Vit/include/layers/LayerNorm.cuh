#pragma once

#include "layers/Layer.cuh"

/**
 * @class LayerNorm
 * @brief Implementa la Normalización de Capa (Layer Normalization).
 *
 * Normaliza las activaciones a lo largo de la dimensión de características para cada
 * muestra de datos de forma independiente. Esto ayuda a estabilizar el entrenamiento
 * en redes profundas, especialmente en Transformers.
 *
 * La operación es: y = gamma * (x - mean(x)) / sqrt(var(x) + epsilon) + beta
 */
class LayerNorm : public Layer
{
public:
  /**
   * @brief Constructor de la capa LayerNorm.
   * @param featureSize El tamaño de la última dimensión (la de características) que se va a normalizar.
   * @param epsilon Un pequeño valor para evitar la división por cero en el cálculo de la varianza.
   */
  LayerNorm(size_t featureSize, float epsilon = 1e-5f);

  /**
   * @brief Realiza el paso hacia adelante de la normalización.
   * @param input Tensor de entrada, típicamente de forma {batch_size, ..., feature_size}.
   * @param isTraining Si es true, almacena los valores intermedios para el backward pass.
   * @return Tensor normalizado, con la misma forma que el de entrada.
   * @override
   */
  Tensor forward(const Tensor &input, bool isTraining) override;

  /**
   * @brief Realiza el paso hacia atrás, calculando los gradientes para gamma, beta y la entrada.
   * @param outputGradient Gradiente de la pérdida respecto a la salida de esta capa (dE/dY).
   * @return Gradiente de la pérdida respecto a la entrada de esta capa (dE/dX).
   * @override
   */
  Tensor backward(const Tensor &outputGradient) override;

  /**
   * @brief Devuelve punteros a los parámetros entrenables (gamma y beta).
   * @return Un vector con punteros a `gamma` y `beta`.
   * @override
   */
  std::vector<Tensor *> getParameters() override;

  /**
   * @brief Devuelve punteros a los gradientes de los parámetros.
   * @return Un vector con punteros a `gammaGradient` y `betaGradient`.
   * @override
   */
  std::vector<Tensor *> getGradients() override;

  /**
   * @brief Devuelve el nombre de la capa.
   * @return El string "LayerNorm".
   * @override
   */
  std::string getName() const override { return "LayerNorm"; }

private:
  size_t featureSize;
  float epsilon;

  // Parámetros entrenables
  Tensor gamma; ///< Parámetro de escala, forma {1, feature_size}.
  Tensor beta;  ///< Parámetro de desplazamiento, forma {1, feature_size}.

  // Gradientes correspondientes
  Tensor gammaGradient;
  Tensor betaGradient;

  // Estado necesario para el backward pass
  Tensor inputTensor;     ///< Copia de la entrada del forward pass.
  Tensor mean;            ///< Media calculada por cada muestra, forma {batch_size, 1}.
  Tensor variance;        ///< Varianza calculada por cada muestra, forma {batch_size, 1}.
  Tensor normalizedInput; ///< Entrada normalizada antes de gamma/beta, (x - mu) / sqrt(var + eps).
};