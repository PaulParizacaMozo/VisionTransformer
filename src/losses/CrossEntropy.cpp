#include "losses/CrossEntropy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Función auxiliar que calcula la activación Softmax de forma estable.
 * @details Softmax convierte un vector de números reales (logits) en una
 *          distribución de probabilidad. Se utiliza el "truco de restar el máximo"
 *          para prevenir el desbordamiento numérico (overflow) con valores grandes
 *          en la función exponencial.
 * @param logits Tensor de logits de forma {batch_size, num_classes}.
 * @return Un tensor de probabilidades con la misma forma.
 */
Tensor softmax(const Tensor &logits) {
  Tensor probabilities(logits.getShape());
  const size_t batchSize = logits.getShape()[0];
  const size_t numClasses = logits.getShape()[1];

#pragma omp parallel for
  for (size_t i = 0; i < batchSize; ++i) {
    // 1. Encontrar el logit máximo en la fila para la estabilidad numérica.
    float maxLogit = -std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < numClasses; ++j) {
      if (logits(i, j) > maxLogit) {
        maxLogit = logits(i, j);
      }
    }

    // 2. Calcular los exponenciales y su suma.
    float sumExp = 0.0f;
    for (size_t j = 0; j < numClasses; ++j) {
      // Restar maxLogit previene que exp() devuelva 'inf'.
      float expVal = std::exp(logits(i, j) - maxLogit);
      probabilities(i, j) = expVal;
      sumExp += expVal;
    }

    // 3. Normalizar para obtener las probabilidades.
    for (size_t j = 0; j < numClasses; ++j) {
      probabilities(i, j) /= sumExp;
    }
  }
  return probabilities;
}

/**
 * @brief Calcula la pérdida de entropía cruzada para un batch.
 */
float CrossEntropy::calculate(const Tensor &yPred, const Tensor &yTrue) {
  if (yPred.getShape() != yTrue.getShape()) {
    throw std::runtime_error("Las formas de predicción y etiquetas verdaderas no coinciden.");
  }

  // 1. Convertir los logits de salida de la red en probabilidades.
  //    Se guarda el resultado para reutilizarlo en el backward pass.
  this->softmaxOutput = softmax(yPred);

  // 2. Calcular la pérdida de entropía cruzada.
  const size_t batchSize = yPred.getShape()[0];
  const size_t numClasses = yPred.getShape()[1];
  float totalLoss = 0.0f;
  const float epsilon = 1e-12; // Pequeño valor para evitar log(0).

#pragma omp parallel for reduction(+ : totalLoss)
  for (size_t i = 0; i < batchSize; ++i) {
    for (size_t j = 0; j < numClasses; ++j) {
      // La pérdida se calcula solo para la clase correcta (donde yTrue es 1).
      if (yTrue(i, j) == 1.0f) {
        totalLoss += -std::log(this->softmaxOutput(i, j) + epsilon);
      }
    }
  }

  return totalLoss / batchSize; // Devolver la pérdida promedio por muestra.
}

/**
 * @brief Calcula el gradiente de (Softmax + CrossEntropy) respecto a los logits.
 */
Tensor CrossEntropy::backward(const Tensor & /*yPred*/, const Tensor &yTrue) {
  // El gradiente combinado de Softmax(yPred) y CrossEntropy es simplemente:
  //   gradiente = softmax(yPred) - yTrue
  // Esta simplificación es la razón principal para combinar ambas funciones.

  // Reutilizamos softmaxOutput calculado en `calculate`.
  // Es importante notar que no se modifica `softmaxOutput` directamente,
  // sino que se crea una copia que será el gradiente.
  Tensor gradient = this->softmaxOutput;

  const size_t batchSize = yTrue.getShape()[0];
  const size_t numClasses = yTrue.getShape()[1];

#pragma omp parallel for
  for (size_t i = 0; i < batchSize; ++i) {
    for (size_t j = 0; j < numClasses; ++j) {
      // Calcula `(probabilidad - etiqueta)` y normaliza por el tamaño del batch.
      // La normalización aquí asegura que la magnitud del gradiente no dependa
      // del tamaño del batch, lo que estabiliza el entrenamiento con diferentes tamaños de batch.
      gradient(i, j) = (gradient(i, j) - yTrue(i, j)) / batchSize;
    }
  }

  return gradient;
}
