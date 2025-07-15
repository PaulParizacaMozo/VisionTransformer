#ifndef ADAM_HPP
#define ADAM_HPP

#include "optimizers/Optimizer.hpp"
#include <vector>

// Implementa el optimizador Adam (Adaptive Moment Estimation).
// Adam combina las ideas de Momentum (primer momento) y RMSprop (segundo momento)
// para adaptar la tasa de aprendizaje para cada parametro individual.
class Adam : public Optimizer {
public:
  // Constructor para el optimizador Adam.
  Adam(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8, float weight_decay = 0.0f);

  // Realiza un unico paso de actualizacion de Adam.
  void update(std::vector<Tensor *> &parameters, const std::vector<Tensor *> &gradients) override;

private:
  // Hiperparametros de Adam.
  float beta1;
  float beta2;
  float epsilon;
  float weight_decay; // Termino de regularizacion L2 (Weight Decay).
  long long t;        // Contador de pasos de tiempo para la correccion de sesgo.

  // Estado del optimizador (se inicializan en la primera llamada a update).
  std::vector<Tensor> m; // Estimacion del primer momento (media de gradientes).
  std::vector<Tensor> v; // Estimacion del segundo momento (media de gradientes^2).

  // Flag para la inicializacion diferida de los tensores de momento.
  bool initialized;
};

#endif // ADAM_HPP
