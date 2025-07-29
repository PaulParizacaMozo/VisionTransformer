#include "optimizers/Adam.hpp"
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

Adam::Adam(float learningRate, float beta1, float beta2, float epsilon, float weight_decay)
    : Optimizer(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), weight_decay(weight_decay), t(0),
      initialized(false) {}

void Adam::update(std::vector<Tensor *> &parameters, const std::vector<Tensor *> &gradients) {
  if (parameters.size() != gradients.size()) {
    throw std::runtime_error("El numero de parametros y gradientes no coincide en Adam::update.");
  }

  // Inicializacion diferida: crea los tensores de momento m y v en la primera llamada.
  if (!initialized) {
    m.reserve(parameters.size());
    v.reserve(parameters.size());
    for (const auto &param : parameters) {
      m.emplace_back(param->getShape());
      v.emplace_back(param->getShape());
    }
    initialized = true;
  }

  t++; // Incrementa el contador de pasos.

  // Correccion de sesgo (bias correction) pre-calculada.
  const float beta1_t = std::pow(beta1, t);
  const float beta2_t = std::pow(beta2, t);

  // Itera sobre cada par de parametro/gradiente.
  for (size_t i = 0; i < parameters.size(); ++i) {
    Tensor *param = parameters[i];
    const Tensor *grad_tensor = gradients[i];
    Tensor &m_i = m[i];
    Tensor &v_i = v[i];

    const auto &shape = param->getShape();

    // Actualizacion de parametros, con bucles especializados por dimensionalidad.
    if (shape.size() == 1 || shape.size() == 2 || shape.size() == 3) {
      if (shape.size() == 1) { // Para Bias, LayerNorm
#pragma omp parallel for
        for (size_t j = 0; j < shape[0]; ++j) {
          float g = (*grad_tensor)(j);
          // Añade el decaimiento de peso (L2 regularization) al gradiente.
          // No se suele aplicar a los biases ni a los parametros de LayerNorm.
          // (Aqui lo aplicamos a todos por simplicidad, se podria refinar).
          if (weight_decay > 0.0f) {
            g += weight_decay * (*param)(j);
          }
          m_i(j) = beta1 * m_i(j) + (1.0f - beta1) * g;
          v_i(j) = beta2 * v_i(j) + (1.0f - beta2) * (g * g);
          float m_hat = m_i(j) / (1.0f - beta1_t);
          float v_hat = v_i(j) / (1.0f - beta2_t);
          (*param)(j) -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
      } else if (shape.size() == 2) { // Para pesos de Dense
#pragma omp parallel for collapse(2)
        for (size_t r = 0; r < shape[0]; ++r) {
          for (size_t c = 0; c < shape[1]; ++c) {
            float g = (*grad_tensor)(r, c);
            if (weight_decay > 0.0f) {
              g += weight_decay * (*param)(r, c);
            }
            m_i(r, c) = beta1 * m_i(r, c) + (1.0f - beta1) * g;
            v_i(r, c) = beta2 * v_i(r, c) + (1.0f - beta2) * (g * g);
            float m_hat = m_i(r, c) / (1.0f - beta1_t);
            float v_hat = v_i(r, c) / (1.0f - beta2_t);
            (*param)(r, c) -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
          }
        }
      } else { // Para embeddings posicionales, etc. (3D)
#pragma omp parallel for collapse(3)
        for (size_t d0 = 0; d0 < shape[0]; ++d0) {
          for (size_t d1 = 0; d1 < shape[1]; ++d1) {
            for (size_t d2 = 0; d2 < shape[2]; ++d2) {
              float g = (*grad_tensor)(d0, d1, d2);
              if (weight_decay > 0.0f) {
                g += weight_decay * (*param)(d0, d1, d2);
              }
              m_i(d0, d1, d2) = beta1 * m_i(d0, d1, d2) + (1.0f - beta1) * g;
              v_i(d0, d1, d2) = beta2 * v_i(d0, d1, d2) + (1.0f - beta2) * (g * g);
              float m_hat = m_i(d0, d1, d2) / (1.0f - beta1_t);
              float v_hat = v_i(d0, d1, d2) / (1.0f - beta2_t);
              (*param)(d0, d1, d2) -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
            }
          }
        }
      }
    }
  }
}
