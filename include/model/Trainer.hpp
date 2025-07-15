#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "losses/CrossEntropy.hpp"
#include "model/VisionTransformer.hpp"
#include "optimizers/Adam.hpp"  // O una interfaz Optimizer si tienes más
#include "utils/DataReader.hpp" // Para recibir los datos
#include <memory>
#include <vector>

struct TrainerConfig {
  int epochs = 10;
  size_t batch_size = 64;
  float learning_rate = 0.001f;
  float weight_decay = 0.01f;
};

class Trainer {
public:
  // Trainer(const ViTConfig &model_config, const TrainerConfig &train_config);
  Trainer(VisionTransformer &model, const TrainerConfig &train_config);
  /**
   * @brief Ejecuta el bucle de entrenamiento completo.
   * @param train_data Par {Imágenes, Etiquetas} para el entrenamiento.
   * @param test_data Par {Imágenes, Etiquetas} para la validación.
   */
  void train(const std::pair<Tensor, Tensor> &train_data, const std::pair<Tensor, Tensor> &test_data);

  const VisionTransformer &getModel() const { return model; }
  VisionTransformer &getModel() { return model; }

private:
  /**
   * @brief Ejecuta una única época de entrenamiento.
   * @return Par {pérdida_promedio, precisión_promedio} de la época.
   */
  std::pair<float, float> train_epoch(const Tensor &X_train, const Tensor &y_train);

  /**
   * @brief Evalúa el modelo en un conjunto de datos.
   * @return Un par {pérdida_promedio, precisión_promedio}.
   */
  std::pair<float, float> evaluate(const Tensor &X_test, const Tensor &y_test);

  // Componentes del entrenamiento
  VisionTransformer &model; // Almacenamos una referencia, no un objeto
  Adam optimizer;
  CrossEntropy loss_fn;

  // Configuración
  TrainerConfig config;
};

#endif // TRAINER_HPP
