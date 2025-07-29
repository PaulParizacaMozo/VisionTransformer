#include "model/Trainer.hpp"
#include "utils/DataAugmentation.hpp"
#include "optimizers/Scheduler.hpp" // Para cosine_warmup_lr
#include <algorithm>                // Para std::random_shuffle
#include <ctime>                    // Para std::time
#include <iomanip>
#include <iostream>
#include <limits>  // Para std::numeric_limits
#include <numeric> // Para std::iota
#include <stdexcept>
#include <random>
#include <cmath>

// --- Función Auxiliar (privada a este archivo usando un namespace anónimo) ---
namespace
{
  /**
   * @brief Calcula la precisión de las predicciones de un batch.
   * @param logits Las salidas del modelo (antes de softmax).
   * @param labels Las etiquetas verdaderas en formato one-hot.
   * @return La precisión como un valor flotante entre 0.0 y 1.0.
   */
  float calculate_accuracy(const Tensor &logits, const Tensor &labels)
  {
    size_t batch_size = logits.getShape()[0];
    if (batch_size == 0)
      return 0.0f;
    size_t correct_predictions = 0;

    for (size_t i = 0; i < batch_size; ++i)
    {
      // Encontrar el índice de la clase con la mayor puntuación (argmax)
      float max_logit = -std::numeric_limits<float>::infinity();
      int pred_class = -1;
      for (size_t j = 0; j < logits.getShape()[1]; ++j)
      {
        if (logits(i, j) > max_logit)
        {
          max_logit = logits(i, j);
          pred_class = j;
        }
      }
      // Comprobar si coincide con la etiqueta verdadera (que es 1.0 en el one-hot)
      if (labels(i, pred_class) == 1.0f)
      {
        correct_predictions++;
      }
    }
    return static_cast<float>(correct_predictions) / batch_size;
  }
} // namespace

/**
 * @brief Constructor del Trainer. Recibe una referencia al modelo y la configuración.
 */
Trainer::Trainer(VisionTransformer &model, const TrainerConfig &train_config, const std::vector<float> &class_weights)
    : model(model), // <-- Guarda la referencia
      optimizer(train_config.learning_rate, 0.9f, 0.999f, 1e-8f, train_config.weight_decay),
      loss_fn(),
      config(train_config), logger("vit_results.csv")
{
  loss_fn.setClassWeights(class_weights);
}

/**
 * @brief Orquesta el proceso de entrenamiento completo a lo largo de varias épocas.
 */
void Trainer::train(const std::pair<Tensor, Tensor> &train_data, const std::pair<Tensor, Tensor> &test_data)
{
  const auto &[X_train, y_train] = train_data;
  const auto &[X_test, y_test] = test_data;

  size_t batches_per_epoch =
      (train_data.first.getShape()[0] + config.batch_size - 1) / config.batch_size;
  total_steps = (long long)batches_per_epoch * config.epochs;

  for (int epoch = 0; epoch < config.epochs; ++epoch)
  {
    // std::cout << "\n--- Época " << epoch + 1 << "/" << config.epochs << " --- | ";

    auto start = std::chrono::high_resolution_clock::now();
    // Ejecutar una época de entrenamiento y obtener sus métricas
    // auto [train_loss, train_acc] = train_epoch(X_train, y_train);
    auto [train_loss, train_acc] = train_epoch(X_train, y_train, epoch);

    // Limpiar la línea de progreso de los batches
    std::cout << "\r" << std::string(80, ' ') << "\r";

    // Evaluar en el conjunto de test para obtener sus métricas
    auto [test_loss, test_acc] = evaluate(X_test, y_test);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    // Registrar en el logger
    logger.log_epoch(epoch, config.epochs, train_loss, train_acc, test_loss, test_acc);

    auto ms_total = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    int h = static_cast<int>(ms_total / 3600000);
    int m = static_cast<int>((ms_total % 3600000) / 60000);
    int s = static_cast<int>((ms_total % 60000) / 1000);
    int ms = static_cast<int>(ms_total % 1000);

    // Formatear el tiempo [hh:mm:ss.ms]
    std::ostringstream time_ss;
    time_ss << std::setfill('0') << "["
            << std::setw(2) << h << ":"
            << std::setw(2) << m << ":"
            << std::setw(2) << s << "."
            << std::setw(3) << ms << "]";

    // Colores ANSI
    const char *COLOR_BOLD_WHITE = "\033[1m";
    const char *COLOR_BLUE = "\033[1;34m";
    const char *COLOR_GREEN = "\033[1;32m";
    const char *COLOR_MAGENTA = "\033[1;35m";
    const char *COLOR_RESET = "\033[0m";

    // Imprimir resumen con colores
    std::cout << COLOR_BOLD_WHITE << "--- Epoch " << (epoch + 1) << "/" << config.epochs << COLOR_RESET;
    std::cout << " | " << COLOR_BLUE << "Train Loss: " << std::fixed << std::setprecision(4) << train_loss
              << " | Train Acc: " << train_acc << COLOR_RESET;
    std::cout << " | " << COLOR_GREEN << "Test Loss: " << test_loss << " | Test Acc: " << test_acc << COLOR_RESET;
    std::cout << " " << COLOR_MAGENTA << time_ss.str() << COLOR_RESET << std::endl;
    // Imprimir el resumen de la época en el formato solicitado
    // std::cout << "--- Época " << epoch + 1 << "/" << config.epochs << " | Train Loss: " << std::fixed << std::setprecision(4)
    //           << train_loss << " | Train Acc: " << train_acc << " | Test Loss: " << test_loss << " | Test Acc: " << test_acc
    //           << std::endl;
  }
}

/**
 * @brief Ejecuta un ciclo completo sobre el dataset de entrenamiento (una época).
 */
std::pair<float, float> Trainer::train_epoch(const Tensor &X_train, const Tensor &y_train, int epoch)
{
  const auto &input_shape = X_train.getShape();
  size_t num_train_samples = input_shape[0];
  size_t channels = input_shape[1];
  size_t height = input_shape[2];
  size_t width = input_shape[3];
  size_t num_classes = y_train.getShape()[1];

  size_t num_batches = (num_train_samples + config.batch_size - 1) / config.batch_size;

  float total_loss = 0.0f;
  float total_accuracy = 0.0f;

  // Crear el objeto de DataAugmentation
  // DataAugmentation augmentor(0.5f, 4, 30.0f, 4.0f); // Flip con probabilidad 0.5 y recorte con padding 4

  // Crear y barajar los índices al inicio de la época
  std::vector<size_t> indices(num_train_samples);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 rng(static_cast<uint32_t>(epoch)); // semilla = epoch
  std::shuffle(indices.begin(), indices.end(), rng);

  // Configurar el aumentador de datos
  DataAugmentation::Config aug_cfg;
  aug_cfg.rotation_prob = 0.5f;
  aug_cfg.translate_prob = 0.5f;
  aug_cfg.zoom_prob = 0.5f;
  DataAugmentation augmentor(aug_cfg);

  for (size_t i = 0; i < num_batches; ++i)
  {
    size_t start_idx_in_indices = i * config.batch_size;
    size_t count = std::min(config.batch_size, num_train_samples - start_idx_in_indices);
    if (count == 0)
      continue;

    // Crear los tensores para el batch actual
    Tensor X_batch({count, channels, height, width});
    Tensor y_batch({count, num_classes});

    // Llenar el batch con los datos correspondientes a los índices barajados
    for (size_t j = 0; j < count; ++j)
    {
      size_t data_idx = indices[start_idx_in_indices + j];
      Tensor x_sample = X_train.slice(0, data_idx, 1);
      Tensor y_sample = y_train.slice(0, data_idx, 1);

      for (size_t c = 0; c < channels; ++c)
      {
        for (size_t h = 0; h < height; ++h)
        {
          for (size_t w = 0; w < width; ++w)
          {
            // Accedemos a los índices correctos en x_sample y X_batch
            X_batch(j, c, h, w) = x_sample(0, c, h, w);
          }
        }
      }
      for (size_t c = 0; c < num_classes; ++c)
      {
        y_batch(j, c) = y_sample(0, c);
      }
    }

    // Aplicar data augmentation al batch completo
    Tensor X_batch_augmented = augmentor.apply(X_batch);
    // Usar X_batch_augmented en lugar de X_batch
    Tensor logits = model.forward(X_batch_augmented, true);

    // --- Ciclo de entrenamiento para el batch ---
    // Tensor logits = model.forward(X_batch, true);
    float batch_loss = loss_fn.calculate(logits, y_batch);

    total_loss += batch_loss;
    // total_loss += loss_fn.calculate(logits, y_batch);
    total_accuracy += calculate_accuracy(logits, y_batch);

    Tensor grad = loss_fn.backward(logits, y_batch);
    model.backward(grad);

    auto params = model.getParameters();
    auto grads = model.getGradients();

    // ─── Scheduler ───────────────────────────────
    long long warmup_steps = (long long)(config.warmup_frac * total_steps);
    float lr_now = cosine_warmup_lr(global_step,
                                    warmup_steps,
                                    total_steps,
                                    config.lr_init);
    optimizer.setLearningRate(lr_now);
    // --------------------------------------------

    optimizer.update(params, grads);
    global_step++;
    std::cout << "\rEntrenando... Batch " << (i + 1)
              << "/" << num_batches
              << " [loss: " << std::fixed << std::setprecision(8) << batch_loss << "]"
              << " | LR: " << std::fixed << std::setprecision(8) << optimizer.getLearningRate()
              << std::flush;
    // std::cout << "\rEntrenando... Batch " << i + 1
    //           << "/" << num_batches
    //           << " | LR: " << std::fixed << std::setprecision(8) << optimizer.getLearningRate() << std::flush;
  }

  return {total_loss / num_batches, total_accuracy / num_batches};
}

/**
 * @brief Evalúa el rendimiento del modelo, calculando pérdida y precisión.
 */
std::pair<float, float> Trainer::evaluate(const Tensor &X_test, const Tensor &y_test)
{
  size_t num_test_samples = X_test.getShape()[0];
  size_t num_batches = (num_test_samples + config.batch_size - 1) / config.batch_size;

  float total_loss = 0.0f;
  float total_accuracy = 0.0f;

  for (size_t i = 0; i < num_batches; ++i)
  {
    size_t start = i * config.batch_size;
    size_t count = std::min(config.batch_size, num_test_samples - start);
    if (count == 0)
      continue;

    Tensor X_batch = X_test.slice(0, start, count);
    Tensor y_batch = y_test.slice(0, start, count);

    // Forward pass en modo inferencia
    Tensor logits = model.forward(X_batch, false);

    // Calcular pérdida y precisión para el batch
    total_loss += loss_fn.calculate(logits, y_batch);
    total_accuracy += calculate_accuracy(logits, y_batch);
  }

  return {total_loss / num_batches, total_accuracy / num_batches};
}
