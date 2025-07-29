#include "model/Trainer.hpp"
#include "utils/DataReader.hpp"
#include "utils/ModelUtils.hpp"
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

/* ────────────────────────────────────────────────────────────── *
 * Imprime la matriz de confusion normalizada (por filas) con la
 * diagonal resaltada en color verde                              *
 * ────────────────────────────────────────────────────────────── */
void printConfusionMatrix(const std::vector<std::vector<int>>& confusion)
{
    const size_t num_classes = confusion.size();

    /* -------- Normalizar (por filas) ---------- */
    std::vector<std::vector<float>> norm(num_classes,
                                         std::vector<float>(num_classes, 0.f));

    for (size_t i = 0; i < num_classes; ++i) {
        int row_sum = 0;
        for (int v : confusion[i]) row_sum += v;
        if (row_sum == 0) continue;
        for (size_t j = 0; j < num_classes; ++j)
            norm[i][j] = static_cast<float>(confusion[i][j]) / row_sum;
    }

    /* ----------- Impresión bonita ------------- */
    const int w_idx  = 6;   // ancho para índice de clase
    const int w_cell = 6;   // ancho de cada celda

    std::cout << "\n=== Matriz de Confusión normalizada (por filas) ===\n";

    /* cabecera */
    std::cout << std::setw(w_idx) << " " << " ";
    for (size_t j = 0; j < num_classes; ++j)
        std::cout << std::setw(w_cell) << j;
    std::cout << '\n';

    /* separador */
    auto line = [&]{
        std::cout << std::string(w_idx + 1 + w_cell * num_classes, '-') << '\n';
    };
    line();

    std::cout << std::fixed << std::setprecision(2);
    for (size_t i = 0; i < num_classes; ++i) {
        std::cout << std::setw(w_idx - 1) << i << " |";
        for (size_t j = 0; j < num_classes; ++j) {
            float val = norm[i][j] * 100.f;        // porcentaje

            if (i == j) std::cout << "\033[1;32m"; // verde‑negrita
            std::cout << std::setw(w_cell) << val;
            if (i == j) std::cout << "\033[0m";    // reset
        }
        std::cout << '\n';
    }
    line();
    std::cout << "(valores en % — diagonal resaltada)\n\n";
}


int main() {
  try {
    const std::string model_name = "vit_mnist_30ep_4_8";
    const std::string weights_path = model_name + ".weights";
    const std::string config_path = model_name + ".json";

    // --- 1. Cargar conifguracion del ViT ---
    std::cout << "Cargando configuración desde: " << config_path << std::endl;
    ViTConfig loaded_config = ModelUtils::load_config( "models/" + config_path);

    // --- 2. Crear y Cargar los pesos en el modelo ---
    std::cout << "Construyendo modelo con la arquitectura cargada..." << std::endl;
    VisionTransformer model(loaded_config);

    std::cout << "Cargando pesos desde: " << weights_path << std::endl;
    ModelUtils::load_weights(model, "models/" + weights_path);
    std::cout << "Pesos cargados correctamente.\n";

    // --- 3. Cargar datos de prueba ---
    auto test_data =
        load_csv_data("data/mnist_test.csv", 1.00f, 1, 28, 28, loaded_config.num_classes, 0.1307f, 0.3081f);

    // --- 3. Hacer predicciones ---
    const Tensor &X_test = test_data.first;
    const Tensor &y_test = test_data.second;

    Tensor logits = model.forward(
        X_test, false); // `isTraining` es `false` durante la inferencia
    Tensor probabilities = softmax(logits);

    size_t batch_size = probabilities.getShape()[0];
    size_t num_classes = probabilities.getShape()[1];
    int correct_predictions = 0;
    int total_samples = 0;

    // Matriz de confusión: confusion[true][predicted]
    std::vector<std::vector<int>> confusion(num_classes,
                                            std::vector<int>(num_classes, 0));

    for (size_t i = 0; i < batch_size; ++i) {
      float max_prob = -std::numeric_limits<float>::infinity();
      int predicted_class = -1;

      for (size_t j = 0; j < num_classes; ++j) {
        if (probabilities(i, j) > max_prob) {
          max_prob = probabilities(i, j);
          predicted_class = j;
        }
      }

      int true_label = -1;
      for (size_t j = 0; j < num_classes; ++j) {
        if (y_test(i, j) == 1.0f) {
          true_label = j;
          break;
        }
      }

      // std::cout << "Sample " << i << " | Predicción: " << predicted_class
      //           << " | Etiqueta: " << true_label << std::endl;

      if (predicted_class == true_label) {
        ++correct_predictions;
      }

      confusion[true_label][predicted_class]++;
      ++total_samples;
    }

    // --- 4. Mostrar métricas ---
    float accuracy = static_cast<float>(correct_predictions) / total_samples;
    std::cout << "\nAccuracy del modelo: " << accuracy * 100.0f << "%"
              << std::endl;

    // Parámetros de formato
    const int w_class = 8;
    const int w_val = 15;

    auto print_separator = [&]() {
      std::cout << "+" << std::string(w_class, '-') << "+"
                << std::string(w_val, '-') << "+" << std::string(w_val, '-')
                << "+" << std::string(w_val, '-') << "+"
                << "\n";
    };

    std::cout << "\n=== Métricas por clase ===\n";
    print_separator();
    std::cout << "|" << std::setw(w_class) << std::left << "Clase" << "|"
              << std::setw(w_val) << "Precision" << "|" << std::setw(w_val)
              << "Recall" << "|" << std::setw(w_val) << "F1-score" << "|\n";
    print_separator();

    for (size_t c = 0; c < num_classes; ++c) {
      int TP = confusion[c][c];
      int FP = 0, FN = 0;
      for (size_t i = 0; i < num_classes; ++i) {
        if (i != c) {
          FP += confusion[i][c];
          FN += confusion[c][i];
        }
      }

      float precision =
          (TP + FP) > 0 ? static_cast<float>(TP) / (TP + FP) : 0.0f;
      float recall = (TP + FN) > 0 ? static_cast<float>(TP) / (TP + FN) : 0.0f;
      float f1 = (precision + recall) > 0
                     ? 2 * precision * recall / (precision + recall)
                     : 0.0f;

      std::cout << "|" << std::setw(w_class) << std::left << c << "|"
                << std::setw(w_val) << precision << "|" << std::setw(w_val)
                << recall << "|" << std::setw(w_val) << f1 << "|\n";
    }
    print_separator();

    /* Imprimir la matriz de confusión normalizada */
    printConfusionMatrix(confusion);

  } catch (const std::exception &e) {
    std::cerr << "\nERROR CRÍTICO: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
