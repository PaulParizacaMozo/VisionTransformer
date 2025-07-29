#include "utils/DataReader.hpp"
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <random>

// --- Funciones Auxiliares (privadas a este archivo) ---
namespace {
/**
 * @brief Convierte un vector de etiquetas de clase (enteros) a un formato one-hot.
 */
Tensor oneHotEncode(const std::vector<int> &labels, int num_classes) {
  const size_t num_samples = labels.size();
  std::vector<float> one_hot_data(num_samples * num_classes, 0.0f);

  for (size_t i = 0; i < num_samples; ++i) {
    if (labels[i] >= 0 && labels[i] < num_classes) {
      one_hot_data[i * num_classes + labels[i]] = 1.0f;
    }
  }
  return Tensor({num_samples, static_cast<size_t>(num_classes)}, one_hot_data);
}
} // namespace

// --- Implementación de la Función Principal ---
std::pair<Tensor, Tensor> load_csv_data(const std::string &filePath,
                                        float sample_fraction,
                                        size_t channels,
                                        size_t height,
                                        size_t width,
                                        size_t num_classes,
                                        float mean,
                                        float stddev) {
    std::cout << "--- Cargando " << filePath
              << "  (fracción: " << sample_fraction * 100 << "%, "
              << "μ=" << mean << ", σ=" << stddev << ")"
              << std::endl;

  std::ifstream file(filePath);
  if (!file.is_open()) {
    throw std::runtime_error("Error: No se pudo abrir el archivo: " + filePath);
  }

  const size_t expected_pixels = channels * height * width;

  // 1. Leer todas las líneas del archivo CSV
  std::string line;
  std::getline(file, line); // Ignorar la línea de cabecera

  std::vector<std::vector<float>> all_pixel_data;
  std::vector<int> all_labels;

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string value_str;

    // Leer la etiqueta (primera columna)
    std::getline(ss, value_str, ',');
    all_labels.push_back(std::stoi(value_str));

    // Leer los píxeles y normalizarlos
    std::vector<float> pixels;
    pixels.reserve(expected_pixels);
    while (std::getline(ss, value_str, ',')) {
      // Normalizar el valor del píxel a [0, 1]
      float v = std::stof(value_str) / 255.0f;        // [0,1]
      v = (v - mean) / stddev;                        // normalización Z
      pixels.push_back(v);
    }
    if (pixels.size() != expected_pixels) {
      std::cerr << "Advertencia: Se encontró una fila con un número de píxeles incorrecto. Se ignora." << std::endl;
      all_labels.pop_back(); // Eliminar la etiqueta correspondiente
      continue;
    }
    all_pixel_data.push_back(pixels);
  }
  file.close();

  // 2. Barajar y tomar una fracción de los datos
  size_t total_samples = all_labels.size();
  std::vector<size_t> indices(total_samples);
  std::iota(indices.begin(), indices.end(), 0); // Rellena con 0, 1, 2, ...

  // Barajar los índices para tomar una muestra aleatoria
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));

  size_t samples_to_load = static_cast<size_t>(total_samples * sample_fraction);
  if (samples_to_load == 0 && total_samples > 0)
    samples_to_load = 1; // Cargar al menos 1
  if (samples_to_load > total_samples)
    samples_to_load = total_samples;

  std::vector<float> final_pixel_data;
  final_pixel_data.reserve(samples_to_load * expected_pixels);
  std::vector<int> final_labels;
  final_labels.reserve(samples_to_load);

  for (size_t i = 0; i < samples_to_load; ++i) {
    size_t original_index = indices[i];
    final_pixel_data.insert(final_pixel_data.end(), all_pixel_data[original_index].begin(),
                            all_pixel_data[original_index].end());
    final_labels.push_back(all_labels[original_index]);
  }

  // 3. Crear los tensores finales
  // La forma para las imágenes de entrada del ViT es {N, C, H, W}
  // Para MNIST, C=1, H=28, W=28.
  Tensor X({samples_to_load, channels, height, width}, final_pixel_data);

  // Las etiquetas se convierten a one-hot encoding. MNIST tiene 10 clases (0-9).
  Tensor y = oneHotEncode(final_labels, num_classes);

  std::cout << "Carga completa. " << samples_to_load << " muestras cargadas." << std::endl;
  std::cout << "  -> Forma de X (imágenes): " << X.shapeToString() << std::endl;
  std::cout << "  -> Forma de y (etiquetas): " << y.shapeToString() << std::endl;

  return {std::move(X), std::move(y)};
}

std::pair<XYPair, XYPair>
load_csv_data_train_val(const std::string& filePath,
                        float sample_frac,
                        float train_frac,
                        float val_frac,
                        size_t channels,
                        size_t height,
                        size_t width,
                        size_t num_classes,
                        float mean,
                        float stddev)
{
    if (train_frac + val_frac > 1.0f + 1e-6f)
        throw std::invalid_argument("train_frac + val_frac no debe superar 1.0");

    // 1. Cargar SOLO la fracción total solicitada (p.ej. 25 %)
    XYPair all_data = load_csv_data(filePath, sample_frac, channels, height, width, num_classes, mean, stddev);
    Tensor& X_all = all_data.first;
    Tensor& y_all = all_data.second;

    const size_t N = X_all.getShape()[0];
    const size_t N_train = static_cast<size_t>(N * train_frac);
    const size_t N_val   = static_cast<size_t>(N * val_frac);

    if (N_train + N_val > N)
        throw std::runtime_error("No hay suficientes muestras para el split solicitado.");

    // 2. Split mediante vistas (slice) — sin copiar memoria
    Tensor X_train = X_all.slice(0, 0, N_train);
    Tensor y_train = y_all.slice(0, 0, N_train);

    Tensor X_val   = X_all.slice(0, N_train, N_val);
    Tensor y_val   = y_all.slice(0, N_train, N_val);

    return { {X_train, y_train}, {X_val, y_val} };
}



std::vector<float> compute_class_weights(const Tensor& y_onehot) {
  const size_t num_samples = y_onehot.getShape()[0];
  const size_t num_classes = y_onehot.getShape()[1];

  // 1. Contar ocurrencias por clase
  std::vector<size_t> class_counts(num_classes, 0);
  for (size_t i = 0; i < num_samples; ++i) {
    for (size_t j = 0; j < num_classes; ++j) {
      if (y_onehot(i, j) == 1.0f) {
        class_counts[j]++;
        break;
      }
    }
  }

  // 2. Calcular pesos inversamente proporcionales a la frecuencia
  std::vector<float> class_weights(num_classes, 0.0f);
  float total_samples = static_cast<float>(num_samples);
  for (size_t i = 0; i < num_classes; ++i) {
    class_weights[i] = total_samples / (num_classes * static_cast<float>(class_counts[i]));
  }

  // 3. Normalizar para que sumen 1
  float sum_weights = std::accumulate(class_weights.begin(), class_weights.end(), 0.0f);
  for (float &w : class_weights) {
    w /= sum_weights;
  }

  return class_weights;
}

void print_classweights(const std::vector<float>& class_weights){
  for(size_t i = 0; i< class_weights.size(); i++){
    std::cout << "Clase "<< i << ": "<< class_weights[i]<<std::endl;
  }
}