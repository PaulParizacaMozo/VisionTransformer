#ifndef DATAREADER_HPP
#define DATAREADER_HPP

#include "core/Tensor.hpp"
#include <string>
#include <utility> // Para std::pair

using XYPair = std::pair<Tensor, Tensor>;   // {X, y}


/**
 * @brief Carga y procesa un dataset tipo MNIST desde un archivo CSV.
 * @details Lee un archivo CSV donde la primera columna es la etiqueta y las 784
 *          columnas siguientes son los píxeles de una imagen de 28x28.
 *          - Normaliza los valores de los píxeles al rango [0, 1].
 *          - Codifica las etiquetas en formato one-hot.
 *          - Remodela los datos de los píxeles a la forma de imagen 4D {N, C, H, W}.
 *
 * @param filePath La ruta al archivo .csv.
 * @param sample_fraction La fracción del dataset a cargar (de 0.0 a 1.0).
 *        Por defecto, 1.0 para cargar todo el dataset.
 * @return Un par de Tensores: {X, y}, donde X contiene las imágenes y y las etiquetas.
 */
// std::pair<Tensor, Tensor> load_csv_data(const std::string &filePath, float sample_fraction = 1.0f);
std::pair<Tensor, Tensor>
load_csv_data(const std::string& filePath,
              float sample_fraction,
              size_t channels,
              size_t height,
              size_t width,
              size_t num_classes,
              float mean  = 0.2860f,   // media MNIST por defecto
              float stddev = 0.3530f); // desviación estándar MNIST

// Devuelve {train_pair, valid_pair}
std::pair<XYPair, XYPair>
load_csv_data_train_val(const std::string& filePath,
                        float sample_frac,   // p. ej. 0.25            (25 % del dataset)
                        float train_frac,    //        0.80            (80 % de ese 25 %)
                        float val_frac,      //        0.20            (20 % de ese 25 %)
                        size_t channels,
                        size_t height,
                        size_t width,
                        size_t num_classes,
                        float mean  = 0.2860f,
                        float stddev = 0.3530f);

std::vector<float> compute_class_weights(const Tensor& y_onehot);

void print_classweights(const std::vector<float>& vec);

#endif // DATAREADER_HPP
