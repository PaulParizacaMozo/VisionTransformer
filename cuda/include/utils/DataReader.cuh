#pragma once

#include "core/Tensor.cuh"
#include <string>
#include <utility> // Para std::pair

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
std::pair<Tensor, Tensor> load_csv_data(const std::string &filePath, float sample_fraction = 1.0f);
