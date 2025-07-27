#ifndef MODELUTILS_HPP
#define MODELUTILS_HPP

#include "model/VisionTransformer.hpp" // Incluimos el modelo para saber de dónde sacar los pesos
#include <string>
#include <vector>

namespace ModelUtils {

/**
 * @brief Guarda los pesos (parámetros) de un modelo en un archivo binario.
 * @details El formato del archivo es simple:
 *          Para cada tensor de parámetro:
 *          1. (size_t) Número de dimensiones de la forma.
 *          2. (size_t*) Las dimensiones de la forma.
 *          3. (float*) Los datos del tensor.
 * @param model El modelo cuyos pesos se guardarán.
 * @param filePath La ruta al archivo donde se guardarán los pesos.
 */
void save_weights(const VisionTransformer &model, const std::string &filePath);

/**
 * @brief Carga los pesos desde un archivo binario a un modelo existente.
 * @details El modelo debe haber sido construido con la misma arquitectura
 *          (mismas formas de tensores) que el modelo que se guardó.
 *          La función verifica que las formas coincidan antes de cargar.
 * @param model El modelo al que se le cargarán los pesos.
 * @param filePath La ruta al archivo desde donde se cargarán los pesos.
 */
void load_weights(VisionTransformer &model, const std::string &filePath);

/**
 * @brief Guarda la configuración de un modelo en un archivo JSON.
 * @param config La estructura de configuración a guardar.
 * @param filePath La ruta al archivo .json donde se guardará.
 */
void save_config(const ViTConfig& config, const std::string& filePath);

/**
 * @brief Carga la configuración de un modelo desde un archivo JSON.
 * @param filePath La ruta al archivo .json desde donde se cargará.
 * @return Una estructura ViTConfig con los valores cargados.
 */
ViTConfig load_config(const std::string& filePath);

} // namespace ModelUtils

#endif // MODELUTILS_HPP
