#ifndef PATCHEMBEDDING_HPP
#define PATCHEMBEDDING_HPP

#include "layers/Dense.hpp" // La usaremos para la proyección lineal
#include "layers/Layer.hpp"
#include <memory> // Para std::unique_ptr

/**
 * @class PatchEmbedding
 * @brief Convierte un lote de imágenes en una secuencia de embeddings de parches.
 *
 * Esta capa realiza los dos primeros pasos de un Vision Transformer:
 * 1. Divide las imágenes de entrada en parches fijos.
 * 2. Aplana cada parche y lo proyecta a una dimensión de embedding a través
 *    de una capa lineal (Dense) entrenable.
 */
class PatchEmbedding : public Layer {
public:
  /**
   * @brief Constructor de la capa PatchEmbedding.
   * @param image_height Altura de las imágenes de entrada.
   * @param image_width Ancho de las imágenes de entrada.
   * @param patch_size Tamaño de cada parche cuadrado (ej. 7 para parches de 7x7).
   * @param in_channels Número de canales de la imagen de entrada (ej. 1 para MNIST).
   * @param embedding_dim La dimensionalidad del espacio de embedding de salida.
   */
  PatchEmbedding(size_t image_height, size_t image_width, size_t patch_size, size_t in_channels, size_t embedding_dim);

  /**
   * @brief Realiza el paso hacia adelante: parcheo y proyección.
   * @param input Tensor de entrada con forma {batch, channels, height, width}.
   * @param isTraining Booleano que indica el modo de entrenamiento.
   * @return Tensor de salida con forma {batch, num_patches, embedding_dim}.
   * @override
   */
  Tensor forward(const Tensor &input, bool isTraining) override;

  /**
   * @brief Realiza el paso hacia atrás.
   * @param outputGradient Gradiente que viene de la siguiente capa.
   * @return Gradiente con respecto a la entrada de la imagen.
   * @override
   */
  Tensor backward(const Tensor &outputGradient) override;

  /**
   * @brief Devuelve punteros a los parámetros de la capa de proyección.
   * @override
   */
  std::vector<Tensor *> getParameters() override;

  /**
   * @brief Devuelve punteros a los gradientes de la capa de proyección.
   * @override
   */
  std::vector<Tensor *> getGradients() override;

  /**
   * @brief Devuelve el nombre de la capa.
   * @override
   */
  std::string getName() const override { return "PatchEmbedding"; }

  /**
   * @brief Devuelve el número de parches generados.
   */
  size_t getNumPatches() const { return num_patches; }

private:
  size_t image_height;
  size_t image_width;
  size_t patch_size;
  size_t in_channels;
  size_t embedding_dim;

  size_t patch_dim;     // Dimensión del parche aplanado (patch_size * patch_size * in_channels)
  size_t num_patches;   // Número total de parches por imagen
  size_t num_patches_h; // Número de parches a lo largo de la altura
  size_t num_patches_w; // Número de parches a lo largo del ancho

  // Usamos composición: la capa PatchEmbedding "contiene" una capa Dense.
  // std::unique_ptr es una buena forma de gestionar su ciclo de vida.
  std::unique_ptr<Dense> projectionLayer;

  // Tensor para guardar la entrada aplanada y parcheada para el backward pass.
  Tensor flattenedPatches;
};

#endif // PATCHEMBEDDING_HPP
