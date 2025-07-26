#ifndef VISIONTRANSFORMER_HPP
#define VISIONTRANSFORMER_HPP

#include "layers/Dense.hpp"
#include "layers/Embeddings.hpp"
#include "layers/Layer.hpp"
#include "layers/LayerNorm.hpp"
#include "model/TransformerEncoderBlock.hpp"
#include <memory>
#include <vector>

/**
 * @struct ViTConfig
 * @brief Estructura para mantener todos los hiperparámetros de un Vision Transformer.
 */
struct ViTConfig {
  size_t image_size = 28;
  size_t patch_size = 7;
  size_t in_channels = 1;
  size_t num_classes = 10;
  size_t embedding_dim = 128;
  size_t num_heads = 8;
  size_t num_layers = 4;       // Número de bloques encoder
  size_t mlp_hidden_dim = 512; // embedding_dim * 4 es una buena regla general
  float dropout_rate = 0.1;    // (0.1 o 10% es un buen default)
};

/**
 * @class VisionTransformer
 * @brief Implementación completa del modelo Vision Transformer.
 *
 * Encapsula la capa de embeddings, la pila de bloques codificadores y la
 * cabeza de clasificación (MLP Head) en una única interfaz de Layer.
 */
class VisionTransformer : public Layer {
public:
  /**
   * @brief Constructor que construye el modelo a partir de una configuración.
   * @param config Estructura con todos los hiperparámetros del modelo.
   */
  explicit VisionTransformer(const ViTConfig &config);

  /**
   * @brief Realiza un forward pass completo a través de todo el modelo.
   * @param input Tensor de imágenes de entrada de forma {batch, channels, height, width}.
   * @param isTraining Booleano que indica el modo de entrenamiento.
   * @return Tensor de logits de salida de forma {batch, num_classes}.
   * @override
   */
  Tensor forward(const Tensor &input, bool isTraining) override;

  /**
   * @brief Realiza un backward pass completo a través de todo el modelo.
   * @param outputGradient Gradiente que viene de la función de pérdida.
   * @return Tensor de gradiente con respecto a la imagen de entrada (generalmente no se usa).
   * @override
   */
  Tensor backward(const Tensor &outputGradient) override;

  /**
   * @brief Recolecta y devuelve los parámetros de todas las capas del modelo.
   * @override
   */
  std::vector<Tensor *> getParameters() override;

  /**
   * @brief Recolecta y devuelve los gradientes de todas las capas del modelo.
   * @override
   */
  std::vector<Tensor *> getGradients() override;

  /**
   * @brief Devuelve el nombre del modelo.
   * @override
   */
  std::string getName() const override { return "VisionTransformer"; }

private:
  ViTConfig config;

  // Las partes del modelo
  Embeddings embeddings;
  std::vector<TransformerEncoderBlock> encoder_blocks;
  LayerNorm final_norm;
  Dense mlp_head;

  // Tensor guardado para el backward pass
  size_t num_tokens;
  Tensor final_norm_output;
};

#endif // VISIONTRANSFORMER_HPP
