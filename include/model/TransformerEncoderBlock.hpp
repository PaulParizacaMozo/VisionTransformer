#ifndef TRANSFORMERENCODERBLOCK_HPP
#define TRANSFORMERENCODERBLOCK_HPP

#include "layers/Dropout.hpp"
#include "layers/FeedForward.hpp"
#include "layers/Layer.hpp"
#include "layers/LayerNorm.hpp"
#include "layers/MultiHeadAttention.hpp"
#include <vector>

/**
 * @class TransformerEncoderBlock
 * @brief Implementa un bloque codificador completo de un Transformer.
 *
 * Este bloque contiene dos sub-capas principales:
 * 1. Una capa de Multi-Head Self-Attention.
 * 2. Una red Feed-Forward (MLP).
 * Cada sub-capa está precedida por una normalización de capa (LayerNorm)
 * y seguida por una conexión residual (skip connection).
 */
class TransformerEncoderBlock : public Layer {
public:
  /**
   * @brief Constructor del bloque codificador.
   * @param embedding_dim La dimensión de los embeddings (D).
   * @param num_heads El número de cabezas de atención (h).
   * @param mlp_hidden_dim La dimensión oculta de la red Feed-Forward.
   * @param dropout_rate La tasa de dropout para ambas capas de dropout.
   */
  TransformerEncoderBlock(size_t embedding_dim, size_t num_heads, size_t mlp_hidden_dim, float dropout_rate);

  /**
   * @brief Realiza el paso hacia adelante a través del bloque completo.
   * @param input Tensor de entrada de forma {batch, tokens, embedding_dim}.
   * @param isTraining Booleano que indica el modo de entrenamiento.
   * @return Tensor de salida con la misma forma que la entrada.
   * @override
   */
  Tensor forward(const Tensor &input, bool isTraining) override;

  /**
   * @brief Realiza el paso hacia atrás a través del bloque completo.
   * @param outputGradient Gradiente que viene del siguiente bloque.
   * @return Gradiente con respecto a la entrada de este bloque.
   * @override
   */
  Tensor backward(const Tensor &outputGradient) override;

  /**
   * @brief Recolecta y devuelve los parámetros de todas las sub-capas.
   * @override
   */
  std::vector<Tensor *> getParameters() override;

  /**
   * @brief Recolecta y devuelve los gradientes de todas las sub-capas.
   * @override
   */
  std::vector<Tensor *> getGradients() override;

  /**
   * @brief Devuelve el nombre de la capa.
   * @override
   */
  std::string getName() const override { return "TransformerEncoderBlock"; }

private:
  // Componentes del bloque
  LayerNorm norm1;
  MultiHeadAttention attention;
  Dropout attention_dropout;
  LayerNorm norm2;
  FeedForward ffn;

  // Tensores guardados para las conexiones residuales en el backward pass
  Tensor input_skip1; // Entrada a la primera conexión residual (la entrada del bloque)
  Tensor input_skip2; // Entrada a la segunda conexión residual
};

#endif // TRANSFORMERENCODERBLOCK_HPP
