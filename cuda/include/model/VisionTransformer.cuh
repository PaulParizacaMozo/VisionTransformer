#pragma once

#include "layers/Layer.cuh"
#include "layers/Embeddings.cuh"
#include "layers/LayerNorm.cuh"
#include "layers/Dense.cuh"
#include "model/TransformerEncoderBlock.cuh"
#include <vector>
#include <memory>
#include <string>

/**
 * @struct ViTConfig
 * @brief Estructura con los hiperparámetros del modelo Vision Transformer.
 */
struct ViTConfig
{
    size_t image_size = 28;
    size_t patch_size = 7;
    size_t in_channels = 1;
    size_t num_classes = 10;
    size_t embedding_dim = 128;
    size_t num_heads = 8;
    size_t num_layers = 4;       // Número de bloques encoder
    size_t mlp_hidden_dim = 512; // Por convención: 4 * embedding_dim
};

/**
 * @class VisionTransformer
 * @brief Modelo completo Vision Transformer: embeddings + encoder stack + MLP head.
 */
class VisionTransformer : public Layer
{
private:
    ViTConfig config;

    Embeddings embeddings;
    std::vector<TransformerEncoderBlock> encoder_blocks;
    LayerNorm final_norm;
    Dense mlp_head;

    // Guardado para backward
    size_t num_tokens;
    Tensor final_norm_output;

public:
    explicit VisionTransformer(const ViTConfig &config);

    Tensor forward(const Tensor &input, bool isTraining) override;
    Tensor backward(const Tensor &outputGradient) override;

    std::vector<Tensor *> getParameters() override;
    std::vector<Tensor *> getGradients() override;

    std::string getName() const override { return "VisionTransformer"; }
};
