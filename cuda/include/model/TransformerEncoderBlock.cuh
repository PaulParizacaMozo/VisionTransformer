#pragma once

#include "layers/Layer.cuh"
#include "layers/LayerNorm.cuh"
#include "layers/MultiHeadAttention.cuh"
#include "layers/FeedForward.cuh"
#include <vector>
#include <memory>
#include <string>

class TransformerEncoderBlock : public Layer
{
private:
    LayerNorm norm1;
    MultiHeadAttention attention;
    LayerNorm norm2;
    FeedForward ffn;

    // Tensores para las conexiones residuales en el backward
    Tensor input_skip1;
    Tensor input_skip2;

public:
    TransformerEncoderBlock(size_t embedding_dim, size_t num_heads, size_t mlp_hidden_dim);

    Tensor forward(const Tensor &input, bool isTraining) override;
    Tensor backward(const Tensor &outputGradient) override;

    std::vector<Tensor *> getParameters() override;
    std::vector<Tensor *> getGradients() override;

    std::string getName() const override { return "TransformerEncoderBlock"; }
};
