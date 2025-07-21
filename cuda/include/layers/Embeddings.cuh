#pragma once

#include "layers/Layer.cuh"
#include "layers/PatchEmbedding.cuh"
#include <memory>
#include <vector>
#include <string>

class Embeddings : public Layer
{
private:
    std::unique_ptr<PatchEmbedding> patcher;

    Tensor clsToken;           // Forma: [1, 1, embedding_dim]
    Tensor positionalEncoding; // Forma: [1, num_patches + 1, embedding_dim]

    Tensor clsTokenGradient;
    Tensor positionalEncodingGradient;

    size_t num_patches;
    size_t embedding_dim;

public:
    Embeddings(size_t image_height,
               size_t image_width,
               size_t patch_size,
               size_t in_channels,
               size_t embedding_dim);

    Tensor forward(const Tensor &input, bool isTraining) override;
    Tensor backward(const Tensor &outputGradient) override;

    std::vector<Tensor *> getParameters() override;
    std::vector<Tensor *> getGradients() override;

    std::string getName() const override { return "Embeddings"; }
};
