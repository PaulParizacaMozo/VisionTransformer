#pragma once

#include "layers/Dense.cuh"
#include "layers/Layer.cuh"
#include <memory>
#include <vector>
#include <string>

class PatchEmbedding : public Layer
{
private:
    size_t image_height;
    size_t image_width;
    size_t patch_size;
    size_t in_channels;
    size_t embedding_dim;

    size_t patch_dim;   // patch_size * patch_size * in_channels
    size_t num_patches; // num_patches_h * num_patches_w
    size_t num_patches_h;
    size_t num_patches_w;

    std::unique_ptr<Dense> projectionLayer;

    Tensor flattenedPatches; // Cache para el backward
public:
    PatchEmbedding(size_t image_height, size_t image_width, size_t patch_size, size_t in_channels, size_t embedding_dim);
    Tensor forward(const Tensor &input, bool isTraining) override;
    Tensor backward(const Tensor &outputGradient) override;
    std::vector<Tensor *> getParameters() override;
    std::vector<Tensor *> getGradients() override;
    std::string getName() const override { return "PatchEmbedding"; }
    size_t getNumPatches() const { return num_patches; }
};
