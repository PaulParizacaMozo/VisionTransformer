#pragma once

#include "layers/Layer.cuh"
#include "layers/Dense.cuh"
#include <vector>
#include <memory>
#include <string>

class MultiHeadAttention : public Layer
{
private:
    size_t embedding_dim;
    size_t num_heads;
    size_t head_dim; // Dimensión de cada cabeza (D / h)

    // Capas lineales para Q, K, V y la salida
    std::unique_ptr<Dense> q_proj;
    std::unique_ptr<Dense> k_proj;
    std::unique_ptr<Dense> v_proj;
    std::unique_ptr<Dense> out_proj;

    // Función auxiliar para la atención escalada por producto punto
    Tensor scaledDotProductAttention(const Tensor &q, const Tensor &k, const Tensor &v);

    // Tensores guardados para el backward pass
    Tensor inputTensor;
    Tensor q_split;
    Tensor k_split;
    Tensor v_split;
    Tensor v_split2;
    Tensor attention_weights; // Los pesos de softmax
public:
    MultiHeadAttention(size_t embedding_dim, size_t num_heads);

    Tensor forward(const Tensor &input, bool isTraining) override;
    Tensor backward(const Tensor &outputGradient) override;

    std::vector<Tensor *> getParameters() override;
    std::vector<Tensor *> getGradients() override;

    std::string getName() const override { return "MultiHeadAttention"; }
};
