#include "model/TransformerEncoderBlock.cuh"

TransformerEncoderBlock::TransformerEncoderBlock(size_t embedding_dim, size_t num_heads, size_t mlp_hidden_dim)
    : norm1(embedding_dim), attention(embedding_dim, num_heads), norm2(embedding_dim), ffn(embedding_dim, mlp_hidden_dim) {}

Tensor TransformerEncoderBlock::forward(const Tensor &input, bool isTraining)
{
    if (isTraining)
        input_skip1 = input;

    Tensor x = norm1.forward(input, isTraining);
    x = attention.forward(x, isTraining);
    Tensor residual1 = input + x;

    if (isTraining)
        input_skip2 = residual1;

    Tensor y = norm2.forward(residual1, isTraining);
    y = ffn.forward(y, isTraining);
    return residual1 + y;
}

Tensor TransformerEncoderBlock::backward(const Tensor &outputGradient)
{
    Tensor grad_skip2 = outputGradient;

    Tensor grad_ffn = ffn.backward(outputGradient);
    grad_ffn = norm2.backward(grad_ffn);

    Tensor grad1 = grad_skip2 + grad_ffn;

    Tensor grad_mha = attention.backward(grad1);
    grad_mha = norm1.backward(grad_mha);

    return grad1 + grad_mha;
}

std::vector<Tensor *> TransformerEncoderBlock::getParameters()
{
    std::vector<Tensor *> params;
    auto p1 = norm1.getParameters();
    auto p2 = attention.getParameters();
    auto p3 = norm2.getParameters();
    auto p4 = ffn.getParameters();

    params.insert(params.end(), p1.begin(), p1.end());
    params.insert(params.end(), p2.begin(), p2.end());
    params.insert(params.end(), p3.begin(), p3.end());
    params.insert(params.end(), p4.begin(), p4.end());

    return params;
}

std::vector<Tensor *> TransformerEncoderBlock::getGradients()
{
    std::vector<Tensor *> grads;
    auto g1 = norm1.getGradients();
    auto g2 = attention.getGradients();
    auto g3 = norm2.getGradients();
    auto g4 = ffn.getGradients();

    grads.insert(grads.end(), g1.begin(), g1.end());
    grads.insert(grads.end(), g2.begin(), g2.end());
    grads.insert(grads.end(), g3.begin(), g3.end());
    grads.insert(grads.end(), g4.begin(), g4.end());

    return grads;
}