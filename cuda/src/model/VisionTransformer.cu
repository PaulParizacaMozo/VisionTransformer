#include "model/VisionTransformer.cuh"

VisionTransformer::VisionTransformer(const ViTConfig &config)
    : config(config),
      embeddings(config.image_size, config.image_size, config.patch_size, config.in_channels, config.embedding_dim),
      final_norm(config.embedding_dim),
      mlp_head(config.embedding_dim, config.num_classes)
{
    this->num_tokens = 1 + (config.image_size / config.patch_size) * (config.image_size / config.patch_size);
    for (size_t i = 0; i < config.num_layers; ++i)
        encoder_blocks.emplace_back(config.embedding_dim, config.num_heads, config.mlp_hidden_dim);
}

Tensor VisionTransformer::forward(const Tensor &input, bool isTraining)
{
    std::cout << "VIT: paso 1 - Embeddings" << std::endl;
    Tensor x = embeddings.forward(input, isTraining);

    std::cout << "VIT: paso 2 - Añadir token CLS" << std::endl;
    for (auto &block : encoder_blocks)
    {
        x = block.forward(x, isTraining);
    }
    std::cout << "VIT: paso 3 - Normalización final" << std::endl;

    x = final_norm.forward(x, isTraining);

    if (isTraining)
    {
        this->final_norm_output = x; // Guardamos para backward
    }

    // Extraer token CLS (posición 0)
    Tensor x1 = x.slice(1, 0, 1); // Toma solo el primer "patch" a lo largo de dim=1
    Tensor x2 = x1.contiguous();  // Copia los datos a memoria continua, elimina views
    size_t new_shape_host[2] = {input.dim(0), config.embedding_dim};
    Tensor cls_token = x2.reshape(new_shape_host, 2);
    std::cout << "VIT: paso 4 - Token CLS extraído" << std::endl;
    return mlp_head.forward(cls_token, isTraining);
}

__global__ void insertCLSGradKernel(const float *grad, float *grad_seq,
                                    size_t batchSize, size_t embDim,
                                    size_t strideB_seq, size_t strideT_seq)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batchSize)
    {
        for (size_t d = 0; d < embDim; ++d)
        {
            size_t index_seq = b * strideB_seq + 0 * strideT_seq + d;
            size_t index_grad = b * embDim + d;
            grad_seq[index_seq] = grad[index_grad];
        }
    }
}

Tensor VisionTransformer::backward(const Tensor &outputGradient)
{
    Tensor grad = mlp_head.backward(outputGradient);
    grad.printFirstElements("VIT::Gradiente de la cabeza MLP");
    size_t batchSize = outputGradient.dim(0);
    size_t embDim = config.embedding_dim;
    size_t numTokens = this->num_tokens;

    // Crear tensor de gradientes con ceros para toda la secuencia
    Tensor grad_seq({batchSize, numTokens, embDim});
    grad_seq.fill(0.0f);
    grad_seq.printFirstElements("VIT::Gradiente secuencia inicial");

    // --- Transferencia del gradiente del token CLS a la posición [b, 0, d] ---
    dim3 blockSize(128);
    dim3 gridSize((batchSize + blockSize.x - 1) / blockSize.x);
    cudaGetLastError();
    insertCLSGradKernel<<<gridSize, blockSize>>>(
        grad.getData(), grad_seq.getData(),
        batchSize, embDim,
        grad_seq.stride(0), grad_seq.stride(1));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("[CUDA ERROR - scaleKernel (backward)] %s\n", cudaGetErrorString(err));
        throw std::runtime_error("MultiHeadAttention::backward: error al aplicar escalamiento inverso");
    }
    grad_seq.printFirstElements("VIT::Gradiente secuencia después de transferir CLS");

    // --- Backward ---
    grad = final_norm.backward(grad_seq);
    grad.printFirstElements("VIT::Gradiente después de LayerNorm final");

    for (int i = static_cast<int>(encoder_blocks.size()) - 1; i >= 0; --i)
    {
        grad = encoder_blocks[i].backward(grad);
        grad.printFirstElements("VIT::Gradiente después de bloque encoder " + std::to_string(i));
    }
    grad = embeddings.backward(grad);
    grad.printFirstElements("VIT::Gradiente después de Embeddings");
    return grad;
}

std::vector<Tensor *> VisionTransformer::getParameters()
{
    std::vector<Tensor *> params;

    auto emb_params = embeddings.getParameters();
    params.insert(params.end(), emb_params.begin(), emb_params.end());

    for (auto &block : encoder_blocks)
    {
        auto block_params = block.getParameters();
        params.insert(params.end(), block_params.begin(), block_params.end());
    }

    auto norm_params = final_norm.getParameters();
    params.insert(params.end(), norm_params.begin(), norm_params.end());

    auto head_params = mlp_head.getParameters();
    params.insert(params.end(), head_params.begin(), head_params.end());

    return params;
}

std::vector<Tensor *> VisionTransformer::getGradients()
{
    std::vector<Tensor *> grads;

    auto emb_grads = embeddings.getGradients();
    grads.insert(grads.end(), emb_grads.begin(), emb_grads.end());

    for (auto &block : encoder_blocks)
    {
        auto block_grads = block.getGradients();
        grads.insert(grads.end(), block_grads.begin(), block_grads.end());
    }

    auto norm_grads = final_norm.getGradients();
    grads.insert(grads.end(), norm_grads.begin(), norm_grads.end());

    auto head_grads = mlp_head.getGradients();
    grads.insert(grads.end(), head_grads.begin(), head_grads.end());

    return grads;
}