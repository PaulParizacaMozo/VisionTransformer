#include "layers/Embeddings.cuh"
#include "core/Tensor.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

Embeddings::Embeddings(size_t image_height, size_t image_width, size_t patch_size, size_t in_channels, size_t embedding_dim)
    : embedding_dim(embedding_dim)
{

    patcher = std::make_unique<PatchEmbedding>(image_height, image_width, patch_size, in_channels, embedding_dim);
    this->num_patches = patcher->getNumPatches();

    float stddev = 0.02f;

    clsToken = Tensor({1, 1, embedding_dim});
    clsToken.randomizeNormal(0.0f, stddev);

    positionalEncoding = Tensor({1, 1 + this->num_patches, embedding_dim});
    positionalEncoding.randomizeNormal(0.0f, stddev);
    auto cls_shape = clsToken.getShapeHost();
    auto pos_shape = positionalEncoding.getShapeHost();
    clsTokenGradient = Tensor({cls_shape[0], cls_shape[1], cls_shape[2]});
    positionalEncodingGradient = Tensor({pos_shape[0], pos_shape[1], pos_shape[2]});
}

Tensor Embeddings::forward(const Tensor &input, bool isTraining)
{
    size_t batchSize = input.dim(0);
    std::cout << "Batch size: " << batchSize << std::endl;

    Tensor patch_embeddings = this->patcher->forward(input, isTraining); // [B, N, D]

    if (patch_embeddings.dims() != 3)
        throw std::runtime_error("Patch embeddings should have 3 dimensions: [B, N, D]");

    // Expandir clsToken a [B, 1, D]
    Tensor cls_token_batch({batchSize, 1, this->embedding_dim}); //{B, 1, D}

    Tensor concat_inputs[] = {cls_token_batch, patch_embeddings};
    Tensor embeddings_with_cls = concatenate(concat_inputs, 2, 1); // [B, N+1, D]

    // Sumar positionalEncoding broadcasted
    embeddings_with_cls.addBroadcast(this->positionalEncoding); // [B, N+1, D] + [1, N+1, D]

    return embeddings_with_cls;
}

__global__ void copy3DStridedKernel(const float *__restrict__ src, float *dst,
                                    size_t *shape, size_t *src_strides, size_t *dst_strides)
{
    size_t B = shape[0], N = shape[1], D = shape[2];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = B * N * D;

    if (idx >= total)
        return;

    size_t i = idx / (N * D);
    size_t j = (idx / D) % N;
    size_t k = idx % D;

    size_t src_offset = i * src_strides[0] + j * src_strides[1] + k * src_strides[2];
    size_t dst_offset = i * dst_strides[0] + j * dst_strides[1] + k * dst_strides[2];

    dst[dst_offset] = src[src_offset];
}

Tensor Embeddings::backward(const Tensor &outputGradient)
{
    // Sumar gradiente posicional
    this->positionalEncodingGradient = outputGradient.sum(0); // [1, N+1, D]

    // Separar gradientes
    Tensor grad_cls = outputGradient.slice(1, 0, 1);                          // [B, 1, D]
    Tensor grad_patches_view = outputGradient.slice(1, 1, this->num_patches); // [B, N, D] (no contiguo)

    // 3. Gradiente para clsToken
    this->clsTokenGradient = grad_cls.sum(0); // [1, 1, D]

    // 4. Crear tensor contiguo manualmente con metadatos de host
    auto shape_host = grad_patches_view.getShapeHost();
    Tensor grad_patches_contiguous({shape_host[0], shape_host[1], shape_host[2]}); // [B, N, D]

    // 5. Lanzar kernel que copia datos usando strides
    size_t total = grad_patches_view.size();
    dim3 blockSize(256);
    dim3 gridSize((total + blockSize.x - 1) / blockSize.x);

    copy3DStridedKernel<<<gridSize, blockSize>>>(
        grad_patches_view.getData(),
        grad_patches_contiguous.getData(),
        grad_patches_view.getShape(),
        grad_patches_view.getStrides(),
        grad_patches_contiguous.getStrides());
    cudaDeviceSynchronize(); // esperar copia

    // Llamar backward al patcher con tensor ya contiguo
    Tensor input_image_gradient = this->patcher->backward(grad_patches_contiguous);
    return input_image_gradient;
}

std::vector<Tensor *> Embeddings::getParameters()
{
    auto params = this->patcher->getParameters();
    params.push_back(&this->clsToken);
    params.push_back(&this->positionalEncoding);
    return params;
}

std::vector<Tensor *> Embeddings::getGradients()
{
    auto grads = this->patcher->getGradients();
    grads.push_back(&this->clsTokenGradient);
    grads.push_back(&this->positionalEncodingGradient);
    return grads;
}
