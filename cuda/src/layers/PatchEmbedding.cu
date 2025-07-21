#include "layers/PatchEmbedding.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

PatchEmbedding::PatchEmbedding(size_t image_height, size_t image_width, size_t patch_size,
                               size_t in_channels, size_t embedding_dim)
    : image_height(image_height), image_width(image_width), patch_size(patch_size),
      in_channels(in_channels), embedding_dim(embedding_dim)
{

    if (image_height % patch_size != 0 || image_width % patch_size != 0)
        throw std::invalid_argument("Las dimensiones de la imagen deben ser divisibles por el tamaño del parche.");

    this->num_patches_h = image_height / patch_size;
    this->num_patches_w = image_width / patch_size;
    this->num_patches = num_patches_h * num_patches_w;
    this->patch_dim = patch_size * patch_size * in_channels;

    this->projectionLayer = std::make_unique<Dense>(this->patch_dim, this->embedding_dim);
}

__global__ void extractPatchesKernel(
    const float *input, float *patches,
    size_t batchSize, size_t in_channels,
    size_t image_height, size_t image_width,
    size_t patch_size, size_t patch_dim,
    size_t num_patches_h, size_t num_patches_w)
{
    int patch_global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (patch_global_idx >= batchSize * num_patches_h * num_patches_w)
        return;
    int b = patch_global_idx / (num_patches_h * num_patches_w);
    int patch_idx_in_image = patch_global_idx % (num_patches_h * num_patches_w);
    int ph = patch_idx_in_image / num_patches_w;
    int pw = patch_idx_in_image % num_patches_w;
    int h_start = ph * patch_size;
    int w_start = pw * patch_size;
    float *out_ptr = patches + patch_global_idx * patch_dim;
    int idx = 0;
    for (int c = 0; c < in_channels; ++c)
        for (int h = 0; h < patch_size; ++h)
            for (int w = 0; w < patch_size; ++w)
            {
                int in_idx =
                    (((b * in_channels + c) * image_height + (h_start + h)) * image_width + (w_start + w));
                out_ptr[idx++] = input[in_idx];
            }
}

Tensor PatchEmbedding::forward(const Tensor &input, bool isTraining)
{
    size_t *inputShape = input.getShapeHost();
    int dims = input.getNDim();
    size_t batchSize = inputShape[0];

    Tensor patches_flat({batchSize * this->num_patches, this->patch_dim});
    int totalPatches = batchSize * this->num_patches;

    int threads = 256;
    int blocks = (totalPatches + threads - 1) / threads;

    extractPatchesKernel<<<blocks, threads>>>(
        input.getData(), patches_flat.getData(),
        batchSize, in_channels,
        image_height, image_width,
        patch_size, patch_dim,
        num_patches_h, num_patches_w);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "!! [PatchEmbedding] CUDA Error after extractPatchesKernel: "
                  << cudaGetErrorString(err) << std::endl;
    }

    if (isTraining)
        this->flattenedPatches = patches_flat;

    Tensor projected_patches = this->projectionLayer->forward(patches_flat, isTraining);
    size_t newShape[] = {batchSize, this->num_patches, this->embedding_dim};
    Tensor reshaped = projected_patches.reshape(newShape, 3);
    Tensor copia_reshaped(reshaped);
    return copia_reshaped;
}

__global__ void scatterPatchGradKernel(
    const float *patch_grad, float *input_grad,
    size_t batchSize, size_t in_channels,
    size_t image_height, size_t image_width,
    size_t patch_size, size_t patch_dim,
    size_t num_patches_h, size_t num_patches_w)
{
    int patch_global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (patch_global_idx >= batchSize * num_patches_h * num_patches_w)
        return;

    int b = patch_global_idx / (num_patches_h * num_patches_w);
    int patch_idx_in_image = patch_global_idx % (num_patches_h * num_patches_w);
    int ph = patch_idx_in_image / num_patches_w;
    int pw = patch_idx_in_image % num_patches_w;

    int h_start = ph * patch_size;
    int w_start = pw * patch_size;

    const float *patch_ptr = patch_grad + patch_global_idx * patch_dim;

    int idx = 0;
    for (int c = 0; c < in_channels; ++c)
        for (int h = 0; h < patch_size; ++h)
            for (int w = 0; w < patch_size; ++w)
            {
                int out_idx = (((b * in_channels + c) * image_height + (h_start + h)) * image_width + (w_start + w));
                atomicAdd(&input_grad[out_idx], patch_ptr[idx++]);
            }
}

Tensor PatchEmbedding::backward(const Tensor &outputGradient)
{
    const auto &gradShape = outputGradient.getShapeHost();
    size_t batchSize = gradShape[0];

    size_t grad2DShape[] = {batchSize * this->num_patches, this->embedding_dim};
    Tensor grad2D = outputGradient.reshape(grad2DShape, 2);

    Tensor patch_gradient = this->projectionLayer->backward(grad2D);

    Tensor input_gradient({batchSize, this->in_channels, this->image_height, this->image_width});
    input_gradient.fill(0.0f);

    int totalPatches = batchSize * this->num_patches;
    int threads = 256;
    int blocks = (totalPatches + threads - 1) / threads;

    scatterPatchGradKernel<<<blocks, threads>>>(
        patch_gradient.getData(), input_gradient.getData(),
        batchSize, this->in_channels,
        this->image_height, this->image_width,
        this->patch_size, this->patch_size * this->patch_size * this->in_channels,
        this->num_patches_h, this->num_patches_w);
    cudaDeviceSynchronize();

    return input_gradient;
}

std::vector<Tensor *> PatchEmbedding::getParameters()
{
    return this->projectionLayer->getParameters();
}

std::vector<Tensor *> PatchEmbedding::getGradients()
{
    return this->projectionLayer->getGradients();
}
