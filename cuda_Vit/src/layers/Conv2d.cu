#include "layers/Conv2d.cuh"
#include <cmath>
#include <stdexcept>
#include <vector>
#include "utils/CudaUtils.cuh"
#include <iostream>

Conv2d::Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding)
    : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding)
{

    // Inicialización de pesos Kaiming/He, mejor para arquitecturas con ReLU/GELU.
    float fan_in = static_cast<float>(in_channels * kernel_size * kernel_size);
    float stddev = std::sqrt(2.0f / fan_in);

    // Forma de los pesos: {out_channels, in_channels, kernel_size, kernel_size}
    this->weights = Tensor({out_channels, in_channels, kernel_size, kernel_size});
    this->weights.randomizeNormal(0.0f, stddev);

    // Bias se inicializa como {1, out_channels, 1, 1} para broadcasting 4D
    this->bias = Tensor({1, out_channels, 1, 1});
    this->bias.fill(0.0f);

    // Inicializar gradientes
    this->weightGradients = Tensor(this->weights.getShape());
    this->biasGradients = Tensor(this->bias.getShape());
}

Tensor Conv2d::forward(const Tensor &input, bool isTraining)
{
    if (isTraining)
    {
        this->inputTensor = input;
    }

    const auto &in_shape = input.getShape();
    size_t batch_size = in_shape[0];
    size_t in_h = in_shape[2];
    size_t in_w = in_shape[3];

    // 1. Calcular dimensiones de salida
    size_t out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    size_t out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    // 2. Transformar la entrada a una matriz de columnas (im2col)
    size_t patch_dim = this->in_channels * this->kernel_size * this->kernel_size;
    size_t num_patches = out_h * out_w;

    Tensor im2col_matrix = im2col_cuda(input, {patch_dim, batch_size, num_patches}, this->kernel_size, this->stride, this->padding);

    // 3. Remodelar los pesos para la multiplicación (sin copia)
    // De {out_C, in_C, kH, kW} a {out_C, in_C * kH * kW}
    Tensor reshaped_weights = this->weights.reshape({this->out_channels, patch_dim});

    // 4. Convolución como multiplicación de matrices
    // {out_C, patch_dim} @ {patch_dim, B * num_patches} -> {out_C, B * num_patches}
    Tensor conv_result = matrixMultiply_cuda(reshaped_weights, im2col_matrix);

    // 5. Remodelar la salida y añadir el bias
    // {out_C, B * num_patches} -> {out_C, B, num_patches} -> transpose -> {B, out_C, num_patches}
    Tensor output = conv_result.reshape({this->out_channels, batch_size, num_patches});
    output = output.transpose(0, 1);  // -> Crea una vista NO CONTIGUA
    output = contiguous_cuda(output); // Antes del reshape final, hacemos la vista contigua.
    // {B, out_C, num_patches} -> {B, out_C, out_H, out_W}
    output = output.reshape({batch_size, this->out_channels, out_h, out_w});
    output = addBroadcast_cuda(output, this->bias);
    return output;
}

Tensor Conv2d::backward(const Tensor &outputGradient)
{
    const auto &in_shape = this->inputTensor.getShape();
    size_t batch_size = in_shape[0];

    const auto &out_grad_shape = outputGradient.getShape();
    size_t out_h = out_grad_shape[2];
    size_t out_w = out_grad_shape[3];
    size_t num_patches = out_h * out_w;
    size_t patch_dim = this->in_channels * this->kernel_size * this->kernel_size;

    // --- 1. Calcular gradiente del bias ---
    // Sumamos el gradiente de salida a lo largo de las dimensiones B, H, W
    this->biasGradients = outputGradient.sum(0).sum(2).sum(3).reshape(this->bias.getShape());

    // --- 2. Preparar gradiente de salida y entrada (im2col) ---
    // {B, out_C, out_H, out_W} -> {B, out_C, num_patches} -> transpose -> {out_C, B, num_patches} -> {out_C, B * num_patches}
    // Preparamos el gradiente de salida, asegurando contigüidad
    Tensor reshaped_out_grad = outputGradient.reshape({batch_size, this->out_channels, num_patches});
    reshaped_out_grad = reshaped_out_grad.transpose(0, 1);  // -> Vista NO CONTIGUA
    reshaped_out_grad = contiguous_cuda(reshaped_out_grad); // Aseguramos que sea contiguo
    reshaped_out_grad = reshaped_out_grad.reshape({this->out_channels, batch_size * num_patches});
    Tensor im2col_matrix = im2col_cuda(this->inputTensor, {patch_dim, batch_size, num_patches}, this->kernel_size, this->stride, this->padding);

    // --- 3. Calcular gradiente de los pesos (dE/dW) ---
    // dW = dY @ X_im2col^T
    Tensor im2col_transposed = im2col_matrix.transpose(0, 1);
    im2col_transposed = contiguous_cuda(im2col_transposed);
    Tensor dW_flat = matrixMultiply_cuda(reshaped_out_grad, im2col_transposed);
    this->weightGradients = dW_flat.reshape(this->weights.getShape());

    // --- 4. Calcular gradiente de la entrada (dE/dX) ---
    // dX_col = W^T @ dY
    Tensor reshaped_weights = this->weights.reshape({this->out_channels, patch_dim});
    Tensor weights_transposed = reshaped_weights.transpose(0, 1);
    weights_transposed = contiguous_cuda(weights_transposed);
    Tensor dX_col = matrixMultiply_cuda(weights_transposed, reshaped_out_grad);
    return col2im_cuda(dX_col, in_shape, this->kernel_size, this->stride, this->padding);
}

// --- Getters ---
std::vector<Tensor *> Conv2d::getParameters() { return {&this->weights, &this->bias}; }
std::vector<Tensor *> Conv2d::getGradients() { return {&this->weightGradients, &this->biasGradients}; }
