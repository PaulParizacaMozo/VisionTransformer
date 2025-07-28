#include "layers/PatchEmbedding.hpp"
#include <stdexcept>

/**
 * @brief Constructor de la capa PatchEmbedding, ahora usando Conv2d.
 */
PatchEmbedding::PatchEmbedding(size_t image_height, size_t image_width, size_t patch_size, size_t in_channels,
                               size_t embedding_dim)
    : patch_size(patch_size), embedding_dim(embedding_dim) {

  if (image_height % patch_size != 0 || image_width % patch_size != 0) {
    throw std::invalid_argument("Las dimensiones de la imagen deben ser divisibles por el tamaño del parche.");
  }

  // Calculamos las dimensiones de la secuencia de salida
  this->num_patches_h = image_height / patch_size;
  this->num_patches_w = image_width / patch_size;
  this->num_patches = num_patches_h * num_patches_w;

  // --- El Corazón del Cambio ---
  // Instanciamos una capa Conv2d para que haga el trabajo de parcheo y proyección.
  // El truco consiste en usar un kernel_size y un stride del mismo tamaño que el parche.
  // Esto hace que la convolución opere sobre parches no superpuestos, que es exactamente lo que queremos.
  // - in_channels: Canales de la imagen de entrada (1 para MNIST, 3 para RGB).
  // - embedding_dim: Canales de salida, que es nuestra dimensión de embedding (D).
  // - kernel_size: patch_size
  // - stride: patch_size
  // - padding: 0
  this->projectionLayer = std::make_unique<Conv2d>(in_channels, embedding_dim, patch_size, patch_size, 0);
}

/**
 * @brief Realiza el forward pass usando la capa Conv2d.
 */
Tensor PatchEmbedding::forward(const Tensor &input, bool isTraining) {
  const size_t batchSize = input.getShape()[0];

  // 1. Aplicar la convolución
  // La entrada es el batch de imágenes: {B, C_in, H, W}.
  // La salida de la Conv2d es un mapa de características: {B, D, H_out, W_out}.
  // Como H_out = (H - P)/P + 1 = H/P, la forma es {B, D, num_patches_h, num_patches_w}.
  Tensor x = this->projectionLayer->forward(input, isTraining);

  // 2. Aplanar las dimensiones espaciales
  // Queremos pasar de {B, D, H/P, W/P} a {B, D, N}, donde N = H/P * W/P es el número de parches.
  x = x.reshape({batchSize, this->embedding_dim, this->num_patches});

  // 3. Transponer para obtener el formato de secuencia que espera el Transformer
  // Queremos {B, N, D} para que la dimensión de la secuencia sea la segunda.
  // {B, D, N} -> {B, N, D}
  x = x.transpose(1, 2);
  x.contiguous();
  return x;
}

/**
 * @brief Realiza el backward pass revirtiendo las operaciones del forward.
 */
Tensor PatchEmbedding::backward(const Tensor &outputGradient) {
  Tensor grad_in = outputGradient.isContiguous() ? outputGradient : outputGradient.contiguous();
  const size_t batchSize = outputGradient.getShape()[0];
  
  // El gradiente de entrada (outputGradient) tiene la forma {B, N, D}.
  // Debemos revertir las operaciones para obtener un gradiente de la forma {B, C_in, H, W}.

  // 1. Invertir la transposición: .transpose(1, 2)
  // {B, N, D} -> {B, D, N}
  Tensor grad = outputGradient.transpose(1, 2);

  grad = grad.contiguous(); // Necesario antes del reshape
  // 2. Invertir el aplanamiento (reshape)
  // {B, D, N} -> {B, D, num_patches_h, num_patches_w}
  grad = grad.reshape({batchSize, this->embedding_dim, this->num_patches_h, this->num_patches_w});
  
  // 3. Propagar el gradiente hacia atrás a través de la capa de convolución.
  // La capa Conv2d se encargará de la compleja operación de col2im.
  return this->projectionLayer->backward(grad);
}

/**
 * @brief Delega la obtención de parámetros a la capa Conv2d interna.
 */
std::vector<Tensor *> PatchEmbedding::getParameters() {
  return this->projectionLayer->getParameters();
}

/**
 * @brief Delega la obtención de gradientes a la capa Conv2d interna.
 */
std::vector<Tensor *> PatchEmbedding::getGradients() {
  return this->projectionLayer->getGradients();
}
