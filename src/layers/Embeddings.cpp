#include "layers/Embeddings.hpp"
#include "layers/Conv2D.hpp"
#include "core/Tensor.hpp" // Para expand, concatenate
#include <iostream>

Embeddings::Embeddings(size_t image_height, size_t image_width, size_t patch_size, size_t in_channels, size_t embedding_dim)
    : embedding_dim(embedding_dim) {

  // patcher = std::make_unique<PatchEmbedding>(image_height, image_width, patch_size, in_channels, embedding_dim);
  // this->num_patches = patcher->getNumPatches();

  // Inicializamos la capa Conv2D en lugar de PatchEmbedding
  conv2d = std::make_unique<Conv2D>(in_channels, embedding_dim, patch_size, patch_size, 0);
  
  // El número de parches debe basarse en las dimensiones de la salida de la convolución
  size_t output_height = (image_height - patch_size) / patch_size + 1;
  size_t output_width = (image_width - patch_size) / patch_size + 1;
  this->num_patches = output_height * output_width; // Número de parches

  float stddev = 0.02f;
  // Inicializar parámetros entrenables con valores pequeños aleatorios
  clsToken = Tensor({1, 1, embedding_dim});
  clsToken.randomizeNormal(0.0f, stddev);

  positionalEncoding = Tensor({1, 1+this->num_patches, embedding_dim});
  positionalEncoding.randomizeNormal(0.0f, stddev);

  // Inicializar gradientes
  clsTokenGradient = Tensor(clsToken.getShape());
  positionalEncodingGradient = Tensor(positionalEncoding.getShape());
}

Tensor Embeddings::forward(const Tensor &input, bool isTraining) {
  size_t batchSize = input.getShape()[0];

  // 1. Obtener los embeddings de los parches a través de la convolución
  Tensor patch_embeddings = this->conv2d->forward(input, isTraining); // -> {B, D, H_out, W_out}
  // std::cout << "Patches shape: " << patch_embeddings.shapeToString() << std::endl;

  // Calcular el número de parches
  size_t outputHeight = patch_embeddings.getShape()[2];
  size_t outputWidth = patch_embeddings.getShape()[3];
  size_t num_patches = outputHeight * outputWidth;  // Número total de parches
  std::cout << "Number of patches: " << num_patches << " | ";
  std::cout << "Input shape: " << input.shapeToString() << " | ";
  std::cout << "Patch embeddings shape: " << patch_embeddings.shapeToString() << " | ";

  
  // Verificar que el número total de elementos coincida
  size_t total_elements = batchSize * num_patches * embedding_dim;
  size_t original_elements = patch_embeddings.getShape()[0] * patch_embeddings.getShape()[1] * patch_embeddings.getShape()[2] * patch_embeddings.getShape()[3];

  if (original_elements != total_elements) {
    throw std::runtime_error("El número total de elementos no coincide entre el tensor original y la forma objetivo del reshape.");
  }

  // Aplanamos la salida de la convolución para convertirla en una secuencia de parches
  Tensor patch_embeddings_flat = patch_embeddings.reshape({batchSize, num_patches, embedding_dim}); // -> {B, N, D}
  std::cout << "Reshaped patches shape: " << patch_embeddings_flat.shapeToString() << std::endl;

  // 2. Añadir la codificación posicional
  // patch_embeddings_flat.addBroadcast(this->positionalEncoding);  // (B, N, D) + (1, N, D)
  
  Tensor cls_token_batch({batchSize, 1, this->embedding_dim});
  Tensor embeddings_with_cls = concatenate({cls_token_batch, patch_embeddings_flat}, 1); // -> {B, N+1, D}
  std::cout << "embeddings_with_cls shape: " << embeddings_with_cls.shapeToString() << std::endl;
  embeddings_with_cls.addBroadcast(this->positionalEncoding);
  std::cout << "Embeddings with positional encoding shape: " << patch_embeddings_flat.shapeToString() << std::endl;
  
  // return patch_embeddings_flat; // Devolvemos los parches codificados posicionalmente
  return embeddings_with_cls;
}


Tensor Embeddings::backward(const Tensor &outputGradient) {
    this->positionalEncodingGradient = outputGradient.sum(0);
    Tensor grad_before_pos = outputGradient;

    // Des-concatenar el gradiente
    Tensor grad_cls = grad_before_pos.slice(1, 0, 1);
    Tensor grad_patches_view = grad_before_pos.slice(1, 1, this->num_patches); // Vista no contigua

    this->clsTokenGradient = grad_cls.sum(0);
    // Paso 1: Calcular el gradiente de la codificación posicional
    // this->positionalEncodingGradient = outputGradient.sum(0);  // (B, N, D) -> (1, N, D)

    // Paso 2: Eliminar el gradiente del CLS, ya que no existe
    // Simplemente, el gradiente es directamente el gradiente de los parches
    // Tensor grad_patches_view = outputGradient;  // (B, N, D) -> (B, N, D) (sin token CLS)

    // Paso 3: Garantizar que el tensor de gradientes de los parches sea contiguo
    Tensor grad_patches_contiguous(grad_patches_view.getShape());
    const auto &shape = grad_patches_view.getShape();

    // Usamos el operator() que sabe cómo manejar los strides de la vista
    #pragma omp parallel for collapse(3)
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            for (size_t k = 0; k < shape[2]; ++k) {
                grad_patches_contiguous(i, j, k) = grad_patches_view(i, j, k);
            }
        }
    }

    // Paso 4: Pasamos el gradiente contiguo a la capa Conv2D para calcular el gradiente de la entrada
    Tensor input_image_gradient = this->conv2d->backward(grad_patches_contiguous);

    return input_image_gradient;
}


std::vector<Tensor *> Embeddings::getParameters() {
  // Obtenemos los parámetros de la capa interna (pesos y bias de la Dense)
  // auto params = this->patcher->getParameters();
  auto params = this->conv2d->getParameters();
  // Añadimos nuestros propios parámetros
  params.push_back(&this->clsToken);
  params.push_back(&this->positionalEncoding);
  return params;
}

std::vector<Tensor *> Embeddings::getGradients() {
  // auto grads = this->patcher->getGradients();
  auto grads = this->conv2d->getGradients();
  grads.push_back(&this->clsTokenGradient);
  grads.push_back(&this->positionalEncodingGradient);
  return grads;
}
