#include "layers/Embeddings.hpp"
#include "core/Tensor.cuh" // Para expand, concatenate
#include <iostream>

Embeddings::Embeddings(size_t image_height, size_t image_width, size_t patch_size, size_t in_channels, size_t embedding_dim)
    : embedding_dim(embedding_dim)
{

  patcher = std::make_unique<PatchEmbedding>(image_height, image_width, patch_size, in_channels, embedding_dim);
  this->num_patches = patcher->getNumPatches();

  float stddev = 0.02f;
  // Inicializar parámetros entrenables con valores pequeños aleatorios
  clsToken = Tensor({1, 1, embedding_dim});
  clsToken.randomizeNormal(0.0f, stddev);

  positionalEncoding = Tensor({1, 1 + this->num_patches, embedding_dim});
  positionalEncoding.randomizeNormal(0.0f, stddev);

  // Inicializar gradientes
  clsTokenGradient = Tensor(clsToken.getShape());
  positionalEncodingGradient = Tensor(positionalEncoding.getShape());
}

Tensor Embeddings::forward(const Tensor &input, bool isTraining)
{
  // std::cout << "--Embeddings::forward--" << std::endl;
  size_t batchSize = input.getShape()[0];
  // std::cout << "Batch size: " << batchSize << std::endl;

  // 1. Obtener los embeddings de los parches
  Tensor patch_embeddings = this->patcher->forward(input, isTraining); // -> {B, N, D}
  // std::cout << "--Embeddings::Parches embeddings: " << patch_embeddings.shapeToString() << std::endl;
  // 2. Expandir el token CLS para que coincida con el tamaño del batch
  // expand() crea una vista {B, 1, D} sin copiar memoria.
  // Tensor cls_token_batch({batchSize, 1, this->embedding_dim});

  // 2. Expandir el token CLS para que coincida con el tamaño del batch
  Tensor cls_token_batch = clsToken.expand({batchSize, 1, embedding_dim});
  // std::cout << "--Embeddings::CLS token batch: " << cls_token_batch.shapeToString() << std::endl;

  // 3. Concatenar el CLS token y los parches a lo largo del eje de la secuencia (axis=1)
  Tensor embeddings_with_cls = concatenate({cls_token_batch, patch_embeddings}, 1); // -> {B, N+1, D}
  // std::cout << "--Embeddings::Embeddings con CLS: " << embeddings_with_cls.shapeToString() << std::endl;
  // 4. Añadir la codificación posicional
  // addBroadcast suma {1, N+1, D} a cada muestra de {B, N+1, D}
  embeddings_with_cls.addBroadcast(this->positionalEncoding);
  // std::cout << "--Embeddings::Embeddings con codificación posicional: " << embeddings_with_cls.shapeToString() << std::endl;

  return embeddings_with_cls;
}

Tensor Embeddings::backward(const Tensor &outputGradient)
{
  this->positionalEncodingGradient = outputGradient.sum(0);
  Tensor grad_before_pos = outputGradient;

  // Des-concatenar el gradiente
  Tensor grad_cls = grad_before_pos.slice(1, 0, 1);
  Tensor grad_patches_view = grad_before_pos.slice(1, 1, this->num_patches); // Vista no contigua

  this->clsTokenGradient = grad_cls.sum(0);

  // std::cout << this->clsTokenGradient.shapeToString() << std::endl;
  // clsTokenGradient.printDebugInfo("clsTokenGradient");
  // --- LA SOLUCIÓN DIRECTA ---
  // En lugar de llamar a .contiguous(), creamos un nuevo tensor y copiamos los datos.
  // Esto garantiza que el tensor que pasamos es 100% contiguo.
  Tensor grad_patches_contiguous(grad_patches_view.getShape());
  grad_patches_contiguous.embedding_backward(grad_patches_view);

  // Usamos el operator() que sabe cómo manejar los strides de la vista
  //   const auto &shape = grad_patches_view.getShape();
  // #pragma omp parallel for collapse(3)
  //   for (size_t i = 0; i < shape[0]; ++i)
  //   {
  //     for (size_t j = 0; j < shape[1]; ++j)
  //     {
  //       for (size_t k = 0; k < shape[2]; ++k)
  //       {
  //         grad_patches_contiguous(i, j, k) = grad_patches_view(i, j, k);
  //       }
  //     }
  //   }

  // Ahora pasamos este tensor garantizado-contiguo
  Tensor input_image_gradient = this->patcher->backward(grad_patches_contiguous);

  return input_image_gradient;
}

std::vector<Tensor *> Embeddings::getParameters()
{
  // Obtenemos los parámetros de la capa interna (pesos y bias de la Dense)
  auto params = this->patcher->getParameters();
  // Añadimos nuestros propios parámetros
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
