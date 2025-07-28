#include "model/VisionTransformer.hpp"
#include <iostream>
#include "utils/CudaUtils.hpp"

/**
 * @brief Constructor que inicializa todas las capas del modelo.
 */
VisionTransformer::VisionTransformer(const ViTConfig &config)
    : config(config),
      embeddings(config.image_size, config.image_size, config.patch_size, config.in_channels, config.embedding_dim),
      final_norm(config.embedding_dim), mlp_head(config.embedding_dim, config.num_classes)
{
  this->num_tokens = 1 + (config.image_size / config.patch_size) * (config.image_size / config.patch_size);

  for (size_t i = 0; i < config.num_layers; ++i)
  {
    encoder_blocks.emplace_back(config.embedding_dim, config.num_heads, config.mlp_hidden_dim, config.dropout_rate);
  }
}

/**
 * @brief Encadena el forward pass de todo el modelo.
 */
Tensor VisionTransformer::forward(const Tensor &input, bool isTraining)
{
  Tensor x = embeddings.forward(input, isTraining);

  for (auto &block : encoder_blocks)
  {
    x = block.forward(x, isTraining);
  }

  x = final_norm.forward(x, isTraining);

  if (isTraining)
  {
    // Guardamos la salida normalizada para el backward pass
    this->final_norm_output = x;
  }

  // Extraemos solo el token CLS (en la posición 0) para la clasificación
  Tensor cls_token = x.slice(1, 0, 1);
  cls_token = contiguous_cuda(cls_token);
  // cls_token = cls_token.contiguous();
  // if (verify(cls_token, cls_token_cuda, 1e-5f) == false)
  // {
  //   std::cerr << "Error en la verificación de contiguous para cls_token\n";
  // }
  cls_token = cls_token.reshape({input.getShape()[0], config.embedding_dim});

  return mlp_head.forward(cls_token, isTraining);
}

/**
 * @brief Encadena el backward pass de todo el modelo en orden inverso.
 */
Tensor VisionTransformer::backward(const Tensor &outputGradient)
{
  Tensor grad = mlp_head.backward(outputGradient);
  size_t batchSize = outputGradient.getShape()[0];

  // El gradiente está solo para el token CLS. Necesitamos "re-inyectarlo"
  // en una secuencia completa de gradientes (con ceros para los otros tokens).
  Tensor grad_seq({batchSize, this->num_tokens, config.embedding_dim});

  grad_seq.fill(0.0f);
  for (size_t b = 0; b < grad_seq.getShape()[0]; ++b)
  {
    for (size_t d = 0; d < config.embedding_dim; ++d)
    {
      grad_seq(b, 0, d) = grad(b, d);
    }
  }

  grad = final_norm.backward(grad_seq);

  // Propagamos hacia atrás a través de los bloques codificadores en orden inverso
  for (int i = encoder_blocks.size() - 1; i >= 0; --i)
  {
    grad = encoder_blocks[i].backward(grad);
  }

  grad = embeddings.backward(grad);

  // Devolvemos el gradiente final (con respecto a la imagen de entrada), aunque no suele usarse.
  return grad;
}

/**
 * @brief Recolecta los parámetros de todas las capas del modelo.
 */
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

/**
 * @brief Recolecta los gradientes de todas las capas del modelo.
 */
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
