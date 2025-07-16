#include "model/TransformerEncoderBlock.hpp"
#include <iostream> 

/**
 * @brief Constructor que inicializa todas las sub-capas del bloque.
 */
TransformerEncoderBlock::TransformerEncoderBlock(size_t embedding_dim, size_t num_heads, size_t mlp_hidden_dim)
    : norm1(embedding_dim), attention(embedding_dim, num_heads), norm2(embedding_dim), ffn(embedding_dim, mlp_hidden_dim) {
  // El cuerpo del constructor está vacío.
}

/**
 * @brief Define el flujo de datos forward del bloque, incluyendo las conexiones residuales.
 */
Tensor TransformerEncoderBlock::forward(const Tensor &input, bool isTraining) {
  if (isTraining) {
    // Guardamos la entrada para la primera conexión residual en el backward pass
    this->input_skip1 = input;
  }

  // Sub-capa 1: Multi-Head Attention
  Tensor x = norm1.forward(input, isTraining);
  std::cout<<"TransformerEncoderBlock - norm1 - Despues";  x.printFirstN(20);
  x = attention.forward(x, isTraining);
  std::cout<<"TransformerEncoderBlock - attention - Despues"; x.printFirstN(20);

  Tensor residual1 = input + x;

  if (isTraining) {
    // Guardamos la entrada de la segunda conexión residual
    this->input_skip2 = residual1;
  }

  // Sub-capa 2: Feed-Forward Network
  Tensor y = norm2.forward(residual1, isTraining);
  std::cout<< "TransformerEncoderBlock - norm2 - Despues"; y.printFirstN(20);
  y = ffn.forward(y, isTraining);
  return residual1 + y;
}

/**
 * @brief Define el flujo de gradientes hacia atrás, manejando las conexiones residuales.
 */
Tensor TransformerEncoderBlock::backward(const Tensor &outputGradient) {
  // El gradiente 'outputGradient' (dL/dY) fluye hacia atrás desde la última suma.

  // --- Inversa de la segunda conexión residual: Z = X + FFN(Norm(X)) ---
  // El gradiente se bifurca: uno va por la rama FFN, otro por la rama skip.
  Tensor grad_skip2 = outputGradient;

  Tensor grad_ffn = outputGradient;
  grad_ffn = ffn.backward(grad_ffn);
  grad_ffn = norm2.backward(grad_ffn);

  // Se suman los gradientes de ambas ramas para obtener el gradiente de la entrada de este bloque.
  Tensor grad1 = grad_skip2 + grad_ffn; // Gradiente con respecto a 'residual1'

  // --- Inversa de la primera conexión residual: residual1 = input + MHA(Norm(input)) ---
  Tensor grad_skip1 = grad1;

  Tensor grad_mha = grad1;
  grad_mha = attention.backward(grad_mha);
  grad_mha = norm1.backward(grad_mha);

  // Sumamos los gradientes para obtener el gradiente final con respecto a 'input'.
  return grad_skip1 + grad_mha;
}

/**
 * @brief Recolecta los parámetros de todas las sub-capas.
 */
std::vector<Tensor *> TransformerEncoderBlock::getParameters() {
  std::vector<Tensor *> params;
  auto p1 = norm1.getParameters();
  params.insert(params.end(), p1.begin(), p1.end());
  auto p2 = attention.getParameters();
  params.insert(params.end(), p2.begin(), p2.end());
  auto p3 = norm2.getParameters();
  params.insert(params.end(), p3.begin(), p3.end());
  auto p4 = ffn.getParameters();
  params.insert(params.end(), p4.begin(), p4.end());
  return params;
}

/**
 * @brief Recolecta los gradientes de todas las sub-capas.
 */
std::vector<Tensor *> TransformerEncoderBlock::getGradients() {
  std::vector<Tensor *> grads;
  auto g1 = norm1.getGradients();
  grads.insert(grads.end(), g1.begin(), g1.end());
  auto g2 = attention.getGradients();
  grads.insert(grads.end(), g2.begin(), g2.end());
  auto g3 = norm2.getGradients();
  grads.insert(grads.end(), g3.begin(), g3.end());
  auto g4 = ffn.getGradients();
  grads.insert(grads.end(), g4.begin(), g4.end());
  return grads;
}
