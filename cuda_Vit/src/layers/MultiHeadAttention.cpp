#include "layers/MultiHeadAttention.hpp"
#include "core/Tensor.hpp" // Para las funciones libres
#include <cmath>           // Para std::sqrt
#include "utils/CudaUtils.hpp"
#include <iostream>

// Declaración de la función softmax que usaremos
// Tensor softmax(const Tensor &logits, int axis);

// Tensor softmax_backward(const Tensor &grad_output, const Tensor &softmax_output);

MultiHeadAttention::MultiHeadAttention(size_t embedding_dim, size_t num_heads)
    : embedding_dim(embedding_dim), num_heads(num_heads)
{

  if (embedding_dim % num_heads != 0)
  {
    throw std::invalid_argument("embedding_dim debe ser divisible por num_heads.");
  }
  this->head_dim = embedding_dim / num_heads;

  q_proj = std::make_unique<Dense>(embedding_dim, embedding_dim);
  k_proj = std::make_unique<Dense>(embedding_dim, embedding_dim);
  v_proj = std::make_unique<Dense>(embedding_dim, embedding_dim);
  out_proj = std::make_unique<Dense>(embedding_dim, embedding_dim);
}

Tensor MultiHeadAttention::forward(const Tensor &input, bool isTraining)
{
  if (isTraining)
  {
    this->inputTensor = input;
  }

  const auto &s = input.getShape(); // {B, N, D}
  size_t B = s[0], N = s[1];

  // 1. Proyecciones Lineales
  Tensor q = q_proj->forward(input, isTraining); // -> {B, N, D}
  Tensor k = k_proj->forward(input, isTraining); // -> {B, N, D}
  Tensor v = v_proj->forward(input, isTraining); // -> {B, N, D}

  // 2. Dividir en cabezas
  // La maniobra estándar: reshape a 4D -> transpose -> reshape a 3D para BMM

  // {B, N, D} -> {B, N, h, d_h}
  q = q.reshape({B, N, this->num_heads, this->head_dim});
  k = k.reshape({B, N, this->num_heads, this->head_dim});
  v = v.reshape({B, N, this->num_heads, this->head_dim});

  // {B, N, h, d_h} -> {B, h, N, d_h}
  q = q.transpose(1, 2);
  k = k.transpose(1, 2);
  v = v.transpose(1, 2);

  // Ahora q, k, v son vistas no contiguas.
  // Para el siguiente reshape, necesitamos hacerlas contiguas.
  q = contiguous_cuda(q);
  k = contiguous_cuda(k);
  v = contiguous_cuda(v);

  // {B, h, N, d_h} -> {B*h, N, d_h}
  q = q.reshape({B * this->num_heads, N, this->head_dim});
  k = k.reshape({B * this->num_heads, N, this->head_dim});
  v = v.reshape({B * this->num_heads, N, this->head_dim});

  if (isTraining)
  {
    this->q_split = q;
    this->k_split = k;
    this->v_split = v;
  }

  // 3. Atención Escalada por Producto Punto
  Tensor context = scaledDotProductAttention(q, k, v); // -> {B*h, N, d_h}

  // 4. Re-ensamblar cabezas
  // Invertimos el proceso de división
  // {B*h, N, d_h} -> {B, h, N, d_h}
  context = context.reshape({B, this->num_heads, N, this->head_dim});

  // {B, h, N, d_h} -> {B, N, h, d_h}
  context = context.transpose(1, 2); // <- ¡Esta operación crea la vista no contigua!
  context = contiguous_cuda(context);

  // Ahora este reshape es seguro.
  // {B, N, h, d_h} -> {B, N, D}
  context = context.reshape({B, N, this->embedding_dim});

  // 5. Proyección de salida final
  return out_proj->forward(context, isTraining);
}

Tensor MultiHeadAttention::scaledDotProductAttention(const Tensor &q, const Tensor &k, const Tensor &v)
{
  float scale_factor = 1.0f / std::sqrt(static_cast<float>(this->head_dim));

  Tensor attention_c;
  Tensor output_c = scaledDotProductAttention_cuda(q, k, v, scale_factor, attention_c);
  this->attention_weights = attention_c;
  return output_c;
}

// --- Métodos restantes (por ahora vacíos o delegando) ---

Tensor MultiHeadAttention::backward(const Tensor &outputGradient)
{
  // outputGradient (dL/dY) tiene forma {B, N, D}
  const auto &inputShape = this->inputTensor.getShape();
  size_t B = inputShape[0], N = inputShape[1];

  // ----------------------------------------------------------------------
  // 1. Inversa de la Proyección de Salida (out_proj)
  // ----------------------------------------------------------------------
  Tensor grad = this->out_proj->backward(outputGradient); // -> {B, N, D}

  // ----------------------------------------------------------------------
  // 2. Inversa del Re-ensamblaje de Cabezas
  // ----------------------------------------------------------------------
  // FORWARD: context {B*h,N,d_h} -> reshape {B,h,N,d_h} -> transpose {B,N,h,d_h} -> contiguous -> reshape {B,N,D}
  // BACKWARD:

  grad = grad.reshape({B, N, this->num_heads, this->head_dim});
  grad = grad.transpose(1, 2);
  grad = contiguous_cuda(grad);
  grad = grad.reshape({B * this->num_heads, N, this->head_dim});
  // 'grad' es ahora dL/d(attention_output) con forma {B*h, N, d_h}

  // ----------------------------------------------------------------------
  // 3. Inversa de la Multiplicación Final de la Atención
  // ----------------------------------------------------------------------
  // FORWARD: attention_output = attention_weights @ V
  // Tensor V_T = this->v_split.transpose(1, 2);
  // Tensor d_attention_weights = batchMatrixMultiply(grad, V_T);
  Tensor V_T_contiguous = this->v_split.transpose(1, 2);
  V_T_contiguous = contiguous_cuda(V_T_contiguous);
  Tensor d_attention_weights = batchMatrixMultiply_cuda(grad, V_T_contiguous);

  Tensor attention_weights_T_contiguous = this->attention_weights.transpose(1, 2);
  attention_weights_T_contiguous = contiguous_cuda(attention_weights_T_contiguous);
  Tensor dV = batchMatrixMultiply_cuda(attention_weights_T_contiguous, grad);

  // ----------------------------------------------------------------------
  // 4. Inversa del Softmax
  // ----------------------------------------------------------------------
  // Usamos la nueva función para obtener el gradiente con respecto a las puntuaciones (scores)
  Tensor d_scores = softmax_backward_cuda(d_attention_weights, this->attention_weights);

  // ----------------------------------------------------------------------
  // 5. Inversa del Escalamiento y Q @ K^T
  // ----------------------------------------------------------------------

  // 5.1 Invertir el escalamiento
  float scale_factor = 1.0f / std::sqrt(static_cast<float>(this->head_dim));
  d_scores = scale_tensor_cuda(d_scores, scale_factor);

  // 5.2 Propagar a través de Q @ K^T
  // Forward: scores = Q @ K^T
  // dL/dQ = dL/d(scores) @ K
  // Tensor dQ = batchMatrixMultiply(d_scores, this->k_split);
  Tensor k_contiguous = contiguous_cuda(this->k_split);
  Tensor dQ = batchMatrixMultiply_cuda(d_scores, k_contiguous);

  // dL/dK = Q^T @ dL/d(scores)
  // Tensor Q_T = this->q_split.transpose(1, 2);
  // Tensor dK = batchMatrixMultiply(Q_T, d_scores);
  Tensor Q_T_contiguous = this->q_split.transpose(1, 2);
  Q_T_contiguous = contiguous_cuda(Q_T_contiguous);
  Tensor dK = batchMatrixMultiply_cuda(Q_T_contiguous, d_scores);

  // ----------------------------------------------------------------------
  // 6. Inversa de la División de Cabezas (Re-ensamblaje de Gradientes)
  // ----------------------------------------------------------------------
  auto reassemble_grads = [&](Tensor &g)
  {
    g = g.reshape({B, this->num_heads, N, this->head_dim});

    g = g.transpose(1, 2);
    g = contiguous_cuda(g);
    return g.reshape({B, N, this->embedding_dim});
  };

  dQ = reassemble_grads(dQ); // -> {B, N, D}
  dK = reassemble_grads(dK); // -> {B, N, D}
  dV = reassemble_grads(dV); // -> {B, N, D}

  // ----------------------------------------------------------------------
  // 7. Inversa de las Proyecciones de Entrada
  // ----------------------------------------------------------------------
  Tensor dInput_q = this->q_proj->backward(dQ);
  Tensor dInput_k = this->k_proj->backward(dK);
  Tensor dInput_v = this->v_proj->backward(dV); // ¡Este calculará gradientes reales para w_v, b_v!

  // ----------------------------------------------------------------------
  // 8. Suma de Gradientes
  // ----------------------------------------------------------------------
  // El gradiente de entrada es la suma de los gradientes de las 3 ramas.
  Tensor final_grad = dInput_q + dInput_k + dInput_v;

  return final_grad;
}

std::vector<Tensor *> MultiHeadAttention::getParameters()
{
  auto q_params = q_proj->getParameters();
  auto k_params = k_proj->getParameters();
  auto v_params = v_proj->getParameters();
  auto out_params = out_proj->getParameters();

  std::vector<Tensor *> all_params;
  all_params.insert(all_params.end(), q_params.begin(), q_params.end());
  all_params.insert(all_params.end(), k_params.begin(), k_params.end());
  all_params.insert(all_params.end(), v_params.begin(), v_params.end());
  all_params.insert(all_params.end(), out_params.begin(), out_params.end());
  return all_params;
}

std::vector<Tensor *> MultiHeadAttention::getGradients()
{
  auto q_grads = q_proj->getGradients();
  auto k_grads = k_proj->getGradients();
  auto v_grads = v_proj->getGradients();
  auto out_grads = out_proj->getGradients();

  std::vector<Tensor *> all_grads;
  all_grads.insert(all_grads.end(), q_grads.begin(), q_grads.end());
  all_grads.insert(all_grads.end(), k_grads.begin(), k_grads.end());
  all_grads.insert(all_grads.end(), v_grads.begin(), v_grads.end());
  all_grads.insert(all_grads.end(), out_grads.begin(), out_grads.end());
  return all_grads;
}
