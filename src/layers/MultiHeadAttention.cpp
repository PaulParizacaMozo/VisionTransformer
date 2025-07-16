#include "layers/MultiHeadAttention.hpp"
#include "core/Tensor.hpp" // Para las funciones libres
#include <cmath>           // Para std::sqrt
#include <iostream>

// Declaración de la función softmax que usaremos
Tensor softmax(const Tensor &logits, int axis);

Tensor softmax_backward(const Tensor &grad_output, const Tensor &softmax_output);

MultiHeadAttention::MultiHeadAttention(size_t embedding_dim, size_t num_heads)
    : embedding_dim(embedding_dim), num_heads(num_heads) {

  if (embedding_dim % num_heads != 0) {
    throw std::invalid_argument("embedding_dim debe ser divisible por num_heads.");
  }
  this->head_dim = embedding_dim / num_heads;

  q_proj = std::make_unique<Dense>(embedding_dim, embedding_dim);
  k_proj = std::make_unique<Dense>(embedding_dim, embedding_dim);
  v_proj = std::make_unique<Dense>(embedding_dim, embedding_dim);
  out_proj = std::make_unique<Dense>(embedding_dim, embedding_dim);
}

Tensor MultiHeadAttention::forward(const Tensor &input, bool isTraining) {
  if (isTraining) {
    this->inputTensor = input;
  }

  const auto &s = input.getShape(); // {B, N, D}
  size_t B = s[0], N = s[1];

  // 1. Proyecciones Lineales
  Tensor q = q_proj->forward(input, isTraining); // -> {B, N, D}
  Tensor k = k_proj->forward(input, isTraining); // -> {B, N, D}
  Tensor v = v_proj->forward(input, isTraining); // -> {B, N, D}

  std::cout<<"q - Despues";  q.printFirstN(20);
  std::cout<<"k - Despues";  k.printFirstN(20);
  std::cout<<"v - Despues";  v.printFirstN(20);

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
  q = q.contiguous();
  k = k.contiguous();
  v = v.contiguous();

  // {B, h, N, d_h} -> {B*h, N, d_h}
  q = q.reshape({B * this->num_heads, N, this->head_dim});
  k = k.reshape({B * this->num_heads, N, this->head_dim});
  v = v.reshape({B * this->num_heads, N, this->head_dim});

  if (isTraining) {
    this->q_split = q;
    this->k_split = k;
    this->v_split = v;
  }

  // 3. Atención Escalada por Producto Punto
  Tensor context = scaledDotProductAttention(q, k, v); // -> {B*h, N, d_h}
  std::cout<<"context - Despues";  context.printFirstN(20); std::cout<<std::flush;

  // 4. Re-ensamblar cabezas
  // Invertimos el proceso de división
  // {B*h, N, d_h} -> {B, h, N, d_h}
  context = context.reshape({B, this->num_heads, N, this->head_dim});

  // {B, h, N, d_h} -> {B, N, h, d_h}
  context = context.transpose(1, 2); // <- ¡Esta operación crea la vista no contigua!

  // --- LA SOLUCIÓN AL ERROR ---
  // Antes del reshape final, hacemos que el tensor sea contiguo en memoria.
  context = context.contiguous();

  // Ahora este reshape es seguro.
  // {B, N, h, d_h} -> {B, N, D}
  context = context.reshape({B, N, this->embedding_dim});

  // 5. Proyección de salida final
  return out_proj->forward(context, isTraining);
}

Tensor MultiHeadAttention::scaledDotProductAttention(const Tensor &q,
                                                     const Tensor &k,
                                                     const Tensor &v) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(this->head_dim));

    /* 1. Escalamos Q antes del producto punto  ----  evita overflow */
    Tensor q_scaled = q;                      // copia liviana (misma vista)
    if (q_scaled.isContiguous()) {
        float* qdata = q_scaled.getData();
        #pragma omp parallel for
        for (size_t i = 0; i < q_scaled.getSize(); ++i)
            qdata[i] *= scale;
    } else {
        for (size_t b = 0; b < q_scaled.getShape()[0]; ++b)
            for (size_t n = 0; n < q_scaled.getShape()[1]; ++n)
                for (size_t d = 0; d < q_scaled.getShape()[2]; ++d)
                    q_scaled(b,n,d) *= scale;
    }

    /* 2. Producto punto */
    Tensor kT = k.transpose(1,2);             // (B·h, d_h, N)
    Tensor scores = batchMatrixMultiply(q_scaled, kT);  // (B·h, N, N)

    /* 3. Soft‑max estable */
    Tensor attn = softmax(scores, /*axis=*/2);
    this->attention_weights = attn;           // (para backward / visualización)

    /* 4. Contexto final */
    return batchMatrixMultiply(attn, v);      // (B·h, N, d_h)
}

// Implementación de softmax (debería ir a un archivo de utilidades)

Tensor softmax(const Tensor &logits, int axis) {
  const auto &shape = logits.getShape();
  if (axis < 0)
    axis = shape.size() + axis;

  Tensor probabilities(shape);

  if (axis == 2 && shape.size() == 3) { // Nuestro caso de uso
#pragma omp parallel for collapse(2)
    for (size_t b = 0; b < shape[0]; ++b) {
      for (size_t n = 0; n < shape[1]; ++n) {
        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t d = 0; d < shape[2]; ++d) {
          if (logits(b, n, d) > max_logit)
            max_logit = logits(b, n, d);
        }

        float sum_exp = 0.0f;
        for (size_t d = 0; d < shape[2]; ++d) {
          float exp_val = std::exp(logits(b, n, d) - max_logit);
          probabilities(b, n, d) = exp_val;
          sum_exp += exp_val;
        }

        for (size_t d = 0; d < shape[2]; ++d) {
          probabilities(b, n, d) /= sum_exp;
        }
      }
    }
  } else {
    throw std::runtime_error("Softmax a lo largo de este eje no está implementado.");
  }
  return probabilities;
}

// --- Métodos restantes (por ahora vacíos o delegando) ---

Tensor MultiHeadAttention::backward(const Tensor &outputGradient) {
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
  grad = grad.transpose(1, 2).contiguous();

  grad = grad.reshape({B * this->num_heads, N, this->head_dim});
  // 'grad' es ahora dL/d(attention_output) con forma {B*h, N, d_h}

  // ----------------------------------------------------------------------
  // 3. Inversa de la Multiplicación Final de la Atención
  // ----------------------------------------------------------------------
  // FORWARD: attention_output = attention_weights @ V
  Tensor V_T = this->v_split.transpose(1, 2);
  Tensor d_attention_weights = batchMatrixMultiply(grad, V_T);

  Tensor attention_weights_T = this->attention_weights.transpose(1, 2);
  Tensor dV = batchMatrixMultiply(attention_weights_T, grad); // ¡Este ya es un gradiente real!

  // ----------------------------------------------------------------------
  // 4. Inversa del Softmax
  // ----------------------------------------------------------------------
  // Usamos la nueva función para obtener el gradiente con respecto a las puntuaciones (scores)
  Tensor d_scores = softmax_backward(d_attention_weights, this->attention_weights);

  // ----------------------------------------------------------------------
  // 5. Inversa del Escalamiento y Q @ K^T
  // ----------------------------------------------------------------------

  // 5.1 Invertir el escalamiento
  float scale_factor = 1.0f / std::sqrt(static_cast<float>(this->head_dim));
  if (d_scores.isContiguous()) {
    float *d_scores_data = d_scores.getData();
#pragma omp parallel for
    for (size_t i = 0; i < d_scores.getSize(); ++i)
      d_scores_data[i] *= scale_factor;
  } // (Podríamos añadir un else para el caso no contiguo si fuera necesario)

  // 5.2 Propagar a través de Q @ K^T
  // Forward: scores = Q @ K^T
  // dL/dQ = dL/d(scores) @ K
  Tensor dQ = batchMatrixMultiply(d_scores, this->k_split);

  // dL/dK = Q^T @ dL/d(scores)
  Tensor Q_T = this->q_split.transpose(1, 2);
  Tensor dK = batchMatrixMultiply(Q_T, d_scores);

  // ----------------------------------------------------------------------
  // 6. Inversa de la División de Cabezas (Re-ensamblaje de Gradientes)
  // ----------------------------------------------------------------------
  auto reassemble_grads = [&](Tensor &g) {
    g = g.reshape({B, this->num_heads, N, this->head_dim});

    g = g.transpose(1, 2).contiguous();

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

std::vector<Tensor *> MultiHeadAttention::getParameters() {
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

std::vector<Tensor *> MultiHeadAttention::getGradients() {
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

Tensor softmax_backward(const Tensor &grad_output, const Tensor &softmax_output) {
  // grad_output es dL/dS, softmax_output es S
  const auto &shape = grad_output.getShape();
  Tensor grad_input(shape); // dL/dZ

  // Asumimos que el softmax se aplicó en el último eje (axis=2)
  if (shape.size() == 3) {
#pragma omp parallel for collapse(2)
    for (size_t b = 0; b < shape[0]; ++b) {
      for (size_t n = 0; n < shape[1]; ++n) {
        // Para cada fila de la matriz de atención

        // 1. Calcular el término sum(dL/dS_j * S_j)
        float dot_product = 0.0f;
        for (size_t k = 0; k < shape[2]; ++k) {
          dot_product += grad_output(b, n, k) * softmax_output(b, n, k);
        }

        // 2. Calcular dL/dZ_i para cada elemento de la fila
        for (size_t i = 0; i < shape[2]; ++i) {
          float s_i = softmax_output(b, n, i);
          grad_input(b, n, i) = s_i * (grad_output(b, n, i) - dot_product);
        }
      }
    }
  } else {
    throw std::runtime_error("softmax_backward no implementado para este rank.");
  }
  return grad_input;
}
