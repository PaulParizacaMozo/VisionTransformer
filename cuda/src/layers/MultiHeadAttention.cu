#include "layers/MultiHeadAttention.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

MultiHeadAttention::MultiHeadAttention(size_t embedding_dim, size_t num_heads)
    : embedding_dim(embedding_dim), num_heads(num_heads)
{

    if (embedding_dim % num_heads != 0)
    {
        printf("Error: embedding_dim (%zu) debe ser divisible por num_heads (%zu).\n",
               embedding_dim, num_heads);
        exit(EXIT_FAILURE);
    }

    this->head_dim = embedding_dim / num_heads;

    q_proj = std::make_unique<Dense>(embedding_dim, embedding_dim);
    k_proj = std::make_unique<Dense>(embedding_dim, embedding_dim);
    v_proj = std::make_unique<Dense>(embedding_dim, embedding_dim);
    out_proj = std::make_unique<Dense>(embedding_dim, embedding_dim);
}

__global__ void scaleKernel(float *data, float scale, size_t totalSize)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalSize)
        data[idx] *= scale;
}

Tensor MultiHeadAttention::scaledDotProductAttention(const Tensor &q, const Tensor &k, const Tensor &v)
{
    // Transponer k para obtener (B*h, d_h, N)
    Tensor k_transposed = k.transpose(1, 2);

    // scores: (B*h, N, d_h) x (B*h, d_h, N) -> (B*h, N, N)
    Tensor scores = batchMatrixMultiply(q, k_transposed);
    Tensor contiguous_scores = scores.isContiguous() ? scores : scores.contiguous();

    float scale_factor = 1.0f / std::sqrt(static_cast<float>(this->head_dim));

    // Aplicar escalamiento a scores usando CUDA
    float *scores_data = contiguous_scores.getData();
    size_t totalSize = contiguous_scores.size();

    constexpr int threadsPerBlock = 128;
    int blocks = (totalSize + threadsPerBlock - 1) / threadsPerBlock;
    cudaGetLastError();
    scaleKernel<<<blocks, threadsPerBlock>>>(scores_data, scale_factor, totalSize);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("[CUDA ERROR - scaleKernel] %s\n", cudaGetErrorString(err));
        throw std::runtime_error("scaledDotProductAttention: Error al escalar los scores");
    }

    // Softmax sobre la última dimensión (dim=2)
    Tensor attention = softmax(contiguous_scores, 2);
    this->attention_weights = attention;

    // attention: (B*h, N, N), v: (B*h, N, d_h) → out: (B*h, N, d_h)
    return batchMatrixMultiply(attention, v);
}

Tensor MultiHeadAttention::forward(const Tensor &input, bool isTraining)
{
    Tensor c1(input);
    Tensor c2(c1);
    Tensor c3(c2);

    if (isTraining)
        this->inputTensor.deepCopyFrom(input);
    // c1.printContents("MultiHeadAttention::forward - Input c1");
    size_t B = input.dim(0), N = input.dim(1);

    // std::cout << ">> MultiHeadAttention: Paso 1" << std::endl;
    // 1. Proyecciones lineales
    Tensor q = q_proj->forward(c1, isTraining); // -> {B, N, D}
    Tensor k = k_proj->forward(c2, isTraining);
    Tensor v = v_proj->forward(c3, isTraining);
    // k.printContents("K Proj Output");
    // q.printContents("Q Proj Output");
    // v.printContents("V Proj Output");

    // std::cout << ">> MultiHeadAttention: Paso 2" << std::endl;
    // 2. Dividir en cabezas: reshape a {B, N, h, d_h}
    size_t q1[4] = {B, N, num_heads, head_dim};
    q = q.reshape(q1, 4);
    k = k.reshape(q1, 4);
    v = v.reshape(q1, 4);
    // q.printContents("Q reshaped [B,N,h,d_h]");
    // k.printContents("K reshaped [B,N,h,d_h]");
    // v.printContents("V reshaped [B,N,h,d_h]");

    Tensor q_t(q);
    Tensor k_t(k);
    Tensor v_t(v);

    // std::cout << ">> MultiHeadAttention: Paso 2 Transpose" << std::endl;
    // Transponer: {B, N, h, d_h} -> {B, h, N, d_h}
    q = q_t.transpose(1, 2);
    k = k_t.transpose(1, 2);
    v = v_t.transpose(1, 2);
    // q.printContents("Q transposed [B,h,N,d_h]");
    // k.printContents("K transposed [B,h,N,d_h]");
    // v.printContents("V transposed [B,h,N,d_h]");

    // std::cout << ">> MultiHeadAttention: Paso 2 Contiguous" << std::endl;
    // Importante: hacer contiguos para el reshape que sigue
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    // q.printContents("Q contiguous");
    // k.printContents("K contiguous");
    // v.printContents("V contiguous");

    Tensor q4_split(q);
    Tensor k4_split(k);
    Tensor v4_split(v);

    // std::cout << ">> MultiHeadAttention: Paso 3" << std::endl;
    // 3. Unificar batch y heads: {B, h, N, d_h} -> {B*h, N, d_h}
    size_t q2[3] = {B * num_heads, N, head_dim};
    q = q4_split.reshape(q2, 3);
    k = k4_split.reshape(q2, 3);
    v = v4_split.reshape(q2, 3);
    // q.printContents("Q reshaped [B*h,N,d_h]");
    // k.printContents("K reshaped [B*h,N,d_h]");
    // v.printContents("V reshaped [B*h,N,d_h]");

    // std::cout << ">> MultiHeadAttention: Paso 3 train" << std::endl;
    if (isTraining)
    {
        Tensor v_copy6(v);
        Tensor k_copy6(k);
        Tensor q_copy6(q);
        // std::cout << "Dirección de v antes de copiar: " << v.getData() << std::endl;
        // std::cout << "Dirección de v_split antes de copiar: " << this->v_split.getData() << std::endl;

        this->q_split.deepCopyFrom(q_copy6);
        this->k_split.deepCopyFrom(k_copy6);
        v_copy6.printFirstElements("MHA::FORWARD v_copy6 [B*h,N,d_h] antes de copiar");
        this->v_split = v_copy6.clone();
        v_split2 = v_copy6.clone();
        v_split.printFirstElements("MHA::FORWARD v_split [B*h,N,d_h] después de copiar");
        v_split2.printFirstElements("MHA::FORWARD v_split2 [B*h,N,d_h] después de copiar");

        std::cout << "Dirección de v_split después de copiar: " << this->v_split.getData() << std::endl;
    }

    // q.printContents("Q final [B*h,N,d_h]");
    // k.printContents("K final [B*h,N,d_h]");
    // v.printContents("V final [B*h,N,d_h]");

    // std::cout << ">> MultiHeadAttention: Paso 4" << std::endl;
    // 4. Atención escalada
    Tensor context = scaledDotProductAttention(q, k, v); // -> {B*h, N, d_h}
    // context.printContents("Context [B*h,N,d_h]");

    // std::cout << ">> MultiHeadAttention: Paso 5" << std::endl;
    // 5. Reunir cabezas: {B*h, N, d_h} -> {B, h, N, d_h}
    size_t q3[4] = {B, num_heads, N, head_dim};
    context = context.reshape(q3, 4); // {B, h, N, d_h}
    // context.printContents("Context [B,h,N,d_h] antes de transponer");
    Tensor context_t(context);
    context = context_t.transpose(1, 2); // -> {B, N, h, d_h}
    // context.printContents("Context [B,N,h,d_h] después de transponer");
    context = context.contiguous(); // Necesario antes del reshape final
    // context.printContents("Context Final [B,N,h,d_h]");

    // std::cout << ">> MultiHeadAttention: Paso 6" << std::endl;
    // 6. Combinar cabezas: {B, N, h, d_h} -> {B, N, D}
    size_t q4[3] = {B, N, embedding_dim}; // D = h * d_h
    context = context.reshape(q4, 3);
    // context.printContents("Context [B,N,D] final antes de out_proj");

    // std::cout << ">> MultiHeadAttention: Paso 7" << std::endl;
    // 6. Proyección de salida
    Tensor out = out_proj->forward(context, isTraining);
    // out.printContents("Output final MultiHeadAttention");

    // v_split.printFirstElements("MHA::FORWARD v_split [B*h,N,d_h] después de copiar");
    // v_split2.printFirstElements("MHA::FORWARD v_split2 [B*h,N,d_h] después de copiar");

    // std::cout << "Dirección de v_split después de copiar: " << this->v_split.getData() << std::endl;
    return out;
}

Tensor MultiHeadAttention::backward(const Tensor &outputGradient)
{
    // outputGradient.printContents("MultiHeadAttention::backward - Output Gradient");
    size_t B = inputTensor.dim(0);
    size_t N = inputTensor.dim(1);

    std::cout << ">> MultiHeadAttention: Paso 1" << std::endl;
    std::cout << "Batch size: " << B << ", Sequence length: " << N << std::endl;

    std::cout << "v_split antes del backward: " << v_split.getData() << std::endl;
    v_split.printDebugInfo("v_split antes del backward");
    v_split.printFirstElements("v_split antes del backward");
    v_split2.printDebugInfo("v_split2 antes del backward");
    v_split2.printFirstElements("v_split2 antes del backward");
    Tensor grad = this->out_proj->backward(outputGradient);
    // std::cout << "v_split después del backward: " << v_split.getData() << std::endl;

    grad.printFirstElements("MHA::Gradiente después de out_proj");

    size_t s1[4] = {B, N, this->num_heads, this->head_dim};
    grad = grad.reshape(s1, 4);
    grad.printDebugInfo("MHA::Gradiente después de reshape a [B, N, h, d_h]");
    grad.printFirstElements("MHA::Gradiente después de reshape a [B, N, h, d_h]");
    Tensor grad_t(grad);
    grad = grad_t.transpose(1, 2);
    grad.printDebugInfo("MHA::Gradiente después de transponer a [B, h, N, d_h]");
    grad.printFirstElements("MHA::Gradiente después de transponer a [B, h, N, d_h]");
    grad = grad.contiguous();
    grad.printFirstElements("MHA::Gradiente después de hacer contiguo");

    size_t s2[3] = {B * this->num_heads, N, this->head_dim};
    grad = grad.reshape(s2, 3);
    // std::cout << "Dirección de grad en backward: " << grad.getData() << std::endl;
    // std::cout << "Dirección de v_split: " << this->v_split.getData() << std::endl;

    grad.printDebugInfo("MHA::Gradiente después de unificar batch y heads a [B*h, N, d_h]");
    grad.printFirstElements("MHA::Gradiente después de unificar batch y heads a [B*h, N, d_h]");
    v_split2.printDebugInfo("MHA::v_split2 [B*h, N, d_h] antes de usar en backward");
    v_split2.printFirstElements("MHA::v_split2 [B*h, N, d_h]");

    Tensor V_T = this->v_split2.transpose(1, 2);
    V_T.printFirstElements("MHA::V_T [B*h, d_h, N]");

    Tensor d_attention_weights = batchMatrixMultiply(grad, V_T);                    // resuolt 0's
    d_attention_weights.printFirstElements("MHA::d_attention_weights [B*h, N, N]"); // 0's

    Tensor attention_weights_T = this->attention_weights.transpose(1, 2);
    attention_weights_T.printFirstElements("MHA::attention_weights_T [B*h, N, N]");
    Tensor dV = batchMatrixMultiply(attention_weights_T, grad); // result 0's
    dV.printFirstElements("MHA::dV [B*h, N, d_h]");

    Tensor d_scores = softmax_backward(d_attention_weights, this->attention_weights);
    d_scores.printFirstElements("MHA::d_scores [B*h, N, N]"); // 0's

    float scale_factor = 1.0f / std::sqrt(static_cast<float>(this->head_dim));
    d_scores.printDebugInfo("MHA::d_scores antes de aplicar escalamiento inverso");
    Tensor contiguous_scores = d_scores.isContiguous() ? d_scores : d_scores.contiguous();
    contiguous_scores.printFirstElements("MHA::Contiguous scores para escalamiento inverso"); // 0's

    float *scores_data = contiguous_scores.getData();
    size_t totalSize = contiguous_scores.size();

    constexpr int threadsPerBlock = 128;
    int blocks = (totalSize + threadsPerBlock - 1) / threadsPerBlock;
    cudaGetLastError();
    scaleKernel<<<blocks, threadsPerBlock>>>(scores_data, scale_factor, totalSize);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("[CUDA ERROR - scaleKernel (backward)] %s\n", cudaGetErrorString(err));
        throw std::runtime_error("MultiHeadAttention::backward: error al aplicar escalamiento inverso");
    }
    contiguous_scores.printDebugInfo("MHA::d_scores después de aplicar escalamiento inverso");     // 0's
    contiguous_scores.printFirstElements("MHA::d_scores después de aplicar escalamiento inverso"); // 0's
    Tensor dQ = batchMatrixMultiply(contiguous_scores, this->k_split);                             // contiguos 0's
    dQ.printFirstElements("MHA::dQ [B*h, N, d_h]");
    Tensor Q_T = this->q_split.transpose(1, 2);
    Q_T.printFirstElements("MHA::Q_T [B*h, d_h, N]");
    Tensor dK = batchMatrixMultiply(Q_T, contiguous_scores);
    dK.printFirstElements("MHA::dK [B*h, N, d_h]");

    auto reassemble_grads = [&](Tensor &g)
    {
        size_t r1[4] = {B, this->num_heads, N, this->head_dim};
        g = g.reshape(r1, 4);
        g = g.transpose(1, 2).contiguous();

        size_t r2[3] = {B, N, this->embedding_dim};
        return g.reshape(r2, 3);
    };

    dQ = reassemble_grads(dQ);
    dQ.printFirstElements("MHA::dQ reensamblado [B, N, D]");

    dK = reassemble_grads(dK);
    dK.printFirstElements("MHA::dK reensamblado [B, N, D]");
    dV = reassemble_grads(dV);
    dV.printFirstElements("MHA::dV reensamblado [B, N, D]");

    Tensor dInput_q = this->q_proj->backward(dQ);
    Tensor dInput_k = this->k_proj->backward(dK);
    Tensor dInput_v = this->v_proj->backward(dV);

    dInput_q.printFirstElements("MHA::dInput_q [B, N, D]");
    dInput_k.printFirstElements("MHA::dInput_k [B, N, D]");
    dInput_v.printFirstElements("MHA::dInput_v [B, N, D]");
    Tensor final_grad = dInput_q + dInput_k + dInput_v;
    final_grad.printFirstElements("MHA::Gradiente final [B, N, D]");
    return final_grad;
}

std::vector<Tensor *> MultiHeadAttention::getParameters()
{
    std::vector<Tensor *> all_params;

    auto q_params = q_proj->getParameters();
    auto k_params = k_proj->getParameters();
    auto v_params = v_proj->getParameters();
    auto out_params = out_proj->getParameters();

    all_params.insert(all_params.end(), q_params.begin(), q_params.end());
    all_params.insert(all_params.end(), k_params.begin(), k_params.end());
    all_params.insert(all_params.end(), v_params.begin(), v_params.end());
    all_params.insert(all_params.end(), out_params.begin(), out_params.end());

    return all_params;
}

std::vector<Tensor *> MultiHeadAttention::getGradients()
{
    std::vector<Tensor *> all_grads;

    auto q_grads = q_proj->getGradients();
    auto k_grads = k_proj->getGradients();
    auto v_grads = v_proj->getGradients();
    auto out_grads = out_proj->getGradients();

    all_grads.insert(all_grads.end(), q_grads.begin(), q_grads.end());
    all_grads.insert(all_grads.end(), k_grads.begin(), k_grads.end());
    all_grads.insert(all_grads.end(), v_grads.begin(), v_grads.end());
    all_grads.insert(all_grads.end(), out_grads.begin(), out_grads.end());

    return all_grads;
}