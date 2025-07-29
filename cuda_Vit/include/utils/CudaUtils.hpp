#ifndef CUDAUTILS_HPP
#define CUDAUTILS_HPP

#include "core/Tensor.hpp"

struct LayerNormResult
{
    Tensor input2D;
    Tensor output;
    Tensor normalized;
    Tensor mean;
    Tensor invStd; // 1 / sqrt(var + epsilon)
};

// funciones que se ejecutaran en la GPU
Tensor matrixMultiply_cuda(const Tensor &a, const Tensor &b);
Tensor batchMatrixMultiply_cuda(const Tensor &a, const Tensor &b);
Tensor concatenate_cuda(const std::vector<Tensor> &tensors, size_t axis);
Tensor addBroadcast_cuda(const Tensor &A, const Tensor &B);
Tensor contiguous_cuda(const Tensor &input);
Tensor softmax_cuda(const Tensor &logits);
Tensor softmax_cuda(const Tensor &logits, int axis);
Tensor softmax_backward_cuda(const Tensor &grad_output, const Tensor &softmax_output);
Tensor scale_tensor_cuda(const Tensor &scores, float scale_factor);
Tensor scaledDotProductAttention_cuda(const Tensor &q, const Tensor &k, const Tensor &v, float scale_factor, Tensor &out_attention_weights);
Tensor denseForward_cuda(const Tensor &input, const Tensor &weights, const Tensor &bias);
Tensor tensorAdd_cuda(const Tensor &a, const Tensor &b);
Tensor tensorSquare_cuda(const Tensor &a);
Tensor tensorSum_cuda(const Tensor &a, size_t axis);
Tensor dropout_backward_cuda(const Tensor &grad_out, const Tensor &mask);
LayerNormResult layernorm_forward_cuda(const Tensor &input,
                                       const Tensor &gamma,
                                       const Tensor &beta,
                                       float epsilon,
                                       bool isTraining);
Tensor im2col_cuda(const Tensor &input, const std::vector<size_t> &shape, size_t kernel_size, size_t stride, size_t padding);
Tensor col2im_cuda(const Tensor &col_matrix,
                   const std::vector<size_t> &in_shape,
                   size_t kernel_size,
                   size_t stride,
                   size_t padding);
float cross_entropy_cuda(const Tensor &softmax_output, const Tensor &y_true);
Tensor ce_backward_cuda(const Tensor &softmax_output_cpu, const Tensor &y_true_cpu);
#endif // CUDAUTILS_HPP
