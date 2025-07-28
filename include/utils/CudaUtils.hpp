#ifndef CUDAUTILS_HPP
#define CUDAUTILS_HPP

#include "core/Tensor.hpp"

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
#endif // CUDAUTILS_HPP
