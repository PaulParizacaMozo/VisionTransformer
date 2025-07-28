#ifndef CUDAUTILS_HPP
#define CUDAUTILS_HPP

#include "core/Tensor.hpp"

// funciones que se ejecutaran en la GPU
Tensor matrixMultiply_cuda(const Tensor &a, const Tensor &b);
Tensor batchMatrixMultiply_cuda(const Tensor &a, const Tensor &b);
Tensor concatenate_cuda(const std::vector<Tensor> &tensors, size_t axis);

#endif // CUDAUTILS_HPP
