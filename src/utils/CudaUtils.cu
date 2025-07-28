#include "utils/CudaUtils.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// Macro de utilidad para verificar errores de CUDA y cuBLAS
#define CUDA_CHECK(call)                                                                               \
    do                                                                                                 \
    {                                                                                                  \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "CUDA Error en %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do                                                                      \
    {                                                                       \
        cublasStatus_t status = call;                                       \
        if (status != CUBLAS_STATUS_SUCCESS)                                \
        {                                                                   \
            fprintf(stderr, "cuBLAS Error en %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

Tensor matrixMultiply_cuda(const Tensor &a, const Tensor &b)
{
    // 1. Validaciones
    if (!a.isContiguous() || !b.isContiguous())
    {
        throw std::runtime_error("matrixMultiply_cuda requiere tensores de entrada contiguos.");
    }
    const auto &aShape = a.getShape();
    const auto &bShape = b.getShape();
    if (aShape.size() != 2 || bShape.size() != 2 || aShape[1] != bShape[0])
    {
        throw std::invalid_argument("Dimensiones de matriz incompatibles para multiplicación.");
    }
    const int m = aShape[0];
    const int n = aShape[1];
    const int p = bShape[1];

    // 2. Crear tensor de resultado en la CPU
    Tensor result_cpu({(size_t)m, (size_t)p});

    // 3. Asignar memoria en la GPU
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, a.getSize() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, b.getSize() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, result_cpu.getSize() * sizeof(float)));

    // 4. Copiar datos de CPU (Host) a GPU (Device)
    CUDA_CHECK(cudaMemcpy(d_a, a.getData(), a.getSize() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.getData(), b.getSize() * sizeof(float), cudaMemcpyHostToDevice));

    // 5. Ejecutar la multiplicación de matrices en la GPU usando cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // NOTA: cuBLAS usa Column-Major por defecto. Para usar nuestros datos Row-Major,
    // calculamos C^T = B^T @ A^T. Esto es un truco común y eficiente.
    // C(m,p) = A(m,n) @ B(n,p)
    // C^T(p,m) = B^T(p,n) @ A^T(n,m)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             p, m, n, // p, m, n
                             &alpha,
                             d_b, p, // Matriz B, leading dim p
                             d_a, n, // Matriz A, leading dim n
                             &beta,
                             d_c, p)); // Matriz C, leading dim p

    CUBLAS_CHECK(cublasDestroy(handle));

    // 6. Copiar el resultado de GPU (Device) de vuelta a CPU (Host)
    CUDA_CHECK(cudaMemcpy(result_cpu.getData(), d_c, result_cpu.getSize() * sizeof(float), cudaMemcpyDeviceToHost));

    // 7. Liberar memoria de la GPU
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return result_cpu;
}

Tensor batchMatrixMultiply_cuda(const Tensor &a, const Tensor &b)
{
    // 1. Validaciones
    const auto &aShape = a.getShape();
    const auto &bShape = b.getShape();
    if (aShape.size() != 3 || bShape.size() != 3 || aShape[0] != bShape[0] || aShape[2] != bShape[1])
    {
        throw std::invalid_argument("Dimensiones incompatibles para BMM en CUDA.");
    }
    const int batchSize = aShape[0];
    const int m = aShape[1];
    const int n = aShape[2];
    const int p = bShape[2];

    // 2. Crear tensores en CPU y GPU
    Tensor result_cpu({(size_t)batchSize, (size_t)m, (size_t)p});
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, a.getSize() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, b.getSize() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, result_cpu.getSize() * sizeof(float)));

    // 3. Copiar datos a la GPU
    CUDA_CHECK(cudaMemcpy(d_a, a.getData(), a.getSize() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.getData(), b.getSize() * sizeof(float), cudaMemcpyHostToDevice));

    // 4. Ejecutar BMM en la GPU
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Para la versión por lotes, necesitamos un array de punteros
    std::vector<const float *> a_array(batchSize, nullptr);
    std::vector<const float *> b_array(batchSize, nullptr);
    std::vector<float *> c_array(batchSize, nullptr);
    for (int i = 0; i < batchSize; ++i)
    {
        a_array[i] = d_a + i * m * n;
        b_array[i] = d_b + i * n * p;
        c_array[i] = d_c + i * m * p;
    }

    const float **d_a_array, **d_b_array;
    float **d_c_array;
    CUDA_CHECK(cudaMalloc(&d_a_array, batchSize * sizeof(float *)));
    CUDA_CHECK(cudaMalloc(&d_b_array, batchSize * sizeof(float *)));
    CUDA_CHECK(cudaMalloc(&d_c_array, batchSize * sizeof(float *)));
    CUDA_CHECK(cudaMemcpy(d_a_array, a_array.data(), batchSize * sizeof(float *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_array, b_array.data(), batchSize * sizeof(float *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_array, c_array.data(), batchSize * sizeof(float *), cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    p, m, n, &alpha,
                                    d_b_array, p,
                                    d_a_array, n, &beta,
                                    d_c_array, p,
                                    batchSize));

    CUBLAS_CHECK(cublasDestroy(handle));

    // 5. Copiar resultado de vuelta a la CPU
    CUDA_CHECK(cudaMemcpy(result_cpu.getData(), d_c, result_cpu.getSize() * sizeof(float), cudaMemcpyDeviceToHost));

    // 6. Liberar toda la memoria de la GPU
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_a_array));
    CUDA_CHECK(cudaFree(d_b_array));
    CUDA_CHECK(cudaFree(d_c_array));

    return result_cpu;
}

__global__ void validate_and_sum_axis_kernel(
    const size_t *shapes, // [num_tensors][rank]
    size_t num_tensors,
    size_t rank,
    size_t axis,
    const size_t *ref_shape,
    int *error_flag,
    size_t *axis_sizes)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_tensors)
        return;

    for (size_t i = 0; i < rank; ++i)
    {
        size_t val = shapes[tid * rank + i];
        if (i != axis && val != ref_shape[i])
        {
            *error_flag = 1;
            return;
        }
    }

    axis_sizes[tid] = shapes[tid * rank + axis];
}

__global__ void copy_tensor_to_concat_kernel(
    const float *input, float *output,
    size_t B, size_t N, size_t D,
    size_t out_stride_B,
    size_t out_stride_N,
    size_t out_stride_D,
    size_t offset_axis,
    size_t axis)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = B * N * D;
    if (tid >= total)
        return;

    size_t i = tid / (N * D);
    size_t j = (tid / D) % N;
    size_t k = tid % D;

    size_t out_i = i, out_j = j, out_k = k;

    if (axis == 0)
        out_i += offset_axis;
    else if (axis == 1)
        out_j += offset_axis;
    else if (axis == 2)
        out_k += offset_axis;

    size_t out_idx = out_i * out_stride_B +
                     out_j * out_stride_N +
                     out_k * out_stride_D;

    output[out_idx] = input[tid];
}
__global__ void copy_tensor_kernel_3d_strided(
    const float *input, float *output,
    size_t B, size_t N, size_t D,
    size_t in_stride0, size_t in_stride1, size_t in_stride2,
    size_t out_stride0, size_t out_stride1, size_t out_stride2,
    size_t in_offset, size_t out_offset)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = B * N * D;

    if (tid >= total)
        return;

    size_t i = tid / (N * D);
    size_t j = (tid / D) % N;
    size_t k = tid % D;

    size_t in_idx = in_offset + i * in_stride0 + j * in_stride1 + k * in_stride2;
    size_t out_idx = out_offset + i * out_stride0 + j * out_stride1 + k * out_stride2;

    output[out_idx] = input[in_idx];
}

Tensor concatenate_cuda(const std::vector<Tensor> &tensors, size_t axis)
{
    if (tensors.empty())
        return Tensor();
    if (tensors.size() == 1)
        return tensors[0];

    const size_t num_tensors = tensors.size();
    const auto &refShape = tensors[0].getShape();
    const size_t rank = refShape.size();

    if (axis >= rank)
        throw std::invalid_argument("Eje fuera de rango.");

    // Preparar shapes en CPU
    std::vector<size_t> all_shapes(num_tensors * rank);
    for (size_t i = 0; i < num_tensors; ++i)
    {
        const auto &s = tensors[i].getShape();
        if (s.size() != rank)
            throw std::invalid_argument("Todos los tensores deben tener el mismo rank.");

        for (size_t j = 0; j < rank; ++j)
            all_shapes[i * rank + j] = s[j];
    }

    // Copiar a GPU
    size_t *d_shapes, *d_ref_shape, *d_axis_sizes;
    int *d_error_flag;
    CUDA_CHECK(cudaMalloc(&d_shapes, sizeof(size_t) * all_shapes.size()));
    CUDA_CHECK(cudaMalloc(&d_ref_shape, sizeof(size_t) * rank));
    CUDA_CHECK(cudaMalloc(&d_axis_sizes, sizeof(size_t) * num_tensors));
    CUDA_CHECK(cudaMalloc(&d_error_flag, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_shapes, all_shapes.data(), sizeof(size_t) * all_shapes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref_shape, refShape.data(), sizeof(size_t) * rank, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_error_flag, 0, sizeof(int)));

    // Ejecutar kernel
    int threads = 128;
    int blocks = (num_tensors + threads - 1) / threads;
    validate_and_sum_axis_kernel<<<blocks, threads>>>(
        d_shapes, num_tensors, rank, axis,
        d_ref_shape, d_error_flag, d_axis_sizes);
    CUDA_CHECK(cudaGetLastError());

    // Leer resultados
    int error_flag_h = 0;
    CUDA_CHECK(cudaMemcpy(&error_flag_h, d_error_flag, sizeof(int), cudaMemcpyDeviceToHost));
    if (error_flag_h)
        throw std::runtime_error("Las dimensiones no son compatibles para concatenación.");

    std::vector<size_t> axis_sizes_h(num_tensors);
    CUDA_CHECK(cudaMemcpy(axis_sizes_h.data(), d_axis_sizes, sizeof(size_t) * num_tensors, cudaMemcpyDeviceToHost));

    // Liberar memoria
    CUDA_CHECK(cudaFree(d_shapes));
    CUDA_CHECK(cudaFree(d_ref_shape));
    CUDA_CHECK(cudaFree(d_error_flag));
    CUDA_CHECK(cudaFree(d_axis_sizes));

    // --- Crear tensor de resultado ---
    size_t newDimSize = 0;
    for (size_t v : axis_sizes_h)
        newDimSize += v;

    std::vector<size_t> newShape = refShape;
    newShape[axis] = newDimSize;
    Tensor result(newShape);
    size_t result_total_size = result.getSize();
    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float) * result_total_size));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float) * result_total_size)); // opcional

    size_t offset_on_axis = 0;
    for (const auto &t : tensors)
    {
        const auto &shape = t.getShape();
        const auto &in_strides = t.getStrides();
        const auto &out_strides = result.getStrides(); // usamos output completo

        size_t B = shape[0], N = shape[1], D = shape[2];
        size_t input_size = t.getSize();

        float *d_input;
        CUDA_CHECK(cudaMalloc(&d_input, sizeof(float) * input_size));
        CUDA_CHECK(cudaMemcpy(d_input, t.getData(), sizeof(float) * input_size, cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks = (B * N * D + threads - 1) / threads;

        copy_tensor_kernel_3d_strided<<<blocks, threads>>>(
            d_input,
            d_output,
            B, N, D,
            in_strides[0], in_strides[1], in_strides[2],
            out_strides[0], out_strides[1], out_strides[2],
            t.getDataOffset(),
            result.getStrides()[axis] * offset_on_axis // nuevo offset dinámico
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(d_input));

        offset_on_axis += shape[axis];
    }
    CUDA_CHECK(cudaMemcpy(result.getDataPtr()->data(), d_output, sizeof(float) * result_total_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output));
    return result;
}