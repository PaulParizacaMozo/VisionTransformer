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

__global__ void concatenate_kernel(
    const float *input_data,
    float *output_data,
    const size_t *shapes,
    const size_t *strides,
    const size_t *offsets,
    const size_t *out_strides,
    const size_t *axis_sizes,
    size_t rank, size_t axis,
    size_t num_tensors,
    size_t tensor_id)
{
    const size_t *shape = &shapes[tensor_id * rank];
    const size_t *stride = &strides[tensor_id * rank];
    size_t in_offset = offsets[tensor_id];

    size_t B = shape[0], N = shape[1], D = shape[2];
    size_t total = B * N * D;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total)
        return;

    size_t i = tid / (N * D);
    size_t j = (tid / D) % N;
    size_t k = tid % D;

    size_t in_idx = in_offset + i * stride[0] + j * stride[1] + k * stride[2];

    size_t out_axis_offset = 0;
    for (int t = 0; t < tensor_id; ++t)
        out_axis_offset += axis_sizes[t];

    size_t out_offset = out_axis_offset * out_strides[axis];
    size_t out_idx = out_offset + i * out_strides[0] + j * out_strides[1] + k * out_strides[2];

    output_data[out_idx] = input_data[in_idx];
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

    // --- CPU: Preparar metadata ---
    std::vector<size_t> h_shapes(num_tensors * rank);
    std::vector<size_t> h_strides(num_tensors * rank);
    std::vector<size_t> h_offsets(num_tensors);
    std::vector<size_t> h_axis_sizes(num_tensors);
    std::vector<float> flat_input_data;
    size_t offset = 0;
    for (size_t i = 0; i < num_tensors; ++i)
    {
        const auto &t = tensors[i];
        const auto &shape = t.getShape();
        const auto &strides = t.getStrides();

        if (shape.size() != rank)
            throw std::invalid_argument("Todos los tensores deben tener el mismo rank.");

        for (size_t j = 0; j < rank; ++j)
        {
            if (j != axis && shape[j] != refShape[j])
                throw std::runtime_error("Dimensiones incompatibles para concatenación.");
            h_shapes[i * rank + j] = shape[j];
            h_strides[i * rank + j] = strides[j];
        }

        size_t size = t.getSize();
        h_offsets[i] = offset;
        h_axis_sizes[i] = shape[axis];

        const float *src = t.getData();
        flat_input_data.insert(flat_input_data.end(), src, src + size);
        offset += size;
    }

    // --- GPU: Reservar y copiar metadata ---
    size_t *d_shapes, *d_strides, *d_offsets, *d_axis_sizes, *d_out_strides;
    float *d_input_data, *d_output_data;

    CUDA_CHECK(cudaMalloc(&d_shapes, sizeof(size_t) * h_shapes.size()));
    CUDA_CHECK(cudaMalloc(&d_strides, sizeof(size_t) * h_strides.size()));
    CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(size_t) * h_offsets.size()));
    CUDA_CHECK(cudaMalloc(&d_axis_sizes, sizeof(size_t) * h_axis_sizes.size()));
    CUDA_CHECK(cudaMalloc(&d_input_data, sizeof(float) * flat_input_data.size()));

    CUDA_CHECK(cudaMemcpy(d_shapes, h_shapes.data(), sizeof(size_t) * h_shapes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strides, h_strides.data(), sizeof(size_t) * h_strides.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), sizeof(size_t) * h_offsets.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_axis_sizes, h_axis_sizes.data(), sizeof(size_t) * h_axis_sizes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_data, flat_input_data.data(), sizeof(float) * flat_input_data.size(), cudaMemcpyHostToDevice));

    // --- Crear tensor resultado ---
    size_t newDimSize = std::accumulate(h_axis_sizes.begin(), h_axis_sizes.end(), size_t(0));
    std::vector<size_t> newShape = refShape;
    newShape[axis] = newDimSize;
    Tensor result(newShape);
    size_t result_size = result.getSize();
    CUDA_CHECK(cudaMalloc(&d_output_data, sizeof(float) * result_size));
    CUDA_CHECK(cudaMemset(d_output_data, 0, sizeof(float) * result_size));

    // --- Copiar strides de salida ---
    const auto &out_strides = result.getStrides();
    CUDA_CHECK(cudaMalloc(&d_out_strides, sizeof(size_t) * rank));
    CUDA_CHECK(cudaMemcpy(d_out_strides, out_strides.data(), sizeof(size_t) * rank, cudaMemcpyHostToDevice));

    // --- Lanzar kernel ---
    int threads = 256;
    for (size_t t = 0; t < num_tensors; ++t)
    {
        const size_t B = h_shapes[t * rank + 0];
        const size_t N = h_shapes[t * rank + 1];
        const size_t D = h_shapes[t * rank + 2];
        size_t total = B * N * D;

        int blocks = (total + threads - 1) / threads;

        concatenate_kernel<<<blocks, threads>>>(
            d_input_data, d_output_data,
            d_shapes, d_strides, d_offsets,
            d_out_strides, d_axis_sizes,
            rank, axis, num_tensors,
            t // ← tensor_id
        );

        CUDA_CHECK(cudaGetLastError());
    }

    // --- Copiar resultado a CPU ---
    CUDA_CHECK(cudaMemcpy(result.getDataPtr()->data(), d_output_data, sizeof(float) * result_size, cudaMemcpyDeviceToHost));

    // --- Liberar ---
    CUDA_CHECK(cudaFree(d_shapes));
    CUDA_CHECK(cudaFree(d_strides));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_axis_sizes));
    CUDA_CHECK(cudaFree(d_input_data));
    CUDA_CHECK(cudaFree(d_output_data));
    CUDA_CHECK(cudaFree(d_out_strides));

    return result;
}

// Kernel para broadcasting de {1, N} sobre {M, N}
__global__ void addBroadcast2D(const float *A, const float *B, float *out, size_t M, size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N)
    {
        out[i * N + j] = A[i * N + j] + B[j]; // B[0, j] es B[j]
    }
}

// Kernel para broadcasting de {1, N, D} sobre {B, N, D}
__global__ void addBroadcast3D(const float *A, const float *B, float *out, size_t Bsize, size_t N, size_t D)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = Bsize * N * D;

    if (idx < total)
    {
        size_t b = idx / (N * D);
        size_t rem = idx % (N * D);
        size_t n = rem / D;
        size_t d = rem % D;

        size_t bidx = n * D + d; // índice en B[0, n, d]
        out[idx] = A[idx] + B[bidx];
    }
}

// Kernel para broadcasting de {1, C, H, W} sobre {N, C, H, W}
__global__ void addBias4D(const float *A, const float *B, float *out,
                          size_t N, size_t C, size_t H, size_t W)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = N * C * H * W;
    if (idx < total)
    {
        // idx -> (n,c,h,w)
        size_t rem1 = idx;
        size_t n = rem1 / (C * H * W);
        rem1 %= (C * H * W);
        size_t c = rem1 / (H * W);
        // size_t rem2 = rem1 % (H*W);
        // size_t h = rem2 / W;
        // size_t w = rem2 % W;
        size_t bidx = c; // B[0,c,0,0]
        out[idx] = A[idx] + B[bidx];
    }
}

Tensor addBroadcast_cuda(const Tensor &A, const Tensor &B)
{
    const std::vector<size_t> &shapeA = A.getShape();
    const std::vector<size_t> &shapeB = B.getShape();

    // Validación de compatibilidad para casos comunes
    bool is2D = (shapeA.size() == 2 && shapeB.size() == 2 &&
                 shapeB[0] == 1 && shapeA[1] == shapeB[1]);

    bool is3D = (shapeA.size() == 3 && shapeB.size() == 3 &&
                 shapeB[0] == 1 && shapeA[1] == shapeB[1] && shapeA[2] == shapeB[2]);

    bool is4D = (shapeA.size() == 4 && shapeB.size() == 4 &&
                 shapeB[0] == 1 &&
                 shapeA[1] == shapeB[1] &&
                 1 == shapeB[2] &&
                 1 == shapeB[3]);

    if (!is2D && !is3D && !is4D)
    {
        throw std::runtime_error("Broadcasting no implementado para estas formas.");
    }

    Tensor out(shapeA); // La salida tendrá la misma forma que A
    size_t totalSize = A.getSize();

    // --- Copiar datos a device ---
    float *d_A, *d_B, *d_out;
    CUDA_CHECK(cudaMalloc(&d_A, totalSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, B.getSize() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, totalSize * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A,
                          A.getDataPtr()->data() + A.getDataOffset(),
                          totalSize * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_B,
                          B.getDataPtr()->data() + B.getDataOffset(),
                          B.getSize() * sizeof(float),
                          cudaMemcpyHostToDevice));

    if (is2D)
    {
        size_t M = shapeA[0], N = shapeA[1];
        dim3 threads(16, 16);
        dim3 blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
        addBroadcast2D<<<blocks, threads>>>(d_A, d_B, d_out, M, N);
    }
    else if (is3D)
    {
        size_t Bsize = shapeA[0], N = shapeA[1], D = shapeA[2];
        size_t total = Bsize * N * D;

        size_t threadsPerBlock = 256;
        size_t numBlocks = (total + threadsPerBlock - 1) / threadsPerBlock;

        addBroadcast3D<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_out, Bsize, N, D);
    }
    else if (is4D)
    {
        size_t Nn = shapeA[0], C = shapeA[1], H = shapeA[2], W = shapeA[3];
        size_t tot = Nn * C * H * W;
        size_t tp = 256, nb = (tot + tp - 1) / tp;
        addBias4D<<<nb, tp>>>(d_A, d_B, d_out, Nn, C, H, W);
    }
    else
    {
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_out));
        throw std::runtime_error("Broadcasting no implementado para esas formas");
    }
    CUDA_CHECK(cudaMemcpy(out.getDataPtr()->data() + out.getDataOffset(),
                          d_out,
                          totalSize * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_out));
    return out;
}

__global__ void copy1D(const float *in, float *out, size_t stride0, size_t offset, size_t dim0)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim0)
    {
        size_t idx = offset + i * stride0;
        out[i] = in[idx];
    }
}

__global__ void copy2D(const float *in, float *out,
                       size_t stride0, size_t stride1,
                       size_t offset, size_t dim0, size_t dim1)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim0 * dim1)
    {
        size_t d0 = i / dim1;
        size_t d1 = i % dim1;
        size_t idx = offset + d0 * stride0 + d1 * stride1;
        out[i] = in[idx];
    }
}

__global__ void copy3D(const float *in, float *out,
                       size_t stride0, size_t stride1, size_t stride2,
                       size_t offset, size_t dim0, size_t dim1, size_t dim2)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim0 * dim1 * dim2)
    {
        size_t d0 = i / (dim1 * dim2);
        size_t rem = i % (dim1 * dim2);
        size_t d1 = rem / dim2;
        size_t d2 = rem % dim2;
        size_t idx = offset + d0 * stride0 + d1 * stride1 + d2 * stride2;
        out[i] = in[idx];
    }
}

__global__ void copy4D(const float *in, float *out,
                       size_t stride0, size_t stride1, size_t stride2, size_t stride3,
                       size_t offset, size_t dim0, size_t dim1, size_t dim2, size_t dim3)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim0 * dim1 * dim2 * dim3)
    {
        size_t d0 = i / (dim1 * dim2 * dim3);
        size_t rem = i % (dim1 * dim2 * dim3);
        size_t d1 = rem / (dim2 * dim3);
        rem = rem % (dim2 * dim3);
        size_t d2 = rem / dim3;
        size_t d3 = rem % dim3;
        size_t idx = offset + d0 * stride0 + d1 * stride1 + d2 * stride2 + d3 * stride3;
        out[i] = in[idx];
    }
}

Tensor contiguous_cuda(const Tensor &input)
{
    const std::vector<size_t> &shape = input.getShape();
    const std::vector<size_t> &strides = input.getStrides();
    size_t ndim = shape.size();
    size_t totalSize = input.getSize();
    size_t offset = input.getDataOffset();

    if (input.isContiguous() && offset == 0)
        return input;

    Tensor output(shape);

    // Reservamos y copiamos memoria al device
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, input.getDataPtr()->size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, totalSize * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, input.getDataPtr()->data(),
                          input.getDataPtr()->size() * sizeof(float), cudaMemcpyHostToDevice));

    // Lanzar kernel adecuado por dimensión
    size_t threads = 256;
    size_t blocks = (totalSize + threads - 1) / threads;

    if (ndim == 1)
    {
        copy1D<<<blocks, threads>>>(
            d_in, d_out,
            strides[0], offset, shape[0]);
    }
    else if (ndim == 2)
    {
        copy2D<<<blocks, threads>>>(
            d_in, d_out,
            strides[0], strides[1],
            offset,
            shape[0], shape[1]);
    }
    else if (ndim == 3)
    {
        copy3D<<<blocks, threads>>>(
            d_in, d_out,
            strides[0], strides[1], strides[2],
            offset,
            shape[0], shape[1], shape[2]);
    }
    else if (ndim == 4)
    {
        copy4D<<<blocks, threads>>>(
            d_in, d_out,
            strides[0], strides[1], strides[2], strides[3],
            offset,
            shape[0], shape[1], shape[2], shape[3]);
    }
    else
    {
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        throw std::runtime_error("contiguous_cuda() no implementado para ndim > 4.");
    }

    // Copiar resultado de vuelta
    CUDA_CHECK(cudaMemcpy(output.getDataPtr()->data(), d_out, totalSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Liberar
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return output;
}

__global__ void softmax2D_kernel(const float *logits, float *probs,
                                 size_t batchSize, size_t numClasses,
                                 size_t stride0, size_t stride1, size_t offset)
{
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batchSize)
        return;

    // Cargar logits de esta fila
    const float *row_ptr = logits + offset + row * stride0;
    float *out_ptr = probs + row * numClasses;

    // 1. Máximo logit (para estabilidad numérica)
    float maxLogit = -INFINITY;
    for (size_t j = 0; j < numClasses; ++j)
    {
        float val = row_ptr[j * stride1];
        if (val > maxLogit)
            maxLogit = val;
    }

    // 2. Exponenciales y suma
    float sumExp = 0.0f;
    for (size_t j = 0; j < numClasses; ++j)
    {
        float expVal = expf(row_ptr[j * stride1] - maxLogit);
        out_ptr[j] = expVal; // guardamos temporalmente exp
        sumExp += expVal;
    }

    // 3. Normalizar
    for (size_t j = 0; j < numClasses; ++j)
    {
        out_ptr[j] /= sumExp;
    }
}
Tensor softmax_cuda(const Tensor &logits)
{
    const std::vector<size_t> &shape = logits.getShape();
    const std::vector<size_t> &strides = logits.getStrides();
    size_t offset = logits.getDataOffset();

    if (shape.size() != 2)
        throw std::runtime_error("softmax_cuda solo soporta tensores 2D (batch_size x num_classes)");

    size_t batchSize = shape[0];
    size_t numClasses = shape[1];
    size_t stride0 = strides[0];
    size_t stride1 = strides[1];

    Tensor output(shape);

    // --- Reservar memoria en device ---
    float *d_logits, *d_probs;
    size_t totalInputSize = logits.getDataPtr()->size();
    size_t totalOutputSize = output.getSize();

    CUDA_CHECK(cudaMalloc(&d_logits, totalInputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_probs, totalOutputSize * sizeof(float)));

    // Copiar logits al device
    CUDA_CHECK(cudaMemcpy(d_logits, logits.getDataPtr()->data(),
                          totalInputSize * sizeof(float), cudaMemcpyHostToDevice));

    // --- Ejecutar kernel ---
    size_t threads = 256;
    size_t blocks = (batchSize + threads - 1) / threads;

    softmax2D_kernel<<<blocks, threads>>>(
        d_logits, d_probs,
        batchSize, numClasses,
        stride0, stride1, offset);

    // --- Copiar de vuelta ---
    CUDA_CHECK(cudaMemcpy(output.getDataPtr()->data(), d_probs,
                          totalOutputSize * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_probs));

    return output;
}

__global__ void softmax3D_axis2_kernel(const float *logits, float *probs,
                                       size_t stride0, size_t stride1, size_t stride2,
                                       size_t B, size_t N, size_t D,
                                       size_t offset)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * N)
        return;

    size_t b = i / N;
    size_t n = i % N;

    // --- Calcular puntero base para esta "fila lógica"
    const float *row_ptr = logits + offset + b * stride0 + n * stride1;
    float *out_ptr = probs + b * N * D + n * D;

    // 1. Máximo logit para estabilidad numérica
    float max_logit = -INFINITY;
    for (size_t d = 0; d < D; ++d)
    {
        float val = row_ptr[d * stride2];
        if (val > max_logit)
            max_logit = val;
    }

    // 2. Calcular exponenciales y suma
    float sum_exp = 0.0f;
    for (size_t d = 0; d < D; ++d)
    {
        float exp_val = expf(row_ptr[d * stride2] - max_logit);
        out_ptr[d] = exp_val;
        sum_exp += exp_val;
    }

    // 3. Normalizar
    for (size_t d = 0; d < D; ++d)
    {
        out_ptr[d] /= sum_exp;
    }
}
Tensor softmax_cuda(const Tensor &logits, int axis)
{
    const auto &shape = logits.getShape();
    const auto &strides = logits.getStrides();
    size_t offset = logits.getDataOffset();

    if (axis < 0)
        axis += shape.size();

    if (axis != 2 || shape.size() != 3)
        throw std::runtime_error("softmax_cuda solo implementado para tensores 3D en axis=2.");

    size_t B = shape[0];
    size_t N = shape[1];
    size_t D = shape[2];

    size_t stride0 = strides[0];
    size_t stride1 = strides[1];
    size_t stride2 = strides[2];

    Tensor output(shape);

    // --- Reservar memoria ---
    float *d_logits, *d_probs;
    size_t totalInSize = logits.getDataPtr()->size();
    size_t totalOutSize = output.getSize();

    CUDA_CHECK(cudaMalloc(&d_logits, totalInSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_probs, totalOutSize * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_logits, logits.getDataPtr()->data(),
                          totalInSize * sizeof(float), cudaMemcpyHostToDevice));

    // --- Ejecutar kernel ---
    size_t threads = 256;
    size_t blocks = (B * N + threads - 1) / threads;

    softmax3D_axis2_kernel<<<blocks, threads>>>(
        d_logits, d_probs,
        stride0, stride1, stride2,
        B, N, D, offset);

    // --- Copiar de vuelta ---
    CUDA_CHECK(cudaMemcpy(output.getDataPtr()->data(), d_probs,
                          totalOutSize * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_probs));

    return output;
}

__global__ void softmax_backward_axis2_kernel(const float *grad_output,
                                              const float *softmax_output,
                                              float *grad_input,
                                              size_t stride0, size_t stride1, size_t stride2,
                                              size_t B, size_t N, size_t D,
                                              size_t offset)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * N)
        return;

    size_t b = i / N;
    size_t n = i % N;

    // Punteros base para esta fila
    const float *go_row = grad_output + offset + b * stride0 + n * stride1;
    const float *s_row = softmax_output + offset + b * stride0 + n * stride1;
    float *gi_row = grad_input + b * N * D + n * D; // salida contigua

    // Paso 1: dot product entre grad_output y softmax
    float dot = 0.0f;
    for (size_t k = 0; k < D; ++k)
    {
        dot += go_row[k * stride2] * s_row[k * stride2];
    }

    // Paso 2: calcular dL/dZ_i = s_i * (dL/dS_i - dot)
    for (size_t i = 0; i < D; ++i)
    {
        float s_i = s_row[i * stride2];
        gi_row[i] = s_i * (go_row[i * stride2] - dot);
    }
}
Tensor softmax_backward_cuda(const Tensor &grad_output, const Tensor &softmax_output)
{
    const auto &shape = grad_output.getShape();
    const auto &strides = grad_output.getStrides();
    size_t offset = grad_output.getDataOffset();

    if (shape.size() != 3)
        throw std::runtime_error("softmax_backward_cuda solo implementado para tensores 3D.");

    size_t B = shape[0], N = shape[1], D = shape[2];
    size_t stride0 = strides[0];
    size_t stride1 = strides[1];
    size_t stride2 = strides[2];

    Tensor grad_input(shape);

    size_t totalSizeIn = grad_output.getDataPtr()->size();
    size_t totalSizeOut = grad_input.getSize();

    // --- Reservar y copiar memoria ---
    float *d_go, *d_softmax, *d_gi;

    CUDA_CHECK(cudaMalloc(&d_go, totalSizeIn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_softmax, totalSizeIn * sizeof(float))); // misma forma
    CUDA_CHECK(cudaMalloc(&d_gi, totalSizeOut * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_go, grad_output.getDataPtr()->data(),
                          totalSizeIn * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_softmax, softmax_output.getDataPtr()->data(),
                          totalSizeIn * sizeof(float), cudaMemcpyHostToDevice));

    // --- Ejecutar kernel ---
    size_t threads = 256;
    size_t blocks = (B * N + threads - 1) / threads;

    softmax_backward_axis2_kernel<<<blocks, threads>>>(
        d_go, d_softmax, d_gi,
        stride0, stride1, stride2,
        B, N, D, offset);

    // --- Copiar resultado a host ---
    CUDA_CHECK(cudaMemcpy(grad_input.getDataPtr()->data(), d_gi,
                          totalSizeOut * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Liberar ---
    CUDA_CHECK(cudaFree(d_go));
    CUDA_CHECK(cudaFree(d_softmax));
    CUDA_CHECK(cudaFree(d_gi));

    return grad_input;
}

// Kernel para tensor contiguo
__global__ void scale_contiguous_kernel(const float *input, float *output, float scale, size_t totalSize)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalSize)
    {
        output[i] = input[i] * scale;
    }
}

// Kernel para tensor 3D con strides
__global__ void scale_strided3D_kernel(const float *input, float *output, float scale,
                                       size_t stride0, size_t stride1, size_t stride2,
                                       size_t offset,
                                       size_t dim0, size_t dim1, size_t dim2)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim0 * dim1 * dim2)
    {
        size_t b = i / (dim1 * dim2);
        size_t rem = i % (dim1 * dim2);
        size_t n = rem / dim2;
        size_t d = rem % dim2;

        size_t idx = offset + b * stride0 + n * stride1 + d * stride2;
        output[idx] = input[idx] * scale;
    }
}
Tensor scale_tensor_cuda(const Tensor &scores, float scale_factor)
{
    const auto &shape = scores.getShape();
    const auto &strides = scores.getStrides();
    size_t ndim = shape.size();
    size_t offset = scores.getDataOffset();
    size_t totalSize = scores.getSize();

    if (ndim != 3)
        throw std::runtime_error("scale_tensor_cuda solo implementado para tensores 3D");

    const float *h_input = scores.getDataPtr()->data();
    size_t totalSizeWithOffset = scores.getDataPtr()->size();

    Tensor result(shape);
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, totalSizeWithOffset * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, totalSizeWithOffset * sizeof(float)));

    // Copiar input a GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input,
                          totalSizeWithOffset * sizeof(float), cudaMemcpyHostToDevice));

    // Ejecutar kernel
    size_t threads = 256;
    size_t blocks = (totalSize + threads - 1) / threads;

    if (scores.isContiguous() && offset == 0)
    {
        scale_contiguous_kernel<<<blocks, threads>>>(d_input, d_output, scale_factor, totalSize);
    }
    else
    {
        scale_strided3D_kernel<<<blocks, threads>>>(
            d_input, d_output, scale_factor,
            strides[0], strides[1], strides[2],
            offset,
            shape[0], shape[1], shape[2]);
    }

    // Copiar resultado desde GPU
    CUDA_CHECK(cudaMemcpy(result.getDataPtr()->data(), d_output,
                          totalSizeWithOffset * sizeof(float), cudaMemcpyDeviceToHost));

    // Liberar memoria
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return result;
}
Tensor scaledDotProductAttention_cuda(const Tensor &q, const Tensor &k, const Tensor &v, float scale_factor, Tensor &out_attention_weights)
{
    // 1. Transponer k (en CPU o GPU, según cómo esté implementado)
    Tensor k_transposed = k.transpose(1, 2);
    const auto &qShape = q.getShape(); // [B, N, D]
    const auto &kShape = k.getShape(); // [B, D, N] después de transponer
    const auto &vShape = v.getShape(); // [B, N, D]
    if (qShape.size() != 3 || kShape.size() != 3 || vShape.size() != 3)
        throw std::invalid_argument("scaledDotProductAttention_cuda requiere tensores 3D");

    size_t B = qShape[0];
    size_t N = qShape[1];
    size_t D = qShape[2];

    // 2. Subir q, k_transposed y v a GPU
    float *d_q, *d_k, *d_v;
    size_t qSize = q.getSize();
    size_t kSize = k_transposed.getSize();
    size_t vSize = v.getSize();

    CUDA_CHECK(cudaMalloc(&d_q, qSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, kSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, vSize * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_q, q.getData(), qSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, k_transposed.getData(), kSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, v.getData(), vSize * sizeof(float), cudaMemcpyHostToDevice));

    // 3. Ejecutar BMM: scores = q @ k_transposed
    float *d_scores;
    size_t scoresSize = B * N * N;
    CUDA_CHECK(cudaMalloc(&d_scores, scoresSize * sizeof(float)));

    // Configurar BMM con cublas
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    std::vector<const float *> q_array(B, nullptr), k_array(B, nullptr);
    std::vector<float *> scores_array(B, nullptr);
    for (size_t i = 0; i < B; ++i)
    {
        q_array[i] = d_q + i * N * D;
        k_array[i] = d_k + i * D * N;
        scores_array[i] = d_scores + i * N * N;
    }
    const float **d_q_array, **d_k_array;
    float **d_scores_array;
    CUDA_CHECK(cudaMalloc(&d_q_array, B * sizeof(float *)));
    CUDA_CHECK(cudaMalloc(&d_k_array, B * sizeof(float *)));
    CUDA_CHECK(cudaMalloc(&d_scores_array, B * sizeof(float *)));
    CUDA_CHECK(cudaMemcpy(d_q_array, q_array.data(), B * sizeof(float *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_array, k_array.data(), B * sizeof(float *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores_array, scores_array.data(), B * sizeof(float *), cudaMemcpyHostToDevice));

    // BMM: (B, N, D) x (B, D, N) = (B, N, N)
    CUBLAS_CHECK(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    N, N, D,
                                    &alpha,
                                    d_k_array, N,
                                    d_q_array, D,
                                    &beta,
                                    d_scores_array, N,
                                    B));

    // 4. Escalar: scores *= scale
    int threads = 256;
    int blocks = (scoresSize + threads - 1) / threads;
    scale_contiguous_kernel<<<blocks, threads>>>(d_scores, d_scores, scale_factor, scoresSize);

    // 5. Softmax: scores = softmax(scores)
    float *d_softmax;
    CUDA_CHECK(cudaMalloc(&d_softmax, scoresSize * sizeof(float)));
    softmax3D_axis2_kernel<<<(B * N + threads - 1) / threads, threads>>>(
        d_scores, d_softmax,
        N * N, N, 1, B, N, N, 0); // stride0, stride1, stride2, B,N,D

    out_attention_weights = Tensor({B, N, N});
    CUDA_CHECK(cudaMemcpy(out_attention_weights.getData(), d_softmax, scoresSize * sizeof(float), cudaMemcpyDeviceToHost));

    // 6. BMM final: output = softmax @ v
    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, B * N * D * sizeof(float)));

    std::vector<const float *> softmax_array(B), v_array(B);
    std::vector<float *> output_array(B);
    for (size_t i = 0; i < B; ++i)
    {
        softmax_array[i] = d_softmax + i * N * N;
        v_array[i] = d_v + i * N * D;
        output_array[i] = d_output + i * N * D;
    }
    const float **d_softmax_array, **d_v_array;
    float **d_output_array;
    CUDA_CHECK(cudaMalloc(&d_softmax_array, B * sizeof(float *)));
    CUDA_CHECK(cudaMalloc(&d_v_array, B * sizeof(float *)));
    CUDA_CHECK(cudaMalloc(&d_output_array, B * sizeof(float *)));
    CUDA_CHECK(cudaMemcpy(d_softmax_array, softmax_array.data(), B * sizeof(float *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_array, v_array.data(), B * sizeof(float *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_array, output_array.data(), B * sizeof(float *), cudaMemcpyHostToDevice));

    // BMM: (B, N, N) x (B, N, D) = (B, N, D)
    CUBLAS_CHECK(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    D, N, N,
                                    &alpha,
                                    d_v_array, D,
                                    d_softmax_array, N,
                                    &beta,
                                    d_output_array, D,
                                    B));

    // 7. Copiar d_output a CPU Tensor
    Tensor result({B, N, D});
    CUDA_CHECK(cudaMemcpy(result.getData(), d_output, result.getSize() * sizeof(float), cudaMemcpyDeviceToHost));

    // 8. Liberar toda la memoria GPU usada
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_softmax));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_q_array));
    CUDA_CHECK(cudaFree(d_k_array));
    CUDA_CHECK(cudaFree(d_scores_array));
    CUDA_CHECK(cudaFree(d_softmax_array));
    CUDA_CHECK(cudaFree(d_v_array));
    CUDA_CHECK(cudaFree(d_output_array));
    CUBLAS_CHECK(cublasDestroy(handle));
    return result;
}
Tensor denseForward_cuda(const Tensor &input, const Tensor &weights, const Tensor &bias)
{
    const auto &inputShape = input.getShape();
    size_t inputRank = inputShape.size();

    Tensor input_cpu = input;
    Tensor reshaped_input;
    std::vector<size_t> finalOutputShape;

    // --- Paso 1: Preprocesar entrada si es 3D ---
    if (inputRank == 3)
    {
        size_t B = inputShape[0];
        size_t N = inputShape[1];
        size_t D = inputShape[2];
        reshaped_input = input_cpu.reshape({B * N, D});
        finalOutputShape = {B, N, bias.getShape()[1]};
    }
    else if (inputRank == 2)
    {
        reshaped_input = input_cpu;
        finalOutputShape = {inputShape[0], bias.getShape()[1]};
    }
    else
    {
        throw std::runtime_error("denseForward_cuda: solo se admiten tensores 2D o 3D.");
    }

    // --- Paso 2: Subir input, weights y bias a la GPU ---
    float *d_input, *d_weights, *d_bias, *d_output;
    size_t inputSize = reshaped_input.getSize();
    size_t weightsSize = weights.getSize();
    size_t biasSize = bias.getSize();
    size_t outputSize = reshaped_input.getShape()[0] * weights.getShape()[1];

    CUDA_CHECK(cudaMalloc(&d_input, inputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, weightsSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, biasSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, outputSize * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, reshaped_input.getData(), inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights.getData(), weightsSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias.getData(), biasSize * sizeof(float), cudaMemcpyHostToDevice));

    // --- Paso 3: Multiplicación en GPU (X @ W) ---
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;

    int M = reshaped_input.getShape()[0];
    int K = reshaped_input.getShape()[1];
    int N = weights.getShape()[1];

    // cuBLAS: C^T = B^T @ A^T
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_weights, N,
                             d_input, K,
                             &beta,
                             d_output, N));

    // --- Paso 4: Sumar bias con broadcasting ---

    float *d_result;
    CUDA_CHECK(cudaMalloc(&d_result, outputSize * sizeof(float)));
    const std::vector<size_t> &shapeA = finalOutputShape; // salida sin bias
    int ndim = shapeA.size();

    if (ndim == 2)
    {
        size_t M = shapeA[0], N = shapeA[1];
        dim3 threads(16, 16);
        dim3 blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
        addBroadcast2D<<<blocks, threads>>>(d_output, d_bias, d_result, M, N);
    }
    else if (ndim == 3)
    {
        size_t B = shapeA[0], N = shapeA[1], D = shapeA[2];
        size_t total = B * N * D;
        size_t threadsPerBlock = 256;
        size_t numBlocks = (total + threadsPerBlock - 1) / threadsPerBlock;
        addBroadcast3D<<<numBlocks, threadsPerBlock>>>(d_output, d_bias, d_result, B, N, D);
    }
    else if (ndim == 4)
    {
        size_t Nn = shapeA[0], C = shapeA[1], H = shapeA[2], W = shapeA[3];
        size_t total = Nn * C * H * W;
        size_t threadsPerBlock = 256;
        size_t numBlocks = (total + threadsPerBlock - 1) / threadsPerBlock;
        addBias4D<<<numBlocks, threadsPerBlock>>>(d_output, d_bias, d_result, Nn, C, H, W);
    }
    else
    {
        throw std::runtime_error("Broadcast de bias no soportado para tensores de " + std::to_string(ndim) + " dimensiones.");
    }
    // --- Paso 5: Descargar resultado a CPU ---
    Tensor result_cpu(finalOutputShape); // CORRECTO
    CUDA_CHECK(cudaMemcpy(result_cpu.getData(), d_result, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Paso 6: Liberar memoria GPU ---
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_result));
    CUBLAS_CHECK(cublasDestroy(handle));

    // --- Paso 7: Si era entrada 3D, rehacer reshape final ---
    if (inputRank == 3)
    {
        return result_cpu.reshape(finalOutputShape);
    }
    return result_cpu;
}

__global__ void tensorAdd_kernel(const float *a, const float *b, float *c, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void tensorAdd_strided_kernel(
    const float *a, const float *b, float *c,
    const size_t *shape, const size_t *a_strides,
    const size_t *b_strides, const size_t *c_strides,
    size_t rank, size_t a_offset, size_t b_offset, size_t c_offset)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calcular índice multidimensional
    size_t total = 1;
    for (int i = 0; i < rank; ++i)
        total *= shape[i];

    if (idx >= total)
        return;

    size_t a_idx = a_offset;
    size_t b_idx = b_offset;
    size_t c_idx = c_offset;

    size_t remainder = idx;
    for (int d = 0; d < rank; ++d)
    {
        size_t coord = remainder / shape[d];
        remainder = remainder % shape[d];
        a_idx += coord * a_strides[d];
        b_idx += coord * b_strides[d];
        c_idx += coord * c_strides[d];
    }

    c[c_idx] = a[a_idx] + b[b_idx];
}

Tensor tensorAdd_cuda(const Tensor &a, const Tensor &b)
{
    if (a.getShape() != b.getShape())
        throw std::invalid_argument("tensorAdd_cuda: formas incompatibles");

    size_t size = a.getSize();
    Tensor result(a.getShape());

    const bool contiguo = a.getDataOffset() == 0 && b.getDataOffset() == 0 && result.getDataOffset() == 0;

    if (contiguo)
    {
        // --- CONTIGUO ---
        float *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc(&d_a, sizeof(float) * size));
        CUDA_CHECK(cudaMalloc(&d_b, sizeof(float) * size));
        CUDA_CHECK(cudaMalloc(&d_c, sizeof(float) * size));

        CUDA_CHECK(cudaMemcpy(d_a, a.getData(), sizeof(float) * size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, b.getData(), sizeof(float) * size, cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        tensorAdd_kernel<<<blocks, threads>>>(d_a, d_b, d_c, size);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(result.getData(), d_c, sizeof(float) * size, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }
    else
    {
        // --- NO CONTIGUO ---
        const auto &shape = a.getShape();
        const auto &a_strides = a.getStrides();
        const auto &b_strides = b.getStrides();
        const auto &c_strides = result.getStrides();
        const size_t rank = shape.size();

        float *d_a, *d_b, *d_c;
        size_t *d_shape, *d_a_strides, *d_b_strides, *d_c_strides;

        CUDA_CHECK(cudaMalloc(&d_a, sizeof(float) * a.getSize()));
        CUDA_CHECK(cudaMalloc(&d_b, sizeof(float) * b.getSize()));
        CUDA_CHECK(cudaMalloc(&d_c, sizeof(float) * result.getSize()));
        CUDA_CHECK(cudaMemcpy(d_a, a.getData(), sizeof(float) * a.getSize(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, b.getData(), sizeof(float) * b.getSize(), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_shape, sizeof(size_t) * rank));
        CUDA_CHECK(cudaMalloc(&d_a_strides, sizeof(size_t) * rank));
        CUDA_CHECK(cudaMalloc(&d_b_strides, sizeof(size_t) * rank));
        CUDA_CHECK(cudaMalloc(&d_c_strides, sizeof(size_t) * rank));

        CUDA_CHECK(cudaMemcpy(d_shape, shape.data(), sizeof(size_t) * rank, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_a_strides, a_strides.data(), sizeof(size_t) * rank, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b_strides, b_strides.data(), sizeof(size_t) * rank, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_c_strides, c_strides.data(), sizeof(size_t) * rank, cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        tensorAdd_strided_kernel<<<blocks, threads>>>(
            d_a, d_b, d_c,
            d_shape, d_a_strides, d_b_strides, d_c_strides,
            rank,
            a.getDataOffset(), b.getDataOffset(), result.getDataOffset());
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(result.getData(), d_c, sizeof(float) * result.getSize(), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
        CUDA_CHECK(cudaFree(d_shape));
        CUDA_CHECK(cudaFree(d_a_strides));
        CUDA_CHECK(cudaFree(d_b_strides));
        CUDA_CHECK(cudaFree(d_c_strides));
    }

    return result;
}

__global__ void tensorSquare_kernel(const float *a, float *c, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        c[idx] = a[idx] * a[idx];
    }
}
__global__ void tensorSquare_strided_kernel(
    const float *a, float *c,
    const size_t *shape,
    const size_t *a_strides,
    const size_t *c_strides,
    size_t rank,
    size_t a_offset,
    size_t c_offset)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calcular tamaño total
    size_t total = 1;
    for (int i = 0; i < rank; ++i)
        total *= shape[i];

    if (idx >= total)
        return;

    size_t a_idx = a_offset;
    size_t c_idx = c_offset;

    size_t remainder = idx;
    for (int d = 0; d < rank; ++d)
    {
        size_t coord = remainder / shape[d];
        remainder %= shape[d];
        a_idx += coord * a_strides[d];
        c_idx += coord * c_strides[d];
    }

    c[c_idx] = a[a_idx] * a[a_idx];
}

Tensor tensorSquare_cuda(const Tensor &a)
{
    size_t size = a.getSize();
    Tensor result(a.getShape());

    const bool contiguo = a.getDataOffset() == 0 && result.getDataOffset() == 0;

    if (contiguo)
    {
        // --- CONTIGUO ---
        float *d_a, *d_c;
        CUDA_CHECK(cudaMalloc(&d_a, sizeof(float) * size));
        CUDA_CHECK(cudaMalloc(&d_c, sizeof(float) * size));

        CUDA_CHECK(cudaMemcpy(d_a, a.getData(), sizeof(float) * size, cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        tensorSquare_kernel<<<blocks, threads>>>(d_a, d_c, size);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(result.getData(), d_c, sizeof(float) * size, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_c));
    }
    else
    {
        // --- NO CONTIGUO ---
        const auto &shape = a.getShape();
        const auto &a_strides = a.getStrides();
        const auto &c_strides = result.getStrides();
        const size_t rank = shape.size();

        float *d_a, *d_c;
        size_t *d_shape, *d_a_strides, *d_c_strides;

        CUDA_CHECK(cudaMalloc(&d_a, sizeof(float) * a.getSize()));
        CUDA_CHECK(cudaMalloc(&d_c, sizeof(float) * result.getSize()));
        CUDA_CHECK(cudaMemcpy(d_a, a.getData(), sizeof(float) * a.getSize(), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_shape, sizeof(size_t) * rank));
        CUDA_CHECK(cudaMalloc(&d_a_strides, sizeof(size_t) * rank));
        CUDA_CHECK(cudaMalloc(&d_c_strides, sizeof(size_t) * rank));

        CUDA_CHECK(cudaMemcpy(d_shape, shape.data(), sizeof(size_t) * rank, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_a_strides, a_strides.data(), sizeof(size_t) * rank, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_c_strides, c_strides.data(), sizeof(size_t) * rank, cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        tensorSquare_strided_kernel<<<blocks, threads>>>(
            d_a, d_c,
            d_shape, d_a_strides, d_c_strides,
            rank,
            a.getDataOffset(),
            result.getDataOffset());
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(result.getData(), d_c, sizeof(float) * result.getSize(), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_c));
        CUDA_CHECK(cudaFree(d_shape));
        CUDA_CHECK(cudaFree(d_a_strides));
        CUDA_CHECK(cudaFree(d_c_strides));
    }

    return result;
}

__global__ void sum2D_kernel(const float *input, float *output,
                             size_t dim0, size_t dim1,
                             size_t axis,
                             size_t in_stride0, size_t in_stride1,
                             size_t out_stride0, size_t out_stride1,
                             size_t in_offset, size_t out_offset)
{
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= dim0 || j >= dim1)
        return;

    size_t out_idx = out_offset + i * out_stride0 + j * out_stride1;

    float sum = 0.0f;
    for (size_t k = 0; k < (axis == 0 ? dim0 : dim1); ++k)
    {
        size_t in_i = axis == 0 ? k : i;
        size_t in_j = axis == 1 ? k : j;
        size_t in_idx = in_offset + in_i * in_stride0 + in_j * in_stride1;
        sum += input[in_idx];
    }

    output[out_idx] = sum;
}

__global__ void sum3D_kernel(const float *input, float *output,
                             size_t d0, size_t d1, size_t d2,
                             size_t axis,
                             const size_t *in_strides,
                             const size_t *out_strides,
                             size_t in_offset, size_t out_offset)
{
    size_t i = blockIdx.z * blockDim.z + threadIdx.z;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= d0 || j >= d1 || k >= d2)
        return;

    size_t coord[3] = {i, j, k};
    coord[axis] = 0;

    size_t out_idx = out_offset;
    for (int d = 0; d < 3; ++d)
        out_idx += coord[d] * out_strides[d];

    float sum = 0.0f;
    for (size_t l = 0; l < (axis == 0 ? d0 : (axis == 1 ? d1 : d2)); ++l)
    {
        coord[axis] = l;
        size_t in_idx = in_offset;
        for (int d = 0; d < 3; ++d)
            in_idx += coord[d] * in_strides[d];
        sum += input[in_idx];
    }

    output[out_idx] = sum;
}

__global__ void sum4D_kernel(const float *input, float *output,
                             size_t d0, size_t d1, size_t d2, size_t d3,
                             size_t axis,
                             const size_t *in_strides,
                             const size_t *out_strides,
                             size_t in_offset, size_t out_offset)
{
    size_t i = blockIdx.z * blockDim.z + threadIdx.z;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= d0 || j >= d1 || k >= d2 * d3)
        return;

    size_t l = k % d3;
    size_t m = k / d3;

    size_t coord[4] = {i, j, m, l};
    coord[axis] = 0;

    size_t out_idx = out_offset;
    for (int d = 0; d < 4; ++d)
        out_idx += coord[d] * out_strides[d];

    float sum = 0.0f;
    for (size_t n = 0; n < (axis == 0 ? d0 : (axis == 1 ? d1 : (axis == 2 ? d2 : d3))); ++n)
    {
        coord[axis] = n;
        size_t in_idx = in_offset;
        for (int d = 0; d < 4; ++d)
            in_idx += coord[d] * in_strides[d];
        sum += input[in_idx];
    }

    output[out_idx] = sum;
}

Tensor tensorSum_cuda2(const Tensor &a, size_t axis)
{
    const auto &shape = a.getShape();
    const auto &strides = a.getStrides();

    if (axis >= shape.size())
        throw std::out_of_range("tensorSum_cuda: eje fuera de rango");

    std::vector<size_t> outputShape = shape;
    outputShape[axis] = 1;
    Tensor result(outputShape);

    float *d_in, *d_out;
    size_t rank = shape.size();

    CUDA_CHECK(cudaMalloc(&d_in, sizeof(float) * a.getSize()));
    CUDA_CHECK(cudaMemcpy(d_in, a.getData(), sizeof(float) * a.getSize(), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * result.getSize()));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float) * result.getSize()));

    size_t in_offset = a.getDataOffset();
    size_t out_offset = result.getDataOffset();

    if (rank == 2)
    {
        size_t d0 = shape[0], d1 = shape[1];
        dim3 block(16, 16);
        dim3 grid((d1 + 15) / 16, (d0 + 15) / 16);

        sum2D_kernel<<<grid, block>>>(
            d_in, d_out,
            d0, d1, axis,
            strides[0], strides[1],
            result.getStrides()[0], result.getStrides()[1],
            in_offset, out_offset);
    }
    else if (rank == 3)
    {
        size_t *d_in_strides, *d_out_strides;
        CUDA_CHECK(cudaMalloc(&d_in_strides, sizeof(size_t) * 3));
        CUDA_CHECK(cudaMalloc(&d_out_strides, sizeof(size_t) * 3));
        CUDA_CHECK(cudaMemcpy(d_in_strides, strides.data(), sizeof(size_t) * 3, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out_strides, result.getStrides().data(), sizeof(size_t) * 3, cudaMemcpyHostToDevice));

        dim3 block(4, 4, 4);
        dim3 grid((shape[2] + 3) / 4, (shape[1] + 3) / 4, (shape[0] + 3) / 4);

        sum3D_kernel<<<grid, block>>>(
            d_in, d_out,
            shape[0], shape[1], shape[2],
            axis,
            d_in_strides, d_out_strides,
            in_offset, out_offset);

        CUDA_CHECK(cudaFree(d_in_strides));
        CUDA_CHECK(cudaFree(d_out_strides));
    }
    else if (rank == 4)
    {
        size_t *d_in_strides, *d_out_strides;
        CUDA_CHECK(cudaMalloc(&d_in_strides, sizeof(size_t) * 4));
        CUDA_CHECK(cudaMalloc(&d_out_strides, sizeof(size_t) * 4));
        CUDA_CHECK(cudaMemcpy(d_in_strides, strides.data(), sizeof(size_t) * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out_strides, result.getStrides().data(), sizeof(size_t) * 4, cudaMemcpyHostToDevice));

        size_t d0 = shape[0], d1 = shape[1], d2 = shape[2], d3 = shape[3];
        dim3 block(8, 4, 4);
        dim3 grid((d2 * d3 + 7) / 8, (d1 + 3) / 4, (d0 + 3) / 4);

        sum4D_kernel<<<grid, block>>>(
            d_in, d_out,
            d0, d1, d2, d3,
            axis,
            d_in_strides, d_out_strides,
            in_offset, out_offset);

        CUDA_CHECK(cudaFree(d_in_strides));
        CUDA_CHECK(cudaFree(d_out_strides));
    }
    else
    {
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        throw std::runtime_error("tensorSum_cuda solo implementado para 2D, 3D y 4D.");
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(result.getData(), d_out, sizeof(float) * result.getSize(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return result;
}

Tensor tensorSum_cuda(const Tensor &a, size_t axis)
{
    const auto &shape = a.getShape();
    const auto &strides = a.getStrides();

    if (axis >= shape.size())
        throw std::out_of_range("tensorSum_cuda: eje fuera de rango");

    std::vector<size_t> outputShape = shape;
    outputShape[axis] = 1;
    Tensor result(outputShape);

    size_t rank = shape.size();
    size_t totalSize = a.getSize();
    size_t offset = a.getDataOffset();

    float *d_in_raw, *d_out;

    // --- 1. Copiar tensor a GPU (contiguo si necesario) ---
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * result.getSize()));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float) * result.getSize()));

    bool isContig = a.isContiguous() && offset == 0;

    if (isContig)
    {
        // Ya es contiguo, copiar directamente
        CUDA_CHECK(cudaMalloc(&d_in_raw, sizeof(float) * totalSize));
        CUDA_CHECK(cudaMemcpy(d_in_raw, a.getData(), sizeof(float) * totalSize, cudaMemcpyHostToDevice));
    }
    else
    {
        // NO contiguo → usar copyXD para hacer layout contiguo en GPU
        CUDA_CHECK(cudaMalloc(&d_in_raw, sizeof(float) * totalSize));

        float *d_temp;
        size_t fullDataSize = a.getDataPtr()->size();
        CUDA_CHECK(cudaMalloc(&d_temp, sizeof(float) * fullDataSize));
        CUDA_CHECK(cudaMemcpy(d_temp, a.getDataPtr()->data(), sizeof(float) * fullDataSize, cudaMemcpyHostToDevice));

        size_t threads = 256;
        size_t blocks = (totalSize + threads - 1) / threads;

        if (rank == 1)
        {
            copy1D<<<blocks, threads>>>(d_temp, d_in_raw, strides[0], offset, shape[0]);
        }
        else if (rank == 2)
        {
            copy2D<<<blocks, threads>>>(d_temp, d_in_raw, strides[0], strides[1], offset, shape[0], shape[1]);
        }
        else if (rank == 3)
        {
            copy3D<<<blocks, threads>>>(d_temp, d_in_raw, strides[0], strides[1], strides[2], offset, shape[0], shape[1], shape[2]);
        }
        else if (rank == 4)
        {
            copy4D<<<blocks, threads>>>(d_temp, d_in_raw, strides[0], strides[1], strides[2], strides[3], offset, shape[0], shape[1], shape[2], shape[3]);
        }
        else
        {
            CUDA_CHECK(cudaFree(d_temp));
            CUDA_CHECK(cudaFree(d_in_raw));
            CUDA_CHECK(cudaFree(d_out));
            throw std::runtime_error("tensorSum_cuda: ndim > 4 no soportado");
        }

        CUDA_CHECK(cudaFree(d_temp));
    }

    // --- 2. Lanzar kernel de suma según dimensiones ---
    size_t out_offset = result.getDataOffset();

    if (rank == 2)
    {
        dim3 block(16, 16);
        dim3 grid((shape[1] + 15) / 16, (shape[0] + 15) / 16);

        sum2D_kernel<<<grid, block>>>(
            d_in_raw, d_out,
            shape[0], shape[1], axis,
            shape[1], 1, // Input ya contiguo
            result.getStrides()[0], result.getStrides()[1],
            0, out_offset);
    }
    else if (rank == 3)
    {
        size_t *d_in_strides, *d_out_strides;
        CUDA_CHECK(cudaMalloc(&d_in_strides, sizeof(size_t) * 3));
        CUDA_CHECK(cudaMalloc(&d_out_strides, sizeof(size_t) * 3));

        std::vector<size_t> input_strides = {shape[1] * shape[2], shape[2], 1}; // contiguo layout
        CUDA_CHECK(cudaMemcpy(d_in_strides, input_strides.data(), sizeof(size_t) * 3, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out_strides, result.getStrides().data(), sizeof(size_t) * 3, cudaMemcpyHostToDevice));

        dim3 block(4, 4, 4);
        dim3 grid((shape[2] + 3) / 4, (shape[1] + 3) / 4, (shape[0] + 3) / 4);

        sum3D_kernel<<<grid, block>>>(
            d_in_raw, d_out,
            shape[0], shape[1], shape[2],
            axis,
            d_in_strides, d_out_strides,
            0, out_offset);

        CUDA_CHECK(cudaFree(d_in_strides));
        CUDA_CHECK(cudaFree(d_out_strides));
    }
    else if (rank == 4)
    {
        size_t *d_in_strides, *d_out_strides;
        CUDA_CHECK(cudaMalloc(&d_in_strides, sizeof(size_t) * 4));
        CUDA_CHECK(cudaMalloc(&d_out_strides, sizeof(size_t) * 4));

        std::vector<size_t> input_strides = {
            shape[1] * shape[2] * shape[3],
            shape[2] * shape[3],
            shape[3],
            1};
        CUDA_CHECK(cudaMemcpy(d_in_strides, input_strides.data(), sizeof(size_t) * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out_strides, result.getStrides().data(), sizeof(size_t) * 4, cudaMemcpyHostToDevice));

        dim3 block(8, 4, 4);
        dim3 grid((shape[2] * shape[3] + 7) / 8, (shape[1] + 3) / 4, (shape[0] + 3) / 4);

        sum4D_kernel<<<grid, block>>>(
            d_in_raw, d_out,
            shape[0], shape[1], shape[2], shape[3],
            axis,
            d_in_strides, d_out_strides,
            0, out_offset);

        CUDA_CHECK(cudaFree(d_in_strides));
        CUDA_CHECK(cudaFree(d_out_strides));
    }
    else
    {
        CUDA_CHECK(cudaFree(d_in_raw));
        CUDA_CHECK(cudaFree(d_out));
        throw std::runtime_error("tensorSum_cuda solo implementado para 2D, 3D y 4D.");
    }

    // --- 3. Copiar resultado final a CPU ---
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(result.getData(), d_out, sizeof(float) * result.getSize(), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in_raw));
    CUDA_CHECK(cudaFree(d_out));

    return result;
}
__global__ void dropout_backward_kernel(const float *grad_out, const float *mask, float *grad_in, size_t totalSize)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalSize)
    {
        grad_in[i] = grad_out[i] * mask[i];
    }
}
Tensor dropout_backward_cuda(const Tensor &grad_out, const Tensor &mask)
{
    if (!grad_out.isContiguous() || !mask.isContiguous())
        throw std::runtime_error("dropout_backward_cuda solo soporta tensores contiguos por ahora.");

    size_t totalSize = grad_out.getSize();
    if (mask.getSize() != totalSize)
        throw std::runtime_error("dropout_backward_cuda: tamaños no coinciden entre grad_out y mask.");

    // Crear tensor de salida
    Tensor grad_in(grad_out.getShape());

    // Punteros en host
    const float *h_grad_out = grad_out.getDataPtr()->data();
    const float *h_mask = mask.getDataPtr()->data();
    float *h_grad_in = grad_in.getDataPtr()->data();

    // Punteros en device
    float *d_grad_out, *d_mask, *d_grad_in;
    CUDA_CHECK(cudaMalloc(&d_grad_out, totalSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mask, totalSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_in, totalSize * sizeof(float)));

    // Copiar datos al device
    CUDA_CHECK(cudaMemcpy(d_grad_out, h_grad_out, totalSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, h_mask, totalSize * sizeof(float), cudaMemcpyHostToDevice));

    // Lanzar kernel
    size_t threads = 256;
    size_t blocks = (totalSize + threads - 1) / threads;
    dropout_backward_kernel<<<blocks, threads>>>(d_grad_out, d_mask, d_grad_in, totalSize);
    CUDA_CHECK(cudaGetLastError());

    // Copiar resultado a host
    CUDA_CHECK(cudaMemcpy(h_grad_in, d_grad_in, totalSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Liberar
    CUDA_CHECK(cudaFree(d_grad_out));
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_grad_in));

    return grad_in;
}
__global__ void layernorm_forward_kernel(
    const float *input,
    float *output,
    float *normalized,
    float *mean_out,
    float *inv_std_out,
    const float *gamma,
    const float *beta,
    size_t batchSize,
    size_t featureSize,
    float epsilon)
{
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batchSize)
        return;

    const float *in_row = input + row * featureSize;
    float *out_row = output + row * featureSize;
    float *norm_row = normalized ? (normalized + row * featureSize) : nullptr;

    // --- 1. Mean ---
    float mean = 0.0f;
    for (size_t j = 0; j < featureSize; ++j)
        mean += in_row[j];
    mean /= featureSize;

    if (mean_out)
        mean_out[row] = mean;

    // --- 2. Variance ---
    float var = 0.0f;
    for (size_t j = 0; j < featureSize; ++j)
    {
        float diff = in_row[j] - mean;
        var += diff * diff;
    }
    var /= featureSize;
    float inv_std = rsqrtf(var + epsilon);
    if (inv_std_out)
        inv_std_out[row] = inv_std;

    // --- 3. Normalize and apply gamma/beta ---
    for (size_t j = 0; j < featureSize; ++j)
    {
        float x_hat = (in_row[j] - mean) * inv_std;
        if (norm_row)
            norm_row[j] = x_hat;
        out_row[j] = gamma[j] * x_hat + beta[j];
    }
}
LayerNormResult layernorm_forward_cuda(const Tensor &input,
                                       const Tensor &gamma,
                                       const Tensor &beta,
                                       float epsilon,
                                       bool isTraining)
{
    const auto &shape = input.getShape();
    if (shape.size() < 1)
        throw std::runtime_error("input must have at least 1 dimension");

    size_t featureSize = shape.back();
    size_t batchSize = input.getSize() / featureSize;
    Tensor input2D = input.reshape({batchSize, featureSize});

    const float *h_input = input2D.getDataPtr()->data();
    const float *h_gamma = gamma.getDataPtr()->data();
    const float *h_beta = beta.getDataPtr()->data();

    size_t totalSize = input2D.getSize();

    Tensor output({batchSize, featureSize});
    Tensor normalized, mean, invStd;

    float *d_input, *d_output, *d_normalized = nullptr;
    float *d_mean = nullptr, *d_invstd = nullptr;
    float *d_gamma, *d_beta;

    CUDA_CHECK(cudaMalloc(&d_input, totalSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, totalSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, featureSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, featureSize * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, totalSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, featureSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta, featureSize * sizeof(float), cudaMemcpyHostToDevice));

    if (isTraining)
    {
        normalized = Tensor({batchSize, featureSize});
        mean = Tensor({batchSize, 1});
        invStd = Tensor({batchSize, 1});
        CUDA_CHECK(cudaMalloc(&d_normalized, totalSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mean, batchSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_invstd, batchSize * sizeof(float)));
    }

    size_t threads = 256;
    size_t blocks = (batchSize + threads - 1) / threads;

    layernorm_forward_kernel<<<blocks, threads>>>(
        d_input,
        d_output,
        d_normalized,
        d_mean,
        d_invstd,
        d_gamma,
        d_beta,
        batchSize,
        featureSize,
        epsilon);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(output.getDataPtr()->data(), d_output, totalSize * sizeof(float), cudaMemcpyDeviceToHost));
    if (isTraining)
    {
        CUDA_CHECK(cudaMemcpy(normalized.getDataPtr()->data(), d_normalized, totalSize * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mean.getDataPtr()->data(), d_mean, batchSize * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(invStd.getDataPtr()->data(), d_invstd, batchSize * sizeof(float), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    if (isTraining)
    {
        CUDA_CHECK(cudaFree(d_normalized));
        CUDA_CHECK(cudaFree(d_mean));
        CUDA_CHECK(cudaFree(d_invstd));
    }

    return {input2D, output.reshape(shape), normalized, mean, invStd};
}

__global__ void im2col_kernel_strided(
    const float *input,
    float *col,
    size_t offset,
    size_t s_batch, size_t s_channel, size_t s_y, size_t s_x,
    size_t batch_size, size_t channels,
    size_t height, size_t width,
    size_t kernel_size, size_t stride, size_t padding,
    size_t out_h, size_t out_w)
{
    size_t col_index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_cols = batch_size * out_h * out_w;
    if (col_index >= total_cols)
        return;

    size_t w_out = col_index % out_w;
    size_t h_out = (col_index / out_w) % out_h;
    size_t b = col_index / (out_h * out_w);

    for (size_t c = 0; c < channels; ++c)
    {
        for (size_t kh = 0; kh < kernel_size; ++kh)
        {
            for (size_t kw = 0; kw < kernel_size; ++kw)
            {
                int h_in = int(h_out * stride + kh) - int(padding);
                int w_in = int(w_out * stride + kw) - int(padding);

                float val = 0.0f;
                if (h_in >= 0 && h_in < int(height) && w_in >= 0 && w_in < int(width))
                {
                    size_t input_idx = offset +
                                       b * s_batch +
                                       c * s_channel +
                                       h_in * s_y +
                                       w_in * s_x;
                    val = input[input_idx];
                }

                size_t row = c * kernel_size * kernel_size + kh * kernel_size + kw;
                size_t col_pos = b * (out_h * out_w) + h_out * out_w + w_out;
                col[row * total_cols + col_pos] = val;
            }
        }
    }
}

Tensor im2col_cuda(const Tensor &input, const std::vector<size_t> &shape, size_t kernel_size, size_t stride, size_t padding)
{
    if (shape.size() != 3)
        throw std::runtime_error("La forma debe tener tamaño 3: {patch_dim, batch_size, num_patches}");

    const auto &in_shape = input.getShape();
    if (in_shape.size() != 4)
        throw std::runtime_error("El tensor de entrada debe ser 4D");

    const size_t batch_size = in_shape[0];
    const size_t in_channels = in_shape[1];
    const size_t in_h = in_shape[2];
    const size_t in_w = in_shape[3];

    const size_t out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    const size_t out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    const size_t patch_dim = shape[0];
    const size_t batch_size_nominal = shape[1];
    const size_t num_patches = shape[2];

    if (batch_size_nominal != batch_size)
        throw std::runtime_error("Batch size no coincide");

    if (patch_dim != in_channels * kernel_size * kernel_size)
        throw std::runtime_error("Dim patch incorrecto");

    if (num_patches != out_h * out_w)
        throw std::runtime_error("num_patches no coincide con out_h * out_w");

    Tensor col_matrix({patch_dim, batch_size * num_patches});

    const auto &in_strides = input.getStrides();
    const size_t offset = input.getDataOffset();
    const size_t ndim = in_shape.size();
    const size_t input_size = input.getSize();
    const size_t total_input_elements = input.getDataPtr()->size();
    const size_t col_size = col_matrix.getSize();

    float *d_input_raw = nullptr;
    float *d_input_contig = nullptr;
    float *d_col = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input_raw, sizeof(float) * total_input_elements));
    CUDA_CHECK(cudaMemcpy(d_input_raw, input.getDataPtr()->data(),
                          sizeof(float) * total_input_elements, cudaMemcpyHostToDevice));

    if (input.isContiguous() && offset == 0)
    {
        d_input_contig = d_input_raw;
    }
    else
    {
        CUDA_CHECK(cudaMalloc(&d_input_contig, sizeof(float) * input_size));
        const size_t threads = 256;
        const size_t blocks = (input_size + threads - 1) / threads;

        switch (ndim)
        {
        case 1:
            copy1D<<<blocks, threads>>>(d_input_raw, d_input_contig, in_strides[0], offset, in_shape[0]);
            break;
        case 2:
            copy2D<<<blocks, threads>>>(d_input_raw, d_input_contig, in_strides[0], in_strides[1], offset, in_shape[0], in_shape[1]);
            break;
        case 3:
            copy3D<<<blocks, threads>>>(d_input_raw, d_input_contig, in_strides[0], in_strides[1], in_strides[2], offset, in_shape[0], in_shape[1], in_shape[2]);
            break;
        case 4:
            copy4D<<<blocks, threads>>>(d_input_raw, d_input_contig, in_strides[0], in_strides[1], in_strides[2], in_strides[3], offset, in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
            break;
        default:
            CUDA_CHECK(cudaFree(d_input_raw));
            throw std::runtime_error("Contiguous no soportado para ndim > 4.");
        }

        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMalloc(&d_col, sizeof(float) * col_size));

    const int threads = 256;
    const int blocks = (batch_size * out_h * out_w + threads - 1) / threads;

    im2col_kernel_strided<<<blocks, threads>>>(
        d_input_contig,
        d_col,
        0, // offset ya aplicado
        in_channels * in_h * in_w,
        in_h * in_w,
        in_w,
        1, // contiguo
        batch_size, in_channels, in_h, in_w,
        kernel_size, stride, padding,
        out_h, out_w);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(col_matrix.getData(), d_col, sizeof(float) * col_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input_raw));
    if (d_input_contig != d_input_raw)
        CUDA_CHECK(cudaFree(d_input_contig));
    CUDA_CHECK(cudaFree(d_col));

    return col_matrix;
}

__global__ void col2im_kernel(
    const float *__restrict__ d_col,
    float *__restrict__ d_output,
    size_t batch_size,
    size_t in_channels,
    size_t in_h,
    size_t in_w,
    size_t kernel_size,
    size_t stride,
    size_t padding,
    size_t out_h,
    size_t out_w,
    size_t col_stride_0, // row-major stride de col_matrix
    size_t col_stride_1,
    size_t out_stride_0, // strides de output_image
    size_t out_stride_1,
    size_t out_stride_2,
    size_t out_stride_3)
{
    size_t col_c = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_cols = batch_size * out_h * out_w;
    if (col_c >= total_cols)
        return;

    size_t w_out = col_c % out_w;
    size_t h_out = (col_c / out_w) % out_h;
    size_t b_idx = col_c / (out_h * out_w);

    size_t row_idx = 0;
    for (size_t c_in = 0; c_in < in_channels; ++c_in)
    {
        for (size_t kh = 0; kh < kernel_size; ++kh)
        {
            for (size_t kw = 0; kw < kernel_size; ++kw)
            {
                int h_in = static_cast<int>(h_out * stride + kh) - static_cast<int>(padding);
                int w_in = static_cast<int>(w_out * stride + kw) - static_cast<int>(padding);

                if (h_in >= 0 && h_in < static_cast<int>(in_h) &&
                    w_in >= 0 && w_in < static_cast<int>(in_w))
                {
                    size_t out_index =
                        b_idx * out_stride_0 +
                        c_in * out_stride_1 +
                        h_in * out_stride_2 +
                        w_in * out_stride_3;

                    size_t col_index = row_idx * col_stride_0 + col_c * col_stride_1;

                    // Accumulación atómica
                    atomicAdd(&d_output[out_index], d_col[col_index]);
                }
                row_idx++;
            }
        }
    }
}
Tensor col2im_cuda(const Tensor &col_matrix,
                   const std::vector<size_t> &in_shape,
                   size_t kernel_size,
                   size_t stride,
                   size_t padding)
{
    if (in_shape.size() != 4)
        throw std::runtime_error("in_shape debe ser de tamaño 4 (N, C, H, W)");

    size_t batch_size = in_shape[0];
    size_t in_channels = in_shape[1];
    size_t in_h = in_shape[2];
    size_t in_w = in_shape[3];

    size_t out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    size_t out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    size_t total_cols = batch_size * out_h * out_w;

    // Crear tensor de salida inicializado en 0
    Tensor output_image(in_shape);
    std::fill(output_image.getDataPtr()->begin(), output_image.getDataPtr()->end(), 0.0f);

    // Reservar y copiar memoria a GPU
    float *d_col = nullptr;
    float *d_output = nullptr;

    const float *h_col = col_matrix.getData();
    const size_t col_size = col_matrix.getSize();
    const size_t out_size = output_image.getSize();

    CUDA_CHECK(cudaMalloc(&d_col, sizeof(float) * col_size));
    CUDA_CHECK(cudaMemcpy(d_col, h_col, sizeof(float) * col_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float) * out_size));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float) * out_size)); // Cero inicializado

    const auto &col_strides = col_matrix.getStrides();
    const auto &out_strides = output_image.getStrides();

    dim3 blockDim(256);
    dim3 gridDim((total_cols + blockDim.x - 1) / blockDim.x);

    col2im_kernel<<<gridDim, blockDim>>>(
        d_col,
        d_output,
        batch_size,
        in_channels,
        in_h,
        in_w,
        kernel_size,
        stride,
        padding,
        out_h,
        out_w,
        col_strides[0],
        col_strides[1],
        out_strides[0],
        out_strides[1],
        out_strides[2],
        out_strides[3]);

    CUDA_CHECK(cudaGetLastError());

    // Copiar resultado a CPU
    CUDA_CHECK(cudaMemcpy(output_image.getData(), d_output, sizeof(float) * out_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFree(d_output));

    return output_image;
}
__global__ void cross_entropy_kernel(
    const float *softmax_output,
    const float *y_true,
    float *losses,
    size_t batch_size,
    size_t num_classes,
    size_t stride_softmax_0,
    size_t stride_softmax_1,
    size_t stride_ytrue_0,
    size_t stride_ytrue_1,
    float epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
        return;

    float sample_loss = 0.0f;

    for (size_t j = 0; j < num_classes; ++j)
    {
        size_t y_idx = idx * stride_ytrue_0 + j * stride_ytrue_1;
        size_t s_idx = idx * stride_softmax_0 + j * stride_softmax_1;
        if (y_true[y_idx] == 1.0f)
        {
            sample_loss = -logf(softmax_output[s_idx] + epsilon);
            break; // Solo una clase es 1
        }
    }

    losses[idx] = sample_loss;
}

float cross_entropy_cuda(const Tensor &softmax_outpu, const Tensor &y_tru)
{
    // === Parte 1: Preparar tensores contiguos en GPU ===
    const std::vector<size_t> &shape = softmax_outpu.getShape();
    const std::vector<size_t> &strides_soft = softmax_outpu.getStrides();
    const std::vector<size_t> &strides_true = y_tru.getStrides();

    size_t batch_size = shape[0];
    size_t num_classes = shape[1];
    size_t total_size_soft = softmax_outpu.getSize();
    size_t total_size_true = y_tru.getSize();

    size_t offset_soft = softmax_outpu.getDataOffset();
    size_t offset_true = y_tru.getDataOffset();

    // Asumimos ambos tensores tienen mismo ndim
    size_t ndim = shape.size();

    // --- Allocar y copiar datos originales a GPU ---
    float *d_in_soft, *d_in_true;
    CUDA_CHECK(cudaMalloc(&d_in_soft, softmax_outpu.getDataPtr()->size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_in_true, y_tru.getDataPtr()->size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in_soft, softmax_outpu.getDataPtr()->data(),
                          softmax_outpu.getDataPtr()->size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_true, y_tru.getDataPtr()->data(),
                          y_tru.getDataPtr()->size() * sizeof(float), cudaMemcpyHostToDevice));

    // --- Allocar resultados contiguos ---
    float *d_soft, *d_true;
    CUDA_CHECK(cudaMalloc(&d_soft, total_size_soft * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_true, total_size_true * sizeof(float)));

    // --- Lanzar kernels para reordenar contiguamente ---
    dim3 threads(256);
    dim3 blocks_soft((total_size_soft + 255) / 256);
    dim3 blocks_true((total_size_true + 255) / 256);

    if (ndim == 2)
    {
        copy2D<<<blocks_soft, threads>>>(
            d_in_soft, d_soft,
            strides_soft[0], strides_soft[1],
            offset_soft,
            shape[0], shape[1]);

        copy2D<<<blocks_true, threads>>>(
            d_in_true, d_true,
            strides_true[0], strides_true[1],
            offset_true,
            shape[0], shape[1]);
    }
    else
    {
        CUDA_CHECK(cudaFree(d_in_soft));
        CUDA_CHECK(cudaFree(d_in_true));
        CUDA_CHECK(cudaFree(d_soft));
        CUDA_CHECK(cudaFree(d_true));
        throw std::runtime_error("cross_entropy_cuda2() solo soporta tensores 2D.");
    }

    // === Parte 2: Calcular cross entropy en GPU ===
    float *d_losses;
    CUDA_CHECK(cudaMalloc(&d_losses, sizeof(float) * batch_size));

    const float epsilon = 1e-12f;

    cross_entropy_kernel<<<(batch_size + 255) / 256, 256>>>(
        d_soft, d_true, d_losses,
        batch_size,
        num_classes,
        num_classes, 1, // Contiguo: fila mayor
        num_classes, 1,
        epsilon);

    CUDA_CHECK(cudaGetLastError());

    // === Parte 3: Reducir pérdidas ===
    std::vector<float> h_losses(batch_size);
    CUDA_CHECK(cudaMemcpy(h_losses.data(), d_losses, sizeof(float) * batch_size, cudaMemcpyDeviceToHost));

    float total_loss = std::accumulate(h_losses.begin(), h_losses.end(), 0.0f);
    total_loss /= static_cast<float>(batch_size);

    // === Liberar memoria ===
    CUDA_CHECK(cudaFree(d_in_soft));
    CUDA_CHECK(cudaFree(d_in_true));
    CUDA_CHECK(cudaFree(d_soft));
    CUDA_CHECK(cudaFree(d_true));
    CUDA_CHECK(cudaFree(d_losses));

    return total_loss;
}

__global__ void crossentropy_backward_kernel(
    const float *softmax_output,
    const float *y_true,
    float *gradient_out,
    size_t batch_size,
    size_t num_classes)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // batch idx
    int j = blockIdx.x * blockDim.x + threadIdx.x; // class idx

    if (i < batch_size && j < num_classes)
    {
        int idx = i * num_classes + j;
        gradient_out[idx] = (softmax_output[idx] - y_true[idx]) / static_cast<float>(batch_size);
    }
}
Tensor ce_backward_cuda(const Tensor &softmax_output_cpu, const Tensor &y_true_cpu)
{
    // Convertir a layout contiguo para acceso directo
    // Tensor softmax_output_cpu = contiguous_cuda(softmax_output_cp);
    // Tensor y_true_cpu = contiguous_cuda(y_true_cp);

    const std::vector<size_t> &shape = softmax_output_cpu.getShape();
    size_t batch_size = shape[0];
    size_t num_classes = shape[1];
    size_t total_size = batch_size * num_classes;

    const float *h_soft = softmax_output_cpu.getDataPtr()->data();
    const float *h_true = y_true_cpu.getDataPtr()->data();

    // Allocate and copy input to device
    float *d_soft, *d_true, *d_grad;
    CUDA_CHECK(cudaMalloc(&d_soft, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_true, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad, total_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_soft, h_soft, total_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_true, h_true, total_size * sizeof(float), cudaMemcpyHostToDevice));

    // Configurar ejecución
    dim3 threads(32, 32);
    dim3 blocks((num_classes + threads.x - 1) / threads.x,
                (batch_size + threads.y - 1) / threads.y);

    // Lanzar kernel
    crossentropy_backward_kernel<<<blocks, threads>>>(
        d_soft, d_true, d_grad,
        batch_size, num_classes);

    CUDA_CHECK(cudaGetLastError());

    // Copiar resultado a host
    Tensor grad_cpu(shape);
    CUDA_CHECK(cudaMemcpy(grad_cpu.getData(), d_grad,
                          total_size * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_soft));
    CUDA_CHECK(cudaFree(d_true));
    CUDA_CHECK(cudaFree(d_grad));

    return grad_cpu;
}
