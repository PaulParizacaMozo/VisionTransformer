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