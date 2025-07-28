#include "utils/CudaUtils.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// Macro de utilidad para verificar errores de CUDA y cuBLAS
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error en %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error en %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


Tensor matrixMultiply_cuda(const Tensor& a, const Tensor& b) {
    // 1. Validaciones
    if (!a.isContiguous() || !b.isContiguous()) {
        throw std::runtime_error("matrixMultiply_cuda requiere tensores de entrada contiguos.");
    }
    const auto& aShape = a.getShape();
    const auto& bShape = b.getShape();
    if (aShape.size() != 2 || bShape.size() != 2 || aShape[1] != bShape[0]) {
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
                             p, m, n,          // p, m, n
                             &alpha,
                             d_b, p,           // Matriz B, leading dim p
                             d_a, n,           // Matriz A, leading dim n
                             &beta,
                             d_c, p));          // Matriz C, leading dim p

    CUBLAS_CHECK(cublasDestroy(handle));

    // 6. Copiar el resultado de GPU (Device) de vuelta a CPU (Host)
    CUDA_CHECK(cudaMemcpy(result_cpu.getData(), d_c, result_cpu.getSize() * sizeof(float), cudaMemcpyDeviceToHost));

    // 7. Liberar memoria de la GPU
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    
    return result_cpu;
}

Tensor batchMatrixMultiply_cuda(const Tensor& a, const Tensor& b) {
    // 1. Validaciones
    const auto& aShape = a.getShape();
    const auto& bShape = b.getShape();
    if (aShape.size() != 3 || bShape.size() != 3 || aShape[0] != bShape[0] || aShape[2] != bShape[1]) {
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
    std::vector<const float*> a_array(batchSize, nullptr);
    std::vector<const float*> b_array(batchSize, nullptr);
    std::vector<float*> c_array(batchSize, nullptr);
    for(int i=0; i<batchSize; ++i) {
        a_array[i] = d_a + i * m * n;
        b_array[i] = d_b + i * n * p;
        c_array[i] = d_c + i * m * p;
    }
    
    const float **d_a_array, **d_b_array;
    float **d_c_array;
    CUDA_CHECK(cudaMalloc(&d_a_array, batchSize * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_b_array, batchSize * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_c_array, batchSize * sizeof(float*)));
    CUDA_CHECK(cudaMemcpy(d_a_array, a_array.data(), batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_array, b_array.data(), batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_array, c_array.data(), batchSize * sizeof(float*), cudaMemcpyHostToDevice));

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
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_a_array)); CUDA_CHECK(cudaFree(d_b_array)); CUDA_CHECK(cudaFree(d_c_array));

    return result_cpu;
}
