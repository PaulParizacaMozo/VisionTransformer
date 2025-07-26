#include "core/Tensor.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cfloat>
#include <ctime>
#include <atomic>

static std::atomic<unsigned long long> global_seed_counter(0);

unsigned long long generateUniqueSeed()
{
    unsigned long long time_part = static_cast<unsigned long long>(time(NULL));
    unsigned long long clock_part = static_cast<unsigned long long>(clock());
    unsigned long long counter = global_seed_counter.fetch_add(1);
    return time_part ^ (clock_part << 16) ^ counter;
}

__global__ void computeStridesKernel(const size_t *shape, size_t *strides, size_t ndim)
{
    int i = threadIdx.x;
    if (i >= ndim)
        return;
    __shared__ size_t sharedShape[16]; // MAX 16 dims
    if (i < ndim)
        sharedShape[i] = shape[i];
    __syncthreads();
    if (i < ndim)
    {
        size_t stride = 1;
        for (int j = ndim - 1; j > i; --j)
            stride *= sharedShape[j];
        strides[i] = stride;
    }
}

void Tensor::allocateMetadata(const size_t *shape_host, size_t ndim_)
{
    this->ndim = ndim_;
    cudaMalloc(&shape, ndim * sizeof(size_t));
    cudaMemcpy(shape, shape_host, ndim * sizeof(size_t), cudaMemcpyHostToDevice);
    size_t *strides_host = new size_t[ndim];
    totalSize = 1;
    for (int i = ndim - 1; i >= 0; --i)
    {
        strides_host[i] = totalSize;
        totalSize *= shape_host[i];
    }
    cudaMalloc(&strides, ndim * sizeof(size_t));
    cudaMemcpy(strides, strides_host, ndim * sizeof(size_t), cudaMemcpyHostToDevice);
    delete[] strides_host;
}

void Tensor::computeStrides()
{
    if (!shape || ndim == 0)
        return;
    cudaMalloc(&strides, ndim * sizeof(size_t));
    computeStridesKernel<<<1, ndim>>>(shape, strides, ndim);
}

// --- Implementación de Constructores ---
Tensor::Tensor() : data(nullptr), shape(nullptr), strides(nullptr), ndim(0), totalSize(0), dataOffset(0) {}
Tensor::Tensor(std::initializer_list<size_t> dims)
    : Tensor(std::vector<size_t>(dims).data(), dims.size()) {}

// Constructor principal (shape_host y strides_host_copy)
Tensor::Tensor(const size_t *shape_host, size_t ndim_)
    : ndim(ndim_), dataOffset(0)
{
    // --- Copia shape en CPU ---
    shape_host_copy = new size_t[ndim];
    std::memcpy(shape_host_copy, shape_host, ndim * sizeof(size_t));

    // --- Copia shape en GPU ---
    cudaMalloc(&shape, ndim * sizeof(size_t));
    cudaMemcpy(shape, shape_host, ndim * sizeof(size_t), cudaMemcpyHostToDevice);

    // --- Calcular y guardar strides ---
    size_t *strides_host = new size_t[ndim];
    totalSize = 1;
    for (int i = ndim - 1; i >= 0; --i)
    {
        strides_host[i] = totalSize;
        totalSize *= shape_host[i];
    }

    // --- Copia strides en GPU ---
    cudaMalloc(&strides, ndim * sizeof(size_t));
    cudaMemcpy(strides, strides_host, ndim * sizeof(size_t), cudaMemcpyHostToDevice);

    // --- Copia strides en CPU ---
    strides_host_copy = new size_t[ndim];
    std::memcpy(strides_host_copy, strides_host, ndim * sizeof(size_t));
    delete[] strides_host;

    // --- Reservar memoria para los datos ---
    cudaMalloc(&data, totalSize * sizeof(float));
    cudaMemset(data, 0, totalSize * sizeof(float));
}

// Constructor con punteros externos
Tensor::Tensor(float *dataPtr, const size_t *shape_host, const size_t *strides_host, size_t ndim_, size_t offset)
    : data(dataPtr), ndim(ndim_), dataOffset(offset)
{
    shape_host_copy = new size_t[ndim];
    std::memcpy(shape_host_copy, shape_host, ndim * sizeof(size_t));

    strides_host_copy = new size_t[ndim];
    std::memcpy(strides_host_copy, strides_host, ndim * sizeof(size_t));

    cudaMalloc(&shape, ndim * sizeof(size_t));
    cudaMemcpy(shape, shape_host, ndim * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMalloc(&strides, ndim * sizeof(size_t));
    cudaMemcpy(strides, strides_host, ndim * sizeof(size_t), cudaMemcpyHostToDevice);

    totalSize = 1;
    for (size_t i = 0; i < ndim; ++i)
        totalSize *= shape_host[i];
}

// Constructor de copia
Tensor::Tensor(const Tensor &other)
    : ndim(other.ndim), totalSize(other.totalSize), dataOffset(other.dataOffset)
{
    // Copia en CPU
    shape_host_copy = new size_t[ndim];
    strides_host_copy = new size_t[ndim];
    std::memcpy(shape_host_copy, other.shape_host_copy, ndim * sizeof(size_t));
    std::memcpy(strides_host_copy, other.strides_host_copy, ndim * sizeof(size_t));

    // Copia en GPU
    cudaMalloc(&shape, ndim * sizeof(size_t));
    cudaMemcpy(shape, other.shape, ndim * sizeof(size_t), cudaMemcpyDeviceToDevice);

    cudaMalloc(&strides, ndim * sizeof(size_t));
    cudaMemcpy(strides, other.strides, ndim * sizeof(size_t), cudaMemcpyDeviceToDevice);

    cudaMalloc(&data, totalSize * sizeof(float));
    cudaMemcpy(data, other.data, totalSize * sizeof(float), cudaMemcpyDeviceToDevice);
}

// Constructor de movimiento
Tensor::Tensor(Tensor &&other) noexcept
    : data(other.data), shape(other.shape), strides(other.strides),
      ndim(other.ndim), totalSize(other.totalSize), dataOffset(other.dataOffset),
      shape_host_copy(other.shape_host_copy), strides_host_copy(other.strides_host_copy)
{
    other.data = nullptr;
    other.shape = nullptr;
    other.strides = nullptr;
    other.ndim = 0;
    other.totalSize = 0;
    other.dataOffset = 0;
    other.shape_host_copy = nullptr;
    other.strides_host_copy = nullptr;
}

// Asignación por copia
Tensor &Tensor::operator=(const Tensor &other)
{
    // printf("===> Copia de tensor\n");
    if (this != &other)
    {
        if (data)
            cudaFree(data);
        if (shape)
            cudaFree(shape);
        if (strides)
            cudaFree(strides);
        if (shape_host_copy)
            delete[] shape_host_copy;
        if (strides_host_copy)
            delete[] strides_host_copy;

        ndim = other.ndim;
        totalSize = other.totalSize;
        dataOffset = other.dataOffset;

        shape_host_copy = new size_t[ndim];
        strides_host_copy = new size_t[ndim];
        std::memcpy(shape_host_copy, other.shape_host_copy, ndim * sizeof(size_t));
        std::memcpy(strides_host_copy, other.strides_host_copy, ndim * sizeof(size_t));

        cudaMalloc(&shape, ndim * sizeof(size_t));
        cudaMemcpy(shape, other.shape, ndim * sizeof(size_t), cudaMemcpyDeviceToDevice);

        cudaMalloc(&strides, ndim * sizeof(size_t));
        cudaMemcpy(strides, other.strides, ndim * sizeof(size_t), cudaMemcpyDeviceToDevice);

        cudaMalloc(&data, totalSize * sizeof(float));
        cudaMemcpy(data, other.data, totalSize * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    return *this;
}

void Tensor::deepCopyFrom(const Tensor &other)
{
    // printf("===> Copia de tensor\n");
    if (this == &other)
        return;

    // Liberar memoria previa
    if (data)
        cudaFree(data);
    if (shape)
        cudaFree(shape);
    if (strides)
        cudaFree(strides);
    if (shape_host_copy)
        delete[] shape_host_copy;
    if (strides_host_copy)
        delete[] strides_host_copy;

    // Copiar metadatos
    ndim = other.ndim;
    totalSize = other.totalSize;
    dataOffset = other.dataOffset;

    // Copiar forma y strides en host
    shape_host_copy = new size_t[ndim];
    strides_host_copy = new size_t[ndim];
    std::memcpy(shape_host_copy, other.shape_host_copy, ndim * sizeof(size_t));
    std::memcpy(strides_host_copy, other.strides_host_copy, ndim * sizeof(size_t));

    // Copiar forma y strides en GPU
    cudaMalloc(&shape, ndim * sizeof(size_t));
    cudaMemcpy(shape, other.shape, ndim * sizeof(size_t), cudaMemcpyDeviceToDevice);

    cudaMalloc(&strides, ndim * sizeof(size_t));
    cudaMemcpy(strides, other.strides, ndim * sizeof(size_t), cudaMemcpyDeviceToDevice);

    // Copiar datos en GPU
    cudaMalloc(&data, totalSize * sizeof(float));
    cudaMemcpy(data, other.data, totalSize * sizeof(float), cudaMemcpyDeviceToDevice);
}

// Asignación por movimiento
Tensor &Tensor::operator=(Tensor &&other) noexcept
{
    // printf("===> Movimiento de tensor\n");
    if (this != &other)
    {
        if (data)
            cudaFree(data);
        if (shape)
            cudaFree(shape);
        if (strides)
            cudaFree(strides);
        if (shape_host_copy)
            delete[] shape_host_copy;
        if (strides_host_copy)
            delete[] strides_host_copy;

        data = other.data;
        shape = other.shape;
        strides = other.strides;
        ndim = other.ndim;
        totalSize = other.totalSize;
        dataOffset = other.dataOffset;
        shape_host_copy = other.shape_host_copy;
        strides_host_copy = other.strides_host_copy;

        other.data = nullptr;
        other.shape = nullptr;
        other.strides = nullptr;
        other.ndim = 0;
        other.totalSize = 0;
        other.dataOffset = 0;
        other.shape_host_copy = nullptr;
        other.strides_host_copy = nullptr;
    }
    return *this;
}

Tensor::~Tensor()
{
    if (data)
        cudaFree(data);
    if (shape)
        cudaFree(shape);
    if (strides)
        cudaFree(strides);
    if (shape_host_copy)
        delete[] shape_host_copy;
    if (strides_host_copy)
        delete[] strides_host_copy;
}

// --- Getters ---
bool Tensor::isContiguous() const
{
    if (ndim == 0)
        return true;

    size_t expectedStride = 1;
    for (int i = ndim - 1; i >= 0; --i)
    {
        if (strides_host_copy[i] != expectedStride)
            return false;
        expectedStride *= shape_host_copy[i];
    }
    return true;
}

// --- Inicialización ---
__global__ void fillKernel(float *data, size_t totalSize, float value)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalSize)
        data[idx] = value;
}
void Tensor::fill(float value)
{
    if (!isContiguous())
        throw std::runtime_error("fill() solo se puede usar en tensores contiguos.");

    int threads = 256;
    int blocks = (totalSize + threads - 1) / threads;
    fillKernel<<<blocks, threads>>>(data + dataOffset, totalSize, value);
    cudaDeviceSynchronize();
}

__global__ void randomUniformKernel(float *data, size_t totalSize, float min, float max, unsigned long long seed)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize)
        return;

    curandState state;
    curand_init(seed, idx, 0, &state);
    float r = curand_uniform(&state);
    data[idx] = min + r * (max - min);
}
void Tensor::randomize(float min, float max)
{
    if (!isContiguous())
        throw std::runtime_error("randomize() solo se puede usar en tensores contiguos.");
    int threads = 256;
    int blocks = (totalSize + threads - 1) / threads;
    unsigned long long seed = generateUniqueSeed();
    randomUniformKernel<<<blocks, threads>>>(data + dataOffset, totalSize, min, max, seed);
    cudaDeviceSynchronize();
}

__global__ void randomNormalKernel(float *data, size_t totalSize, float mean, float stddev, unsigned long long seed)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize)
        return;

    curandState state;
    curand_init(seed, idx, 0, &state);
    float r = curand_normal(&state);
    data[idx] = mean + stddev * r;
}
void Tensor::randomizeNormal(float mean, float stddev)
{
    if (!isContiguous())
        throw std::runtime_error("randomizeNormal() solo se puede usar en tensores contiguos.");

    int threads = 256;
    int blocks = (totalSize + threads - 1) / threads;
    unsigned long long seed = generateUniqueSeed();
    randomNormalKernel<<<blocks, threads>>>(data + dataOffset, totalSize, mean, stddev, seed);
    cudaDeviceSynchronize();
}

// --- Transformaciones ---
Tensor Tensor::slice(size_t axis, size_t start, size_t count) const
{
    if (axis >= ndim)
        throw std::out_of_range("slice: eje fuera de rango");

    if (start + count > shape_host_copy[axis])
        throw std::out_of_range("slice: fuera de los límites");

    // Crear copias modificadas en host (no en GPU)
    size_t *new_shape = new size_t[ndim];
    size_t *new_strides = new size_t[ndim];
    for (size_t i = 0; i < ndim; ++i)
    {
        new_shape[i] = shape_host_copy[i];
        new_strides[i] = strides_host_copy[i];
    }

    new_shape[axis] = count;
    size_t newOffset = dataOffset + start * new_strides[axis];

    // Crear nuevo tensor usando la metadata modificada
    Tensor out(data, new_shape, new_strides, ndim, newOffset);

    delete[] new_shape;
    delete[] new_strides;
    return out;
}

Tensor Tensor::reshape(const size_t *newShape_host, size_t newNdim) const
{
    if (!isContiguous())
        throw std::runtime_error("reshape() requiere un tensor contiguo.");

    // Calcular nuevo totalSize en host
    size_t newTotalSize = 1;
    for (size_t i = 0; i < newNdim; ++i)
        newTotalSize *= newShape_host[i];

    if (newTotalSize != totalSize)
        throw std::runtime_error("reshape() requiere mismo número de elementos.");

    // Crear nuevos strides (en host)
    size_t *newStrides_host = new size_t[newNdim];
    size_t stride = 1;
    for (int i = static_cast<int>(newNdim) - 1; i >= 0; --i)
    {
        newStrides_host[i] = stride;
        stride *= newShape_host[i];
    }

    // Crear nuevo tensor como view
    Tensor out(data, newShape_host, newStrides_host, newNdim, dataOffset);
    delete[] newStrides_host;
    Tensor copy(out); // For debugging purposes
    return copy;
}

Tensor Tensor::transpose(size_t dim1, size_t dim2) const
{
    if (dim1 >= ndim || dim2 >= ndim)
        throw std::out_of_range("transpose: dimensiones fuera de rango");

    // Clonar metadatos en CPU
    size_t *new_shape_host = new size_t[ndim];
    size_t *new_strides_host = new size_t[ndim];
    std::copy(shape_host_copy, shape_host_copy + ndim, new_shape_host);
    std::copy(strides_host_copy, strides_host_copy + ndim, new_strides_host);

    // Intercambiar dimensiones
    std::swap(new_shape_host[dim1], new_shape_host[dim2]);
    std::swap(new_strides_host[dim1], new_strides_host[dim2]);

    // Crear nuevo tensor con misma data y offset, pero nuevo shape/strides
    Tensor out(data, new_shape_host, new_strides_host, ndim, dataOffset);

    delete[] new_shape_host;
    delete[] new_strides_host;
    return out;
}

__global__ void copyContiguousKernel(
    float *dst, const float *src, size_t size, size_t offset)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        dst[idx] = src[offset + idx];
}

Tensor Tensor::contiguous() const
{
    if (isContiguous() && dataOffset == 0)
        return *this;

    // Creamos un nuevo tensor usando el shape ya disponible en CPU
    Tensor out(shape_host_copy, ndim);

    // Copiamos los datos desde el offset
    int threads = 256;
    int blocks = (totalSize + threads - 1) / threads;
    cudaGetLastError();
    copyContiguousKernel<<<blocks, threads>>>(out.data, this->data, totalSize, dataOffset);
    cudaDeviceSynchronize();
    // Verificación de errores CUDA
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel error in contiguous: ") + cudaGetErrorString(err));
    return out;
}

// --- Operaciones tensoriales ---
__global__ void addKernel(const float *A, const float *B, float *C, size_t offsetA, size_t offsetB, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        C[idx] = A[offsetA + idx] + B[offsetB + idx];
}

Tensor Tensor::operator+(const Tensor &other) const
{
    if (ndim != other.ndim)
        throw std::runtime_error("Dimensiones incompatibles para suma");

    for (size_t i = 0; i < ndim; ++i)
        if (shape_host_copy[i] != other.shape_host_copy[i])
            throw std::runtime_error("Shapes incompatibles para suma");

    Tensor result(shape_host_copy, ndim);
    int threads = 256;
    int blocks = (totalSize + threads - 1) / threads;
    cudaGetLastError();
    addKernel<<<blocks, threads>>>(
        this->data,
        other.data,
        result.data,
        this->dataOffset,
        other.dataOffset,
        totalSize);

    // Chequear errores CUDA
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel error: ") + cudaGetErrorString(err));

    return result;
}

__global__ void squareKernel(const float *input, float *output, size_t offset, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        output[idx] = input[offset + idx] * input[offset + idx];
}

Tensor Tensor::square() const
{
    Tensor result(shape_host_copy, ndim);

    int threads = 256;
    int blocks = (totalSize + threads - 1) / threads;

    squareKernel<<<blocks, threads>>>(data, result.data, dataOffset, totalSize);
    cudaDeviceSynchronize();
    return result;
}

#define MAX_NDIM 10
__global__ void sumGenericKernel(
    const float *__restrict__ input, float *output,
    const size_t *shape, const size_t *strides,
    const size_t *out_shape, const size_t *out_strides,
    size_t ndim, size_t axis, size_t outputSize, size_t offset)
{
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= outputSize)
        return;

    // Calcular índices multidimensionales del tensor de salida
    size_t idx[MAX_NDIM];
    size_t temp = out_idx;
    for (int i = 0; i < ndim; ++i)
    {
        if (i == axis)
        {
            idx[i] = 0; // colapsado
            continue;
        }
        idx[i] = temp / out_strides[i];
        temp %= out_strides[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < shape[axis]; ++i)
    {
        idx[axis] = i;
        size_t flat_index = 0;
        for (int d = 0; d < ndim; ++d)
            flat_index += idx[d] * strides[d];
        sum += input[offset + flat_index];
    }
    output[out_idx] = sum;
}

Tensor Tensor::sum(size_t axis) const
{
    if (axis >= ndim)
        throw std::runtime_error("Eje fuera de rango en sum().");

    size_t *new_shape = new size_t[ndim];
    size_t *new_strides = new size_t[ndim];

    // Copiar y modificar shape
    for (size_t i = 0; i < ndim; ++i)
    {
        new_shape[i] = shape_host_copy[i];
        new_strides[i] = (i < ndim - 1) ? 1 : 0; // dummy
    }
    new_shape[axis] = 1;

    Tensor result(new_shape, ndim);
    delete[] new_shape;
    delete[] new_strides;

    // Calcular tamaño total del tensor de salida
    size_t outSize = 1;
    for (size_t i = 0; i < ndim; ++i)
        outSize *= result.dim(i);

    int threads = 256;
    int blocks = (outSize + threads - 1) / threads;

    sumGenericKernel<<<blocks, threads>>>(
        data, result.data,
        shape, strides,
        result.getShape(), result.getStrides(),
        ndim, axis, outSize, dataOffset);
    cudaDeviceSynchronize();
    return result;
}

__global__ void broadcast2D(float *A, const float *B, size_t rows, size_t cols)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
        A[idx] += B[idx % cols];
}

__global__ void broadcast3D(float *A, const float *B, size_t B_, size_t N, size_t D)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B_ * N * D)
    {
        size_t b = idx / (N * D);
        size_t nd = idx % (N * D);
        A[idx] += B[nd];
    }
}

void Tensor::addBroadcast(const Tensor &other)
{
    if (ndim != other.ndim)
        throw std::runtime_error("Dimensiones distintas para broadcast");

    int threads = 256;

    if (ndim == 2 && other.dim(0) == 1 && dim(1) == other.dim(1))
    {
        size_t rows = dim(0);
        size_t cols = dim(1);
        int blocks = (rows * cols + threads - 1) / threads;

        broadcast2D<<<blocks, threads>>>(
            this->data + dataOffset,
            other.data + other.dataOffset,
            rows, cols);
    }
    else if (ndim == 3 && other.dim(0) == 1 &&
             dim(1) == other.dim(1) && dim(2) == other.dim(2))
    {
        size_t B_ = dim(0);
        size_t N = dim(1);
        size_t D = dim(2);
        int blocks = (B_ * N * D + threads - 1) / threads;

        broadcast3D<<<blocks, threads>>>(
            this->data + dataOffset,
            other.data + other.dataOffset,
            B_, N, D);
    }
    else
        throw std::runtime_error("Broadcasting no soportado en estas formas.");
    cudaDeviceSynchronize();
}

// --- Debug info opcional ---
void Tensor::printDebugInfo(const std::string &name) const
{
    std::cout << "--- Tensor Debug: " << name << " ---\n";
    if (!shape_host_copy || !strides_host_copy)
    {
        std::cerr << "  [ERROR] shape_host_copy o strides_host_copy no inicializados.\n";
        return;
    }

    std::cout << "  Forma: (";
    for (size_t i = 0; i < ndim; ++i)
        std::cout << shape_host_copy[i] << (i + 1 < ndim ? ", " : "");
    std::cout << ")\n";

    std::cout << "  Contiguo: " << (isContiguous() ? "Sí" : "NO") << "\n";
    std::cout << "  Offset: " << dataOffset << "\n";
    std::cout << "  totalSize: " << totalSize << "\n";

    std::cout << "  Strides: (";
    for (size_t i = 0; i < ndim; ++i)
        std::cout << strides_host_copy[i] << (i + 1 < ndim ? ", " : "");
    std::cout << ")\n";
    std::cout << "-------------------------\n";
}

__device__ inline float &Tensor::operator()(size_t i)
{
    return data[dataOffset + i * strides[0]];
}

__device__ inline float &Tensor::operator()(size_t i, size_t j)
{
    return data[dataOffset + i * strides[0] + j * strides[1]];
}

__device__ inline float &Tensor::operator()(size_t i, size_t j, size_t k)
{
    return data[dataOffset + i * strides[0] + j * strides[1] + k * strides[2]];
}

__device__ inline float &Tensor::operator()(size_t i, size_t j, size_t k, size_t l)
{
    return data[dataOffset + i * strides[0] + j * strides[1] + k * strides[2] + l * strides[3]];
}

// --- Funciones auxiliares GPU ---
Tensor matrixMultiply(const Tensor &a, const Tensor &b)
{
    // Validar dimensiones
    if (a.dims() != 2 || b.dims() != 2)
        throw std::runtime_error("matrixMultiply: solo se permite 2D");

    size_t m = a.dim(0);   // filas de A
    size_t k = a.dim(1);   // columnas de A = filas de B
    size_t k_b = b.dim(0); // filas de B
    size_t n = b.dim(1);   // columnas de B
    if (k != k_b)
        throw std::runtime_error("matrixMultiply: dimensiones incompatibles");

    // Crear tensor de salida
    size_t shapeOut[2] = {m, n};
    Tensor result(shapeOut, 2);

    // cuBLAS GEMM: C = alpha * A * B + beta * C
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N, // No transpuestas
        n, m, k,                  // C[n x m] = A[k x m] * B[n x k]
        &alpha,
        b.getData(), n, // B: [k x n], lda = n
        a.getData(), k, // A: [m x k], lda = k
        &beta,
        result.getData(), n); // C: [m x n], ldc = n

    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("matrixMultiply: error en cublasSgemm");

    cublasDestroy(handle);
    return result;
}

Tensor batchMatrixMultiply(const Tensor &a, const Tensor &b)
{
    if (a.dims() != 3 || b.dims() != 3)
        throw std::runtime_error("batchMatrixMultiply: solo soporta tensores 3D");

    const size_t B = a.dim(0);
    const size_t M = a.dim(1);
    const size_t K = a.dim(2);
    const size_t Kb = b.dim(1);
    const size_t N = b.dim(2);

    if (B != b.dim(0) || K != Kb)
        throw std::runtime_error("batchMatrixMultiply: dimensiones incompatibles entre a y b");

    Tensor result({B, M, N});

    const float *A = a.getData() + a.getDataOffset();
    const float *Bptr = b.getData() + b.getDataOffset();
    float *C = result.getData();

    const size_t strideA = M * K;
    const size_t strideB = K * N;
    const size_t strideC = M * N;

    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasStatus_t status = cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N, // A: [M,K], B: [K,N]
        N, M, K,                  // Cada C_i: [M,N]
        &alpha,
        Bptr, ldb, strideB,
        A, lda, strideA,
        &beta,
        C, ldc, strideC,
        B);

    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("batchMatrixMultiply: error en cublasSgemmStridedBatched");

    cublasDestroy(handle);
    return result;
}

// Kernel para concatenar tensores 3D en GPU// Kernel para concatenar tensores 3D en GPU
__global__ void concat3DKernel(const float *src, float *dst,
                               size_t offset, size_t dim0, size_t dim1, size_t dim2,
                               size_t axis, size_t axisLen)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * axisLen * dim2;

    if (idx >= total)
        return;

    size_t b = idx / (axisLen * dim2);
    size_t ai = (idx / dim2) % axisLen;
    size_t d = idx % dim2;

    size_t dst_idx = 0, src_idx = 0;

    if (axis == 0)
    {
        dst_idx = (offset + b) * dim1 * dim2 + ai * dim2 + d;
        src_idx = b * axisLen * dim2 + ai * dim2 + d;
    }
    else if (axis == 1)
    {
        dst_idx = b * dim1 * dim2 + (offset + ai) * dim2 + d;
        src_idx = b * axisLen * dim2 + ai * dim2 + d;
    }
    else if (axis == 2)
    {
        dst_idx = b * dim1 * dim2 + ai * dim2 + (offset + d);
        src_idx = b * dim1 * axisLen + ai * axisLen + d;
    }

    dst[dst_idx] = src[src_idx];
}

Tensor concatenate(const Tensor *tensors, size_t numTensors, size_t axis)
{
    if (numTensors == 0)
        return Tensor();
    if (numTensors == 1)
        return tensors[0];

    const size_t ndim = tensors[0].dims();
    if (ndim != 3)
        throw std::runtime_error("concatenate: solo soporta tensores 3D");

    // Validación y acumulación del tamaño final en el eje de concatenación
    size_t totalAxis = 0;
    for (size_t i = 0; i < numTensors; ++i)
    {
        for (size_t d = 0; d < ndim; ++d)
        {
            if (d != axis && tensors[i].dim(d) != tensors[0].dim(d))
                throw std::runtime_error("concatenate: incompatibilidad en dimensiones");
        }
        totalAxis += tensors[i].dim(axis);
    }

    // Dimensiones del resultado
    size_t shape[3] = {
        tensors[0].dim(0),
        tensors[0].dim(1),
        tensors[0].dim(2)};
    shape[axis] = totalAxis;

    Tensor result(shape, 3);
    float *dst = result.getData();

    // Para el kernel, necesitamos pasar las dimensiones del resultado
    size_t kernelDims[3] = {
        shape[0], shape[1], shape[2]};

    size_t axisOffset = 0;
    for (size_t t = 0; t < numTensors; ++t)
    {
        const Tensor &src = tensors[t];
        size_t axisLen = src.dim(axis);
        size_t totalElems = src.dim(0) * axisLen * src.dim(2);
        constexpr int threadsPerBlock = 256;
        int blocks = (totalElems + threadsPerBlock - 1) / threadsPerBlock;
        cudaGetLastError();
        concat3DKernel<<<blocks, threadsPerBlock>>>(
            src.getData(), dst,
            axisOffset,
            kernelDims[0], kernelDims[1], kernelDims[2],
            axis, axisLen);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("Error tras lanzar kernel");
        }

        cudaDeviceSynchronize();
        axisOffset += axisLen;
    }

    return result;
}

__global__ void expandKernel(const float *in, float *out,
                             size_t in_dim1, size_t in_dim2,
                             size_t out_dim0, size_t out_dim1, size_t out_dim2)
{
    int i = blockIdx.x;
    int j = threadIdx.y;
    int k = threadIdx.x;

    if (i < out_dim0 && j < out_dim1 && k < out_dim2)
    {
        size_t in_idx = j * in_dim2 + k;
        size_t out_idx = i * out_dim1 * out_dim2 + j * out_dim2 + k;
        out[out_idx] = in[in_idx]; // broadcasting en la dimensión expandida
    }
}

Tensor expand(const Tensor &tensor, size_t dim, size_t size)
{
    if (tensor.dims() != 2)
        throw std::runtime_error("expand: solo soporta tensor 2D para broadcasting a 3D");

    if (dim > tensor.dims())
        throw std::invalid_argument("expand: dimensión mayor al rank");

    size_t in_dim1 = tensor.dim(0);
    size_t in_dim2 = tensor.dim(1);

    // Definir nueva forma (broadcast hacia 3D)
    size_t out_dim0 = (dim == 0) ? size : in_dim1;
    size_t out_dim1 = (dim == 0) ? in_dim1 : size;
    size_t out_dim2 = in_dim2;

    // Crear nuevo tensor expandido
    size_t shape[3] = {out_dim0, out_dim1, out_dim2};
    Tensor expanded(shape, 3); // Constructor que aloque en GPU

    // Lanzar kernel
    dim3 grid(out_dim0);
    dim3 block(out_dim2, out_dim1); // (k, j)

    expandKernel<<<grid, block>>>(
        tensor.getData(), expanded.getData(),
        in_dim1, in_dim2,
        out_dim0, out_dim1, out_dim2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("expand: kernel error: ") + cudaGetErrorString(err));

    cudaDeviceSynchronize();
    return expanded;
}

__global__ void softmax3DAxis2Kernel(const float *logits, float *output,
                                     size_t B, size_t N, size_t D)
{
    // Cada hilo procesa un vector de tamaño D (softmax sobre ese vector)
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N)
        return;

    size_t b = idx / N;
    size_t n = idx % N;

    // Índice base para el vector logits[b, n, :]
    size_t base = (b * N + n) * D;

    // Paso 1: encontrar el máximo
    float max_val = -FLT_MAX;
    for (size_t d = 0; d < D; ++d)
    {
        float val = logits[base + d];
        if (val > max_val)
            max_val = val;
    }

    // Paso 2: calcular suma de exp
    float sum_exp = 0.0f;
    for (size_t d = 0; d < D; ++d)
    {
        float exp_val = expf(logits[base + d] - max_val);
        output[base + d] = exp_val; // temporal
        sum_exp += exp_val;
    }

    // Paso 3: normalizar
    for (size_t d = 0; d < D; ++d)
    {
        output[base + d] /= sum_exp;
    }
}

Tensor softmax(const Tensor &logits, int axis)
{
    if (logits.getNDim() != 3)
        throw std::runtime_error("Softmax: solo se soportan tensores 3D");

    if (axis < 0)
        axis += logits.getNDim();

    if (axis != 2)
        throw std::runtime_error("Softmax: solo se soporta axis == 2 (última dimensión)");

    size_t B = logits.dim(0);
    size_t N = logits.dim(1);
    size_t D = logits.dim(2);

    Tensor result(logits.getShapeHost(), logits.getNDim());

    const float *logits_ptr = logits.getData();
    float *output_ptr = result.getData();

    size_t totalRows = B * N;
    constexpr int threadsPerBlock = 128;
    int blocks = (totalRows + threadsPerBlock - 1) / threadsPerBlock;
    cudaGetLastError();
    softmax3DAxis2Kernel<<<blocks, threadsPerBlock>>>(logits_ptr, output_ptr, B, N, D);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "[CUDA ERROR - softmax] " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("softmax: Error tras lanzar kernel");
    }

    return result;
}

__global__ void softmaxBackward3DAxis2Kernel(const float *grad_output, const float *softmax_output,
                                             float *grad_input,
                                             size_t B, size_t N, size_t D)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N)
        return;

    size_t b = idx / N;
    size_t n = idx % N;

    size_t base = (b * N + n) * D;

    // Paso 1: calcular el producto punto grad_output * softmax_output
    float dot = 0.0f;
    for (size_t d = 0; d < D; ++d)
    {
        dot += grad_output[base + d] * softmax_output[base + d];
    }

    // Paso 2: calcular el gradiente dL/dZ = s_i * (dL/dS_i - dot)
    for (size_t d = 0; d < D; ++d)
    {
        float s = softmax_output[base + d];
        grad_input[base + d] = s * (grad_output[base + d] - dot);
    }
}

Tensor softmax_backward(const Tensor &grad_output, const Tensor &softmax_output)
{
    if (grad_output.getNDim() != 3 || softmax_output.getNDim() != 3)
        throw std::runtime_error("softmax_backward: solo se soportan tensores 3D");

    size_t B = grad_output.dim(0);
    size_t N = grad_output.dim(1);
    size_t D = grad_output.dim(2);

    Tensor grad_input(grad_output.getShapeHost(), grad_output.getNDim());

    const float *grad_output_ptr = grad_output.getData();
    const float *softmax_output_ptr = softmax_output.getData();
    float *grad_input_ptr = grad_input.getData();

    size_t totalRows = B * N;
    constexpr int threadsPerBlock = 128;
    int blocks = (totalRows + threadsPerBlock - 1) / threadsPerBlock;
    cudaGetLastError();
    softmaxBackward3DAxis2Kernel<<<blocks, threadsPerBlock>>>(
        grad_output_ptr,
        softmax_output_ptr,
        grad_input_ptr,
        B, N, D);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "[CUDA ERROR - softmax_backward] " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("softmax_backward: Error tras lanzar kernel");
    }

    return grad_input;
}
