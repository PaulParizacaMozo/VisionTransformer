#include "core/Tensor.cuh"
#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <cstring>
#include <memory>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <curand_kernel.h>

// Calcula los pasos (strides) para navegar entre dimensiones
void Tensor::computeStrides()
{
    strides.resize(shape.size());
    std::size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
        strides[i] = stride;
        stride *= shape[i];
    }
}

Tensor::Tensor() : dataOffset(0), totalSize(0) {}

Tensor::Tensor(const std::vector<std::size_t> &newShape) : shape(newShape), dataOffset(0)
{
    totalSize = newShape.empty() ? 0 : std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies<std::size_t>());
    cudaMalloc(&data, totalSize * sizeof(float));
    cudaMemset(data, 0, totalSize * sizeof(float));
    computeStrides();
}

Tensor::Tensor(const std::vector<std::size_t> &newShape,
               const std::vector<float> &initialData) : shape(newShape), dataOffset(0)
{
    totalSize = std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies<std::size_t>());
    if (totalSize != initialData.size())
    {
        throw std::invalid_argument("El tamaño de los datos iniciales no coincide con la forma del tensor.");
    }
    cudaError_t err = cudaMalloc(&data, totalSize * sizeof(float));
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Fallo en cudaMalloc: " + std::string(cudaGetErrorString(err)));
    }
    err = cudaMemcpy(data, initialData.data(), totalSize * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Fallo en cudaMemcpy (HostToDevice): " + std::string(cudaGetErrorString(err)));
    }
    computeStrides();
}

Tensor::Tensor(const std::vector<float> &ptr,
               const std::vector<size_t> &newShape,
               const std::vector<size_t> &newStrides,
               size_t offset)
    : shape(newShape), strides(newStrides), dataOffset(offset),
      totalSize(newShape.empty() ? 0 : std::accumulate(newShape.begin(), newShape.end(), size_t(1), std::multiplies<size_t>()))
{
    if (totalSize == 0)
    {
        throw std::invalid_argument("Tensor::Tensor: totalSize no puede ser cero.");
    }

    if (offset + totalSize > ptr.size())
    {
        std::cerr << "Tensor::Tensor: offset = " << offset
                  << ", totalSize = " << totalSize
                  << ", ptr.size() = " << ptr.size() << std::endl;
        throw std::out_of_range("Offset + totalSize excede el tamaño del vector original.");
    }

    // std::cout << "Tensor::Tensor: Copiando " << totalSize << " elementos desde offset " << offset << std::endl;

    // Reservar en GPU
    // std::cout << "[DEBUG] Reservando " << (totalSize * sizeof(float)) / (1024.0 * 1024.0) << " MB" << std::endl;

    cudaError_t allocErr = cudaMalloc(&data, totalSize * sizeof(float));
    if (allocErr != cudaSuccess)
    {
        std::cerr << "cudaMalloc error: " << cudaGetErrorString(allocErr) << std::endl;
        throw std::runtime_error("Fallo en cudaMalloc para Tensor::data");
    }

    // Copiar desde CPU a GPU
    const float *src_ptr = ptr.data() + offset;
    cudaError_t copyErr = cudaMemcpy(data, src_ptr, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    if (copyErr != cudaSuccess)
    {
        std::cerr << "cudaMemcpy error: " << cudaGetErrorString(copyErr) << std::endl;
        cudaFree(data);
        throw std::runtime_error("Fallo en cudaMemcpy desde ptr al device");
    }

    computeStrides();
}

float *Tensor::getData()
{
    return data;
}

const float *Tensor::getData() const
{
    return data;
}

bool Tensor::isContiguous() const
{
    if (shape.empty())
        return true;

    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i)
    {
        if (strides[i] != stride)
        {
            return false;
        }
        stride *= shape[i];
    }
    return true;
}

std::string Tensor::shapeToString() const
{
    if (shape.empty())
        return "()";
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        ss << shape[i] << (i == shape.size() - 1 ? "" : ", ");
    }
    ss << ")";
    return ss.str();
}
__global__ void fillKernel(float *data, size_t offset, size_t size, float value)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[offset + idx] = value;
    }
}
void Tensor::fill(float value)
{
    if (!isContiguous())
    {
        throw std::runtime_error("fill() solo se puede usar en tensores contiguos.");
    }
    if (data)
    {
        const int threadsPerBlock = 256;
        const int numBlocks = (totalSize + threadsPerBlock - 1) / threadsPerBlock;

        fillKernel<<<numBlocks, threadsPerBlock>>>(data, dataOffset, totalSize, value);
        cudaDeviceSynchronize();
    }
}

__global__ void randomizeKernel(float *data, size_t size, float min, float max, unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    curandState state;
    curand_init(seed, idx, 0, &state);
    float rnd = curand_uniform(&state); // entre (0,1]
    data[idx] = min + rnd * (max - min);
}

void Tensor::randomize(float min, float max)
{
    if (!isContiguous())
    {
        throw std::runtime_error("randomize() solo se puede usar en tensores contiguos.");
    }
    if (data)
    {
        int threads = 256;
        int blocks = (totalSize + threads - 1) / threads;
        unsigned long seed = SeedGenerator::getNextSeed();

        randomizeKernel<<<blocks, threads>>>(data + dataOffset, totalSize, min, max, seed);
        cudaDeviceSynchronize();
    }
}

__global__ void randomizeNormalKernel(float *data, size_t size, float mean, float stddev, unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    // Cada hilo inicializa su generador curand
    curandState state;
    curand_init(seed, idx, 0, &state);

    // Genera un número normal
    float val = curand_normal(&state) * stddev + mean;
    data[idx] = val;
}

void Tensor::randomizeNormal(float mean, float stddev)
{
    if (!isContiguous())
    {
        throw std::runtime_error("randomizeNormal() solo se puede usar en tensores contiguos.");
    }
    if (data)
    {
        int threads = 256;
        int blocks = (totalSize + threads - 1) / threads;

        unsigned long seed = SeedGenerator::getNextSeed();

        randomizeNormalKernel<<<blocks, threads>>>(data + dataOffset, totalSize, mean, stddev, seed);
        cudaDeviceSynchronize();
    }
}

Tensor Tensor::slice(size_t axis, size_t start, size_t count) const
{
    if (axis >= shape.size())
    {
        throw std::out_of_range("Eje de slice fuera de rango.");
    }
    if (start + count > shape[axis])
    {
        throw std::out_of_range("Slice fuera de los límites de la dimensión " + std::to_string(axis));
    }

    std::vector<size_t> newShape = shape;
    newShape[axis] = count;

    // El nuevo offset se calcula a partir del stride del eje especificado.
    size_t newOffset = dataOffset + start * strides[axis];

    // data de gpu pasarla a cpu
    std::vector<float> dataPtr(totalSize);
    cudaMemcpy(dataPtr.data(), data, totalSize * sizeof(float), cudaMemcpyDeviceToHost);

    Tensor out(dataPtr, newShape, strides, newOffset);
    return out;
}

Tensor Tensor::reshape(const std::vector<size_t> &newShape, bool print) const
{
    if (!isContiguous())
    {
        throw std::runtime_error("reshape() solo se puede usar en un tensor contiguo. Use .contiguous() primero.");
    }
    size_t newTotalSize = newShape.empty() ? 0 : std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies<size_t>());
    if (this->totalSize != newTotalSize)
    {
        throw std::runtime_error("No se puede hacer reshape: el número total de elementos debe ser el mismo.");
    }

    std::vector<float> dataPtr(totalSize);
    cudaMemcpy(dataPtr.data(), data, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
    if (print)
    {
        // mostrar primeros 10 elementos de dataPtr
        // std::cout << "Tensor::reshape: dataPtr (" << dataPtr.size() << ") = ";
        for (size_t i = 0; i < std::min(dataPtr.size(), size_t(10)); ++i)
        {
            std::cout << dataPtr[i] << " ";
        }
        std::cout << std::endl;
    }
    // Tensor out(dataPtr, newShape, strides, dataOffset);
    // out.printFirstRow("Tensor::reshape out");
    Tensor tempForStrides(newShape);
    return Tensor(dataPtr, newShape, tempForStrides.getStrides(), this->dataOffset);
}

Tensor Tensor::transpose(size_t dim1, size_t dim2) const
{
    if (dim1 >= shape.size() || dim2 >= shape.size())
    {
        throw std::out_of_range("Ejes para transpose fuera de rango.");
    }
    std::vector<size_t> newShape = this->shape;
    std::swap(newShape[dim1], newShape[dim2]);
    std::vector<size_t> newStrides = this->strides;
    std::swap(newStrides[dim1], newStrides[dim2]);

    std::vector<float> dataPtr(totalSize);
    cudaMemcpy(dataPtr.data(), data, totalSize * sizeof(float), cudaMemcpyDeviceToHost);

    return Tensor(dataPtr, newShape, newStrides, this->dataOffset);
}

__global__ void makeContiguousKernel(float *dst, const float *src,
                                     const size_t *shape, const size_t *strides,
                                     size_t ndim, size_t totalSize, size_t offset)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize)
        return;

    // Calcula la posición multidimensional del idx
    size_t src_idx = 0;
    size_t remaining = idx;
    for (int d = 0; d < ndim; ++d)
    {
        size_t dim = shape[d];
        size_t coord = remaining;
        for (int j = d + 1; j < ndim; ++j)
        {
            coord /= shape[j];
        }
        coord = coord % dim;
        src_idx += coord * strides[d];
    }

    dst[idx] = src[src_idx + offset];
}

Tensor Tensor::contiguous() const
{
    if (isContiguous() && dataOffset == 0)
    {
        return *this;
    }

    Tensor new_tensor(this->shape);
    // Usa los operadores () que ya saben manejar strides para copiar
    // del tensor no contiguo al nuevo tensor contiguo.
    size_t ndim = shape.size();
    size_t *d_shape;
    size_t *d_strides;
    cudaMalloc(&d_shape, ndim * sizeof(size_t));
    cudaMemcpy(d_shape, shape.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMalloc(&d_strides, ndim * sizeof(size_t));
    cudaMemcpy(d_strides, strides.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (totalSize + threads - 1) / threads;
    makeContiguousKernel<<<blocks, threads>>>(
        new_tensor.getData(), this->data,
        d_shape, d_strides,
        ndim, totalSize, dataOffset);

    cudaFree(d_shape);
    cudaFree(d_strides);
    cudaDeviceSynchronize();

    return new_tensor;
}

Tensor Tensor::expand(const std::vector<size_t> &newShape) const
{
    if (newShape.size() != shape.size())
        throw std::invalid_argument("expand: la dimensionalidad no coincide.");

    std::vector<size_t> newStrides = strides;

    for (size_t d = 0; d < shape.size(); ++d)
    {
        if (newShape[d] == shape[d])
            continue; // sin cambio
        if (shape[d] != 1)
            throw std::invalid_argument(
                "expand: solo se puede expandir dimensiones de tamaño 1.");
        newStrides[d] = 0; // truco clásico de broadcasting: stride 0
    }
    auto newTotalSize = std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies<size_t>());
    std::vector<float> dataPtr(newTotalSize); // Inicializar todo con ceros
    // std::cout << "Tensor::expand: dataPtr (" << dataPtr.size() << ") " << std::endl;

    cudaMemcpy(dataPtr.data(), data, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
    // Intentamos copiar desde GPU si hay suficientes datos válidos desde el offset
    // std::cout << "Tensor::expand: dataPtr (" << dataPtr.size() << ") = ";
    // for (const auto &val : dataPtr)
    //     std::cout << val << " ";
    // std::cout << std::endl;
    // print dataPtr
    return Tensor(dataPtr, newShape, newStrides, dataOffset);
}

__global__ void add2D_kernel(const float *a, const float *b, float *out,
                             size_t rows, size_t cols,
                             size_t strideA, size_t strideB, size_t strideOut,
                             size_t offsetA, size_t offsetB, size_t offsetOut)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols)
    {
        size_t idxA = offsetA + row * strideA + col;
        size_t idxB = offsetB + row * strideB + col;
        size_t idxOut = offsetOut + row * strideOut + col;
        out[idxOut] = a[idxA] + b[idxB];
    }
}

__global__ void add3D_kernel(const float *a, const float *b, float *out,
                             size_t d1, size_t d2, size_t d3,
                             size_t sA1, size_t sA2,
                             size_t sB1, size_t sB2,
                             size_t sO1, size_t sO2,
                             size_t offsetA, size_t offsetB, size_t offsetOut)
{
    size_t i = blockIdx.z * blockDim.z + threadIdx.z;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d1 && j < d2 && k < d3)
    {
        size_t idxA = offsetA + i * sA1 + j * sA2 + k;
        size_t idxB = offsetB + i * sB1 + j * sB2 + k;
        size_t idxOut = offsetOut + i * sO1 + j * sO2 + k;
        out[idxOut] = a[idxA] + b[idxB];
    }
}
__global__ void addWithOffset_kernel(const float *a, const float *b, float *out,
                                     size_t size, size_t offsetA, size_t offsetB)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        out[i] = a[offsetA + i] + b[offsetB + i];
    }
}

Tensor Tensor::operator+(const Tensor &other) const
{
    if (this->shape != other.getShape())
    {
        throw std::invalid_argument("Los tensores deben tener la misma forma para la suma. " + this->shapeToString() + " vs " +
                                    other.shapeToString());
    }

    Tensor result(this->shape);

    dim3 blockSize, gridSize;
    // Iteramos sobre el tensor de salida y calculamos cada valor.
    // Esto funciona para cualquier tensor (contiguo o no) porque usamos los operadores ().
    if (this->shape.size() == 2)
    {
        size_t rows = shape[0], cols = shape[1];
        blockSize = dim3(16, 16);
        gridSize = dim3((cols + 15) / 16, (rows + 15) / 16);

        add2D_kernel<<<gridSize, blockSize>>>(
            this->data, other.getData(), result.getData(),
            rows, cols,
            this->strides[0], other.strides[0], result.strides[0],
            this->dataOffset, other.dataOffset, result.dataOffset);
    }
    else if (this->shape.size() == 3)
    {
        size_t d1 = shape[0], d2 = shape[1], d3 = shape[2];
        blockSize = dim3(8, 8, 8);
        gridSize = dim3((d3 + 7) / 8, (d2 + 7) / 8, (d1 + 7) / 8);

        add3D_kernel<<<gridSize, blockSize>>>(
            this->data, other.getData(), result.getData(),
            d1, d2, d3,
            this->strides[0], this->strides[1],
            other.strides[0], other.strides[1],
            result.strides[0], result.strides[1],
            this->dataOffset, other.dataOffset, result.dataOffset);
    }
    else
    { // Fallback para 1D u otras formas
        size_t N = this->totalSize;
        dim3 blockSize(256);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

        addWithOffset_kernel<<<gridSize, blockSize>>>(
            this->data, other.getData(), result.getData(),
            N, this->dataOffset, other.dataOffset);
    }
    cudaDeviceSynchronize();
    return result;
}

__global__ void sum_axis_strided_2d(const float *input, float *output,
                                    size_t dim0, size_t dim1,
                                    size_t axis,
                                    size_t in_stride0, size_t in_stride1,
                                    size_t out_stride0, size_t out_stride1,
                                    size_t in_offset, size_t out_offset)
{
    size_t i = blockIdx.y;
    size_t j = threadIdx.x;

    if (i < dim0 && j < dim1)
    {
        float sum = 0.0f;
        for (size_t k = 0; k < (axis == 0 ? dim0 : dim1); ++k)
        {
            size_t i_k = axis == 0 ? k : i;
            size_t j_k = axis == 1 ? k : j;
            size_t in_idx = in_offset + i_k * in_stride0 + j_k * in_stride1;
            sum += input[in_idx];
        }
        size_t out_idx = out_offset + i * out_stride0 + j * out_stride1;
        output[out_idx] = sum;
    }
}

__global__ void sum_axis_strided_3d(const float *input, float *output,
                                    size_t d0, size_t d1, size_t d2,
                                    size_t axis,
                                    size_t in_s0, size_t in_s1, size_t in_s2,
                                    size_t out_s0, size_t out_s1, size_t out_s2,
                                    size_t in_offset, size_t out_offset)
{
    size_t i = blockIdx.z;
    size_t j = blockIdx.y;
    size_t k = threadIdx.x;

    if (i < d0 && j < d1 && k < d2)
    {
        float sum = 0.0f;
        for (size_t t = 0; t < (axis == 0 ? d0 : axis == 1 ? d1
                                                           : d2);
             ++t)
        {
            size_t ti = axis == 0 ? t : i;
            size_t tj = axis == 1 ? t : j;
            size_t tk = axis == 2 ? t : k;
            size_t in_idx = in_offset + ti * in_s0 + tj * in_s1 + tk * in_s2;
            sum += input[in_idx];
        }
        size_t out_idx = out_offset + i * out_s0 + j * out_s1 + k * out_s2;
        output[out_idx] = sum;
    }
}

Tensor Tensor::sum(size_t axis) const
{
    if (axis >= shape.size())
    {
        throw std::out_of_range("Eje para sum() fuera de rango.");
    }

    std::vector<size_t> outputShape = this->shape;
    outputShape[axis] = 1;
    Tensor result(outputShape); // Se inicializa a ceros

    // Bucle genérico que itera sobre la forma de salida
    // y suma a lo largo del eje colapsado de la entrada.
    // Esto es más lento pero funciona para cualquier dimensionalidad.
    if (shape.size() == 2)
    {
        size_t d0 = shape[0];
        size_t d1 = shape[1];
        dim3 block(256);
        dim3 grid(1, d0); // i = blockIdx.y, j = threadIdx.x

        sum_axis_strided_2d<<<grid, block>>>(
            data, result.getData(),
            shape[0], shape[1],
            axis,
            strides[0], strides[1],
            result.strides[0], result.strides[1],
            dataOffset, result.dataOffset);
    }
    else if (shape.size() == 3)
    {
        size_t d0 = shape[0];
        size_t d1 = shape[1];
        size_t d2 = shape[2];

        dim3 block(128);       // threadIdx.x = k
        dim3 grid(d2, d1, d0); // blockIdx.z = i, blockIdx.y = j

        sum_axis_strided_3d<<<grid, block>>>(
            data, result.getData(),
            shape[0], shape[1], shape[2],
            axis,
            strides[0], strides[1], strides[2],
            result.strides[0], result.strides[1], result.strides[2],
            dataOffset, result.dataOffset);
    }
    else
    {
        throw std::runtime_error("sum() solo está implementado para 2D y 3D por ahora.");
    }
    cudaDeviceSynchronize();
    return result;
}

__global__ void addBroadcast2D_kernel(float *data, const float *broadcast, size_t M, size_t N, size_t offset_data, size_t offset_broadcast)
{
    size_t i = blockIdx.y * blockDim.y + threadIdx.y; // fila
    size_t j = blockIdx.x * blockDim.x + threadIdx.x; // columna

    if (i < M && j < N)
    {
        size_t idx = offset_data + i * N + j;
        size_t idx_b = offset_broadcast + j;
        data[idx] += broadcast[idx_b];
    }
}
__global__ void addBroadcast3D_kernel(float *data, const float *broadcast, size_t B, size_t N, size_t D, size_t offset_data, size_t offset_broadcast)
{
    size_t b = blockIdx.z * blockDim.z + threadIdx.z;
    size_t n = blockIdx.y * blockDim.y + threadIdx.y;
    size_t d = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < B && n < N && d < D)
    {
        size_t idx = offset_data + (b * N + n) * D + d;
        size_t idx_b = offset_broadcast + (0 * N + n) * D + d;
        data[idx] += broadcast[idx_b];
    }
}

void Tensor::addBroadcast(const Tensor &other)
{
    // Caso 1: Broadcasting de {1, N} sobre {M, N}
    if (this->shape.size() == 2 && other.getShape().size() == 2 &&
        other.getShape()[0] == 1 &&
        this->shape[1] == other.getShape()[1])
    {
        size_t M = this->shape[0];
        size_t N = this->shape[1];
        dim3 blockSize(16, 16);
        dim3 gridSize((N + 15) / 16, (M + 15) / 16);

        addBroadcast2D_kernel<<<gridSize, blockSize>>>(
            this->data, other.getData(),
            M, N,
            this->dataOffset, other.dataOffset);
    }
    // Caso 2: Broadcasting de {1, N, D} sobre {B, N, D}
    else if (this->shape.size() == 3 && other.getShape().size() == 3 && other.getShape()[0] == 1 &&
             this->shape[1] == other.getShape()[1] && this->shape[2] == other.getShape()[2])
    {
        size_t B = this->shape[0];
        size_t N = this->shape[1];
        size_t D = this->shape[2];
        dim3 blockSize(8, 8, 8);
        dim3 gridSize((D + 7) / 8, (N + 7) / 8, (B + 7) / 8);

        addBroadcast3D_kernel<<<gridSize, blockSize>>>(
            this->data, other.getData(),
            B, N, D,
            this->dataOffset, other.dataOffset);
    }
    else
    {
        throw std::runtime_error("Broadcasting no implementado para estas formas: " + this->shapeToString() + " y " +
                                 other.shapeToString());
    }
    cudaDeviceSynchronize();
}

void Tensor::printDebugInfo(const std::string &name) const
{
    std::cout << "--- Tensor Debug: " << name << " ---" << std::endl;
    std::cout << "  Forma: " << shapeToString() << std::endl;
    std::cout << "  Contiguo: " << (isContiguous() ? "Sí" : "NO") << std::endl;
    std::cout << "  Offset: " << dataOffset << std::endl;
    std::cout << "  Strides: ";
    for (const auto &s : strides)
        std::cout << s << " ";
    std::cout << std::endl;
    std::cout << "-------------------------" << std::endl;
}

__global__ void matrixMultiplyKernel(const float *a_data, const size_t *a_strides, size_t a_offset,
                                     const float *b_data, const size_t *b_strides, size_t b_offset,
                                     float *out_data, const size_t *out_strides, size_t out_offset,
                                     size_t M, size_t N, size_t P)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < P)
    {
        float sum = 0.0f;
        for (size_t k = 0; k < N; ++k)
        {
            size_t a_idx = a_offset + row * a_strides[0] + k * a_strides[1];
            size_t b_idx = b_offset + k * b_strides[0] + col * b_strides[1];
            sum += a_data[a_idx] * b_data[b_idx];
        }
        size_t out_idx = out_offset + row * out_strides[0] + col * out_strides[1];
        out_data[out_idx] = sum;
    }
}

Tensor matrixMultiply(const Tensor &a, const Tensor &b)
{
    const auto &aShape = a.getShape();
    const auto &bShape = b.getShape();

    if (aShape.size() != 2 || bShape.size() != 2)
    {
        throw std::runtime_error("matrixMultiply solo está implementada para tensores 2D.");
    }
    if (aShape[1] != bShape[0])
    {
        throw std::runtime_error("Dimensiones de matriz incompatibles para la multiplicación: " + a.shapeToString() + " y " +
                                 b.shapeToString());
    }

    size_t M = aShape[0];
    size_t N = aShape[1];
    size_t P = bShape[1];

    Tensor out({M, P});

    size_t *d_a_strides;
    size_t *d_b_strides;
    size_t *d_out_strides;
    cudaMalloc(&d_a_strides, 2 * sizeof(size_t));
    cudaMalloc(&d_b_strides, 2 * sizeof(size_t));
    cudaMalloc(&d_out_strides, 2 * sizeof(size_t));
    cudaMemcpy(d_a_strides, a.getStrides().data(), 2 * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_strides, b.getStrides().data(), 2 * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out.getStrides().data(), 2 * sizeof(size_t), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((P + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matrixMultiplyKernel<<<grid, block>>>(
        a.getData(), d_a_strides, a.getDataOffset(),
        b.getData(), d_b_strides, b.getDataOffset(),
        out.getData(), d_out_strides, out.getDataOffset(),
        M, N, P);

    cudaFree(d_a_strides);
    cudaFree(d_b_strides);
    cudaFree(d_out_strides);
    cudaDeviceSynchronize(); // Opcional para depuración
    return out;
}
__global__ void batchMatMulKernel(
    const float *__restrict__ a_data, const size_t *a_strides, size_t a_offset,
    const float *__restrict__ b_data, const size_t *b_strides, size_t b_offset,
    float *__restrict__ out_data, const size_t *out_strides, size_t out_offset,
    size_t batchSize, size_t m, size_t n, size_t p)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= p)
        return;

    for (size_t batch = 0; batch < batchSize; ++batch)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            size_t a_idx = a_offset + batch * a_strides[0] + row * a_strides[1] + i * a_strides[2];
            size_t b_idx = b_offset + batch * b_strides[0] + i * b_strides[1] + col * b_strides[2];
            sum += a_data[a_idx] * b_data[b_idx];
        }

        size_t out_idx = out_offset + batch * out_strides[0] + row * out_strides[1] + col * out_strides[2];
        out_data[out_idx] = sum;
    }
}

Tensor batchMatrixMultiply(const Tensor &a, const Tensor &b)
{
    // a.printFirstRow("Tensor A");
    // b.printFirstRow("Tensor B");
    const auto &aShape = a.getShape();
    const auto &bShape = b.getShape();

    if (aShape.size() != 3 || bShape.size() != 3)
    {
        throw std::runtime_error("BMM solo está implementado para tensores 3D.");
    }
    if (aShape[0] != bShape[0])
    {
        throw std::runtime_error("El tamaño del batch debe ser el mismo para ambos tensores en BMM.");
    }
    if (aShape[2] != bShape[1])
    {
        throw std::runtime_error("Dimensiones de matriz incompatibles para BMM: " + a.shapeToString() + " y " + b.shapeToString());
    }
    size_t B = aShape[0] /*batchSize*/, M = aShape[1], N = aShape[2], P = bShape[2];
    Tensor out({B, M, P});

    size_t *d_a_shape, *d_a_strides, *d_b_shape,
        *d_b_strides, *d_out_shape, *d_out_strides;
    cudaMalloc(&d_a_shape, aShape.size() * sizeof(size_t));
    cudaMalloc(&d_a_strides, a.getStrides().size() * sizeof(size_t));
    cudaMalloc(&d_b_shape, bShape.size() * sizeof(size_t));
    cudaMalloc(&d_b_strides, b.getStrides().size() * sizeof(size_t));
    cudaMalloc(&d_out_shape, out.getShape().size() * sizeof(size_t));
    cudaMalloc(&d_out_strides, out.getStrides().size() * sizeof(size_t));

    cudaMemcpy(d_a_shape, aShape.data(), aShape.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a.getStrides().data(), a.getStrides().size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape, bShape.data(), bShape.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_strides, b.getStrides().data(), b.getStrides().size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out.getShape().data(), out.getShape().size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out.getStrides().data(), out.getStrides().size() * sizeof(size_t), cudaMemcpyHostToDevice);

    // Lanzar kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((P + 15) / 16, (M + 15) / 16);
    cudaError_t err = cudaGetLastError();
    batchMatMulKernel<<<numBlocks, threadsPerBlock>>>(
        a.getData(), d_a_strides, a.getDataOffset(),
        b.getData(), d_b_strides, b.getDataOffset(),
        out.getData(), d_out_strides, out.getDataOffset(),
        B, M, N, P);

    cudaDeviceSynchronize();
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
    cudaFree(d_b_shape);
    cudaFree(d_b_strides);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;

    // out.printFirstRow("Tensor Resultado BMM");
    return out;
}
__global__ void concatenate3d_kernel(
    float *__restrict__ out,
    const float *__restrict__ in,
    size_t in_dim0,
    size_t in_dim1,
    size_t in_dim2,
    size_t out_dim1,
    size_t out_dim2,
    size_t axis_offset,
    size_t axis)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < in_dim0 && j < in_dim1 && k < in_dim2)
    {
        size_t in_idx = i * in_dim1 * in_dim2 + j * in_dim2 + k;

        size_t out_i = i, out_j = j, out_k = k;
        if (axis == 0)
            out_i += axis_offset;
        else if (axis == 1)
            out_j += axis_offset;
        else if (axis == 2)
            out_k += axis_offset;

        size_t out_idx = out_i * out_dim1 * out_dim2 + out_j * out_dim2 + out_k;

        out[out_idx] = in[in_idx];
    }
}

Tensor concatenate(const std::vector<Tensor> &tensors, size_t axis)
{
    if (tensors.empty())
    {
        return Tensor();
    }
    if (tensors.size() == 1)
    {
        return tensors[0];
    }

    // 1. Validaciones
    const auto &firstShape = tensors[0].getShape();
    size_t newDimSize = 0;
    for (const auto &t : tensors)
    {
        if (t.getShape().size() != firstShape.size() || t.getShape().size() <= axis)
        {
            throw std::invalid_argument("Todos los tensores deben tener el mismo rank y ser compatibles con el eje.");
        }
        for (size_t i = 0; i < firstShape.size(); ++i)
        {
            if (i != axis && t.getShape()[i] != firstShape[i])
            {
                throw std::invalid_argument("Las dimensiones deben ser iguales excepto en el eje de concatenación.");
            }
        }
        newDimSize += t.getShape()[axis];
    }

    // 2. Calcular la nueva forma y crear el tensor resultado
    std::vector<size_t> newShape = firstShape;
    newShape[axis] = newDimSize;
    Tensor result(newShape);

    // 3. Copiar los datos de cada tensor en la sección correcta del resultado
    size_t offset = 0;
    for (const auto &t : tensors)
    {
        dim3 threads(8, 8, 8);
        dim3 blocks(
            (t.getShape()[0] + threads.x - 1) / threads.x,
            (t.getShape()[1] + threads.y - 1) / threads.y,
            (t.getShape()[2] + threads.z - 1) / threads.z);

        concatenate3d_kernel<<<blocks, threads>>>(
            result.getData(),
            t.getData(),
            t.getShape()[0], t.getShape()[1], t.getShape()[2],
            result.getShape()[1], result.getShape()[2],
            offset,
            axis);

        cudaDeviceSynchronize(); // Opcional para detectar errores
        offset += t.getShape()[axis];
    }

    return result;
}

Tensor expand(const Tensor &tensor, size_t dim, size_t size)
{
    if (dim > tensor.getShape().size())
    {
        throw std::invalid_argument("La dimensión para expandir es mayor que el rank del tensor.");
    }

    std::vector<size_t> newShape = tensor.getShape();
    newShape.insert(newShape.begin() + dim, size);

    std::vector<size_t> newStrides = tensor.getStrides();
    newStrides.insert(newStrides.begin() + dim, 0); // El truco mágico: el stride para la nueva dimensión es 0.

    size_t cpuTotalSize = tensor.getSize();
    std::vector<float> dataPtr(cpuTotalSize);
    cudaMemcpy(dataPtr.data(), tensor.getData(), cpuTotalSize * sizeof(float), cudaMemcpyDeviceToHost);

    return Tensor(dataPtr, newShape, newStrides, tensor.getDataOffset());
}

__global__ void relu2d_kernel(const float *input, float *output, size_t rows, size_t cols)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * cols;
    if (idx < total)
    {
        float val = input[idx];
        output[idx] = (val > 0.0f) ? val : 0.0f;
    }
}

void Tensor::relu2d_forward(const Tensor &input)
{
    size_t rows = input.getShape()[0];
    size_t cols = input.getShape()[1];
    size_t total = rows * cols;

    dim3 blockDim(256);
    dim3 gridDim((total + blockDim.x - 1) / blockDim.x);
    relu2d_kernel<<<gridDim, blockDim>>>(input.getData(), this->data, rows, cols);
    cudaDeviceSynchronize(); // si desea asegurarse de que termine
}

__global__ void relu3d_kernel(const float *input, float *output, size_t d1, size_t d2, size_t d3)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = d1 * d2 * d3;
    if (idx < total)
    {
        float val = input[idx];
        output[idx] = (val > 0.0f) ? val : 0.0f;
    }
}

void Tensor::relu3d_forward(const Tensor &input)
{
    size_t d1 = input.getShape()[0];
    size_t d2 = input.getShape()[1];
    size_t d3 = input.getShape()[2];
    size_t total = d1 * d2 * d3;

    dim3 blockDim(256);
    dim3 gridDim((total + blockDim.x - 1) / blockDim.x);
    relu3d_kernel<<<gridDim, blockDim>>>(input.getData(), this->data, d1, d2, d3);
    cudaDeviceSynchronize();
}

__global__ void relu2d_backward_kernel(const float *input, const float *grad_output, float *grad_input,
                                       size_t rows, size_t cols)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * cols;

    if (idx < total)
    {
        float val = input[idx];
        grad_input[idx] = (val > 0.0f) ? grad_output[idx] : 0.0f;
    }
}
void Tensor::relu2d_backward(const Tensor &input, const Tensor &grad_output)
{
    size_t rows = input.getShape()[0];
    size_t cols = input.getShape()[1];
    size_t total = rows * cols;

    dim3 blockDim(256);
    dim3 gridDim((total + blockDim.x - 1) / blockDim.x);
    relu2d_backward_kernel<<<gridDim, blockDim>>>(
        input.getData(), grad_output.getData(), this->data,
        rows, cols);
    cudaDeviceSynchronize();
}

__global__ void relu3d_backward_kernel(const float *input, const float *grad_output, float *grad_input,
                                       size_t d1, size_t d2, size_t d3)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = d1 * d2 * d3;

    if (idx < total)
    {
        float val = input[idx];
        grad_input[idx] = (val > 0.0f) ? grad_output[idx] : 0.0f;
    }
}
void Tensor::relu3d_backward(const Tensor &input, const Tensor &grad_output)
{
    size_t d1 = input.getShape()[0];
    size_t d2 = input.getShape()[1];
    size_t d3 = input.getShape()[2];
    size_t total = d1 * d2 * d3;

    dim3 blockDim(256);
    dim3 gridDim((total + blockDim.x - 1) / blockDim.x);
    relu3d_backward_kernel<<<gridDim, blockDim>>>(
        input.getData(), grad_output.getData(), this->data,
        d1, d2, d3);
    cudaDeviceSynchronize();
}

__global__ void copy_view_to_contiguous_kernel(
    const float *__restrict__ input,
    float *output,
    const size_t *shape,
    const size_t *in_strides,
    const size_t totalSize)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize)
        return;

    // Convert idx a coordenadas [i, j, k]
    size_t D2 = shape[2]; // k
    size_t D1 = shape[1]; // j
    size_t D0 = shape[0]; // i

    size_t i = idx / (D1 * D2);
    size_t j = (idx / D2) % D1;
    size_t k = idx % D2;

    size_t offset = i * in_strides[0] + j * in_strides[1] + k * in_strides[2];
    output[idx] = input[offset];
}
void Tensor::embedding_backward(const Tensor &view)
{
    const auto &shape = view.getShape(); // std::vector<size_t>
    const auto &strides = view.getStrides();
    size_t totalSize = shape[0] * shape[1] * shape[2];

    // Copiar shape y strides a GPU
    size_t *d_shape = nullptr;
    size_t *d_strides = nullptr;

    cudaMalloc(&d_shape, shape.size() * sizeof(size_t));
    cudaMalloc(&d_strides, strides.size() * sizeof(size_t));
    cudaMemcpy(d_shape, shape.data(), shape.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, strides.data(), strides.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (totalSize + threads - 1) / threads;

    copy_view_to_contiguous_kernel<<<blocks, threads>>>(
        view.getData(),
        this->data,
        d_shape,
        d_strides,
        totalSize);

    cudaDeviceSynchronize(); // solo si estás depurando

    // Liberar memoria temporal
    cudaFree(d_shape);
    cudaFree(d_strides);
}

__global__ void argmaxPerRowKernel(const float *data, int *output, size_t rows, size_t cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;

    float max_val = -INFINITY;
    int max_idx = -1;
    for (size_t j = 0; j < cols; ++j)
    {
        float val = data[row * cols + j];
        if (val > max_val)
        {
            max_val = val;
            max_idx = j;
        }
    }
    output[row] = max_idx;
}
std::vector<int> Tensor::argmaxPerRow() const
{
    size_t rows = shape[0];
    size_t cols = shape[1];

    int threads = 256;
    int blocks = (rows + threads - 1) / threads;

    int *d_output;
    cudaMalloc(&d_output, rows * sizeof(int));

    argmaxPerRowKernel<<<blocks, threads>>>(data, d_output, rows, cols);
    cudaDeviceSynchronize();

    std::vector<int> h_output(rows);
    cudaMemcpy(h_output.data(), d_output, rows * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_output);
    return h_output;
}

__global__ void argmaxOneHotKernel(const float *data, int *output, size_t rows, size_t cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;

    for (size_t j = 0; j < cols; ++j)
    {
        if (data[row * cols + j] == 1.0f)
        {
            output[row] = j;
            return;
        }
    }
    output[row] = -1; // fallback
}

std::vector<int> Tensor::argmaxOneHot() const
{
    size_t rows = shape[0];
    size_t cols = shape[1];

    int threads = 256;
    int blocks = (rows + threads - 1) / threads;

    int *d_output;
    cudaMalloc(&d_output, rows * sizeof(int));

    argmaxOneHotKernel<<<blocks, threads>>>(data, d_output, rows, cols);
    cudaDeviceSynchronize();

    std::vector<int> h_output(rows);
    cudaMemcpy(h_output.data(), d_output, rows * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_output);
    return h_output;
}

std::pair<int, float> Tensor::argmaxWithProb() const
{
    if (shape.size() != 2 || shape[0] != 1)
    {
        throw std::runtime_error("argmaxWithProb() solo admite tensores 2D con una fila.");
    }

    size_t num_classes = shape[1];

    std::vector<float> host_probs(num_classes);
    cudaMemcpy(host_probs.data(), data + dataOffset, num_classes * sizeof(float), cudaMemcpyDeviceToHost);

    int argmax = 0;
    float max_val = host_probs[0];
    for (size_t i = 1; i < num_classes; ++i)
    {
        if (host_probs[i] > max_val)
        {
            max_val = host_probs[i];
            argmax = static_cast<int>(i);
        }
    }

    return {argmax, max_val};
}

float Tensor::get(const std::vector<size_t> &indices) const
{
    if (indices.size() != shape.size())
        throw std::invalid_argument("Número de índices incorrecto.");

    size_t flat_index = dataOffset;
    for (size_t i = 0; i < shape.size(); ++i)
        flat_index += indices[i] * strides[i];

    float value;
    cudaMemcpy(&value, data + flat_index, sizeof(float), cudaMemcpyDeviceToHost);
    return value;
}
void Tensor::set(const std::vector<size_t> &indices, float value)
{
    if (indices.size() != shape.size())
        throw std::invalid_argument("Número de índices incorrecto.");

    size_t flat_index = dataOffset;
    for (size_t i = 0; i < shape.size(); ++i)
        flat_index += indices[i] * strides[i];

    cudaMemcpy(data + flat_index, &value, sizeof(float), cudaMemcpyHostToDevice);
}
__global__ void softmax_kernel(const float *logits, float *output, size_t batchSize, size_t numClasses)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batchSize)
        return;

    const float *row_logits = logits + row * numClasses;
    float *row_output = output + row * numClasses;

    // 1. Encontrar el máximo valor
    float maxVal = -INFINITY;
    for (int j = 0; j < numClasses; ++j)
    {
        if (row_logits[j] > maxVal)
            maxVal = row_logits[j];
    }

    // 2. Calcular exponenciales y suma
    float sumExp = 0.0f;
    for (int j = 0; j < numClasses; ++j)
    {
        float expVal = expf(row_logits[j] - maxVal);
        row_output[j] = expVal;
        sumExp += expVal;
    }

    // 3. Normalizar
    for (int j = 0; j < numClasses; ++j)
    {
        row_output[j] /= sumExp;
    }
}
/**
 * @brief Función auxiliar que calcula la activación Softmax de forma estable.
 * @details Softmax convierte un vector de números reales (logits) en una
 *          distribución de probabilidad. Se utiliza el "truco de restar el máximo"
 *          para prevenir el desbordamiento numérico (overflow) con valores grandes
 *          en la función exponencial.
 * @param logits Tensor de logits de forma {batch_size, num_classes}.
 * @return Un tensor de probabilidades con la misma forma.
 */
Tensor softmax(const Tensor &logits)
{
    const std::vector<size_t> &shape = logits.getShape();
    if (shape.size() != 2)
        throw std::runtime_error("Softmax solo soporta tensores 2D [batchSize, numClasses]");

    size_t batchSize = shape[0];
    size_t numClasses = shape[1];

    Tensor output({batchSize, numClasses}); // en GPU

    int threads = 256;
    int blocks = (batchSize + threads - 1) / threads;

    softmax_kernel<<<blocks, threads>>>(
        logits.getData(), output.getData(), batchSize, numClasses);
    cudaDeviceSynchronize(); // opcional para depuración

    return output;
}

__global__ void crossEntropyForwardKernel(const float *softmax, const float *yTrue, float *lossPerSample, size_t batchSize, size_t numClasses)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batchSize)
        return;

    float loss = 0.0f;
    const float epsilon = 1e-12f;

    for (size_t j = 0; j < numClasses; ++j)
    {
        size_t idx = i * numClasses + j;
        if (yTrue[idx] == 1.0f)
        {
            loss = -logf(softmax[idx] + epsilon);
            break;
        }
    }

    lossPerSample[i] = loss;
}
float Tensor::cross_entropy_forward(const Tensor &softmax, const Tensor &yTrue)
{
    size_t batchSize = softmax.getShape()[0];
    size_t numClasses = softmax.getShape()[1];

    float *d_lossPerSample;
    cudaMalloc(&d_lossPerSample, batchSize * sizeof(float));

    int threads = 256;
    int blocks = (batchSize + threads - 1) / threads;

    crossEntropyForwardKernel<<<blocks, threads>>>(
        softmax.getData(),
        yTrue.getData(),
        d_lossPerSample,
        batchSize,
        numClasses);

    cudaDeviceSynchronize();

    std::vector<float> h_loss(batchSize);
    cudaMemcpy(h_loss.data(), d_lossPerSample, batchSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_lossPerSample);

    float totalLoss = 0.0f;
    for (float val : h_loss)
        totalLoss += val;

    return totalLoss / batchSize;
}

__global__ void cross_entropy_backward_kernel(float *grad, const float *yTrue, size_t batchSize, size_t numClasses)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batchSize * numClasses;

    if (idx < total)
    {
        size_t i = idx / numClasses;
        size_t j = idx % numClasses;
        grad[idx] = (grad[idx] - yTrue[idx]) / batchSize;
    }
}

Tensor Tensor::clone() const
{
    Tensor result(this->shape); // Usa el constructor para alocar memoria y calcular strides
    cudaMemcpy(result.getData(), this->data, this->totalSize * sizeof(float), cudaMemcpyDeviceToDevice);
    return result;
}

Tensor Tensor::crossEntropyBackward(const Tensor &yTrue) const
{
    // Verificación de dimensiones
    if (shape != yTrue.shape)
        throw std::runtime_error("crossEntropyBackward: shape mismatch");

    size_t batchSize = shape[0];
    size_t numClasses = shape[1];
    size_t totalSize = batchSize * numClasses;

    // Crear tensor gradiente (copia del actual softmaxOutput)
    Tensor grad = this->clone(); // Asumiendo que clone() crea una copia en GPU

    // Lanzar kernel
    const int threads = 256;
    const int blocks = (totalSize + threads - 1) / threads;

    cross_entropy_backward_kernel<<<blocks, threads>>>(grad.getData(), yTrue.getData(), batchSize, numClasses);
    cudaDeviceSynchronize(); // Opcional para sincronizar

    return grad;
}

__global__ void softmax3d_kernel(const float *logits, float *output,
                                 size_t B, size_t N, size_t D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N;
    if (idx >= total)
        return;

    int b = idx / N;
    int n = idx % N;

    const float *input_ptr = logits + (b * N * D) + (n * D);
    float *output_ptr = output + (b * N * D) + (n * D);

    // 1. Máximo
    float max_val = -INFINITY;
    for (int d = 0; d < D; ++d)
        max_val = fmaxf(max_val, input_ptr[d]);

    // 2. Exponenciales y suma
    float sum = 0.0f;
    for (int d = 0; d < D; ++d)
    {
        float exp_val = expf(input_ptr[d] - max_val);
        output_ptr[d] = exp_val;
        sum += exp_val;
    }

    // 3. Normalización
    for (int d = 0; d < D; ++d)
    {
        output_ptr[d] /= sum;
    }
}
Tensor softmax3d(const Tensor &logits)
{
    const auto &shape = logits.getShape();
    if (shape.size() != 3)
        throw std::runtime_error("softmax3d solo soporta tensores 3D [B, N, D]");

    size_t B = shape[0];
    size_t N = shape[1];
    size_t D = shape[2];

    Tensor output({B, N, D});

    int threads = 256;
    int blocks = (B * N + threads - 1) / threads;

    softmax3d_kernel<<<blocks, threads>>>(
        logits.getData(), output.getData(), B, N, D);
    cudaDeviceSynchronize();

    return output;
}
__global__ void softmax3d_backward_kernel(const float *grad_output, const float *softmax_output,
                                          float *grad_input, size_t B, size_t N, size_t D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N;
    if (idx >= total)
        return;

    int b = idx / N;
    int n = idx % N;

    const float *grad_out_ptr = grad_output + (b * N * D) + (n * D);
    const float *softmax_ptr = softmax_output + (b * N * D) + (n * D);
    float *grad_in_ptr = grad_input + (b * N * D) + (n * D);

    // 1. Calcular dot product: sum_j (grad_out[j] * softmax[j])
    float dot_product = 0.0f;
    for (int k = 0; k < D; ++k)
    {
        dot_product += grad_out_ptr[k] * softmax_ptr[k];
    }

    // 2. Calcular grad_input[i] = s_i * (grad_out[i] - dot)
    for (int i = 0; i < D; ++i)
    {
        grad_in_ptr[i] = softmax_ptr[i] * (grad_out_ptr[i] - dot_product);
    }
}
Tensor softmax3d_backward(const Tensor &grad_output, const Tensor &softmax_output)
{
    const auto &shape = grad_output.getShape();
    if (shape.size() != 3)
        throw std::runtime_error("softmax3d_backward solo soporta tensores 3D [B, N, D]");

    size_t B = shape[0];
    size_t N = shape[1];
    size_t D = shape[2];

    Tensor grad_input({B, N, D}); // Resultado

    int threads = 256;
    int blocks = (B * N + threads - 1) / threads;

    softmax3d_backward_kernel<<<blocks, threads>>>(
        grad_output.getData(),
        softmax_output.getData(),
        grad_input.getData(),
        B, N, D);

    cudaDeviceSynchronize(); // Opcional para depuración

    return grad_input;
}
// Kernel para 1D
__global__ void adam_1d_kernel(float *param, const float *grad, float *m, float *v,
                               float lr, float beta1, float beta2,
                               float beta1_t, float beta2_t,
                               float epsilon, float weight_decay, size_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;

    float g = grad[i];
    if (weight_decay > 0.0f)
        g += weight_decay * param[i];

    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * (g * g);

    float m_hat = m[i] / (1.0f - beta1_t);
    float v_hat = v[i] / (1.0f - beta2_t);

    param[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
}

// Kernel para 2D
__global__ void adam_2d_kernel(float *param, const float *grad, float *m, float *v,
                               float lr, float beta1, float beta2,
                               float beta1_t, float beta2_t,
                               float epsilon, float weight_decay,
                               size_t rows, size_t cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (i >= total)
        return;

    int r = i / cols;
    int c = i % cols;
    int idx = r * cols + c;

    float g = grad[idx];
    if (weight_decay > 0.0f)
        g += weight_decay * param[idx];

    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * (g * g);

    float m_hat = m[idx] / (1.0f - beta1_t);
    float v_hat = v[idx] / (1.0f - beta2_t);

    param[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
}

// Kernel para 3D
__global__ void adam_3d_kernel(float *param, const float *grad, float *m, float *v,
                               float lr, float beta1, float beta2,
                               float beta1_t, float beta2_t,
                               float epsilon, float weight_decay,
                               size_t d0, size_t d1, size_t d2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d0 * d1 * d2;
    if (i >= total)
        return;

    int i0 = i / (d1 * d2);
    int i1 = (i / d2) % d1;
    int i2 = i % d2;
    int idx = i0 * d1 * d2 + i1 * d2 + i2;

    float g = grad[idx];
    if (weight_decay > 0.0f)
        g += weight_decay * param[idx];

    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * (g * g);

    float m_hat = m[idx] / (1.0f - beta1_t);
    float v_hat = v[idx] / (1.0f - beta2_t);

    param[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
}
void Tensor::adamUpdate1D(const Tensor &grad, Tensor &m, Tensor &v,
                          float lr, float beta1, float beta2,
                          float beta1_t, float beta2_t,
                          float epsilon, float weight_decay)
{
    size_t size = this->shape[0];
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    adam_1d_kernel<<<blocks, threads>>>(
        this->data, grad.getData(), m.getData(), v.getData(),
        lr, beta1, beta2, beta1_t, beta2_t,
        epsilon, weight_decay, size);
}

void Tensor::adamUpdate2D(const Tensor &grad, Tensor &m, Tensor &v,
                          float lr, float beta1, float beta2,
                          float beta1_t, float beta2_t,
                          float epsilon, float weight_decay)
{
    size_t rows = this->shape[0];
    size_t cols = this->shape[1];
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    adam_2d_kernel<<<blocks, threads>>>(
        this->data, grad.getData(), m.getData(), v.getData(),
        lr, beta1, beta2, beta1_t, beta2_t,
        epsilon, weight_decay, rows, cols);
}

void Tensor::adamUpdate3D(const Tensor &grad, Tensor &m, Tensor &v,
                          float lr, float beta1, float beta2,
                          float beta1_t, float beta2_t,
                          float epsilon, float weight_decay)
{
    size_t d0 = this->shape[0];
    size_t d1 = this->shape[1];
    size_t d2 = this->shape[2];
    int total = d0 * d1 * d2;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    adam_3d_kernel<<<blocks, threads>>>(
        this->data, grad.getData(), m.getData(), v.getData(),
        lr, beta1, beta2, beta1_t, beta2_t,
        epsilon, weight_decay, d0, d1, d2);
}

// Kernel que copia patch_dim elementos del tensor origen (patch_view) al destino (patches_flat)
__global__ void copy_patch_kernel(const float *src_data, const size_t *src_strides, const size_t *src_shape,
                                  size_t ndim, size_t src_offset,
                                  float *dst_data, size_t dst_offset, size_t total_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements)
        return;

    // Calcular índice multidimensional inverso desde idx (índice lineal)
    size_t remaining = idx;
    size_t linear_index = src_offset;

    for (int d = ndim - 1; d >= 0; --d)
    {
        size_t coord = remaining % src_shape[d];
        remaining /= src_shape[d];
        linear_index += coord * src_strides[d];
    }

    dst_data[dst_offset + idx] = src_data[linear_index];
}

void Tensor::copyToHost(float *host_buffer) const
{
    if (isContiguous())
    {
        size_t totalSizeBytes = totalSize * sizeof(float);
        cudaMemcpy(host_buffer, data + dataOffset, totalSizeBytes, cudaMemcpyDeviceToHost);
        return;
    }

    // std::cout << "Tensor NO contiguo: usando copy_patch_kernel para copiar a host." << std::endl;

    // Paso 1: reservar buffer intermedio en GPU
    float *device_buffer;
    cudaMalloc(&device_buffer, totalSize * sizeof(float));

    // Paso 2: copiar shape y strides a GPU
    size_t *device_shape, *device_strides;
    cudaMalloc(&device_shape, shape.size() * sizeof(size_t));
    cudaMalloc(&device_strides, strides.size() * sizeof(size_t));

    cudaMemcpy(device_shape, shape.data(), shape.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_strides, strides.data(), strides.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    // Paso 3: lanzar kernel
    const int threads = 256;
    const int blocks = (totalSize + threads - 1) / threads;

    copy_patch_kernel<<<blocks, threads>>>(
        data, device_strides, device_shape, shape.size(),
        dataOffset, device_buffer, 0, totalSize);

    cudaDeviceSynchronize();

    // Paso 4: copiar del buffer GPU al host
    cudaMemcpy(host_buffer, device_buffer, totalSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Paso 5: liberar memoria
    cudaFree(device_buffer);
    cudaFree(device_shape);
    cudaFree(device_strides);
}

__global__ void extract_patches_kernel(const float *input,
                                       float *output,
                                       size_t batchSize,
                                       size_t in_channels,
                                       size_t img_height,
                                       size_t img_width,
                                       size_t patch_size,
                                       size_t num_patches_h,
                                       size_t num_patches_w,
                                       size_t patch_dim)
{
    size_t patch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_patches = batchSize * num_patches_h * num_patches_w;
    if (patch_idx >= total_patches)
        return;

    size_t pw = patch_idx % num_patches_w;
    size_t ph = (patch_idx / num_patches_w) % num_patches_h;
    size_t b = patch_idx / (num_patches_h * num_patches_w);

    size_t h_start = ph * patch_size;
    size_t w_start = pw * patch_size;

    for (size_t c = 0; c < in_channels; ++c)
    {
        for (size_t i = 0; i < patch_size; ++i)
        {
            for (size_t j = 0; j < patch_size; ++j)
            {
                size_t in_h = h_start + i;
                size_t in_w = w_start + j;
                size_t input_idx = ((b * in_channels + c) * img_height + in_h) * img_width + in_w;

                size_t local_idx = ((c * patch_size + i) * patch_size + j);
                size_t output_idx = patch_idx * patch_dim + local_idx;

                output[output_idx] = input[input_idx];
            }
        }
    }
}
Tensor Tensor::extractPatches(size_t patch_size,
                              size_t image_height,
                              size_t image_width,
                              size_t in_channels,
                              size_t num_patches_h,
                              size_t num_patches_w) const
{
    if (shape.size() != 4)
        throw std::runtime_error("extractPatches espera un tensor 4D [B, C, H, W]");

    size_t batchSize = shape[0];
    size_t patch_dim = in_channels * patch_size * patch_size;
    size_t total_patches = batchSize * num_patches_h * num_patches_w;

    Tensor patches_flat({total_patches, patch_dim});

    dim3 threads(256);
    dim3 blocks((total_patches + threads.x - 1) / threads.x);

    extract_patches_kernel<<<blocks, threads>>>(
        this->data + this->dataOffset,
        patches_flat.getData(),
        batchSize,
        in_channels,
        image_height,
        image_width,
        patch_size,
        num_patches_h,
        num_patches_w,
        patch_dim);

    cudaDeviceSynchronize();
    return patches_flat;
}

__inline__ __device__ float warpReduceSum(float val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
__inline__ __device__ float blockReduceSum(float val)
{
    static __shared__ float shared[32]; // 32 warps como máximo
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val); // reduce dentro del warp

    if (lane == 0)
        shared[wid] = val; // guardar suma del warp
    __syncthreads();

    // Solo el primer warp realiza la suma final
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0)
        val = warpReduceSum(val);
    return val;
}

__global__ void layerNormKernel(
    const float *__restrict__ input,
    float *__restrict__ output,
    float *__restrict__ meanOut,
    float *__restrict__ varOut,
    float *__restrict__ normOut,
    const float *__restrict__ gamma,
    const float *__restrict__ beta,
    size_t featureSize,
    float epsilon)
{
    size_t batchIdx = blockIdx.x;
    size_t tid = threadIdx.x;

    const float *row = input + batchIdx * featureSize;
    float *outRow = output + batchIdx * featureSize;
    float *normRow = normOut ? normOut + batchIdx * featureSize : nullptr;

    // --- 1. Calcular media ---
    float sum = 0.0f;
    for (size_t j = tid; j < featureSize; j += blockDim.x)
        sum += row[j];
    sum = blockReduceSum(sum);
    __shared__ float mean;
    if (tid == 0)
        mean = sum / featureSize;
    __syncthreads();

    if (meanOut && tid == 0)
        meanOut[batchIdx] = mean;

    // --- 2. Calcular varianza ---
    float varSum = 0.0f;
    for (size_t j = tid; j < featureSize; j += blockDim.x)
    {
        float diff = row[j] - mean;
        varSum += diff * diff;
    }
    varSum = blockReduceSum(varSum);
    __shared__ float var;
    if (tid == 0)
        var = varSum / featureSize;
    __syncthreads();

    float invStd = rsqrtf(var + epsilon);
    if (varOut && tid == 0)
        varOut[batchIdx] = invStd;

    // --- 3. Normalizar y aplicar gamma/beta ---
    for (size_t j = tid; j < featureSize; j += blockDim.x)
    {
        float x_hat = (row[j] - mean) * invStd;
        if (normRow)
            normRow[j] = x_hat;
        outRow[j] = gamma[j] * x_hat + beta[j];
    }
}
Tensor Tensor::layerNorm(const Tensor &gamma,
                         const Tensor &beta,
                         Tensor *meanOut, Tensor *varOut,
                         Tensor *normalizedOut,
                         float epsilon) const
{
    size_t featureSize = gamma.getSize();
    size_t batchSize = this->totalSize / featureSize;
    Tensor output({batchSize, featureSize});

    int threads = 128;
    int blocks = batchSize;

    layerNormKernel<<<blocks, threads>>>(
        this->getData(),
        output.getData(),
        meanOut ? meanOut->getData() : nullptr,
        varOut ? varOut->getData() : nullptr,
        normalizedOut ? normalizedOut->getData() : nullptr,
        gamma.getData(),
        beta.getData(),
        featureSize,
        epsilon);

    cudaDeviceSynchronize(); // opcional, por depuración

    return output;
}
__global__ void scaleKernel(float *data, size_t size, float factor)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] *= factor;
    }
}
void Tensor::scale(float factor)
{
    size_t threads = 256;
    size_t blocks = (totalSize + threads - 1) / threads;
    scaleKernel<<<blocks, threads>>>(data, totalSize, factor);
    cudaDeviceSynchronize(); // Opcional, si necesita bloqueo
}
__global__ void geluKernel(float *out, const float *in, size_t size)
{
    const float SQRT_2_OVER_PI = 0.7978845608f;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float x = in[idx];
        float x_cubed = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + 0.044715f * x_cubed);
        out[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}
Tensor Tensor::gelu_f() const
{
    if (!this->isContiguous())
    {
        throw std::runtime_error("gelu_f: solo implementado para tensores contiguos.");
    }

    Tensor result(this->shape); // nuevo tensor con misma shape

    const size_t size = this->totalSize;
    float *input_ptr = this->data;
    float *output_ptr = result.data;

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    geluKernel<<<blocks, threads>>>(output_ptr, input_ptr, size);
    cudaDeviceSynchronize(); // O usar streams si lo deseas asincrónico

    return result;
}

void Tensor::printData(const std::string &name) const
{
    std::vector<float> host_data(totalSize);
    cudaMemcpy(host_data.data(), data, totalSize * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream file(name + ".txt");
    std::ostringstream oss;

    oss << "--- Tensor: " << name << " (shape: ";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        oss << shape[i];
        if (i != shape.size() - 1)
            oss << "x";
    }
    oss << ") ---\n";

    if (shape.size() == 1)
    {
        for (size_t i = 0; i < shape[0]; ++i)
        {
            oss << host_data[i];
            if (i != shape[0] - 1)
                oss << ", ";
        }
        oss << "\n";
    }
    else if (shape.size() == 2)
    {
        for (size_t i = 0; i < shape[0]; ++i)
        {
            for (size_t j = 0; j < shape[1]; ++j)
            {
                oss << host_data[i * shape[1] + j];
                if (j != shape[1] - 1)
                    oss << ", ";
            }
            oss << "\n";
        }
    }
    else if (shape.size() == 3)
    {
        for (size_t b = 0; b < shape[0]; ++b)
        {
            oss << "Slice [" << b << "]\n";
            for (size_t i = 0; i < shape[1]; ++i)
            {
                for (size_t j = 0; j < shape[2]; ++j)
                {
                    size_t idx = b * shape[1] * shape[2] + i * shape[2] + j;
                    oss << host_data[idx];
                    if (j != shape[2] - 1)
                        oss << ", ";
                }
                oss << "\n";
            }
            oss << "\n";
        }
    }
    else if (shape.size() == 4)
    {
        for (size_t n = 0; n < shape[0]; ++n)
        {
            oss << "Batch [" << n << "]\n";
            for (size_t c = 0; c < shape[1]; ++c)
            {
                oss << " Channel [" << c << "]\n";
                for (size_t h = 0; h < shape[2]; ++h)
                {
                    for (size_t w = 0; w < shape[3]; ++w)
                    {
                        size_t idx = n * shape[1] * shape[2] * shape[3] + c * shape[2] * shape[3] + h * shape[3] + w;
                        oss << host_data[idx];
                        if (w != shape[3] - 1)
                            oss << ", ";
                    }
                    oss << "\n";
                }
                oss << "\n";
            }
            oss << "\n";
        }
    }
    else
    {
        oss << "[printData] Visualización no implementada para dimensiones > 4.\n";
    }

    file << oss.str();
    file.close();
}
void Tensor::printFirstRow(const std::string &name) const
{
    // Mostrar información de depuración
    std::cout << "--- Tensor Debug: " << name << " ---" << std::endl;
    std::cout << "  Forma: " << shapeToString() << std::endl;
    std::cout << "  Contiguo: " << (isContiguous() ? "Sí" : "NO") << std::endl;
    std::cout << "  Offset: " << dataOffset << std::endl;
    std::cout << "  Strides: ";
    for (const auto &s : strides)
        std::cout << s << " ";
    std::cout << std::endl;

    // Copiar datos a CPU
    std::vector<float> host_data(totalSize);
    cudaMemcpy(host_data.data(), data, totalSize * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "--- Primer fila de datos: ---" << std::endl;

    if (shape.size() == 1)
    {
        for (size_t i = 0; i < shape[0]; ++i)
        {
            std::cout << host_data[i];
            if (i != shape[0] - 1)
                std::cout << ", ";
        }
        std::cout << std::endl;
    }
    else if (shape.size() >= 2)
    {
        size_t row_size = shape[shape.size() - 1];
        for (size_t j = 0; j < row_size; ++j)
        {
            std::cout << host_data[j];
            if (j != row_size - 1)
                std::cout << ", ";
        }
        std::cout << std::endl;
    }
    else
    {
        std::cout << "[printFirstRow] Visualización no implementada para shape vacío o desconocido." << std::endl;
    }
    std::cout << "-------------------------" << std::endl;
}
