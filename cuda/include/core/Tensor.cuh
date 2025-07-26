#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <string>
#include <iostream>
#include <vector>
#include <functional>

class Tensor
{
private:
    float *data = nullptr;                                        // Datos en GPU
    size_t *shape = nullptr;                                      // Shape en GPU
    size_t *strides = nullptr;                                    // Strides en GPU
    size_t ndim = 0;                                              // Número de dimensiones
    size_t totalSize = 0;                                         // Producto de shape
    size_t dataOffset = 0;                                        // Para slicing/view
    size_t *shape_host_copy = nullptr;                            // metadato en CPU
    size_t *strides_host_copy = nullptr;                          // metadato en CPU
    void computeStrides();                                        // Versión en GPU
    void allocateMetadata(const size_t *shape_host, size_t ndim); // Copia shape a GPU

public:
    // --- Getters ---
    inline size_t dim(size_t i) const { return shape_host_copy[i]; }
    inline size_t stride(size_t i) const { return strides_host_copy[i]; }
    inline size_t dims() const { return ndim; }
    inline size_t size() const { return totalSize; }
    inline size_t *getShapeHost() const { return shape_host_copy; }
    inline size_t *getStridesHost() const { return strides_host_copy; }
    size_t getDataOffset() const { return dataOffset; }
    size_t *getShape() const { return shape; }
    size_t *getStrides() const { return strides; }
    size_t getNDim() const { return ndim; }
    float *getData() const { return data; }
    std::vector<float> getDataHost() const
    {
        std::vector<float> data_host(totalSize);
        cudaMemcpy(data_host.data(), data, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
        return data_host;
    }
    bool isContiguous() const;

    // --- Constructores y Destructor ---
    Tensor();
    Tensor(std::initializer_list<size_t> dims);
    Tensor(float *dataPtr, const size_t *shape_host,
           const size_t *strides_host, size_t ndim, size_t offset);
    Tensor(const size_t *shape_host, size_t ndim); // Constructor principal
    Tensor(const Tensor &other);                   // Copia profunda GPU-GPU
    Tensor(Tensor &&other) noexcept;
    Tensor &operator=(const Tensor &other);
    void deepCopyFrom(const Tensor &other);
    Tensor &operator=(Tensor &&other) noexcept;
    ~Tensor();
    void copyFromHost(const float *srcHost)
    {
        cudaMemcpy(this->data, srcHost, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    }
    void copyToHost(float *dstHost) const
    {
        cudaMemcpy(dstHost, this->data, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // --- Inicialización ---
    void fill(float value);                                       // Llenar con constante
    void randomize(float min = -1.0f, float max = 1.0f);          // Uniforme
    void randomizeNormal(float mean = 0.0f, float stddev = 1.0f); // Normal

    // --- Transformaciones ---
    Tensor slice(size_t axis, size_t start, size_t count) const;
    Tensor reshape(const size_t *newShape_host, size_t newNdim) const;
    Tensor transpose(size_t dim1, size_t dim2) const;
    Tensor contiguous() const;

    // --- Operaciones tensoriales ---
    Tensor operator+(const Tensor &other) const;
    Tensor square() const;
    Tensor sum(size_t axis) const;
    void addBroadcast(const Tensor &other);

    // --- Debug info opcional ---
    void printDebugInfo(const std::string &name = "") const;
    void printContents(const std::string &name) const
    {
        std::vector<float> host_data(totalSize);
        cudaMemcpy(host_data.data(), data, totalSize * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<size_t> host_shape(ndim);
        cudaMemcpy(host_shape.data(), shape, ndim * sizeof(size_t), cudaMemcpyDeviceToHost);

        std::cout << "--- " << name << " (shape: [";
        for (size_t i = 0; i < ndim; ++i)
        {
            std::cout << host_shape[i];
            if (i < ndim - 1)
                std::cout << ", ";
        }
        std::cout << "]) ---\n";

        if (!isContiguous())
            std::cout << "⚠️  Advertencia: Tensor no contiguo en memoria (strides no estándar).\n";

        // Recursively print tensor contents
        std::function<void(size_t, size_t)> recursivePrint = [&](size_t dim, size_t offset)
        {
            if (dim == ndim - 1)
            {
                std::cout << "[";
                for (size_t i = 0; i < host_shape[dim]; ++i)
                {
                    std::cout << host_data[offset + i];
                    if (i < host_shape[dim] - 1)
                        std::cout << ", ";
                }
                std::cout << "]";
            }
            else
            {
                std::cout << "[";
                size_t step = 1;
                for (size_t i = dim + 1; i < ndim; ++i)
                    step *= host_shape[i];

                for (size_t i = 0; i < host_shape[dim]; ++i)
                {
                    recursivePrint(dim + 1, offset + i * step);
                    if (i < host_shape[dim] - 1)
                        std::cout << ",\n";
                }
                std::cout << "]";
            }
        };

        recursivePrint(0, 0);
        std::cout << "\n\n";
    }
    void printImageAtIndex(const std::string &name, size_t index) const
    {
        std::vector<size_t> host_shape(ndim);
        cudaMemcpy(host_shape.data(), shape, ndim * sizeof(size_t), cudaMemcpyDeviceToHost);

        size_t batch_dim = host_shape[0];
        if (index >= batch_dim)
        {
            std::cerr << "Índice fuera de rango: " << index << " >= " << batch_dim << "\n";
            return;
        }

        // Calcular el tamaño de una muestra (por ejemplo, una imagen)
        size_t sample_size = 1;
        for (size_t i = 1; i < ndim; ++i)
            sample_size *= host_shape[i];

        std::vector<float> host_data(sample_size);
        cudaMemcpy(host_data.data(), data + index * sample_size, sample_size * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "--- " << name << "[" << index << "] (shape: [";
        for (size_t i = 1; i < ndim; ++i)
        {
            std::cout << host_shape[i];
            if (i < ndim - 1)
                std::cout << ", ";
        }
        std::cout << "]) ---\n";

        // Imprimir la imagen (solo soportamos imágenes 2D o 3D por simplicidad)
        if (ndim == 4) // [N, C, H, W]
        {
            size_t C = host_shape[1];
            size_t H = host_shape[2];
            size_t W = host_shape[3];
            for (size_t c = 0; c < C; ++c)
            {
                std::cout << "Canal " << c << ":\n";
                for (size_t i = 0; i < H; ++i)
                {
                    std::cout << "[";
                    for (size_t j = 0; j < W; ++j)
                    {
                        std::cout << (host_data[c * H * W + i * W + j] > 0.5f ? '#' : '.');
                        if (j < W - 1)
                            std::cout << ", ";
                    }
                    std::cout << "]\n";
                }
                std::cout << "\n";
            }
        }
        else if (ndim == 3) // [N, H, W]
        {
            size_t H = host_shape[1];
            size_t W = host_shape[2];
            for (size_t i = 0; i < H; ++i)
            {
                std::cout << "[";
                for (size_t j = 0; j < W; ++j)
                {
                    std::cout << host_data[i * W + j];
                    if (j < W - 1)
                        std::cout << ", ";
                }
                std::cout << "]\n";
            }
        }
        else
        {
            std::cerr << "printImageAtIndex solo soporta tensores 3D o 4D (batch).\n";
        }

        std::cout << "\n";
    }
    void printLabelAtIndex(const std::string &name, size_t index) const
    {
        std::vector<size_t> host_shape(ndim);
        cudaMemcpy(host_shape.data(), shape, ndim * sizeof(size_t), cudaMemcpyDeviceToHost);

        size_t batch_dim = host_shape[0];
        if (index >= batch_dim)
        {
            std::cerr << "Índice fuera de rango: " << index << " >= " << batch_dim << "\n";
            return;
        }

        // Calcular tamaño de una muestra (una etiqueta puede ser escalar o vector)
        size_t label_size = 1;
        for (size_t i = 1; i < ndim; ++i)
            label_size *= host_shape[i];

        std::vector<float> host_data(label_size);
        cudaMemcpy(host_data.data(), data + index * label_size, label_size * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "--- " << name << "[" << index << "] (shape: [";
        for (size_t i = 1; i < ndim; ++i)
        {
            std::cout << host_shape[i];
            if (i < ndim - 1)
                std::cout << ", ";
        }
        std::cout << "]) ---\n";

        // Mostrar etiquetas
        if (ndim == 2) // [N, D]
        {
            std::cout << "[";
            for (size_t i = 0; i < label_size; ++i)
            {
                std::cout << host_data[i];
                if (i < label_size - 1)
                    std::cout << ", ";
            }
            std::cout << "]\n";
        }
        else if (ndim == 1) // [N]
        {
            std::cout << host_data[0] << "\n";
        }
        else
        {
            std::cerr << "printLabelAtIndex solo soporta tensores 1D o 2D (batch de etiquetas).\n";
        }

        std::cout << "\n";
    }

    __device__ inline float &operator()(size_t i);
    __device__ inline float &operator()(size_t i, size_t j);
    __device__ inline float &operator()(size_t i, size_t j, size_t k);
    __device__ inline float &operator()(size_t i, size_t j, size_t k, size_t l);
};

// --- Funciones auxiliares GPU ---
Tensor matrixMultiply(const Tensor &a, const Tensor &b);
Tensor batchMatrixMultiply(const Tensor &a, const Tensor &b);
Tensor concatenate(const Tensor *tensors, size_t numTensors, size_t axis);
Tensor expand(const Tensor &tensor, size_t dim, size_t size);
Tensor softmax(const Tensor &logits, int axis);
Tensor softmax_backward(const Tensor &grad_output, const Tensor &softmax_output);