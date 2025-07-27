#pragma once

#include <vector>
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <atomic>
#include <numeric>    // para std::accumulate
#include <stdexcept>  // para std::invalid_argument
#include <functional> // para std::multiplies
#include <ctime>

class SeedGenerator
{
public:
    static unsigned long getNextSeed()
    {
        static std::atomic<uint64_t> counter{0};
        return static_cast<unsigned long>(time(nullptr)) ^ counter.fetch_add(1);
    }
};

class Tensor; // Declaración adelantada de la clase Tensor
Tensor concatenate(const std::vector<Tensor> &tensors, size_t axis);
Tensor expand(const Tensor &tensor, size_t dim, size_t size);

class Tensor
{
private:
    void computeStrides();

    float *data = nullptr;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t dataOffset;
    size_t totalSize;

public:
    void copyToHost(float *host_buffer) const;
    // --- Constructores y Destructor ---
    Tensor();
    explicit Tensor(const std::vector<size_t> &shape);
    Tensor(const std::vector<size_t> &shape, const std::vector<float> &data);
    Tensor(std::vector<float> dataPtr, const std::vector<size_t> &shape,
           const std::vector<size_t> &strides, size_t offset);
    Tensor(const Tensor &other) = default;
    Tensor(Tensor &&other) noexcept = default;
    Tensor &operator=(const Tensor &other) = default;
    Tensor &operator=(Tensor &&other) noexcept = default;
    ~Tensor() = default;
    // --- Operaciones y Vistas ---
    Tensor slice(size_t axis, size_t start, size_t count) const;
    Tensor reshape(const std::vector<size_t> &newShape) const;
    Tensor transpose(size_t dim1, size_t dim2) const;
    Tensor sum(size_t axis) const;
    void addBroadcast(const Tensor &other);
    Tensor contiguous() const; // AÑADIDO

    Tensor expand(const std::vector<size_t> &newShape) const;

    // --- Operadores Aritméticos ---
    Tensor operator+(const Tensor &other) const;

    // --- Inicialización y Modificación ---
    void fill(float value);
    void randomize(float min = -1.0f, float max = 1.0f);
    void randomizeNormal(float mean = 0.0f, float stddev = 1.0f);

    // --- Getters y Utilidades ---
    const std::vector<size_t> &getShape() const { return shape; }
    size_t getSize() const { return totalSize; }
    const std::vector<size_t> &getStrides() const { return strides; }
    size_t getDataOffset() const { return dataOffset; }
    const float *getData() const;
    float *getData();
    std::string shapeToString() const;
    bool isContiguous() const;
    // const std::shared_ptr<std::vector<float>> &getDataPtr() const { return dataPtr; }

    // depuracion
    void printDebugInfo(const std::string &name) const; // NUEVA
    void relu2d_forward(const Tensor &input);
    void relu3d_forward(const Tensor &input);
    void relu2d_backward(const Tensor &input, const Tensor &grad_output);
    void relu3d_backward(const Tensor &input, const Tensor &grad_output);
    void embedding_backward(const Tensor &view);
    std::vector<int> argmaxPerRow() const;
    std::vector<int> argmaxOneHot() const;
    std::pair<int, float> argmaxWithProb() const;
    float get(const std::vector<size_t> &indices) const;
    void set(const std::vector<size_t> &indices, float value);
    float cross_entropy_forward(const Tensor &softmax, const Tensor &yTrue);
    Tensor crossEntropyBackward(const Tensor &yTrue) const;
    Tensor clone() const;
    void adamUpdate1D(const Tensor &grad, Tensor &m, Tensor &v,
                      float lr, float beta1, float beta2,
                      float beta1_t, float beta2_t,
                      float epsilon, float weight_decay);

    void adamUpdate2D(const Tensor &grad, Tensor &m, Tensor &v,
                      float lr, float beta1, float beta2,
                      float beta1_t, float beta2_t,
                      float epsilon, float weight_decay);

    void adamUpdate3D(const Tensor &grad, Tensor &m, Tensor &v,
                      float lr, float beta1, float beta2,
                      float beta1_t, float beta2_t,
                      float epsilon, float weight_decay);
    Tensor extractPatches(size_t patch_size,
                          size_t image_height,
                          size_t image_width,
                          size_t in_channels,
                          size_t num_patches_h,
                          size_t num_patches_w) const;
    Tensor layerNorm(const Tensor &gamma,
                     const Tensor &beta,
                     Tensor *meanOut, Tensor *varOut,
                     Tensor *normalizedOut,
                     float epsilon) const;
    void scale(float factor);
};
// --- Funciones Libres para Operaciones de Tensor ---

/** @brief Realiza la multiplicación de matrices entre dos tensores 2D. */
Tensor matrixMultiply(const Tensor &a, const Tensor &b);

/** @brief Realiza la multiplicación de matrices por lotes (BMM) en tensores 3D. */
Tensor batchMatrixMultiply(const Tensor &a, const Tensor &b);
Tensor softmax(const Tensor &logits);
Tensor softmax3d(const Tensor &logits);
Tensor softmax3d_backward(const Tensor &grad_output, const Tensor &softmax_output);