// #ifndef TENSOR_HPP
// #define TENSOR_HPP

// #include <memory>
// #include <numeric>
// #include <stdexcept>
// #include <string>
// #include <vector>

// // --- Declaraciones adelantadas de funciones libres ---
// class Tensor; // Declaración adelantada de la clase Tensor
// Tensor concatenate(const std::vector<Tensor> &tensors, size_t axis);
// Tensor expand(const Tensor &tensor, size_t dim, size_t size);

// /**
//  * @class Tensor
//  * @brief Un tensor N-dimensional, especializado para operaciones 2D y 3D.
//  *
//  * Gestiona un bloque de datos multidimensional usando un puntero compartido
//  * para permitir vistas eficientes (sin copia de datos) mediante el uso de
//  * "strides" y un "offset".
//  */
// class Tensor
// {
// public:
//   // --- Constructores y Destructor ---
//   Tensor();
//   explicit Tensor(const std::vector<size_t> &shape);
//   Tensor(const std::vector<size_t> &shape, const std::vector<float> &data);
//   Tensor(const Tensor &other) = default;
//   Tensor(Tensor &&other) noexcept = default;
//   Tensor &operator=(const Tensor &other) = default;
//   Tensor &operator=(Tensor &&other) noexcept = default;
//   ~Tensor() = default;

//   // --- Acceso a Elementos (Optimizados) ---
//   float &operator()(size_t i);
//   const float &operator()(size_t i) const;
//   float &operator()(size_t i, size_t j);
//   const float &operator()(size_t i, size_t j) const;
//   float &operator()(size_t i, size_t j, size_t k);
//   const float &operator()(size_t i, size_t j, size_t k) const;
//   float &operator()(size_t d0, size_t d1, size_t d2, size_t d3);             // RE-INTRODUCIDO
//   const float &operator()(size_t d0, size_t d1, size_t d2, size_t d3) const; // RE-INTRODUCIDO

//   // --- Operaciones y Vistas ---
//   Tensor slice(size_t axis, size_t start, size_t count) const;
//   Tensor reshape(const std::vector<size_t> &newShape) const;
//   Tensor transpose(size_t dim1, size_t dim2) const;
//   Tensor square() const;
//   Tensor sum(size_t axis) const;
//   void addBroadcast(const Tensor &other);
//   Tensor contiguous() const; // AÑADIDO

//   Tensor expand(const std::vector<size_t> &newShape) const;
//   void copyFrom(const Tensor &src);

//   // --- Operadores Aritméticos ---
//   Tensor operator+(const Tensor &other) const;

//   // --- Inicialización y Modificación ---
//   void fill(float value);
//   void randomize(float min = -1.0f, float max = 1.0f);
//   void randomizeNormal(float mean = 0.0f, float stddev = 1.0f);

//   // --- Getters y Utilidades ---
//   const std::vector<size_t> &getShape() const { return shape; }
//   size_t getSize() const { return totalSize; }
//   const std::vector<size_t> &getStrides() const { return strides; }
//   size_t getDataOffset() const { return dataOffset; }
//   const std::shared_ptr<std::vector<float>> &getDataPtr() const { return dataPtr; }
//   const float *getData() const;
//   float *getData();
//   std::string shapeToString() const;
//   bool isContiguous() const;

//   // depuracion
//   void printDebugInfo(const std::string &name) const; // NUEVA
//   Tensor(std::shared_ptr<std::vector<float>> dataPtr, const std::vector<size_t> &shape, const std::vector<size_t> &strides,
//          size_t offset);

// private:
//   void computeStrides();

//   std::shared_ptr<std::vector<float>> dataPtr;
//   std::vector<size_t> shape;
//   std::vector<size_t> strides;
//   size_t dataOffset;
//   size_t totalSize;
// };
// // --- Funciones Libres para Operaciones de Tensor ---

// /** @brief Realiza la multiplicación de matrices entre dos tensores 2D. */
// Tensor matrixMultiply(const Tensor &a, const Tensor &b);

// /** @brief Realiza la multiplicación de matrices por lotes (BMM) en tensores 3D. */
// Tensor batchMatrixMultiply(const Tensor &a, const Tensor &b);

// // --- Implementaciones Inline (para rendimiento) ---

// inline float &Tensor::operator()(size_t i)
// {
// #ifndef NDEBUG
//   if (shape.size() != 1 || i >= shape[0])
//     throw std::out_of_range("Acceso 1D fuera de rango.");
// #endif
//   return (*dataPtr)[dataOffset + i * strides[0]];
// }

// inline const float &Tensor::operator()(size_t i) const
// {
// #ifndef NDEBUG
//   if (shape.size() != 1 || i >= shape[0])
//     throw std::out_of_range("Acceso 1D fuera de rango.");
// #endif
//   return (*dataPtr)[dataOffset + i * strides[0]];
// }

// inline float &Tensor::operator()(size_t i, size_t j)
// {
// #ifndef NDEBUG
//   if (shape.size() != 2 || i >= shape[0] || j >= shape[1])
//     throw std::out_of_range("Acceso 2D fuera de rango.");
// #endif
//   return (*dataPtr)[dataOffset + i * strides[0] + j * strides[1]];
// }

// inline const float &Tensor::operator()(size_t i, size_t j) const
// {
// #ifndef NDEBUG
//   if (shape.size() != 2 || i >= shape[0] || j >= shape[1])
//     throw std::out_of_range("Acceso 2D fuera de rango.");
// #endif
//   return (*dataPtr)[dataOffset + i * strides[0] + j * strides[1]];
// }

// inline float &Tensor::operator()(size_t i, size_t j, size_t k)
// {
// #ifndef NDEBUG
//   if (shape.size() != 3 || i >= shape[0] || j >= shape[1] || k >= shape[2])
//     throw std::out_of_range("Acceso 3D fuera de rango.");
// #endif
//   return (*dataPtr)[dataOffset + i * strides[0] + j * strides[1] + k * strides[2]];
// }

// inline const float &Tensor::operator()(size_t i, size_t j, size_t k) const
// {
// #ifndef NDEBUG
//   if (shape.size() != 3 || i >= shape[0] || j >= shape[1] || k >= shape[2])
//     throw std::out_of_range("Acceso 3D fuera de rango.");
// #endif
//   return (*dataPtr)[dataOffset + i * strides[0] + j * strides[1] + k * strides[2]];
// }

// // --- RE-INTRODUCIDO: Acceso 4D ---
// inline float &Tensor::operator()(size_t d0, size_t d1, size_t d2, size_t d3)
// {
// #ifndef NDEBUG
//   if (shape.size() != 4 || d0 >= shape[0] || d1 >= shape[1] || d2 >= shape[2] || d3 >= shape[3])
//     throw std::out_of_range("Acceso 4D fuera de rango.");
// #endif
//   return (*dataPtr)[dataOffset + d0 * strides[0] + d1 * strides[1] + d2 * strides[2] + d3 * strides[3]];
// }
// inline const float &Tensor::operator()(size_t d0, size_t d1, size_t d2, size_t d3) const
// {
// #ifndef NDEBUG
//   if (shape.size() != 4 || d0 >= shape[0] || d1 >= shape[1] || d2 >= shape[2] || d3 >= shape[3])
//     throw std::out_of_range("Acceso 4D fuera de rango.");
// #endif
//   return (*dataPtr)[dataOffset + d0 * strides[0] + d1 * strides[1] + d2 * strides[2] + d3 * strides[3]];
// }

// #endif // TENSOR_HPP
