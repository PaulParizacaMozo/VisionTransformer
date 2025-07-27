#include "layers/Conv2d.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// --- Declaraciones de funciones privadas de la clase ---
// Es buena práctica declararlas en la cabecera si fueran parte de la API pública.
// Como son utilidades internas, las dejamos aquí, pero necesitarán declararse en Conv2d.hpp
// para que el compilador las conozca.
// Por ahora, las dejaremos como funciones libres en este archivo para simplificar.
namespace {
    void im2col(const Tensor& input, Tensor& col_matrix, size_t kernel_size, size_t stride, size_t padding);
    void col2im(const Tensor& col_matrix, Tensor& output_image, size_t kernel_size, size_t stride, size_t padding);
}

Conv2d::Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding)
    : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding) {
    
    // Inicialización de pesos Kaiming/He, mejor para arquitecturas con ReLU/GELU.
    float fan_in = static_cast<float>(in_channels * kernel_size * kernel_size);
    float stddev = std::sqrt(2.0f / fan_in);

    // Forma de los pesos: {out_channels, in_channels, kernel_size, kernel_size}
    this->weights = Tensor({out_channels, in_channels, kernel_size, kernel_size});
    this->weights.randomizeNormal(0.0f, stddev);

    // Bias se inicializa como {1, out_channels, 1, 1} para broadcasting 4D
    this->bias = Tensor({1, out_channels, 1, 1});
    this->bias.fill(0.0f);

    // Inicializar gradientes
    this->weightGradients = Tensor(this->weights.getShape());
    this->biasGradients = Tensor(this->bias.getShape());
}

Tensor Conv2d::forward(const Tensor& input, bool isTraining) {
    if (isTraining) {
        this->inputTensor = input;
    }
    
    const auto& in_shape = input.getShape();
    size_t batch_size = in_shape[0];
    size_t in_h = in_shape[2];
    size_t in_w = in_shape[3];

    // 1. Calcular dimensiones de salida
    size_t out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    size_t out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    // 2. Transformar la entrada a una matriz de columnas (im2col)
    size_t patch_dim = this->in_channels * this->kernel_size * this->kernel_size;
    size_t num_patches = out_h * out_w;
    Tensor im2col_matrix({patch_dim, batch_size * num_patches});
    im2col(input, im2col_matrix, this->kernel_size, this->stride, this->padding);

    // 3. Remodelar los pesos para la multiplicación (sin copia)
    // De {out_C, in_C, kH, kW} a {out_C, in_C * kH * kW}
    Tensor reshaped_weights = this->weights.reshape({this->out_channels, patch_dim});
    
    // 4. Convolución como multiplicación de matrices
    // {out_C, patch_dim} @ {patch_dim, B * num_patches} -> {out_C, B * num_patches}
    Tensor conv_result = matrixMultiply(reshaped_weights, im2col_matrix);

    // 5. Remodelar la salida y añadir el bias
    // {out_C, B * num_patches} -> {out_C, B, num_patches} -> transpose -> {B, out_C, num_patches}
    Tensor output = conv_result.reshape({this->out_channels, batch_size, num_patches});
    output = output.transpose(0, 1); // -> Crea una vista NO CONTIGUA
    // Antes del reshape final, hacemos la vista contigua.
    output = output.contiguous();

    // {B, out_C, num_patches} -> {B, out_C, out_H, out_W}
    output = output.reshape({batch_size, this->out_channels, out_h, out_w});

    // Añadir el bias. Necesitaríamos una función addBroadcast para 4D.
    // Lo hacemos manualmente por ahora.
    #pragma omp parallel for collapse(4)
    for(size_t b=0; b<batch_size; ++b)
     for(size_t c=0; c<out_channels; ++c)
      for(size_t h=0; h<out_h; ++h)
       for(size_t w=0; w<out_w; ++w)
        output(b,c,h,w) += this->bias(0,c,0,0);
        
    return output;
}

Tensor Conv2d::backward(const Tensor& outputGradient) {
    const auto& in_shape = this->inputTensor.getShape();
    size_t batch_size = in_shape[0];
    
    const auto& out_grad_shape = outputGradient.getShape();
    size_t out_h = out_grad_shape[2];
    size_t out_w = out_grad_shape[3];
    size_t num_patches = out_h * out_w;
    size_t patch_dim = this->in_channels * this->kernel_size * this->kernel_size;

    // --- 1. Calcular gradiente del bias ---
    // Sumamos el gradiente de salida a lo largo de las dimensiones B, H, W
    this->biasGradients = outputGradient.sum(0).sum(2).sum(3).reshape(this->bias.getShape());

    // --- 2. Preparar gradiente de salida y entrada (im2col) ---
    // {B, out_C, out_H, out_W} -> {B, out_C, num_patches} -> transpose -> {out_C, B, num_patches} -> {out_C, B * num_patches}
    // Preparamos el gradiente de salida, asegurando contigüidad
    Tensor reshaped_out_grad = outputGradient.reshape({batch_size, this->out_channels, num_patches});
    reshaped_out_grad = reshaped_out_grad.transpose(0, 1); // -> Vista NO CONTIGUA
    reshaped_out_grad = reshaped_out_grad.contiguous();
    reshaped_out_grad = reshaped_out_grad.reshape({this->out_channels, batch_size * num_patches});
                                           
    Tensor im2col_matrix({patch_dim, batch_size * num_patches});
    im2col(this->inputTensor, im2col_matrix, this->kernel_size, this->stride, this->padding);

    // --- 3. Calcular gradiente de los pesos (dE/dW) ---
    // dW = dY @ X_im2col^T
    Tensor dW_flat = matrixMultiply(reshaped_out_grad, im2col_matrix.transpose(0, 1));
    this->weightGradients = dW_flat.reshape(this->weights.getShape());
    
    // --- 4. Calcular gradiente de la entrada (dE/dX) ---
    // dX_col = W^T @ dY
    Tensor reshaped_weights = this->weights.reshape({this->out_channels, patch_dim});
    Tensor dX_col = matrixMultiply(reshaped_weights.transpose(0, 1), reshaped_out_grad);

    Tensor input_gradient(in_shape);
    col2im(dX_col, input_gradient, this->kernel_size, this->stride, this->padding);
    
    return input_gradient;
}

// --- Getters ---
std::vector<Tensor*> Conv2d::getParameters() { return {&this->weights, &this->bias}; }
std::vector<Tensor*> Conv2d::getGradients() { return {&this->weightGradients, &this->biasGradients}; }

// --- Implementaciones de im2col y col2im (movidas a un namespace anónimo) ---
namespace {

/**
 * @brief Transforma parches de la imagen de entrada en columnas de una matriz.
 * @details Cada columna de la matriz de salida representa un parche aplanado.
 * @param input El tensor de entrada de forma {B, C_in, H_in, W_in}.
 * @param col_matrix El tensor de salida que se llenará, de forma {C_in*kH*kW, B*H_out*W_out}.
 * @param kernel_size El tamaño del kernel (k).
 * @param stride El paso del kernel.
 * @param padding El relleno de la imagen.
 */
void im2col(const Tensor& input, Tensor& col_matrix, size_t kernel_size, size_t stride, size_t padding) {
    const auto& in_shape = input.getShape();
    const size_t batch_size = in_shape[0];
    const size_t in_channels = in_shape[1];
    const size_t in_h = in_shape[2];
    const size_t in_w = in_shape[3];

    const size_t out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    const size_t out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    const size_t col_rows = col_matrix.getShape()[0];
    const size_t col_cols = col_matrix.getShape()[1];

    if (col_rows != in_channels * kernel_size * kernel_size || col_cols != batch_size * out_h * out_w) {
        throw std::runtime_error("Dimensiones de la matriz im2col incorrectas.");
    }
    
    #pragma omp parallel for
    for (size_t col_c = 0; col_c < col_cols; ++col_c) {
        // Calcular a qué parche (b, oh, ow) corresponde esta columna
        size_t w_out = col_c % out_w;
        size_t h_out = (col_c / out_w) % out_h;
        size_t b_idx = col_c / (out_h * out_w);

        size_t row_idx = 0;
        // Rellenar la columna iterando sobre el parche correspondiente
        for (size_t c_in = 0; c_in < in_channels; ++c_in) {
            for (size_t kh = 0; kh < kernel_size; ++kh) {
                for (size_t kw = 0; kw < kernel_size; ++kw) {
                    int h_in = static_cast<int>(h_out * stride + kh) - static_cast<int>(padding);
                    int w_in = static_cast<int>(w_out * stride + kw) - static_cast<int>(padding);

                    float val = 0.0f;
                    // Si el píxel está dentro de los límites (después de considerar el padding), lo tomamos.
                    if (h_in >= 0 && h_in < static_cast<int>(in_h) && w_in >= 0 && w_in < static_cast<int>(in_w)) {
                        val = input(b_idx, c_in, h_in, w_in);
                    }
                    col_matrix(row_idx, col_c) = val;
                    row_idx++;
                }
            }
        }
    }
}

/**
 * @brief Operación inversa a im2col. Transforma una matriz de columnas en una "imagen".
 * @details Acumula (suma) los valores de las columnas en las posiciones correctas de la imagen de salida.
 *          Esencial para calcular el gradiente de entrada (dE/dX).
 * @param col_matrix La matriz de columnas de entrada, forma {C_in*kH*kW, B*H_out*W_out}.
 * @param output_image El tensor de imagen de salida que se llenará, forma {B, C_in, H_in, W_in}.
 * @param kernel_size El tamaño del kernel (k).
 * @param stride El paso del kernel.
 * @param padding El relleno de la imagen.
 */
void col2im(const Tensor& col_matrix, Tensor& output_image, size_t kernel_size, size_t stride, size_t padding) {
    const auto& out_shape = output_image.getShape();
    const size_t batch_size = out_shape[0];
    const size_t in_channels = out_shape[1];
    const size_t in_h = out_shape[2];
    const size_t in_w = out_shape[3];

    const size_t out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    const size_t out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    output_image.fill(0.0f); // Es crucial empezar con ceros porque vamos a acumular gradientes.

    #pragma omp parallel for
    for (size_t col_c = 0; col_c < col_matrix.getShape()[1]; ++col_c) {
        size_t w_out = col_c % out_w;
        size_t h_out = (col_c / out_w) % out_h;
        size_t b_idx = col_c / (out_h * out_w);

        size_t row_idx = 0;
        for (size_t c_in = 0; c_in < in_channels; ++c_in) {
            for (size_t kh = 0; kh < kernel_size; ++kh) {
                for (size_t kw = 0; kw < kernel_size; ++kw) {
                    int h_in = static_cast<int>(h_out * stride + kh) - static_cast<int>(padding);
                    int w_in = static_cast<int>(w_out * stride + kw) - static_cast<int>(padding);

                    if (h_in >= 0 && h_in < static_cast<int>(in_h) && w_in >= 0 && w_in < static_cast<int>(in_w)) {
                        // Un píxel de la imagen de entrada puede ser parte de varios parches,
                        // por lo que sus gradientes deben sumarse (acumularse).
                        // #pragma omp atomic es necesario para evitar condiciones de carrera
                        // cuando múltiples hilos intentan escribir en el mismo píxel de output_image.
                        #pragma omp atomic
                        output_image(b_idx, c_in, h_in, w_in) += col_matrix(row_idx, col_c);
                    }
                    row_idx++;
                }
            }
        }
    }
}

} // namespace
