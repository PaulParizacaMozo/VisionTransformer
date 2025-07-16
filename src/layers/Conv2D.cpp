#include "layers/Conv2D.hpp"
#include "core/Tensor.hpp"
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

Conv2D::Conv2D(size_t input_channels, size_t output_channels, size_t kernel_size, size_t stride, size_t padding)
    : input_channels(input_channels), output_channels(output_channels), kernel_size(kernel_size), stride(stride), padding(padding) {
    // Inicialización de parámetros (filtros y sesgo)
    filters = Tensor({output_channels, input_channels, kernel_size, kernel_size});
    filters.randomizeNormal(0.0f, 0.02f);
    
    bias = Tensor({output_channels});
    bias.randomizeNormal(0.0f, 0.02f);

    filtersGradient = Tensor({output_channels, input_channels, kernel_size, kernel_size});
    // filtersGradient.randomizeNormal(0.0f, 0.02f);
}

Tensor Conv2D::forward(const Tensor &input, bool isTraining) {
    size_t batchSize = input.getShape()[0];
    size_t inputHeight = input.getShape()[2];
    size_t inputWidth = input.getShape()[3];

    // Guardamos el tensor de entrada para usarlo en la retropropagación
    lastInput = input;

    // Calcular el tamaño de la salida considerando el padding y stride
    size_t outputHeight = (inputHeight - kernel_size + 2 * padding) / stride + 1;
    size_t outputWidth = (inputWidth - kernel_size + 2 * padding) / stride + 1;

    // Crear un tensor para almacenar la salida de la convolución
    Tensor output({batchSize, output_channels, outputHeight, outputWidth});

    // Realizar la operación de convolución
    #pragma omp parallel for collapse(4)
    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t c = 0; c < output_channels; ++c) {
            for (size_t h = 0; h < outputHeight; ++h) {
                for (size_t w = 0; w < outputWidth; ++w) {
                    float sum = 0.0f;

                    for (size_t i = 0; i < kernel_size; ++i) {
                        for (size_t j = 0; j < kernel_size; ++j) {
                            for (size_t k = 0; k < input_channels; ++k) {
                                size_t input_h = h * stride + i - padding;
                                size_t input_w = w * stride + j - padding;

                                if (input_h >= 0 && input_h < inputHeight && input_w >= 0 && input_w < inputWidth) {
                                    sum += input(b, k, input_h, input_w) * filters(c, k, i, j);
                                }
                            }
                        }
                    }

                    output(b, c, h, w) = sum + bias(c); // Añadir el sesgo
                }
            }
        }
    }
    return output;
}

Tensor Conv2D::backward(const Tensor &outputGradient) {
    size_t batchSize = outputGradient.getShape()[0];
    size_t outputHeight = outputGradient.getShape()[2];
    size_t outputWidth = outputGradient.getShape()[3];

    // Gradientes de los filtros
    filtersGradient = Tensor({output_channels, input_channels, kernel_size, kernel_size});
    // Reutilizar gradientes
    // if (filtersGradient.getShape() != std::vector<size_t>({output_channels, input_channels, kernel_size, kernel_size})) {
    //     std::cout<<"No Reutiliza porque son diferentes!!! --------------"<<std::endl;
    //     filtersGradient = Tensor({output_channels, input_channels, kernel_size, kernel_size});
    // }
    filtersGradient.fill(0.0f);

    // Gradientes del sesgo
    // biasGradient = Tensor({output_channels});
    if (biasGradient.getShape() != std::vector<size_t>({output_channels})) {
        biasGradient = Tensor({output_channels});
    }
    biasGradient.fill(0.0f);

    // Gradiente de la entrada (conforme con el tamaño de la entrada original)
    Tensor inputGradient = Tensor({batchSize, input_channels, outputHeight * stride, outputWidth * stride});
    inputGradient.fill(0.0f);

    // Calcular gradientes
    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t c = 0; c < output_channels; ++c) {
            for (size_t h = 0; h < outputHeight; ++h) {
                for (size_t w = 0; w < outputWidth; ++w) {
                    // Gradiente para los filtros
                    for (size_t i = 0; i < kernel_size; ++i) {
                        for (size_t j = 0; j < kernel_size; ++j) {
                            for (size_t k = 0; k < input_channels; ++k) {
                                size_t input_h = h * stride + i - padding;
                                size_t input_w = w * stride + j - padding;

                                // Asegurarnos de que estamos dentro de los límites
                                if (input_h >= 0 && input_h < lastInput.getShape()[2] && input_w >= 0 && input_w < lastInput.getShape()[3]) {
                                    filtersGradient(c, k, i, j) += outputGradient(b, c, h, w) * lastInput(b, k, input_h, input_w);
                                }
                            }
                        }
                    }

                    // Gradiente para el sesgo (se suman todos los gradientes de salida en el batch)
                    biasGradient(c) += outputGradient(b, c, h, w);
                }
            }
        }
    }

    // Gradiente de la entrada
    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t c = 0; c < input_channels; ++c) {
            for (size_t h = 0; h < outputHeight; ++h) {
                for (size_t w = 0; w < outputWidth; ++w) {
                    float sum = 0.0f;

                    // Deslizar el filtro en reversa para calcular el gradiente de la entrada
                    for (size_t i = 0; i < kernel_size; ++i) {
                        for (size_t j = 0; j < kernel_size; ++j) {
                            for (size_t k = 0; k < output_channels; ++k) {
                                size_t input_h = h * stride + i - padding;
                                size_t input_w = w * stride + j - padding;

                                // Asegurarnos de que estamos dentro de los límites
                                if (input_h >= 0 && input_h < inputGradient.getShape()[2] && input_w >= 0 && input_w < inputGradient.getShape()[3]) {
                                    sum += filters(k, c, i, j) * outputGradient(b, k, h, w);
                                }
                            }
                        }
                    }

                    inputGradient(b, c, h * stride, w * stride) = sum; // Este es el gradiente de la entrada
                }
            }
        }
    }

    return inputGradient;
}


std::vector<Tensor *> Conv2D::getParameters() {
    return {&filters, &bias};
} 

std::vector<Tensor *> Conv2D::getGradients() {
    return {&filtersGradient, &biasGradient};
}