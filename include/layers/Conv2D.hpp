#ifndef CONV2D_HPP
#define CONV2D_HPP

#include "core/Tensor.hpp"
#include "layers/Layer.hpp"
#include <vector>

class Conv2D : public Layer {
public:
    Conv2D(size_t input_channels, size_t output_channels, size_t kernel_size, size_t stride, size_t padding);

    Tensor forward(const Tensor &input, bool isTraining) override;
    Tensor backward(const Tensor &outputGradient) override;

    std::vector<Tensor *> getParameters() override;
    std::vector<Tensor *> getGradients() override;

    std::string getName() const override { return "Conv2D"; }

private:
    size_t input_channels;
    size_t output_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    
    Tensor filters;    // Filtros para la convolución
    Tensor bias;       // Sesgo de la convolución
    Tensor filtersGradient;
    Tensor biasGradient;

    Tensor lastInput;  // Almacenamos el último tensor de entrada
};

#endif // CONV2D_HPP
