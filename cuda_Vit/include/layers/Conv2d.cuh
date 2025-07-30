#pragma once
#include "layers/Layer.cuh"
#include "core/Tensor.hpp"

/**
 * @class Conv2d
 * @brief Implementa una capa de convolución 2D.
 *
 * Realiza una operación de convolución sobre una entrada 4D {B, C_in, H, W}.
 * Para el caso de PatchEmbedding, se usa sin padding y con un stride igual
 * al tamaño del kernel para lograr el parcheo no superpuesto.
 */
class Conv2d : public Layer
{
public:
    /**
     * @brief Constructor de la capa Conv2d.
     * @param in_channels Número de canales de entrada.
     * @param out_channels Número de canales de salida (filtros).
     * @param kernel_size Tamaño del kernel (asumimos cuadrado).
     * @param stride El paso del kernel al deslizarse por la imagen.
     * @param padding El relleno a añadir a los bordes de la imagen (0 para nuestro caso).
     */
    Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0);

    /**
     * @brief Realiza el forward pass de la convolución.
     * @param input Tensor de entrada de forma {B, C_in, H_in, W_in}.
     * @param isTraining Booleano para el modo de entrenamiento.
     * @return Tensor de salida de forma {B, C_out, H_out, W_out}.
     * @override
     */
    Tensor forward(const Tensor &input, bool isTraining) override;

    /**
     * @brief Realiza el backward pass de la convolución.
     * @details Calcula los gradientes con respecto a la entrada, los pesos y el bias.
     * @param outputGradient Gradiente de la pérdida respecto a la salida de esta capa.
     * @return Gradiente de la pérdida respecto a la entrada de esta capa.
     * @override
     */
    Tensor backward(const Tensor &outputGradient) override;

    /**
     * @brief Devuelve los parámetros entrenables (pesos y bias).
     * @override
     */
    std::vector<Tensor *> getParameters() override;

    /**
     * @brief Devuelve los gradientes de los parámetros.
     * @override
     */
    std::vector<Tensor *> getGradients() override;

    /**
     * @brief Devuelve el nombre de la capa.
     * @override
     */
    std::string getName() const override { return "Conv2d"; }

private:
    // Hiperparámetros
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;

    // Parámetros entrenables
    // Forma: {out_channels, in_channels, kernel_size, kernel_size}
    Tensor weights;
    // Forma: {1, out_channels, 1, 1} para broadcasting
    Tensor bias;

    // Gradientes correspondientes
    Tensor weightGradients;
    Tensor biasGradients;

    // Estado guardado para el backward pass
    Tensor inputTensor;
};