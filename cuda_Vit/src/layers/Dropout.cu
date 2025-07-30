#include "layers/Dropout.cuh"
#include <random>
#include <stdexcept>
#include "utils/CudaUtils.cuh"
#include <iostream>

Dropout::Dropout(float rate) : rate(rate)
{
    if (rate < 0.0f || rate >= 1.0f)
    {
        throw std::invalid_argument("La tasa de Dropout debe estar en el rango [0, 1).");
    }
    // Si la tasa es 0, el dropout no hace nada y la escala es 1.
    if (rate > 0.0f)
    {
        this->scale = 1.0f / (1.0f - this->rate);
    }
    else
    {
        this->scale = 1.0f;
    }
}

Tensor Dropout::forward(const Tensor &input, bool isTraining)
{
    // Durante la inferencia o si la tasa es 0, el dropout no hace nada.
    if (!isTraining || this->rate == 0.0f)
    {
        return input;
    }

    // Durante el entrenamiento, aplicamos la máscara de dropout.
    this->mask = Tensor(input.getShape());
    Tensor output(input.getShape());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // Asumimos que la entrada es contigua para un rendimiento óptimo
    if (input.isContiguous())
    {
        const float *in_data = input.getData();
        float *out_data = output.getData();
        float *mask_data = mask.getData();
        size_t size = input.getSize();

        for (size_t i = 0; i < size; ++i)
        {
            if (dis(gen) < this->rate)
            {
                // Apagar la neurona
                mask_data[i] = 0.0f;
                out_data[i] = 0.0f;
            }
            else
            {
                // Mantener la neurona y escalarla
                mask_data[i] = this->scale;
                out_data[i] = in_data[i] * this->scale;
            }
        }
    }
    else
    {
        // Un fallback para tensores no contiguos sería necesario en una librería de producción
        throw std::runtime_error("Dropout::forward actualmente solo soporta tensores contiguos.");
    }

    return output;
}

Tensor Dropout::backward(const Tensor &outputGradient)
{
    // Si la máscara está vacía, significa que el forward pass se ejecutó en modo
    // inferencia o con tasa 0, por lo que no se aplicó dropout.
    if (mask.getSize() == 0)
    {
        return outputGradient;
    }
    return dropout_backward_cuda(outputGradient, mask);
}
