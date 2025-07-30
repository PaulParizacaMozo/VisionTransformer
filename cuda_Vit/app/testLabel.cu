#include "model/Trainer.cuh"
#include "utils/ModelUtils.hpp"
#include "utils/CudaUtils.cuh"
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "utils/stb_image.h"

Tensor load_image_with_stb(const std::string &path)
{
    int width, height, channels;
    unsigned char *img_data = stbi_load(path.c_str(), &width, &height, &channels, 1);
    if (!img_data)
        throw std::runtime_error("No se pudo cargar la imagen: " + path);

    const int target_size = 28;
    Tensor input({1, 1, target_size, target_size});
    float scale_x = static_cast<float>(width) / target_size;
    float scale_y = static_cast<float>(height) / target_size;

    for (int y = 0; y < target_size; ++y)
    {
        for (int x = 0; x < target_size; ++x)
        {
            int src_x = static_cast<int>(x * scale_x);
            int src_y = static_cast<int>(y * scale_y);
            float pixel = static_cast<float>(img_data[src_y * width + src_x]) / 255.0f;
            input(0, 0, y, x) = (pixel - 0.1307f) / 0.3081f;
        }
    }
    stbi_image_free(img_data);
    return input;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Uso: " << argv[0] << " <ruta_a_imagen>\n";
        return 1;
    }

    try
    {
        // Cargar modelo
        ViTConfig config = ModelUtils::load_config("models/vit_mnist_test.json");
        VisionTransformer model(config);
        ModelUtils::load_weights(model, "models/vit_mnist_test.weights");

        // Procesar imagen
        Tensor input = load_image_with_stb(argv[1]);
        Tensor probabilities = softmax_cuda(model.forward(input, false));

        // Obtener resultados
        int predicted_class = -1;
        float max_prob = 0.0f;
        for (size_t j = 0; j < probabilities.getShape()[1]; ++j)
        {
            if (probabilities(0, j) > max_prob)
            {
                max_prob = probabilities(0, j);
                predicted_class = j;
            }
        }

        // Mostrar salida en formato compacto
        std::cout << predicted_class << "\n"
                  << max_prob * 100.0f << "\n";

        // Vector de probabilidades
        for (size_t j = 0; j < probabilities.getShape()[1]; ++j)
        {
            if (j != 0)
                std::cout << " ";
            std::cout << probabilities(0, j);
        }
        std::cout << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}