// aun no está listo
#include "model/Trainer.hpp"
#include "utils/DataReader.hpp"
#include "utils/ModelUtils.hpp"
#include <iostream>
#include <vector>

int main() {
    try {
        // --- 1. Crear Modelo ---
        ViTConfig model_config;
        model_config.embedding_dim = 64;
        model_config.num_layers = 3;
        model_config.num_heads = 8;
        model_config.patch_size = 7;
        model_config.mlp_hidden_dim = model_config.embedding_dim * 16;

        VisionTransformer model(model_config);

        // Cargar datos de prueba
        auto test_data = load_csv_data("data/mnist_test.csv", 1.00f, 0.1307f, 0.3081f);

        // --- 2. Cargar pesos del Modelo ---
        const std::string weights_path = "vit_mnist.weights.test";
        std::cout << "\nCargando pesos del modelo en: " << weights_path << std::endl;
        ModelUtils::load_weights(model, weights_path);
        std::cout << "\nPesos cargados correctamente.\n";

        // --- 3. Hacer predicciones ---
        // Obtener los datos de entrada y etiquetas del conjunto de prueba
        const Tensor& X_test = test_data.first;
        const Tensor& y_test = test_data.second;

        // Realizar un forward pass para obtener los logits
        Tensor logits = model.forward(X_test, false); // `isTraining` es `false` durante la inferencia

        // Aplicar softmax para obtener las probabilidades
        Tensor probabilities = softmax(logits);

        // Realizar predicciones
        size_t batch_size = probabilities.getShape()[0];
        for (size_t i = 0; i < batch_size; ++i) {
            // Obtener la clase predicha (la clase con la mayor probabilidad)
            float max_prob = -std::numeric_limits<float>::infinity();
            int predicted_class = -1;

            for (size_t j = 0; j < probabilities.getShape()[1]; ++j) {
                if (probabilities(i, j) > max_prob) {
                    max_prob = probabilities(i, j);
                    predicted_class = j;
                }
            }

            // Imprimir la predicción y la etiqueta verdadera
            std::cout << "Sample " << i << " | Predicción: " << predicted_class 
                      << " | Etiqueta: " << y_test(i, predicted_class) << std::endl;
        }

    } catch (const std::exception &e) {
        std::cerr << "\nERROR CRÍTICO: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
