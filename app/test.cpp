#include "model/Trainer.hpp"
#include "utils/DataReader.hpp"
#include "utils/ModelUtils.hpp"
#include <iostream>
#include <vector>

int main() {
    try {
        // --- 1. Crear Modelo ---
        ViTConfig model_config;
        model_config.embedding_dim = 196;
        model_config.num_layers = 2;
        model_config.num_heads = 4;
        model_config.patch_size = 7;
        model_config.mlp_hidden_dim = model_config.embedding_dim * 4;

        VisionTransformer model(model_config);

        // Cargar datos de prueba
        auto test_data = load_csv_data("data/mnist_test.csv", 1.00f, 0.1307f, 0.3081f);

        // --- 2. Cargar pesos del Modelo ---
        const std::string weights_path = "vit_fashion_mnist.weights.30.ep";
        std::cout << "\nCargando pesos del modelo en: " << weights_path << std::endl;
        ModelUtils::load_weights(model, weights_path);
        std::cout << "\nPesos cargados correctamente.\n";

        // --- 3. Hacer predicciones ---
        const Tensor& X_test = test_data.first;
        const Tensor& y_test = test_data.second;

        Tensor logits = model.forward(X_test, false); // `isTraining` es `false` durante la inferencia
        Tensor probabilities = softmax(logits);

        // Calcular accuracy
        size_t batch_size = probabilities.getShape()[0];
        int correct_predictions = 0;
        int total_samples = 0;

        for (size_t i = 0; i < batch_size; ++i) {
            float max_prob = -std::numeric_limits<float>::infinity();
            int predicted_class = -1;

            for (size_t j = 0; j < probabilities.getShape()[1]; ++j) {
                if (probabilities(i, j) > max_prob) {
                    max_prob = probabilities(i, j);
                    predicted_class = j;
                }
            }

            int true_label = -1;
            for (size_t j = 0; j < y_test.getShape()[1]; ++j) {
                if (y_test(i, j) == 1.0f) {
                    true_label = j;
                    break;
                }
            }

            std::cout << "Sample " << i << " | Predicción: " << predicted_class 
                      << " | Etiqueta: " << true_label << std::endl;

            if (predicted_class == true_label) {
                ++correct_predictions;
            }
            ++total_samples;
        }

        float accuracy = static_cast<float>(correct_predictions) / total_samples;
        std::cout << "\nAccuracy del modelo: " << accuracy * 100.0f << "%" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "\nERROR CRÍTICO: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
