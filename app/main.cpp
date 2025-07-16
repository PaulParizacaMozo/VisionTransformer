#include "model/Trainer.hpp"
#include "utils/DataReader.hpp"
#include "utils/ModelUtils.hpp"
#include <iostream>

Tensor softmax(const Tensor &logits, int axis = -1);

int main() {
  try {
    // --- 1. Definir Configuraciones ---
    ViTConfig model_config;
    model_config.embedding_dim = 64;
    model_config.num_layers = 1;
    model_config.num_heads = 4;
    model_config.patch_size = 7;
    model_config.mlp_hidden_dim = 128;

    TrainerConfig train_config;
    train_config.epochs = 50;
    train_config.batch_size = 64;
    train_config.learning_rate = 0.000001f;//2e-4f; // 0.0001f;

    // --- 2. Cargar Datos ---
    std::cout << "--- Cargando Datos de Fashion MNIST ---" << std::endl;
    auto train_data = load_csv_data("data/mnist_train.csv", 0.1f);
    auto test_data = load_csv_data("data/mnist_test.csv", 0.1f);

    // --- 3. Crear Modelo y Entrenador ---
    VisionTransformer model(model_config);
    Trainer trainer(model, train_config);

    // --- 4. Ejecutar el Entrenamiento y la Evaluación ---
    trainer.train(train_data, test_data);

    std::cout << "\n¡Entrenamiento completado!" << std::endl;

    // --- 5. Guardar el Modelo ---
    const std::string weights_path = "vit_fashion_mnist.weights.test";
    std::cout << "\nGuardando pesos del modelo entrenado en: " << weights_path << std::endl;
    ModelUtils::save_weights(model, weights_path);

    std::cout << "\nProceso finalizado." << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "\nERROR CRÍTICO: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
