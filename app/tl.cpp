#include "model/Trainer.hpp"
#include "utils/DataReader.hpp"
#include "utils/ModelUtils.hpp"
#include <iostream>

int main() {
  try {
    const std::string model_name = "vit_mnist_test_64emb_30ep";
    const std::string weights_path = "models/" + model_name + ".weights";
    const std::string config_path = "models/" + model_name + ".json";

    // --- 1. Cargar configuración y pesos del modelo entrenado ---
    std::cout << "Cargando configuración desde: " << config_path << std::endl;
    ViTConfig model_config = ModelUtils::load_config(config_path);

    std::cout << "Cargando modelo..." << std::endl;
    VisionTransformer model(model_config);
    ModelUtils::load_weights(model, weights_path);
    std::cout << "Pesos cargados correctamente.\n";

    // --- 2. Configurar nuevo entrenamiento (fine-tuning) ---
    TrainerConfig train_config;
    train_config.epochs = 10;              // ➜ nuevas épocas de fine-tuning
    train_config.batch_size = 64;
    train_config.learning_rate = 5e-5f;    // más bajo que el inicial
    train_config.weight_decay = 1e-4f;
    train_config.lr_init = train_config.learning_rate;
    train_config.warmup_frac = 0.1f;

    // --- 3. Cargar datos de entrenamiento y validación ---
    std::cout << "Cargando datos MNIST..." << std::endl;
    auto [train_data, valid_data] =
        load_csv_data_train_val("data/mnist_train.csv",
                                0.5f,   // sample_frac (si quieres reducir)
                                0.80f,  // train_frac
                                0.20f,  // val_frac
                                1, 28, 28,
                                model_config.num_classes,
                                0.1307f, 0.3081f);

    // --- 4. Crear entrenador y continuar entrenamiento ---
    Trainer trainer(model, train_config);
    trainer.train(train_data, valid_data);

    // --- 5. Guardar modelo fine-tuned ---
    const std::string new_model_name = model_name + "_finetuned";
    const std::string new_weights_path = "models/" + new_model_name + ".weights";
    const std::string new_config_path = "models/" + new_model_name + ".json";

    std::cout << "Guardando nuevo modelo en: " << new_weights_path << std::endl;
    ModelUtils::save_weights(model, new_weights_path);
    ModelUtils::save_config(model_config, new_config_path);

    std::cout << "\nFine-tuning finalizado.\n";

  } catch (const std::exception &e) {
    std::cerr << "\nERROR CRÍTICO: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
