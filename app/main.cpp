#include "model/Trainer.hpp"
#include "utils/DataReader.hpp"
#include "utils/ModelUtils.hpp"
#include <iostream>

Tensor softmax(const Tensor &logits, int axis = -1);

int main() {
  try {
    // --- 1. Definir Configuraciones ---
    ViTConfig model_config;
    model_config.embedding_dim = 128; // 196 
    model_config.num_layers = 8; // 6 
    model_config.num_heads = 16;
    model_config.patch_size = 14;
    model_config.num_classes = 10;
    model_config.in_channels = 1;
    model_config.mlp_hidden_dim = model_config.embedding_dim * 4;
    model_config.dropout_rate = 0.2;

    TrainerConfig train_config;
    train_config.epochs = 20;
    train_config.batch_size = 64;// 128
    train_config.learning_rate = 3e-4f;
    train_config.weight_decay = 1e-4f; // 0.01f
    train_config.lr_init = train_config.learning_rate;
    train_config.warmup_frac = 0.1f;

    // --- 2. Cargar Datos ---
    std::cout << "--- Cargando Datos de MNIST ---" << std::endl;
    
    // Para mnist y fashin
    // Entrenamiento + validación
    auto [train_data, valid_data] =
    load_csv_data_train_val("data/mnist_train.csv",
                            0.5f,   // sample_frac   → 25 % del total
                            0.80f,   // train_frac    → 80 % de ese 30%
                            0.20f,   // val_frac      → 20 % de ese 30%
                            1,
                            28,
                            28,
                            0.1307f, 0.3081f);

    // Para bloodmnist 3x28x28
    //auto train_data =
    //    load_csv_data("data/bloodmnist_train.csv", 1.00f, 3, 28, 28, 0.1307f, 0.3081f);
    //auto valid_data =
    //    load_csv_data("data/bloodmnist_val.csv", 1.00f, 3, 28, 28, 0.1307f, 0.3081f);
                            
    // --- 3. Crear Modelo y Entrenador ---
    VisionTransformer model(model_config);
    Trainer trainer(model, train_config);

    // --- 4. Ejecutar el Entrenamiento y la Evaluación ---
    trainer.train(train_data, valid_data); // test_data

    std::cout << "\n¡Entrenamiento completado!" << std::endl;

    // --- 5. Guardar el Modelo ---
    const std::string model_name = "vit_mnist_test";
    const std::string weights_path = model_name + ".weights";
    const std::string config_path = model_name + ".json";
    std::cout << "\nGuardando pesos del modelo entrenado en: " << weights_path << std::endl;
    ModelUtils::save_weights(model, weights_path);

    std::cout << "Guardando configuración del modelo en: " << config_path << std::endl;
    ModelUtils::save_config(model_config, config_path);

    std::cout << "\nProceso finalizado." << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "\nERROR CRÍTICO: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
