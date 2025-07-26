#pragma once

#include "losses/CrossEntropy.cuh"
#include "model/VisionTransformer.cuh"
#include "optimizers/Adam.cuh"
#include "utils/DataReader.cuh"

#include <vector>
#include <utility>

/**
 * @struct TrainerConfig
 * @brief Hiperparámetros para el entrenamiento.
 */
struct TrainerConfig
{
    int epochs = 10;
    size_t batch_size = 64;
    float learning_rate = 0.001f;
    float weight_decay = 0.01f;
};

/**
 * @class Trainer
 * @brief Clase encargada de entrenar un modelo VisionTransformer en GPU.
 */
class Trainer
{
private:
    VisionTransformer &model;
    Adam optimizer;
    CrossEntropy loss_fn;
    TrainerConfig config;

    std::pair<float, float> train_epoch(const Tensor &X_train, const Tensor &y_train);
    std::pair<float, float> evaluate(const Tensor &X_test, const Tensor &y_test);

public:
    explicit Trainer(VisionTransformer &model, const TrainerConfig &train_config);

    void train(const std::pair<Tensor, Tensor> &train_data, const std::pair<Tensor, Tensor> &test_data);

    const VisionTransformer &getModel() const { return model; }
    VisionTransformer &getModel() { return model; }
};
