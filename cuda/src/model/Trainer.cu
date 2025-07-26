#include "model/Trainer.cuh"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

__global__ void accuracy_kernel(const float *logits, const float *labels, size_t batch_size, size_t num_classes, int *correct_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
        return;

    float max_logit = -INFINITY;
    int pred_class = -1;

    // Encontrar la clase predicha (argmax)
    for (size_t j = 0; j < num_classes; ++j)
    {
        float val = logits[idx * num_classes + j];
        if (val > max_logit)
        {
            max_logit = val;
            pred_class = j;
        }
    }

    // Comprobar si la predicción es correcta (labels en formato one-hot)
    if (labels[idx * num_classes + pred_class] == 1.0f)
    {
        atomicAdd(correct_count, 1); // acceso atómico
    }
}

// --- Función Auxiliar ---
namespace
{
    float calculate_accuracy(const Tensor &logits, const Tensor &labels)
    {
        size_t batch_size = logits.dim(0);
        size_t num_classes = logits.dim(1);
        if (batch_size == 0 || num_classes == 0)
            return 0.0f;

        int *d_correct_count;
        int h_correct_count = 0;
        cudaMalloc(&d_correct_count, sizeof(int));
        cudaMemset(d_correct_count, 0, sizeof(int));

        const float *logits_ptr = logits.getData();
        const float *labels_ptr = labels.getData();

        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        cudaGetLastError();
        accuracy_kernel<<<blocks, threads>>>(logits_ptr, labels_ptr, batch_size, num_classes, d_correct_count);
        cudaMemcpy(&h_correct_count, d_correct_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_correct_count);
        return static_cast<float>(h_correct_count) / batch_size;
    }
}

// --- Constructor ---
Trainer::Trainer(VisionTransformer &model, const TrainerConfig &train_config)
    : model(model),
      optimizer(train_config.learning_rate, 0.9f, 0.999f, 1e-8f, train_config.weight_decay),
      loss_fn(),
      config(train_config)
{
}

// --- Entrenamiento ---
void Trainer::train(const std::pair<Tensor, Tensor> &train_data, const std::pair<Tensor, Tensor> &test_data)
{
    const auto &[X_train, y_train] = train_data;
    const auto &[X_test, y_test] = test_data;

    for (int epoch = 0; epoch < config.epochs; ++epoch)
    {
        std::cout << ">> Train antes" << std::endl;
        auto [train_loss, train_acc] = train_epoch(X_train, y_train);
        std::cout << ">> Train después" << std::endl;
        std::cout << "\r" << std::string(80, ' ') << "\r";

        auto [test_loss, test_acc] = evaluate(X_test, y_test);

        std::cout << "--- Época " << epoch + 1 << "/" << config.epochs
                  << " | Train Loss: " << std::fixed << std::setprecision(4) << train_loss
                  << " | Train Acc: " << train_acc
                  << " | Test Loss: " << test_loss
                  << " | Test Acc: " << test_acc
                  << std::endl;
    }
}

__global__ void copy_sample_to_batch_kernel(
    const float *X_train, const float *y_train,
    float *X_batch, float *y_batch,
    size_t image_size, size_t num_classes,
    size_t src_idx, size_t dst_idx)
{
    int idx = threadIdx.x;

    // Copiar imagen (28x28 = 784 elementos)
    if (idx < image_size)
    {
        X_batch[dst_idx * image_size + idx] = X_train[src_idx * image_size + idx];
    }

    // Copiar etiqueta one-hot (10 elementos)
    if (idx < num_classes)
    {
        y_batch[dst_idx * num_classes + idx] = y_train[src_idx * num_classes + idx];
    }
}

// --- Época ---
std::pair<float, float> Trainer::train_epoch(const Tensor &X_train, const Tensor &y_train)
{
    size_t num_train_samples = X_train.dim(0);
    size_t num_batches = (num_train_samples + config.batch_size - 1) / config.batch_size;

    float total_loss = 0.0f;
    float total_accuracy = 0.0f;

    std::vector<size_t> indices(num_train_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::srand(static_cast<unsigned int>(std::time(nullptr) + config.epochs));
    std::random_shuffle(indices.begin(), indices.end());

    for (size_t i = 0; i < num_batches; ++i)
    {
        size_t start_idx_in_indices = i * config.batch_size;
        size_t count = std::min(config.batch_size, num_train_samples - start_idx_in_indices);
        if (count == 0)
            continue;

        // Crear los tensores para el batch actual
        Tensor X_batch({count, 1, 28, 28});
        Tensor y_batch({count, 10});

        // Llenar el batch con los datos correspondientes a los índices barajados
        size_t image_size = 28 * 28;
        size_t num_classes = 10;

        for (size_t j = 0; j < count; ++j)
        {
            size_t src_idx = indices[start_idx_in_indices + j];
            size_t dst_idx = j;

            // Lanzamos 784 hilos, suficiente para imagen + etiqueta
            int threads = 784;
            cudaGetLastError();
            copy_sample_to_batch_kernel<<<1, threads>>>(
                X_train.getData(), y_train.getData(),
                X_batch.getData(), y_batch.getData(),
                image_size, num_classes, src_idx, dst_idx);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("Error tras lanzar kernel");
            }
            cudaDeviceSynchronize();
        }

        // --- Ciclo de entrenamiento para el batch ---
        Tensor logits = model.forward(X_batch, true);
        logits.printContents("Logits del batch");
        total_loss += loss_fn.calculate(logits, y_batch);
        logits.printContents("Logits del batch tras calcular pérdida");
        total_accuracy += calculate_accuracy(logits, y_batch);
        logits.printContents("Logits del batch tras calcular precisión");

        Tensor grad = loss_fn.backward(logits, y_batch);
        grad.printContents("Gradiente tras backward");
        model.backward(grad);

        auto params = model.getParameters();
        auto grads = model.getGradients();
        optimizer.update(params, grads);

        std::cout << "\rEntrenando... Batch " << i + 1 << "/" << num_batches << " " << std::flush;
    }

    return {total_loss / num_batches, total_accuracy / num_batches};
}

// --- Evaluación ---
std::pair<float, float> Trainer::evaluate(const Tensor &X_test, const Tensor &y_test)
{
    size_t num_test_samples = X_test.dim(0);
    size_t num_batches = (num_test_samples + config.batch_size - 1) / config.batch_size;

    float total_loss = 0.0f;
    float total_accuracy = 0.0f;

    size_t height = X_test.dim(2);
    size_t width = X_test.dim(3);
    size_t num_classes = y_test.dim(1);
    size_t image_size = height * width;

    for (size_t i = 0; i < num_batches; ++i)
    {
        size_t start = i * config.batch_size;
        size_t count = std::min(config.batch_size, num_test_samples - start);
        if (count == 0)
            continue;

        Tensor X_batch = X_test.slice(0, start, count);
        Tensor y_batch = y_test.slice(0, start, count);

        // Forward pass en modo inferencia
        Tensor logits = model.forward(X_batch, false);

        // Calcular pérdida y precisión para el batch
        total_loss += loss_fn.calculate(logits, y_batch);
        total_accuracy += calculate_accuracy(logits, y_batch);
    }

    return {total_loss / num_batches, total_accuracy / num_batches};
}
