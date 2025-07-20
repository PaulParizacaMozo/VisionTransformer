#pragma once
#include "optimizers/Optimizer.cuh"
#include <vector>

class Adam : public Optimizer
{

private:
    // Hiperparámetros de Adam
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    long long t; // contador de pasos

    // Estados internos
    std::vector<Tensor *>
        m;                   // Primer momento
    std::vector<Tensor *> v; // Segundo momento
    bool initialized;        // Flag de inicialización

public:
    // Constructor
    Adam(float learningRate = 0.001f,
         float beta1 = 0.9f,
         float beta2 = 0.999f,
         float epsilon = 1e-8f,
         float weight_decay = 0.0f);

    // Actualización de parámetros con gradientes
    void update(std::vector<Tensor *> &parameters,
                const std::vector<Tensor *> &gradients) override;
    ~Adam()
    {
        for (auto t : m)
            delete t;
        for (auto t : v)
            delete t;
    }
};