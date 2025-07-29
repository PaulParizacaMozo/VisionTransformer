#ifndef DATA_AUGMENTATION_HPP
#define DATA_AUGMENTATION_HPP

#include "core/Tensor.hpp"
#include <random>
#include <cmath>

class DataAugmentation {
public:
    // Configuración de probabilidades (0.0 a 1.0)
    struct Config {
        float rotation_prob = 0.5f;      // 50% de probabilidad de rotar
        float translate_prob = 0.5f;     // 50% de trasladar
        float zoom_prob = 0.5f;          // 50% de hacer zoom
        float rotation_factor = 10.0f;   // ±10 grados
        float translate_factor = 0.1f;   // ±10% del ancho/alto
        float zoom_min = 0.9f;           // Zoom mínimo (90%)
        float zoom_max = 1.1f;           // Zoom máximo (110%)
    };

    DataAugmentation(const Config& cfg) : config(cfg), rng(std::random_device{}()) {}

    // Aplica aumentos aleatorios a un batch de imágenes
    Tensor apply(const Tensor& batch);

private:
    Config config;
    std::mt19937 rng;  // Motor de números aleatorios

    // Transformaciones individuales
    Tensor random_rotation(const Tensor& img);
    Tensor random_translate(const Tensor& img);
    Tensor random_zoom(const Tensor& img);
};

#endif // DATA_AUGMENTATION_HPP