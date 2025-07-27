#include "utils/DataAugmentation.hpp"
#include <algorithm>

Tensor DataAugmentation::apply(const Tensor& batch) {
    size_t batch_size = batch.getShape()[0];
    Tensor augmented_batch = batch.clone();  // Copia del batch original

    for (size_t i = 0; i < batch_size; ++i) {
        Tensor img = batch.slice(0, i, 1);  // Extrae la i-ésima imagen

        // Aplicar transformaciones aleatorias (cada una con su probabilidad)
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        if (dist(rng) < config.rotation_prob) {
            img = random_rotation(img);
        }
        if (dist(rng) < config.translate_prob) {
            img = random_translate(img);
        }
        if (dist(rng) < config.zoom_prob) {
            img = random_zoom(img);
        }

        // Guardar la imagen aumentada
        for (size_t c = 0; c < img.getShape()[1]; ++c) {
            for (size_t h = 0; h < img.getShape()[2]; ++h) {
                for (size_t w = 0; w < img.getShape()[3]; ++w) {
                    augmented_batch(i, c, h, w) = img(0, c, h, w);
                }
            }
        }
    }

    return augmented_batch;
}

Tensor DataAugmentation::random_rotation(const Tensor& img) {
    std::uniform_real_distribution<float> angle_dist(
        -config.rotation_factor, 
        config.rotation_factor
    );
    float angle = angle_dist(rng);

    // Convertir ángulo a radianes
    float rad = angle * M_PI / 180.0f;
    float cos_theta = std::cos(rad);
    float sin_theta = std::sin(rad);

    size_t height = img.getShape()[2];
    size_t width = img.getShape()[3];
    Tensor rotated_img = img.clone();

    float center_x = width / 2.0f;
    float center_y = height / 2.0f;

    for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
            // Coordenadas relativas al centro
            float x = w - center_x;
            float y = h - center_y;

            // Rotación inversa (para interpolación)
            float src_x = cos_theta * x + sin_theta * y + center_x;
            float src_y = -sin_theta * x + cos_theta * y + center_y;

            // Interpolación bilineal (o nearest-neighbor)
            if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
                for (size_t c = 0; c < img.getShape()[1]; ++c) {
                    rotated_img(0, c, h, w) = img(0, c, (size_t)src_y, (size_t)src_x);
                }
            } else {
                for (size_t c = 0; c < img.getShape()[1]; ++c) {
                    rotated_img(0, c, h, w) = 0.0f;  // Rellenar con 0 (fondo)
                }
            }
        }
    }

    return rotated_img;
}

Tensor DataAugmentation::random_translate(const Tensor& img) {
    std::uniform_real_distribution<float> dist(
        -config.translate_factor, 
        config.translate_factor
    );
    float tx = dist(rng) * img.getShape()[3];  // Desplazamiento en X
    float ty = dist(rng) * img.getShape()[2];  // Desplazamiento en Y

    size_t height = img.getShape()[2];
    size_t width = img.getShape()[3];
    Tensor translated_img = img.clone();

    for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
            int src_h = (int)h - (int)ty;
            int src_w = (int)w - (int)tx;

            if (src_h >= 0 && src_h < height && src_w >= 0 && src_w < width) {
                for (size_t c = 0; c < img.getShape()[1]; ++c) {
                    translated_img(0, c, h, w) = img(0, c, src_h, src_w);
                }
            } else {
                for (size_t c = 0; c < img.getShape()[1]; ++c) {
                    translated_img(0, c, h, w) = 0.0f;  // Rellenar con 0
                }
            }
        }
    }

    return translated_img;
}

Tensor DataAugmentation::random_zoom(const Tensor& img) {
    std::uniform_real_distribution<float> zoom_dist(config.zoom_min, config.zoom_max);
    float zoom = zoom_dist(rng);

    size_t height = img.getShape()[2];
    size_t width = img.getShape()[3];
    Tensor zoomed_img = img.clone();

    float center_x = width / 2.0f;
    float center_y = height / 2.0f;

    for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
            float x = (w - center_x) / zoom + center_x;
            float y = (h - center_y) / zoom + center_y;

            if (x >= 0 && x < width && y >= 0 && y < height) {
                for (size_t c = 0; c < img.getShape()[1]; ++c) {
                    zoomed_img(0, c, h, w) = img(0, c, (size_t)y, (size_t)x);
                }
            } else {
                for (size_t c = 0; c < img.getShape()[1]; ++c) {
                    zoomed_img(0, c, h, w) = 0.0f;  // Rellenar con 0
                }
            }
        }
    }

    return zoomed_img;
}