#include "core/Tensor.hpp"
#include <cmath>
#include <random>
#include <iostream>

// Random Crop
Tensor random_crop(const Tensor& image, size_t crop_size, size_t padding = 0) {
    size_t height = image.getShape()[2];
    size_t width = image.getShape()[3];

    if (height <= crop_size || width <= crop_size) {
        throw std::invalid_argument("Crop size must be smaller than image size.");
    }

    // Añadir padding si es necesario
    size_t pad_top = padding;
    size_t pad_left = padding;
    size_t pad_bottom = height - crop_size - pad_top;
    size_t pad_right = width - crop_size - pad_left;

    // Establecer límites aleatorios para el recorte
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_top(0, pad_bottom);
    std::uniform_int_distribution<> dis_left(0, pad_right);

    size_t top = dis_top(gen);
    size_t left = dis_left(gen);

    return image.slice(2, top, crop_size).slice(3, left, crop_size); // Recorta la imagen
}

// Random Horizontal Flip
Tensor random_flip(const Tensor& image, float prob = 0.5f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    if (dis(gen) < prob) {
        // Flip Horizontal: cambiar el orden de las columnas
        size_t height = image.getShape()[2];
        size_t width = image.getShape()[3];

        Tensor flipped({image.getShape()[0], image.getShape()[1], height, width});

        for (size_t b = 0; b < image.getShape()[0]; ++b) {
            for (size_t c = 0; c < image.getShape()[1]; ++c) {
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        flipped(b, c, h, w) = image(b, c, h, width - w - 1);
                    }
                }
            }
        }

        return flipped;
    }

    return image;  // Si no se aplica flip, devuelve la imagen original
}


// Rotación aleatoria de la imagen en grados entre [-max_angle, max_angle]
Tensor random_rotation(const Tensor& image, float max_angle = 30.0f) {
    size_t height = image.getShape()[2];
    size_t width = image.getShape()[3];

    // Generar un ángulo aleatorio en el rango [-max_angle, max_angle]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-max_angle, max_angle);
    float angle = dis(gen);

    // Convertir el ángulo a radianes
    float radians = angle * M_PI / 180.0f;

    // Matriz de rotación
    float cos_theta = std::cos(radians);
    float sin_theta = std::sin(radians);

    // Nueva imagen rotada
    Tensor rotated_image = Tensor(image.getShape());  // Crear un tensor para la imagen rotada

    // Iterar sobre los píxeles de la imagen y aplicar la rotación
    for (size_t b = 0; b < image.getShape()[0]; ++b) {
        for (size_t c = 0; c < image.getShape()[1]; ++c) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    // Calcular las nuevas coordenadas (x', y') para la rotación
                    float x_new = cos_theta * (w - width / 2.0f) - sin_theta * (h - height / 2.0f) + width / 2.0f;
                    float y_new = sin_theta * (w - width / 2.0f) + cos_theta * (h - height / 2.0f) + height / 2.0f;

                    // Verificar si la nueva coordenada está dentro de los límites de la imagen
                    if (x_new >= 0 && x_new < width && y_new >= 0 && y_new < height) {
                        rotated_image(b, c, h, w) = image(b, c, (size_t)y_new, (size_t)x_new);
                    } else {
                        rotated_image(b, c, h, w) = 0.0f;  // Rellenar con 0 (negro) si está fuera del rango
                    }
                }
            }
        }
    }

    return rotated_image;
}

// Función para aplicar una traslación aleatoria a la imagen
Tensor random_translation(const Tensor& image, size_t max_translation = 4) {
    size_t height = image.getShape()[2];
    size_t width = image.getShape()[3];

    // Generar desplazamientos aleatorios para las direcciones X e Y
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-max_translation, max_translation);
    int tx = dis(gen);  // Desplazamiento en X
    int ty = dis(gen);  // Desplazamiento en Y

    // Nueva imagen para la traducción
    Tensor translated_image = Tensor(image.getShape());  // Crear un tensor para la imagen traslada

    // Rellenar la imagen con 0.0 (negro)
    translated_image.fill(0.0f);

    // Iterar sobre los píxeles de la imagen original y trasladarlos a las nuevas posiciones
    for (size_t b = 0; b < image.getShape()[0]; ++b) {
        for (size_t c = 0; c < image.getShape()[1]; ++c) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    // Nueva posición con la traslación aplicada
                    int new_h = h + ty;
                    int new_w = w + tx;

                    // Asegurarse de que las nuevas coordenadas están dentro de los límites
                    if (new_h >= 0 && new_h < height && new_w >= 0 && new_w < width) {
                        translated_image(b, c, new_h, new_w) = image(b, c, h, w);
                    }
                }
            }
        }
    }

    return translated_image;
}

Tensor random_zoom(const Tensor& image, float min_zoom = 0.8f, float max_zoom = 1.2f) {
    size_t height = image.getShape()[2];
    size_t width = image.getShape()[3];

    // Generar un factor de zoom aleatorio en el rango [min_zoom, max_zoom]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min_zoom, max_zoom);
    float zoom_factor = dis(gen);

    // Calcular las nuevas dimensiones (escala de zoom)
    size_t new_height = static_cast<size_t>(height * zoom_factor);
    size_t new_width = static_cast<size_t>(width * zoom_factor);

    // Si el nuevo tamaño es más pequeño que el original, usaremos relleno.
    size_t pad_top = (height - new_height) / 2;
    size_t pad_left = (width - new_width) / 2;

    // Si la imagen se amplía, necesitamos rellenar con ceros (negro)
    Tensor zoomed_image({image.getShape()[0], image.getShape()[1], height, width});  // Imagen con el tamaño original
    zoomed_image.fill(0.0f);

    // Iterar sobre los píxeles de la imagen original y hacer el zoom
    for (size_t b = 0; b < image.getShape()[0]; ++b) {
        for (size_t c = 0; c < image.getShape()[1]; ++c) {
            for (size_t h = 0; h < new_height; ++h) {
                for (size_t w = 0; w < new_width; ++w) {
                    // Mapear las coordenadas de la imagen original a la imagen con zoom
                    int orig_h = static_cast<int>((h - pad_top) / zoom_factor);
                    int orig_w = static_cast<int>((w - pad_left) / zoom_factor);

                    if (orig_h >= 0 && orig_h < height && orig_w >= 0 && orig_w < width) {
                        zoomed_image(b, c, h + pad_top, w + pad_left) = image(b, c, orig_h, orig_w);
                    }
                }
            }
        }
    }

    return zoomed_image;
}
