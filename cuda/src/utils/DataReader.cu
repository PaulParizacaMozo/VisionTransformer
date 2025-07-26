#include "utils/DataReader.cuh"
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

__global__ void oneHotKernel(const int *labels, float *output, int num_samples, int num_classes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples)
    {
        int label = labels[idx];
        if (label >= 0 && label < num_classes)
        {
            output[idx * num_classes + label] = 1.0f;
        }
    }
}

namespace
{
    Tensor oneHotEncode(const std::vector<int> &labels, int num_classes)
    {
        const size_t num_samples = labels.size();
        const size_t numel = num_samples * num_classes;

        // Crear tensor destino en GPU
        Tensor one_hot({num_samples, static_cast<size_t>(num_classes)});

        // Copiar etiquetas a GPU
        int *d_labels = nullptr;
        cudaMalloc(&d_labels, num_samples * sizeof(int));
        cudaMemcpy(d_labels, labels.data(), num_samples * sizeof(int), cudaMemcpyHostToDevice);

        // Lanzar kernel
        int threads = 256;
        int blocks = (num_samples + threads - 1) / threads;
        oneHotKernel<<<blocks, threads>>>(d_labels, one_hot.getData(), num_samples, num_classes);
        cudaDeviceSynchronize();

        cudaFree(d_labels);
        return one_hot;
    }
} // namespace

std::pair<Tensor, Tensor> load_csv_data(const std::string &filePath, float sample_fraction)
{
    std::cout << "Cargando datos desde: " << filePath << " (fracción a cargar: "
              << sample_fraction * 100 << "%)" << std::endl;

    std::ifstream file(filePath);
    if (!file.is_open())
    {
        throw std::runtime_error("Error: No se pudo abrir el archivo: " + filePath);
    }

    std::string line;
    std::getline(file, line); // Saltar cabecera

    std::vector<std::vector<float>> all_pixel_data;
    std::vector<int> all_labels;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, ',');
        all_labels.push_back(std::stoi(value));

        std::vector<float> pixels;
        pixels.reserve(784);
        while (std::getline(ss, value, ','))
        {
            pixels.push_back(std::stof(value) / 255.0f);
        }

        if (pixels.size() != 784)
        {
            std::cerr << "Fila inválida con " << pixels.size() << " píxeles. Ignorada." << std::endl;
            all_labels.pop_back();
            continue;
        }

        all_pixel_data.push_back(std::move(pixels));
    }

    file.close();

    // Mezclar y muestrear
    size_t total_samples = all_labels.size();
    std::vector<size_t> indices(total_samples);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
    std::shuffle(indices.begin(), indices.end(), rng);

    size_t samples_to_load = static_cast<size_t>(total_samples * sample_fraction);
    if (samples_to_load == 0 && total_samples > 0)
        samples_to_load = 1;
    if (samples_to_load > total_samples)
        samples_to_load = total_samples;

    std::vector<float> final_pixel_data;
    final_pixel_data.reserve(samples_to_load * 784);
    std::vector<int> final_labels;
    final_labels.reserve(samples_to_load);

    for (size_t i = 0; i < samples_to_load; ++i)
    {
        size_t idx = indices[i];
        final_pixel_data.insert(final_pixel_data.end(),
                                all_pixel_data[idx].begin(),
                                all_pixel_data[idx].end());
        final_labels.push_back(all_labels[idx]);
    }

    // // mostrar tres primeras muestras
    // std::cout << "Primeras 3 muestras cargadas:" << std::endl;
    // for (size_t i = 0; i < std::min(samples_to_load, static_cast<size_t>(3)); ++i)
    // {
    //     std::cout << "Etiqueta: " << final_labels[i] << "\n";
    //     std::cout << "Imagen:\n";
    //     for (size_t j = 0; j < 28; ++j)
    //     {
    //         for (size_t k = 0; k < 28; ++k)
    //         {
    //             std::cout << (final_pixel_data[i * 784 + j * 28 + k] > 0.5f ? '#' : '.');
    //         }
    //         std::cout << '\n';
    //     }
    // }

    // --- Crear Tensor X (directamente en GPU) ---
    Tensor X({samples_to_load, 1, 28, 28});
    X.copyFromHost(final_pixel_data.data());

    // One-hot labels
    Tensor y = oneHotEncode(final_labels, 10);

    std::cout << "Carga completa. " << samples_to_load << " muestras cargadas." << std::endl;

    return {std::move(X), std::move(y)};
}
