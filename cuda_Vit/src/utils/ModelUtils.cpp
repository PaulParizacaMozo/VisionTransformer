#include "utils/ModelUtils.hpp"
#include "utils/CudaUtils.cuh"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <stdexcept>
#include "utils/json.hpp"

using json = nlohmann::json;

namespace ModelUtils
{

  void save_weights(const VisionTransformer &model, const std::string &filePath, bool train)
  {
    std::ofstream outFile(filePath, std::ios::binary);
    if (!outFile)
    {
      throw std::runtime_error("No se pudo abrir el archivo para escritura: " + filePath);
    }

    // --- Uso de const_cast ---
    // 'model' es const, pero getParameters() no lo es.
    // Usamos const_cast para eliminar temporalmente la constancia y poder llamar a la función.
    // Esto es seguro porque sabemos que getParameters() no modifica el estado del modelo.
    VisionTransformer &non_const_model = const_cast<VisionTransformer &>(model);
    auto params = non_const_model.getParameters();
    if (train == false)
      std::cout << "Guardando " << params.size() << " tensores de parámetros en " << filePath << "..." << std::endl;

    for (const auto &tensor_ptr : params)
    {
      // El puntero en sí no es const, pero lo tratamos como tal para la lectura.
      const Tensor &tensor = *tensor_ptr;
      const auto &shape = tensor.getShape();
      size_t rank = shape.size();
      size_t num_elements = tensor.getSize();

      // 1. Escribir el número de dimensiones (rank)
      outFile.write(reinterpret_cast<const char *>(&rank), sizeof(size_t));

      // 2. Escribir las dimensiones de la forma
      outFile.write(reinterpret_cast<const char *>(shape.data()), rank * sizeof(size_t));

      // 3. Escribir los datos del tensor
      // Si el tensor no es contiguo, creamos una copia temporal para guardar.
      if (!tensor.isContiguous())
      {
        if (train == false)
          std::cerr << "Advertencia: Guardando un tensor no contiguo. Se creará una copia temporal." << std::endl;
        Tensor temp = contiguous_cuda(tensor);
        // Tensor temp = tensor.contiguous();
        // if (verify(temp, temp_cuda, 1e-5f) == false)
        // {
        //   std::cerr << "Error en la verificación de contiguous para guardar pesos." << std::endl;
        // }
        outFile.write(reinterpret_cast<const char *>(temp.getData()), num_elements * sizeof(float));
      }
      else
      {
        // Accedemos a los datos teniendo en cuenta el offset por si es una vista.
        outFile.write(reinterpret_cast<const char *>(tensor.getData() + tensor.getDataOffset()), num_elements * sizeof(float));
      }
    }

    outFile.close();
    if (train == false)
      std::cout << "Pesos guardados correctamente." << std::endl;
  }

  void load_weights(VisionTransformer &model, const std::string &filePath)
  {
    std::ifstream inFile(filePath, std::ios::binary);
    if (!inFile)
    {
      throw std::runtime_error("No se pudo abrir el archivo para lectura: " + filePath);
    }

    // Aquí 'model' no es const, así que podemos llamar a getParameters() directamente.
    auto params = model.getParameters();

    std::cout << "Cargando " << params.size() << " tensores de parámetros desde " << filePath << "..." << std::endl;

    for (auto &tensor_ptr : params)
    {
      Tensor &tensor = *tensor_ptr;

      size_t file_rank;
      inFile.read(reinterpret_cast<char *>(&file_rank), sizeof(size_t));

      std::vector<size_t> file_shape(file_rank);
      inFile.read(reinterpret_cast<char *>(file_shape.data()), file_rank * sizeof(size_t));

      if (tensor.getShape() != file_shape)
      {
        throw std::runtime_error("Incompatibilidad de formas al cargar pesos. Esperado: " + tensor.shapeToString() +
                                 ", encontrado en archivo: " + Tensor(file_shape).shapeToString());
      }

      size_t num_elements = tensor.getSize();
      if (!tensor.isContiguous())
      {
        // Para cargar en un tensor no contiguo, necesitamos leer a un buffer temporal
        // y luego copiar elemento por elemento.
        std::vector<float> buffer(num_elements);
        inFile.read(reinterpret_cast<char *>(buffer.data()), num_elements * sizeof(float));

        // Copia manual elemento por elemento (requeriría un iterador N-D o bucles anidados)
        // Por ahora, lanzamos un error como medida de seguridad.
        throw std::runtime_error("Cargar pesos a un tensor no contiguo no está implementado aún.");
      }
      else
      {
        inFile.read(reinterpret_cast<char *>(tensor.getData() + tensor.getDataOffset()), num_elements * sizeof(float));
      }

      if (static_cast<size_t>(inFile.gcount()) != num_elements * sizeof(float))
      {
        throw std::runtime_error("Error de lectura: fin de archivo inesperado o datos corruptos.");
      }
    }

    inFile.close();
    std::cout << "Pesos cargados correctamente." << std::endl;
  }

  void save_config(const ViTConfig &config, const std::string &filePath, bool train)
  {
    if (train == false)
      std::cout << "Guardando configuración del modelo en: " << filePath << "..." << std::endl;

    json j = config; // Conversión automática gracias a nuestra función to_json

    std::ofstream outFile(filePath);
    if (!outFile)
    {
      throw std::runtime_error("No se pudo abrir el archivo para escritura: " + filePath);
    }

    // Escribimos el JSON en el archivo con una indentación de 4 espacios para que sea legible
    outFile << std::setw(4) << j << std::endl;
    outFile.close();

    if (train == false)
      std::cout << "Configuración guardada correctamente." << std::endl;
  }

  ViTConfig load_config(const std::string &filePath)
  {
    std::cout << "Cargando configuración del modelo desde: " << filePath << "..." << std::endl;

    std::ifstream inFile(filePath);
    if (!inFile)
    {
      throw std::runtime_error("No se pudo abrir el archivo para lectura: " + filePath);
    }

    json j;
    inFile >> j; // Se parsea el archivo JSON

    // La conversión a ViTConfig
    ViTConfig config = j.get<ViTConfig>();

    std::cout << "Configuración cargada correctamente." << std::endl;
    return config;
  }

  void print_hyperparameters_box(const ViTConfig &model_config, const TrainerConfig &train_config)
  {
    const int name_width = 30;
    const int value_width = 20;

    auto print_row = [&](const std::string &name, const std::string &value)
    {
      std::cout << "║ " << std::left << std::setw(name_width) << name
                << " : " << std::right << std::setw(value_width) << value << " ║\n";
    };

    std::string border(name_width + value_width + 5, '=');

    std::cout << "\n╔" << border << "╗\n";
    std::cout << "║" << std::setw(name_width + value_width + 5) << std::left
              << "            CONFIGURACIÓN DE HIPERPARÁMETROS           " << "║\n";
    std::cout << "╠" << border << "╣\n";

    // Modelo
    print_row("Embedding dim", std::to_string(model_config.embedding_dim));
    print_row("Nro de capas", std::to_string(model_config.num_layers));
    print_row("Nro de heads", std::to_string(model_config.num_heads));
    print_row("Patch size", std::to_string(model_config.patch_size));
    print_row("Nro de clases", std::to_string(model_config.num_classes));
    print_row("Canales de entrada", std::to_string(model_config.in_channels));
    print_row("Dim MLP oculta", std::to_string(model_config.mlp_hidden_dim));
    print_row("Dropout rate", std::to_string(model_config.dropout_rate));

    std::cout << "╠" << border << "╣\n";

    // Entrenamiento
    print_row("Epocas   ", std::to_string(train_config.epochs));
    print_row("Batch size", std::to_string(train_config.batch_size));
    print_row("Learning rate", std::to_string(train_config.learning_rate));
    print_row("Weight decay", std::to_string(train_config.weight_decay));
    print_row("Warmup fraction", std::to_string(train_config.warmup_frac));

    std::cout << "╚" << border << "╝\n";
  }

} // namespace ModelUtils
