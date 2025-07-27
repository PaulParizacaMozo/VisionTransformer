#include "utils/ModelUtils.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "utils/json.hpp"

using json = nlohmann::json;

namespace ModelUtils
{

  void save_weights(const VisionTransformer &model, const std::string &filePath)
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
        std::cerr << "Advertencia: Guardando un tensor no contiguo. Se creará una copia temporal." << std::endl;
        Tensor temp = tensor.contiguous();
        outFile.write(reinterpret_cast<const char *>(temp.getData()), num_elements * sizeof(float));
      }
      else
      {
        // Accedemos a los datos teniendo en cuenta el offset por si es una vista.
        outFile.write(reinterpret_cast<const char *>(tensor.getData() + tensor.getDataOffset()), num_elements * sizeof(float));
      }
    }

    outFile.close();
    std::cout << "Pesos guardados correctamente." << std::endl;
  }

  void load_weights(VisionTransformer &model, const std::string &filePath)
  {
    std::ifstream inFile(filePath, std::ios::binary);
    if (!inFile)
    {
      throw std::runtime_error("No se pudo abrir el archivo para lectura: " + filePath);
    }

    std::cout << "Archivo abierto correctamente." << std::endl;
    // Aquí 'model' no es const, así que podemos llamar a getParameters() directamente.
    auto params = model.getParameters();

    std::cout << "Cargando " << params.size() << " tensores de parámetros desde " << filePath << "..." << std::endl;

    // size_t index = 0;
    for (auto &tensor_ptr : params)
    {
      // std::cout << "\nProcesando tensor #" << index++ << std::endl;
      Tensor &tensor = *tensor_ptr;

      size_t file_rank;
      inFile.read(reinterpret_cast<char *>(&file_rank), sizeof(size_t));
      // std::cout << "Leído file_rank = " << file_rank << std::endl;

      std::vector<size_t> file_shape(file_rank);
      inFile.read(reinterpret_cast<char *>(file_shape.data()), file_rank * sizeof(size_t));
      // std::cout << "Leído file_shape = [";
      // for (size_t i = 0; i < file_shape.size(); ++i)
      //   std::cout << file_shape[i] << (i < file_shape.size() - 1 ? ", " : "");
      // std::cout << "]" << std::endl;
      if (tensor.getShape() != file_shape)
      {
        throw std::runtime_error("Incompatibilidad de formas al cargar pesos. Esperado: " + tensor.shapeToString() +
                                 ", encontrado en archivo: " + Tensor(file_shape).shapeToString());
      }

      size_t num_elements = tensor.getSize();
      // std::cout << "Tamaño esperado: " << num_elements << std::endl;
      if (!tensor.isContiguous())
      {
        // std::cout << "Tensor no es contiguo, leyendo a buffer temporal." << std::endl;
        // Para cargar en un tensor no contiguo, necesitamos leer a un buffer temporal
        // y luego copiar elemento por elemento.
        std::vector<float> buffer(num_elements);
        inFile.read(reinterpret_cast<char *>(buffer.data()), num_elements * sizeof(float));
        // std::cout << "Lectura realizada, pero copiar a tensor no contiguo aún no está implementado." << std::endl;

        // Copia manual elemento por elemento (requeriría un iterador N-D o bucles anidados)
        // Por ahora, lanzamos un error como medida de seguridad.
        throw std::runtime_error("Cargar pesos a un tensor no contiguo no está implementado aún.");
      }
      else
      {
        // std::cout << "Tensor contiguo, leyendo directamente a memoria." << std::endl;
        std::vector<float> host_data(num_elements);
        inFile.read(reinterpret_cast<char *>(host_data.data()), num_elements * sizeof(float));

        // Paso 2: copiar a memoria GPU del tensor
        cudaMemcpy(tensor.getData() + tensor.getDataOffset(), host_data.data(),
                   num_elements * sizeof(float), cudaMemcpyHostToDevice);
        // inFile.read(reinterpret_cast<char *>(tensor.getData() + tensor.getDataOffset()), num_elements * sizeof(float));
      }
      size_t leido = static_cast<size_t>(inFile.gcount());
      // std::cout << "Bytes leídos: " << leido << std::endl;

      if (static_cast<size_t>(inFile.gcount()) != num_elements * sizeof(float))
      {
        throw std::runtime_error("Error de lectura: fin de archivo inesperado o datos corruptos.");
      }
      // std::cout << "Tensor cargado correctamente." << std::endl;
    }

    inFile.close();
    std::cout << "Pesos cargados correctamente." << std::endl;
  }

  void save_config(const ViTConfig &config, const std::string &filePath)
  {
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

} // namespace ModelUtils
