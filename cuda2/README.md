# Vision Transformer

## Estructura del Proyecto

A continuación se muestra la estructura de directorios del proyecto:

```

.
├── app
│   └── main.cpp
├── CMakeLists.txt
├── compile\_commands.json
├── data
│   ├── fashion\_test.csv
│   ├── fashion\_train.csv
│   ├── mnist\_test.csv
│   └── mnist\_train.csv
├── include
│   ├── activations
│   │   ├── GELU.hpp
│   │   └── ReLU.hpp
│   ├── core
│   │   └── Tensor.hpp
│   ├── layers
│   │   ├── Dense.hpp
│   │   ├── Embeddings.hpp
│   │   ├── FeedForward.hpp
│   │   ├── Layer.hpp
│   │   ├── LayerNorm.hpp
│   │   ├── MultiHeadAttention.hpp
│   │   └── PatchEmbedding.hpp
│   ├── losses
│   │   ├── CrossEntropy.hpp
│   │   └── Loss.hpp
│   ├── model
│   │   ├── Trainer.hpp
│   │   ├── TransformerEncoderBlock.hpp
│   │   └── VisionTransformer.hpp
│   ├── optimizers
│   │   ├── Adam.hpp
│   │   ├── Optimizer.hpp
│   │   └── SGD.hpp
│   └── utils
│   │   ├── DataReader.hpp
│   │   └── ModelUtils.hpp
├── README.md
├── run.sh
└── src
├── activations
│   ├── GELU.cpp
│   └── ReLU.cpp
├── core
│   └── Tensor.cpp
├── layers
│   ├── Dense.cpp
│   ├── Embeddings.cpp
│   ├── FeedForward.cpp
│   ├── LayerNorm.cpp
│   ├── MultiHeadAttention.cpp
│   └── PatchEmbedding.cpp
├── losses
│   └── CrossEntropy.cpp
├── model
│   ├── Trainer.cpp
│   ├── TransformerEncoderBlock.cpp
│   └── VisionTransformer.cpp
├── optimizers
│   └── Adam.cpp
└── utils
├── DataReader.cpp
└── ModelUtils.cpp

````

## Instrucciones para ejecutar el proyecto

### 1. Descargar y descomprimir los datos

El proyecto utiliza un archivo comprimido llamado `data.zip` que contiene los conjuntos de datos necesarios para la ejecución (Fashion MNIST y MNIST). Para comenzar, sigue estos pasos:

1. Descarga el archivo `data.zip` desde el repositorio.
2. Descomprime el archivo `data.zip` en el directorio raíz de tu proyecto. Esto creará un directorio `data` que contiene los siguientes archivos CSV:
   - `fashion_test.csv`
   - `fashion_train.csv`
   - `mnist_test.csv`
   - `mnist_train.csv`

### 2. Preparar el script `run.sh`

Antes de ejecutar el script, asegúrate de darle permisos de ejecución:

```bash
chmod +x run.sh
````

### 3. Ejecutar el proyecto

Una vez que hayas dado los permisos necesarios, puedes ejecutar el script `run.sh` con el siguiente comando:

```bash
./run.sh
```

Este script compilará y ejecutará el proyecto. Los datos de entrada se leerán desde el directorio `data` y el modelo Transformer se entrenará o evaluará según el flujo definido en el script.
