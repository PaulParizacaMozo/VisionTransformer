cmake_minimum_required(VERSION 3.18)
project(TensorCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ------------------------------
# Intentar usar GCC 13
# ------------------------------
set(GCC_VER 13)
set(GCC_PATH "/usr/bin/gcc-${GCC_VER}")
set(GPP_PATH "/usr/bin/g++-${GCC_VER}")
set(STD_INC "/usr/include/c++/${GCC_VER}")
set(STD_INC_BACKEND "/usr/include/c++/${GCC_VER}/x86_64-redhat-linux")

if (EXISTS ${GCC_PATH} AND EXISTS ${GPP_PATH})
    message(STATUS "✅ Usando GCC ${GCC_VER}: ${GCC_PATH}")
    set(CMAKE_C_COMPILER ${GCC_PATH})
    set(CMAKE_CXX_COMPILER ${GPP_PATH})

    # Forzar a nvcc a usar gcc-13 y permitir versiones no oficiales
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
        -ccbin=${GCC_PATH} \
        -std=c++17 \
        -Xcompiler=-Wall \
        -Xcompiler=-Wno-unused-function \
        -allow-unsupported-compiler \
        -I${STD_INC} -I${STD_INC_BACKEND} \
        -G -lineinfo"
    )
else()
    message(FATAL_ERROR "❌ GCC ${GCC_VER} no encontrado en /usr/bin")
endif()

# ------------------------------
# Arquitectura CUDA
# ------------------------------
set(CMAKE_CUDA_ARCHITECTURES "native")

# ------------------------------
# Directorios de encabezados
# ------------------------------
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/core
    ${PROJECT_SOURCE_DIR}/include/layers
    ${PROJECT_SOURCE_DIR}/include/activations
    ${PROJECT_SOURCE_DIR}/include/optimizers
    ${PROJECT_SOURCE_DIR}/include/losses
    ${PROJECT_SOURCE_DIR}/include/model
    ${PROJECT_SOURCE_DIR}/include/utils
)

# ------------------------------
# Archivos fuente
# ------------------------------
set(SOURCES
    main.cu
    src/core/Tensor.cu
    src/layers/Dense.cu
    src/layers/PatchEmbedding.cu
    src/layers/Embeddings.cu
    src/layers/FeedForward.cu
    src/layers/MultiHeadAttention.cu
    src/layers/LayerNorm.cu
    src/activations/GELU.cu
    src/activations/RELU.cu
    src/optimizers/Adam.cu
    src/losses/CrossEntropy.cu
    src/model/TransformerEncoderBlock.cu
    src/model/VisionTransformer.cu
    src/model/Trainer.cu
    src/utils/DataReader.cu
)

# ------------------------------
# Ejecutable
# ------------------------------
add_executable(tensor_test ${SOURCES})

set_target_properties(tensor_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

# ------------------------------
# Enlace con bibliotecas necesarias
# ------------------------------
find_package(CUDAToolkit REQUIRED)
target_link_libraries(tensor_test PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    stdc++
    m
)