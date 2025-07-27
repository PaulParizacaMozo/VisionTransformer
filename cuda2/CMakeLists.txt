cmake_minimum_required(VERSION 3.18)

# --- Proyecto híbrido ---
project(ViT LANGUAGES CXX CUDA)

# --- Estándar del Lenguaje ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# --- Forzar uso de GCC específico (opcional) ---
set(GCC_VER 13)
set(GCC_PATH "/usr/bin/gcc-${GCC_VER}")
set(GPP_PATH "/usr/bin/g++-${GCC_VER}")
set(STD_INC "/usr/include/c++/${GCC_VER}")
set(STD_INC_BACKEND "/usr/include/c++/${GCC_VER}/x86_64-redhat-linux")

if (EXISTS ${GCC_PATH} AND EXISTS ${GPP_PATH})
    message(STATUS "✅ Usando GCC ${GCC_VER}: ${GCC_PATH}")
    set(CMAKE_C_COMPILER ${GCC_PATH})
    set(CMAKE_CXX_COMPILER ${GPP_PATH})

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
        -ccbin=${GPP_PATH} \
        -std=c++17 \
        -Xcompiler=-Wall,-Wno-unused-function \
        -Xcompiler=-fPIC \
        -allow-unsupported-compiler \
        -I${STD_INC} -I${STD_INC_BACKEND} \
        -G -lineinfo"
    )
endif()

# --- Arquitectura CUDA ---
set(CMAKE_CUDA_ARCHITECTURES "native")

# --- Flags por tipo de compilación ---
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type: Debug, Release, RelWithDebInfo, MinSizeRel." FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# --- Directorios de encabezados ---
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

# --- Archivos fuente comunes (C++ y CUDA) ---
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE CU_SOURCES "src/*.cu")

# --- Entradas principales ---
set(MAIN_SOURCE "app/main.cpp")
set(TEST_SOURCE "app/test.cpp")
set(TEST_IMAGE_SOURCE "app/testImage.cpp")

# --- Ejecutables ---
add_executable(${PROJECT_NAME} ${MAIN_SOURCE} ${CPP_SOURCES} ${CU_SOURCES})
add_executable(test ${TEST_SOURCE} ${CPP_SOURCES} ${CU_SOURCES})
add_executable(testImage ${TEST_IMAGE_SOURCE} ${CPP_SOURCES} ${CU_SOURCES})

# --- Propiedades necesarias para CUDA ---
foreach(target ${PROJECT_NAME} test testImage)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
endforeach()

# --- Enlazar librerías CUDA ---
find_package(CUDAToolkit REQUIRED)
foreach(target ${PROJECT_NAME} test testImage)
    target_link_libraries(${target} PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
        stdc++
        m
    )
endforeach()

# --- OpenMP (solo para CPU paralela si se necesita) ---
find_package(OpenMP)
if(OpenMP_FOUND)
    message(STATUS "OpenMP encontrado, enlazando...")
    foreach(target ${PROJECT_NAME} test testImage)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endforeach()
endif()

message(STATUS "✔️ CMake configurado correctamente para proyecto híbrido C++/CUDA.")
