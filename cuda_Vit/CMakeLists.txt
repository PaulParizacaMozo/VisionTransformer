cmake_minimum_required(VERSION 3.18)


# --- Información del Proyecto ---
project(ViT LANGUAGES CXX CUDA)

# --- Estándar del Lenguaje ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Establecemos host de CUDA
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- Configuración de Tipos de Compilación y Flags ---
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type: Debug, Release, RelWithDebInfo, MinSizeRel." FORCE)
endif()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")

# Flags específicos por tipo de compilación.
# -O3: Máxima optimización para velocidad. -DNDEBUG: Desactiva las aserciones.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
# -g: Incluye símbolos de depuración. -O0: Desactiva todas las optimizaciones.
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")

# --- Búsqueda de Dependencias ---
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# --- Configuración de Directorios ---
include_directories(include)

# --- Recopilación de Archivos Fuente ---
file(GLOB_RECURSE CPP_SOURCES
    "src/*.cpp"
)
file(GLOB_RECURSE CUDA_SOURCES "src/utils/CudaUtils.cu")

set(MAIN_SOURCE "app/main.cpp")
set(TEST_SOURCE "app/test.cpp")
set(TEST_IMAGE_SOURCE "app/testImage.cpp")
set(TEST_LABEL_SOURCE "app/testLabel.cpp")

# --- Creación del Ejecutable ---
add_executable(${PROJECT_NAME} ${MAIN_SOURCE} ${CPP_SOURCES} ${CUDA_SOURCES})

# --- Creación del Ejecutable para el Test ---
add_executable(test ${TEST_SOURCE} ${CPP_SOURCES} ${CUDA_SOURCES})

# --- Creación del Ejecutable para el TestImage ---
add_executable(testImage ${TEST_IMAGE_SOURCE} ${CPP_SOURCES} ${CUDA_SOURCES})

# --- Creación del Ejecutable para el TestLabel ---
add_executable(testLabel ${TEST_LABEL_SOURCE} ${CPP_SOURCES} ${CUDA_SOURCES})


# --- Enlace de Librerías ---
message(STATUS "Enlazando librerías...")

# Definimos una función para no repetir el código de enlace
function(link_project_libraries target_name)
    # Enlazar OpenMP (para el código de CPU)
    target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
    
    # Enlazar librerías de CUDA
    # CUDA::cublas para la multiplicación de matrices en GPU
    # CUDA::cudart para el runtime de CUDA (cudaMalloc, cudaMemcpy, etc.)
    target_link_libraries(${target_name} PRIVATE CUDA::cublas CUDA::cudart)
endfunction()

# Llamamos a la función para cada uno de nuestros ejecutables
link_project_libraries(${PROJECT_NAME})
link_project_libraries(test)
link_project_libraries(testImage)
link_project_libraries(testLabel)

message(STATUS "Configuración de CMake para ${PROJECT_NAME} completada.")
