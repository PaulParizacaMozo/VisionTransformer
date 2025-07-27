cmake_minimum_required(VERSION 3.10)

# --- Información del Proyecto ---
project(ViT CXX)

# --- Estándar del Lenguaje ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

# --- Configuración de Directorios ---
include_directories(include)

# --- Recopilación de Archivos Fuente ---
file(GLOB_RECURSE SOURCES
    "src/*.cpp"
)

set(MAIN_SOURCE "app/main.cpp")
set(TEST_SOURCE "app/test.cpp")
set(TEST_IMAGE_SOURCE "app/testImage.cpp")
set(TEST_LABEL_SOURCE "app/testLabel.cpp")

# --- Creación del Ejecutable ---
add_executable(${PROJECT_NAME} ${MAIN_SOURCE} ${SOURCES})

# --- Creación del Ejecutable para el Test ---
add_executable(test ${TEST_SOURCE} ${SOURCES}) 

# --- Creación del Ejecutable para el TestImage ---
add_executable(testImage ${TEST_IMAGE_SOURCE} ${SOURCES})

# --- Creación del Ejecutable para el TestLabel ---
add_executable(testLabel ${TEST_LABEL_SOURCE} ${SOURCES})

# --- Enlace de Librerías ---
if(OpenMP_FOUND)
    message(STATUS "OpenMP encontrado, enlazando...")
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(test PRIVATE OpenMP::OpenMP_CXX)  
else()
    message(WARNING "OpenMP no se encontró. La compilación continuará sin paralelización.")
endif()

message(STATUS "Configuración de CMake para ${PROJECT_NAME} completada.")
