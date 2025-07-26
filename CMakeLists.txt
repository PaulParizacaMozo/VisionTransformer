cmake_minimum_required(VERSION 3.10)

# --- Información del Proyecto ---
project(ViT LANGUAGES CXX)

# --- Estándar del Lenguaje ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Tipo de compilación por defecto y flags generales ---
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type: Debug, Release, RelWithDebInfo, MinSizeRel." FORCE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")

# --- Búsqueda de Dependencias ---
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

# --- Directorios de include ---
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${GTK3_INCLUDE_DIRS}
)

# --- Fuentes del núcleo (modelo, utils, layers, etc) ---
file(GLOB_RECURSE CORE_SOURCES
    "src/*.cpp"
    "include/utils/*.hpp"
)

# --- Crear biblioteca estática del core ---
add_library(core STATIC ${CORE_SOURCES})

target_link_libraries(core
  PUBLIC
    OpenMP::OpenMP_CXX
    ${OpenCV_LIBS}
)

# --- Ejecutables ---
add_executable(${PROJECT_NAME} app/main.cpp)
add_executable(test app/test.cpp)
add_executable(testImage app/testImage.cpp)
add_executable(miniplataforma app/miniplataforma.cpp)

# --- Ligar ejecutables con core y dependencias externas ---
foreach(target ${PROJECT_NAME} test testImage)
  target_link_libraries(${target}
    PRIVATE
      core
      ${OpenCV_LIBS}
  )
endforeach()

# Para miniplataforma necesitamos GTK
target_link_libraries(miniplataforma
    PRIVATE
      core
      ${OpenCV_LIBS}
      ${GTK3_LIBRARIES}
)

# Flags específicos para GTK
target_compile_options(miniplataforma PRIVATE ${GTK3_CFLAGS})
target_link_options(miniplataforma PRIVATE ${GTK3_LDFLAGS})

message(STATUS "Configuración de CMake para ${PROJECT_NAME} completada.")