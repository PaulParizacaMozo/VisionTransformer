#!/bin/bash

# Terminar el script inmediatamente si un comando falla.
set -e

# --- Configuración de compilador ---
GCC_VER=13
export CC=/usr/bin/gcc-${GCC_VER}
export CXX=/usr/bin/g++-${GCC_VER}
export CUDAHOSTCXX=${CXX}

# --- Variables de Configuración ---
BUILD_DIR="build"
PROJECT_NAME="ViT"

# Por defecto, se usa Release. Se puede sobreescribir con un argumento.
BUILD_TYPE="Release"

# --- Lógica de Argumentos ---
if [ "$1" == "clean" ]; then
  echo "--- Limpiando el directorio de compilación ---"
  if [ -d "${BUILD_DIR}" ]; then
    rm -rf ${BUILD_DIR}
    echo "Directorio '${BUILD_DIR}' eliminado."
  else
    echo "El directorio '${BUILD_DIR}' no existe. Nada que limpiar."
  fi
  exit 0
fi

if [ "$1" == "debug" ]; then
  BUILD_TYPE="Debug"
fi

# --- Funciones ---
build_project() {
  echo "--- Creando directorio de compilación (${BUILD_DIR}) ---"
  mkdir -p ${BUILD_DIR}
  cd ${BUILD_DIR}

  echo "--- Configurando el proyecto con CMake (Modo: ${BUILD_TYPE}) ---"
  cmake .. \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler -ccbin=${CC}" \
    -DCMAKE_CUDA_ARCHITECTURES="75;86;89"

  echo "--- Compilando el proyecto '${PROJECT_NAME}' ---"
  cmake --build . --config ${BUILD_TYPE} -- -j$(nproc 2>/dev/null || echo 1)

  echo "--- Compilación completada ---"
  cd ..
}

run_app() {
  echo "--- Ejecutando la aplicación '${PROJECT_NAME}' ---"
  ./${BUILD_DIR}/${PROJECT_NAME}
  echo "--- Ejecución finalizada ---"
}

run_test() {
  echo "--- Ejecutando pruebas sobre el conjunto de test ---"
  ./${BUILD_DIR}/test
  echo "--- Pruebas finalizadas ---"
}

run_image() {
  local image_path="$1"
  echo "--- Ejecutando testImage con imagen '${image_path}' ---"
  ./${BUILD_DIR}/testImage "${image_path}"
  echo "--- Predicción completada ---"
}

# --- Flujo Principal ---
echo "Iniciando flujo: Compilar y Ejecutar"
echo "Proyecto: ${PROJECT_NAME}, Tipo de Compilación: ${BUILD_TYPE}"

if [ "$1" == "test" ]; then
  build_project
  run_test
elif [ "$1" == "image" ]; then
  build_project
  run_image "$2"
else
  build_project
  run_app
fi

exit 0
