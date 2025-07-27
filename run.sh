#!/bin/bash

# Terminar el script inmediatamente si un comando falla.
set -e

# --- Variables de Configuración ---
BUILD_DIR="build"
PROJECT_NAME="ViT" # <-- Actualizado de "CNN" a "ViT"

# Por defecto, se usa Release. Se puede sobreescribir con un argumento.
BUILD_TYPE="Release"

# --- Lógica de Argumentos ---
# Permite cambiar el tipo de compilación o limpiar.
# Ejemplos de uso:
#   ./run.sh          (Compila en Release y ejecuta)
#   ./run.sh debug    (Compila en Debug y ejecuta)
#   ./run.sh clean    (Limpia el directorio de compilación)

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

  echo "--- Configurando el proyecto con CMake (Modo: ${BUILD_TYPE}) ---"
  # Entramos al directorio de compilación
  cd ${BUILD_DIR}
  # Ejecutamos cmake para generar los archivos de compilación
  cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ..

  echo "--- Compilando el proyecto '${PROJECT_NAME}' ---"
  # Usamos cmake --build para ser compatibles con cualquier sistema de compilación (Make, Ninja, etc.)
  # El flag -j intenta usar todos los núcleos disponibles para una compilación más rápida.
  # nproc es de gnu-coreutils, el fallback a '1' es para sistemas como macOS.
  cmake --build . --config ${BUILD_TYPE} -- -j$(nproc 2>/dev/null || echo 1)

  echo "--- Compilación completada ---"
  # Volvemos al directorio raíz
  cd ..
}

run_app() {
  echo "--- Ejecutando la aplicación '${PROJECT_NAME}' ---"
  ./${BUILD_DIR}/${PROJECT_NAME}
  echo "--- Ejecución finalizada ---"
}

run_test() {
  echo "--- Ejecutando pruebas sobre el conjunto de test ---"
  ./${BUILD_DIR}/test # Ejecutamos el test de predicción
  echo "--- Pruebas finalizadas ---"
}

run_image() {
  local image_path="$1"
  echo "--- Ejecutando testImage con imagen '${image_path}' ---"
  ./${BUILD_DIR}/testImage "${image_path}"
  echo "--- Predicción completada ---"
}

run_label() {
  local image_path="$1"
  echo "--- Ejecutando testLabel con imagen '${image_path}' ---"
  ./${BUILD_DIR}/testLabel "${image_path}"
  echo "--- Predicción completada ---"
}

# --- Flujo Principal ---
echo "Iniciando flujo: Compilar y Ejecutar"
echo "Proyecto: ${PROJECT_NAME}, Tipo de Compilación: ${BUILD_TYPE}"

# Comprobar si se debe ejecutar un test
if [ "$1" == "test" ]; then
  build_project
  run_test
elif [ "$1" == "image" ]; then
  build_project
  run_image "$2"
elif [ "$1" == "label" ]; then
  build_project
  run_label "$2"
elif [ "$1" == "visualizer" ]; then
  # solo si se tiene opencv instalado
  # cd app
  # g++ realTime.cpp -o realTime `pkg-config --cflags --libs opencv4` 
  # cd ..
  ./app/realTime
else
  build_project
  run_app
fi

exit 0

