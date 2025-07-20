#!/bin/bash
GCC_VER=13
export CC=/usr/bin/gcc-${GCC_VER}
export CXX=/usr/bin/g++-${GCC_VER}

rm -rf build
mkdir build && cd build

cmake .. -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler -ccbin=${CC}" \
         -DCMAKE_CUDA_ARCHITECTURES="75;86;89" || exit 1

make -j || exit 1

echo "✅ Compilado correctamente"
./tensor_test