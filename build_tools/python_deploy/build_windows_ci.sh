#!/usr/bin/env bash
set -eo pipefail

echo "Building torch-mlir"

cmake -GNinja -Bbuild_win \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
  -DPython3_EXECUTABLE="$(which python)" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  $GITHUB_WORKSPACE/externals/llvm-project/llvm

cmake --build build_win --config Release

echo "Build completed successfully"
