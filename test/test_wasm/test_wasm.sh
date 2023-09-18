#!/bin/bash
set -e

# this dir
TEST_WASM_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SRC_DIR=$TEST_WASM_DIR/../..

# setup emsdk
git clone https://github.com/emscripten-core/emsdk
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
cd ..


# build wasm
mkdir -p build
cd build
emcmake cmake -DBUILD_TESTS=ON $SRC_DIR
emmake make -j4
cd ..

python $TEST_WASM_DIR/test_wasm_playwright.py  build/test