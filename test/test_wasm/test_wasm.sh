#!/bin/bash
set -e

# this dir
TEST_WASM_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SRC_DIR=$TEST_WASM_DIR/../..


# the emsdk dir can be passed as optional argument
# if not passed, it will be downloaded in the current dir
if [ $# -eq 0 ]
then
    git clone https://github.com/emscripten-core/emsdk
    cd emsdk
    ./emsdk install latest
    ./emsdk activate latest
    source ./emsdk_env.sh

else
    EMSCRIPTEN_DIR=$1
    source $EMSCRIPTEN_DIR/emsdk_env.sh
fi


export LDFLAGS=""
export CFLAGS=""
export CXXFLAGS=""

# build wasm
mkdir -p build
cd build
emcmake cmake \
    -DBUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=14 \
    -DDOWNLOAD_DOCTEST=ON \
    $SRC_DIR

emmake make -j4
cd ..

# run tests in browser
python $TEST_WASM_DIR/test_wasm_playwright.py  build/test