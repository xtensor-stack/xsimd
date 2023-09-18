#include <xmmintrin.h>
#include <wasm_simd128.h>
#include "xsimd/xsimd.hpp"


#include <emscripten/bind.h>
#include <iostream> // for reporting errors

using namespace emscripten;



int test_mean()
{
    std::cout<<"test_mean"<<std::endl;
    xsimd::batch<float, xsimd::sse2> a = {1.5, 2.5, 3.5, 4.5};
    xsimd::batch<float, xsimd::sse2> b = {2.5, 3.5, 4.5, 5.5};
    auto mean = (a + b) / 2;
    std::cout << mean << std::endl;
    return 0;
}


int run_tests() {
    // todo add actual tests
    if(auto ret = test_mean(); ret != 0) {
        return ret;
    }
    return 0;
}

EMSCRIPTEN_BINDINGS(my_module) {
    emscripten::function("run_tests", &run_tests);
}