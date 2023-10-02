#include <xmmintrin.h>
#include <wasm_simd128.h>
#include "xsimd/xsimd.hpp"

#include <emscripten/bind.h>
#include <iostream> // for reporting errors

using namespace emscripten;

int test_abs()
{
    xsimd::batch<int32_t, xsimd::wasm> a(1, -2, 3, -4);
    auto ans = xsimd::abs(a);
    std::cout << ans << std::endl;
    return 0;
}

int run_tests() {
    // todo add actual tests
    if(auto ret = test_abs(); ret != 0) {
        return ret;
    }
    return 0;
}

EMSCRIPTEN_BINDINGS(my_module) {
    emscripten::function("run_tests", &run_tests);
}