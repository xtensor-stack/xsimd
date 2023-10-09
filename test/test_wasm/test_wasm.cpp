<<<<<<< HEAD
#include "xsimd/xsimd.hpp"
#include <wasm_simd128.h>
#include <xmmintrin.h>
=======
#include <xmmintrin.h>
#include <wasm_simd128.h>
#include "xsimd/xsimd.hpp"
>>>>>>> 1559359ba2d901b5a43d3b7f2021e01e7c9dc7c3

#include <emscripten/bind.h>
#include <iostream> // for reporting errors

using namespace emscripten;

int test_abs()
{
<<<<<<< HEAD
    std::cout << "test_abs" << std::endl;
=======
    xsimd::batch<int32_t, xsimd::wasm> a(1, -2, 3, -4);
>>>>>>> 1559359ba2d901b5a43d3b7f2021e01e7c9dc7c3
    auto ans = xsimd::abs(a);
    std::cout << ans << std::endl;
    return 0;
}

<<<<<<< HEAD
int run_tests()
{
    // todo add actual tests
    if (auto ret = test_abs(); ret != 0)
    {
=======
int run_tests() {
    // todo add actual tests
    if(auto ret = test_abs(); ret != 0) {
>>>>>>> 1559359ba2d901b5a43d3b7f2021e01e7c9dc7c3
        return ret;
    }
    return 0;
}

<<<<<<< HEAD
EMSCRIPTEN_BINDINGS(my_module)
{
=======
EMSCRIPTEN_BINDINGS(my_module) {
>>>>>>> 1559359ba2d901b5a43d3b7f2021e01e7c9dc7c3
    emscripten::function("run_tests", &run_tests);
}