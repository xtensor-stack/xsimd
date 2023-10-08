#include "xsimd/xsimd.hpp"
#include <iostream>

namespace xs = xsimd;

int main(int, char*[])
{
    xs::batch<double, xs::avx> a = { 1.5, 2.5, 3.5, 4.5 };
    xs::batch<double, xs::avx> b = { 2.5, 3.5, 4.5, 5.5 };
    auto mean = (a + b) / 2;
    std::cout << mean << std::endl;
    return 0;
}
