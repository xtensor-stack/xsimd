#include "xsimd/xsimd.hpp"

#include <cstdio>

int main()
{
    xsimd::batch<double> a(1.0), b(2.0);
    double res = (a + b).get(0);
    if (res != 3.0)
    {
        std::fprintf(stderr, "check failed: got %f, expected 3\n", res);
        return 1;
    }
    return 0;
}
