#include <xsimd/xsimd.hpp>

// Basic check: can we instantiate a batch for the given compiler flags?
xsimd::batch<int> come_and_get_some(xsimd::batch<int> x, xsimd::batch<int> y)
{
    return x + y;
}
