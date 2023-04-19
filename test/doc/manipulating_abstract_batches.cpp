#include "xsimd/xsimd.hpp"

namespace xs = xsimd;
xs::batch<float> mean(xs::batch<float> lhs, xs::batch<float> rhs)
{
    return (lhs + rhs) / 2;
}
