#include "xsimd/xsimd.hpp"

namespace xs = xsimd;
template <class T, class Arch>
xs::batch<T, Arch> mean(xs::batch<T, Arch> lhs, xs::batch<T, Arch> rhs)
{
    return (lhs + rhs) / 2;
}
