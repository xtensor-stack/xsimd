#ifndef _SUM_HPP
#define _SUM_HPP
#include "xsimd/xsimd.hpp"

// functor with a call method that depends on `Arch`
struct sum
{
    // It's critical not to use an in-class definition here.
    // In-class and inline definition bypass extern template mechanism.
    template <class Arch, class T>
    T operator()(Arch, T const* data, unsigned size);
};

template <class Arch, class T>
T sum::operator()(Arch, T const* data, unsigned size)
{
    using batch = xsimd::batch<T, Arch>;
    batch acc(static_cast<T>(0));
    const unsigned n = size / batch::size * batch::size;
    for (unsigned i = 0; i != n; i += batch::size)
        acc += batch::load_unaligned(data + i);
    T star_acc = xsimd::reduce_add(acc);
    for (unsigned i = n; i < size; ++i)
        star_acc += data[i];
    return star_acc;
}

// Inform the compiler that sse2 and avx2 implementation are to be found in another compilation unit.
extern template float sum::operator()<xsimd::avx2, float>(xsimd::avx2, float const*, unsigned);
extern template float sum::operator()<xsimd::sse2, float>(xsimd::sse2, float const*, unsigned);
#endif
