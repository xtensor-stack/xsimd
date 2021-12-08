/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "xsimd/xsimd.hpp"
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE

#include <numeric>
#include <random>

#include "test_utils.hpp"

static_assert(xsimd::default_arch::supported(), "default arch must be supported");
static_assert(xsimd::supported_architectures::contains<xsimd::default_arch>(), "default arch is supported");
static_assert(xsimd::all_architectures::contains<xsimd::default_arch>(), "default arch is a valid arch");
// static_assert(!(xsimd::x86_arch::supported() && xsimd::arm::supported()), "either x86 or arm, but not both");

struct check_supported
{
    template <class Arch>
    void operator()(Arch) const
    {
        static_assert(Arch::supported(), "not supported?");
    }
};

TEST(arch, supported)
{
    xsimd::supported_architectures::for_each(check_supported {});
}

TEST(arch, name)
{
    constexpr char const* name = xsimd::default_arch::name();
    (void)name;
}

struct check_available
{
    template <class Arch>
    void operator()(Arch) const
    {
        EXPECT_TRUE(Arch::available());
    }
};

TEST(arch, available)
{
    EXPECT_TRUE(xsimd::default_arch::available());
}

TEST(arch, arch_list_alignment)
{
    static_assert(xsimd::arch_list<xsimd::generic>::alignment() == 0,
                  "generic");
    static_assert(xsimd::arch_list<xsimd::sse2>::alignment()
                      == xsimd::sse2::alignment(),
                  "one architecture");
    static_assert(xsimd::arch_list<xsimd::avx512f, xsimd::sse2>::alignment()
                      == xsimd::avx512f::alignment(),
                  "two architectures");
}

struct sum
{
    template <class Arch, class T>
    T operator()(Arch, T const* data, unsigned size)
    {
        using batch = xsimd::batch<T, Arch>;
        batch acc(static_cast<T>(0));
        const unsigned n = size / batch::size * batch::size;
        for (unsigned i = 0; i != n; i += batch::size)
            acc += batch::load_unaligned(data + i);
        T star_acc = xsimd::hadd(acc);
        for (unsigned i = n; i < size; ++i)
            star_acc += data[i];
        return star_acc;
    }
};

struct get_arch_version
{
    template <class Arch>
    unsigned operator()(Arch) { return Arch::version(); }
};

TEST(arch, dispatcher)
{
    float data[17] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f };
    float ref = std::accumulate(std::begin(data), std::end(data), 0.f);

    // platform specific
    {
        auto dispatched = xsimd::dispatch(sum {});
        float res = dispatched(data, 17);
        EXPECT_EQ(ref, res);
    }

#if XSIMD_WITH_AVX && XSIMD_WITH_SSE2
    static_assert(xsimd::supported_architectures::contains<xsimd::avx>() && xsimd::supported_architectures::contains<xsimd::sse2>(), "consistent supported architectures");
    {
        auto dispatched = xsimd::dispatch<sum, xsimd::arch_list<xsimd::avx, xsimd::sse2>>(sum {});
        float res = dispatched(data, 17);
        EXPECT_EQ(ref, res);
    }

    // check that we pick the most appropriate version
    {
        auto dispatched = xsimd::dispatch<get_arch_version,
                                          xsimd::arch_list<xsimd::sse3, xsimd::sse2>>(get_arch_version {});
        unsigned expected = xsimd::available_architectures().best >= xsimd::sse3::version()
            ? xsimd::sse3::version()
            : xsimd::sse2::version();
        EXPECT_EQ(expected, dispatched());
    }
#endif
}

template <class T>
static bool try_load()
{
    static_assert(std::is_same<xsimd::batch<T>, decltype(xsimd::load_aligned(std::declval<T*>()))>::value,
                  "loading the expected type");
    static_assert(std::is_same<xsimd::batch<T>, decltype(xsimd::load_unaligned(std::declval<T*>()))>::value,
                  "loading the expected type");
    return true;
}

template <class... Tys>
void try_loads()
{
    (void)std::initializer_list<bool> { try_load<Tys>()... };
}

TEST(arch, default_load)
{
    // make sure load_aligned / load_unaligned work for the default arch and
    // return the appropriate type.
    using type_list = xsimd::mpl::type_list<short, int, long, float, double, std::complex<float>, std::complex<double>>;
    try_loads<type_list>();
}

#ifdef XSIMD_ENABLE_FALLBACK
// FIXME: this should be named scalar
TEST(arch, scalar)
{
    float data[17] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 };
    float ref = std::accumulate(std::begin(data), std::end(data), 0);

    float res = sum {}(xsimd::arch::scalar {}, data, 17);
    EXPECT_EQ(ref, res);
}
#endif
#endif
