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
#include <type_traits>

#include "test_sum.hpp"
#include "test_utils.hpp"

#ifndef XSIMD_DEFAULT_ARCH
static_assert(xsimd::default_arch::supported(), "default arch must be supported");
static_assert(std::is_same<xsimd::default_arch, xsimd::best_arch>::value, "default arch is the best available");
static_assert(xsimd::supported_architectures::contains<xsimd::default_arch>(), "default arch is supported");
static_assert(xsimd::all_architectures::contains<xsimd::default_arch>(), "default arch is a valid arch");
#endif

#if !XSIMD_WITH_SVE
static_assert((std::is_same<xsimd::default_arch, xsimd::neon64>::value || !xsimd::neon64::supported()), "on arm, without sve, the best we can do is neon64");
#endif

struct check_supported
{
    template <class Arch>
    void operator()(Arch) const
    {
        static_assert(Arch::supported(), "not supported?");
    }
};

struct check_cpu_has_intruction_set
{
    template <class Arch>
    void operator()(Arch arch) const
    {
        static_assert(std::is_same<decltype(xsimd::available_architectures().has(arch)), bool>::value,
                      "cannot test instruction set availability on CPU");
    }
};

struct check_available
{
    template <class Arch>
    void operator()(Arch) const
    {
        CHECK_UNARY(Arch::available());
    }
};

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

TEST_CASE("[multi arch support]")
{

    SUBCASE("xsimd::supported_architectures")
    {
        xsimd::supported_architectures::for_each(check_supported {});
    }

    SUBCASE("xsimd::available_architectures::has")
    {
        xsimd::all_architectures::for_each(check_cpu_has_intruction_set {});
    }

    SUBCASE("xsimd::default_arch::name")
    {
        constexpr char const* name = xsimd::default_arch::name();
        (void)name;
    }

    SUBCASE("xsimd::default_arch::available")
    {
        CHECK_UNARY(xsimd::default_arch::available());
    }

    SUBCASE("xsimd::arch_list<...>::alignment()")
    {
        static_assert(xsimd::arch_list<xsimd::common>::alignment() == 0,
                      "common");
        static_assert(xsimd::arch_list<xsimd::sse2>::alignment()
                          == xsimd::sse2::alignment(),
                      "one architecture");
        static_assert(xsimd::arch_list<xsimd::avx512f, xsimd::sse2>::alignment()
                          == xsimd::avx512f::alignment(),
                      "two architectures");
    }

    SUBCASE("xsimd::dispatch(...)")
    {
        float data[17] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f };
        float ref = std::accumulate(std::begin(data), std::end(data), 0.f);

        // platform specific
        {
            auto dispatched = xsimd::dispatch(sum {});
            float res = dispatched(data, 17);
            CHECK_EQ(ref, res);
        }

        // only highest available
        {
            auto dispatched = xsimd::dispatch<xsimd::arch_list<xsimd::best_arch>>(sum {});
            float res = dispatched(data, 17);
            CHECK_EQ(ref, res);
        }

#if XSIMD_WITH_AVX && XSIMD_WITH_SSE2
        static_assert(xsimd::supported_architectures::contains<xsimd::avx>() && xsimd::supported_architectures::contains<xsimd::sse2>(), "consistent supported architectures");
        {
            auto dispatched = xsimd::dispatch<xsimd::arch_list<xsimd::avx, xsimd::sse2>>(sum {});
            float res = dispatched(data, 17);
            CHECK_EQ(ref, res);
        }
#endif
    }

    SUBCASE("xsimd::make_sized_batch_t")
    {
        using batch4f = xsimd::make_sized_batch_t<float, 4>;
        using batch2d = xsimd::make_sized_batch_t<double, 2>;
        using batch4c = xsimd::make_sized_batch_t<std::complex<float>, 4>;
        using batch2z = xsimd::make_sized_batch_t<std::complex<double>, 2>;
        using batch4i32 = xsimd::make_sized_batch_t<int32_t, 4>;
        using batch4u32 = xsimd::make_sized_batch_t<uint32_t, 4>;

        using batch8f = xsimd::make_sized_batch_t<float, 8>;
        using batch4d = xsimd::make_sized_batch_t<double, 4>;
        using batch8c = xsimd::make_sized_batch_t<std::complex<float>, 8>;
        using batch4z = xsimd::make_sized_batch_t<std::complex<double>, 4>;
        using batch8i32 = xsimd::make_sized_batch_t<int32_t, 8>;
        using batch8u32 = xsimd::make_sized_batch_t<uint32_t, 8>;

#if XSIMD_WITH_SSE2 || XSIMD_WITH_NEON || XSIMD_WITH_NEON64 || XSIMD_WITH_SVE || (XSIMD_WITH_RVV && XSIMD_RVV_BITS == 128)
        CHECK_EQ(4, size_t(batch4f::size));
        CHECK_EQ(4, size_t(batch4c::size));
        CHECK_EQ(4, size_t(batch4i32::size));
        CHECK_EQ(4, size_t(batch4u32::size));

        CHECK_UNARY(bool(std::is_same<float, batch4f::value_type>::value));
        CHECK_UNARY(bool(std::is_same<std::complex<float>, batch4c::value_type>::value));
        CHECK_UNARY(bool(std::is_same<int32_t, batch4i32::value_type>::value));
        CHECK_UNARY(bool(std::is_same<uint32_t, batch4u32::value_type>::value));

#if XSIMD_WITH_SSE2 || XSIMD_WITH_NEON64 || XSIMD_WITH_SVE || XSIMD_WITH_RVV
        CHECK_EQ(2, size_t(batch2d::size));
        CHECK_EQ(2, size_t(batch2z::size));
        CHECK_UNARY(bool(std::is_same<double, batch2d::value_type>::value));
        CHECK_UNARY(bool(std::is_same<std::complex<double>, batch2z::value_type>::value));
#else
        CHECK_UNARY(bool(std::is_same<void, batch2d>::value));
#endif

#endif
#if !XSIMD_WITH_AVX && !XSIMD_WITH_FMA3 && !(XSIMD_WITH_SVE && XSIMD_SVE_BITS == 256) && !(XSIMD_WITH_RVV && XSIMD_RVV_BITS == 256)
        CHECK_UNARY(bool(std::is_same<void, batch8f>::value));
        CHECK_UNARY(bool(std::is_same<void, batch4d>::value));
        CHECK_UNARY(bool(std::is_same<void, batch8c>::value));
        CHECK_UNARY(bool(std::is_same<void, batch4z>::value));
        CHECK_UNARY(bool(std::is_same<void, batch8i32>::value));
        CHECK_UNARY(bool(std::is_same<void, batch8u32>::value));
#else
        CHECK_EQ(8, size_t(batch8f::size));
        CHECK_EQ(8, size_t(batch8i32::size));
        CHECK_EQ(8, size_t(batch8u32::size));
        CHECK_EQ(4, size_t(batch4d::size));

        CHECK_EQ(8, size_t(batch8c::size));
        CHECK_EQ(4, size_t(batch4z::size));

        CHECK_UNARY(bool(std::is_same<float, batch8f::value_type>::value));
        CHECK_UNARY(bool(std::is_same<double, batch4d::value_type>::value));
        CHECK_UNARY(bool(std::is_same<int32_t, batch8i32::value_type>::value));
        CHECK_UNARY(bool(std::is_same<uint32_t, batch8u32::value_type>::value));
        CHECK_UNARY(bool(std::is_same<std::complex<float>, batch8c::value_type>::value));
        CHECK_UNARY(bool(std::is_same<std::complex<double>, batch4z::value_type>::value));
#endif
    }

    SUBCASE("xsimd::load_(un)aligned(...) return type")
    {
        // make sure load_aligned / load_unaligned work for the default arch and
        // return the appropriate type.
        using type_list = xsimd::mpl::type_list<short, int, long, float, std::complex<float>
#if XSIMD_WITH_NEON64 || !XSIMD_WITH_NEON
                                                ,
                                                double, std::complex<double>
#endif
                                                >;
        try_loads<type_list>();
    }
}

#endif
