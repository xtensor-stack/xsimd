/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 * Copyright (c) Marco Barbone                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "xsimd/xsimd.hpp"
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE

#include "test_utils.hpp"

// Compile time tests for include/xsimd/arch/common/xsimd_common_swizzle.hpp
namespace xsimd
{
    namespace kernel
    {
        namespace detail
        {
            // ────────────────────────────────────────────────────────────────────────
            //  compile-time tests (identity, all-different, dup-lo, dup-hi)
            //  8-lane identity
            static_assert(is_identity<std::uint32_t, 0, 1, 2, 3, 4, 5, 6, 7>(), "identity failed");
            // 8-lane reverse is all-different but not identity
            static_assert(is_all_different<std::uint32_t, 7, 6, 5, 4, 3, 2, 1, 0>(), "all-diff failed");
            static_assert(!is_identity<std::uint32_t, 7, 6, 5, 4, 3, 2, 1, 0>(), "identity on reverse");
            // 8-lane dup-lo (repeat 0..3 twice)
            static_assert(is_dup_lo<std::uint32_t, 0, 1, 2, 3, 0, 1, 2, 3>(), "dup_lo failed");
            static_assert(!is_dup_hi<std::uint32_t, 0, 1, 2, 3, 0, 1, 2, 3>(), "dup_hi on dup_lo");
            // 8-lane dup-hi (repeat 4..7 twice)
            static_assert(is_dup_hi<std::uint32_t, 4, 5, 6, 7, 4, 5, 6, 7>(), "dup_hi failed");
            static_assert(!is_dup_lo<std::uint32_t, 4, 5, 6, 7, 4, 5, 6, 7>(), "dup_lo on dup_hi");
            // ────────────────────────────────────────────────────────────────────────
            //  4-lane identity
            static_assert(is_identity<std::uint32_t, 0, 1, 2, 3>(), "4-lane identity failed");
            // 4-lane reverse all-different but not identity
            static_assert(is_all_different<std::uint32_t, 3, 2, 1, 0>(), "4-lane all-diff failed");
            static_assert(!is_identity<std::uint32_t, 3, 2, 1, 0>(), "4-lane identity on reverse");
            // 4-lane dup-lo (repeat 0..1 twice)
            static_assert(is_dup_lo<std::uint32_t, 0, 1, 0, 1>(), "4-lane dup_lo failed");
            static_assert(!is_dup_hi<std::uint32_t, 0, 1, 0, 1>(), "4-lane dup_hi on dup_lo");
            // 4-lane dup-hi (repeat 2..3 twice)
            static_assert(is_dup_hi<std::uint32_t, 2, 3, 2, 3>(), "4-lane dup_hi failed");
            static_assert(!is_dup_lo<std::uint32_t, 2, 3, 2, 3>(), "4-lane dup_lo on dup_hi");

            static_assert(is_cross_lane<0, 1, 0, 1>(), "dup-lo only → crossing");
            static_assert(is_cross_lane<2, 3, 2, 3>(), "dup-hi only → crossing");
            static_assert(is_cross_lane<0, 3, 3, 3>(), "one low + rest high → crossing");
            static_assert(!is_cross_lane<1, 0, 2, 3>(), "mixed low/high → no crossing");
            static_assert(!is_cross_lane<0, 1, 2, 3>(), "mixed low/high → no crossing");

            static_assert(no_duplicates_v<0, 1, 2, 3>(), "N=4: [0,1,2,3] → distinct");
            static_assert(!no_duplicates_v<0, 1, 2, 2>(), "N=4: [0,1,2,2] → dup");

            static_assert(no_duplicates_v<0, 1, 2, 3, 4, 5, 6, 7>(), "N=8: [0..7] → distinct");
            static_assert(!no_duplicates_v<0, 1, 2, 3, 4, 5, 6, 0>(), "N=8: last repeats 0");
        }
    }
}

namespace xsimd
{
    template <template <class> class Pattern, class Vec>
    void fill_pattern(Vec& dst, const Vec& src)
    {
        using size_type = typename Vec::size_type;
        for (size_type i = 0; i < src.size(); ++i)
        {
            dst[i] = src[Pattern<size_type>::get(i, static_cast<size_type>(src.size()))];
        }
    }

    template <class T>
    struct Reversor
    {
        static constexpr T get(T i, T n) { return n - 1 - i; }
    };
    template <class T>
    struct Last
    {
        static constexpr T get(T, T n) { return n - 1; }
    };
    template <class T>
    struct DupReal
    {
        static constexpr T get(T i, T) { return (i & ~T { 1 }); }
    };

    template <class T>
    struct DupImag
    {
        static constexpr T get(T i, T) { return (i & ~T { 1 }) + 1; }
    };
    template <class T>
    struct SwapRI
    {
        static constexpr T get(T i, T)
        {
            return i ^ T { 1 };
        }
    };
    template <class T>
    struct Identity
    {
        static constexpr T get(T i, T) { return i; }
    };
    template <class T>
    struct DupLowPair
    {
        static constexpr T get(T i, T) { return i / 2; }
    };
    template <class T>
    struct DupHighPair
    {
        static constexpr T get(T i, T n) { return n / 2 + i / 2; }
    };

    template <class T>
    struct RotateRight1
    {
        static constexpr T get(T i, T n) { return (i + n - 1) % n; }
    };
    template <class T>
    struct RotateLeft1
    {
        static constexpr T get(T i, T n) { return (i + 1) % n; }
    };

    template <class T>
    struct ReversePairs
    {
        static constexpr T get(T i, T) { return (i & ~T { 1 }) | (1 - (i & T { 1 })); }
    };
    template <class T>
    struct EvenThenOdd
    {
        static constexpr T get(T i, T n)
        {
            return (i < n / 2 ? 2 * i : 2 * (i - n / 2) + 1);
        }
    };
    template <class T>
    struct OddThenEven
    {
        static constexpr T get(T i, T n)
        {
            return (i < n / 2 ? 2 * i + 1 : 2 * (i - n / 2));
        }
    };
    template <class T>
    struct InterleavePairs
    {
        static constexpr T get(T i, T n)
        {
            return (i & 1) ? (i / 2 + n / 2) : (i / 2);
        }
    };
    template <class T>
    struct as_index
    {
        using type = xsimd::as_unsigned_integer_t<T>;
    };

    template <class T>
    struct as_index<std::complex<T>> : as_index<T>
    {
    };
} // namespace xsimd

//------------------------------------------------------------------------------
// insert_test: unchanged from original
//------------------------------------------------------------------------------
template <class B>
struct insert_test
{
    using batch_type = B;
    using value_type = typename B::value_type;

    void insert_first()
    {
        value_type fill_value = 0;
        value_type sentinel_value = 1;
        batch_type v(fill_value);
        batch_type w = xsimd::insert(v, sentinel_value, xsimd::index<0>());
        std::array<value_type, batch_type::size> data {};
        w.store_unaligned(data.data());
        CHECK_SCALAR_EQ(data.front(), sentinel_value);
        for (std::size_t i = 1; i < batch_type::size; ++i)
            CHECK_SCALAR_EQ(data[i], fill_value);
    }

    void insert_last()
    {
        value_type fill_value = 0;
        value_type sentinel_value = 1;
        batch_type v(fill_value);
        batch_type w = xsimd::insert(v, sentinel_value,
                                     xsimd::index<batch_type::size - 1>());
        std::array<value_type, batch_type::size> data {};
        w.store_unaligned(data.data());
        for (std::size_t i = 0; i < batch_type::size - 1; ++i)
            CHECK_SCALAR_EQ(data[i], fill_value);
        CHECK_SCALAR_EQ(data.back(), sentinel_value);
    }
};

TEST_CASE_TEMPLATE("[insert_test]", B, BATCH_TYPES)
{
    insert_test<B> Test;
    SUBCASE("insert_first") { Test.insert_first(); }
    SUBCASE("insert_last") { Test.insert_last(); }
}

template <class B>
struct swizzle_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    using arch_type = typename B::arch_type;
    static constexpr std::size_t N = B::size;
    using vec_t = std::array<value_type, N>;

    // Build the input [1,3,5,...,2N-1]
    static vec_t make_lhs()
    {
        vec_t v;
        for (std::size_t i = 0; i < N; ++i)
            v[i] = static_cast<value_type>(2 * i + 1);
        return v;
    }

    template <template <class> class Pattern>
    void run()
    {
        vec_t lhs = make_lhs();
        vec_t expect = lhs;
        xsimd::fill_pattern<Pattern>(expect, lhs);

        auto b_lhs = batch_type::load_unaligned(lhs.data());
        auto b_expect = batch_type::load_unaligned(expect.data());

        using idx_t = typename xsimd::as_index<value_type>::type;
        auto idx_batch = xsimd::make_batch_constant<idx_t, Pattern<idx_t>, arch_type>();

        CHECK_BATCH_EQ(xsimd::swizzle(b_lhs, idx_batch), b_expect);
        CHECK_BATCH_EQ(xsimd::swizzle(b_lhs,
                                      static_cast<xsimd::batch<idx_t, arch_type>>(idx_batch)),
                       b_expect);
    }

    void rotate_right()
    {
        vec_t lhs = make_lhs(), expect;
        std::rotate_copy(lhs.begin(), lhs.end() - 1, lhs.end(), expect.begin());
        CHECK_BATCH_EQ(xsimd::rotate_right<1>(batch_type::load_unaligned(lhs.data())),
                       batch_type::load_unaligned(expect.data()));
    }
    void rotate_left()
    {
        vec_t lhs = make_lhs(), expect;
        std::rotate_copy(lhs.begin(), lhs.begin() + 1, lhs.end(), expect.begin());
        CHECK_BATCH_EQ(xsimd::rotate_left<1>(batch_type::load_unaligned(lhs.data())),
                       batch_type::load_unaligned(expect.data()));
    }
    void rotate_left_inv()
    {
        vec_t lhs = make_lhs(), expect;
        std::rotate_copy(lhs.begin(), lhs.end() - 1, lhs.end(), expect.begin());
        CHECK_BATCH_EQ(xsimd::rotate_left<N - 1>(batch_type::load_unaligned(lhs.data())),
                       batch_type::load_unaligned(expect.data()));
    }
};

// Macro to instantiate one SUBCASE per pattern
#define XSIMD_SWIZZLE_PATTERN_CASE(PAT) \
    SUBCASE(#PAT) { swizzle_test<B>().template run<xsimd::PAT>(); }

TEST_CASE_TEMPLATE("[swizzle]", B, BATCH_SWIZZLE_TYPES)
{
    // All existing patterns:
    XSIMD_SWIZZLE_PATTERN_CASE(Reversor);
    XSIMD_SWIZZLE_PATTERN_CASE(Last);
    XSIMD_SWIZZLE_PATTERN_CASE(DupReal);
    XSIMD_SWIZZLE_PATTERN_CASE(DupImag);
    XSIMD_SWIZZLE_PATTERN_CASE(SwapRI);
    XSIMD_SWIZZLE_PATTERN_CASE(Identity);
    XSIMD_SWIZZLE_PATTERN_CASE(DupLowPair);
    XSIMD_SWIZZLE_PATTERN_CASE(DupHighPair);
    XSIMD_SWIZZLE_PATTERN_CASE(RotateRight1);
    XSIMD_SWIZZLE_PATTERN_CASE(RotateLeft1);
    XSIMD_SWIZZLE_PATTERN_CASE(ReversePairs);
    XSIMD_SWIZZLE_PATTERN_CASE(EvenThenOdd);
    XSIMD_SWIZZLE_PATTERN_CASE(OddThenEven);
    XSIMD_SWIZZLE_PATTERN_CASE(InterleavePairs);
    // Rotation checks:
    SUBCASE("rotate_left") { swizzle_test<B>().rotate_left(); }
    SUBCASE("rotate_left_inv") { swizzle_test<B>().rotate_left_inv(); }
    SUBCASE("rotate_right") { swizzle_test<B>().rotate_right(); }
}

#undef XSIMD_SWIZZLE_PATTERN_CASE

#endif /* XSIMD_NO_SUPPORTED_ARCHITECTURE */
