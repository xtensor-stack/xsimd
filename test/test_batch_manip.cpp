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

#include "test_utils.hpp"

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

    /*  Existing patterns kept for backward compatibility  */

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
    struct DupReal /* 0,0,2,2,… */
    {
        static constexpr T get(T i, T) { return (i & ~T { 1 }); }
    };

    /*  New patterns requested  */

    template <class T>
    struct DupImag /* 1,1,3,3,… */
    {
        static constexpr T get(T i, T) { return (i & ~T { 1 }) + 1; }
    };

    template <class T>
    struct SwapRI /* 1,0,3,2,… swap real <-> imag per complex number */
    {
        static constexpr T get(T i, T)
        {
            return i ^ T { 1 };
        }
    };

    template <class T>
    struct Identity /* 0,1,2,3,… */
    {
        static constexpr T get(T i, T) { return i; }
    };

    template <class T>
    struct DupLowPair /* 0,0,1,1,… */
    {
        static constexpr T get(T i, T) { return i / 2; }
    };

    template <class T>
    struct DupHighPair /* n/2,n/2,n/2+1,n/2+1,… */
    {
        static constexpr T get(T i, T n) { return n / 2 + i / 2; }
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

    template <typename T, std::size_t N>
    struct init_swizzle_base
    {
        using swizzle_vector_type = std::array<T, N>;

        swizzle_vector_type lhs_in {},
            exped_reverse {}, exped_fill {}, exped_dup {},
            exped_ror {}, exped_rol {}, exped_rol2 {},
            /* new patterns */
            exped_dup_imag {}, exped_swap_ri {}, exped_identity {},
            exped_dup_low {}, exped_dup_high {}, exped_generic {};

        template <int... Indices>
        std::vector<swizzle_vector_type> create_swizzle_vectors()
        {
            std::vector<swizzle_vector_type> vects;

            /* 0) Generate input data */
            for (std::size_t i = 0; i < N; ++i)
            {
                lhs_in[i] = static_cast<T>(2 * i + 1);
            }
            vects.push_back(lhs_in);

            /* 1‑3) Original expectations */
            for (std::size_t i = 0; i < N; ++i)
            {
                exped_reverse[i] = lhs_in[N - 1 - i];
                exped_fill[i] = lhs_in[N - 1];
                exped_dup[i] = lhs_in[(i & ~std::size_t { 1 })];
                exped_ror[i] = lhs_in[(i + N - 1) % N];
                exped_rol[i] = lhs_in[(i + 1) % N];
                exped_rol2[i] = lhs_in[(i + N - 1) % N]; // rotate_left<N-1>
            }

            /*  New expectations built through generic helper  */
            fill_pattern<DupImag>(exped_dup_imag, lhs_in);
            fill_pattern<SwapRI>(exped_swap_ri, lhs_in);
            fill_pattern<Identity>(exped_identity, lhs_in);
            fill_pattern<DupLowPair>(exped_dup_low, lhs_in);
            fill_pattern<DupHighPair>(exped_dup_high, lhs_in);

            /* Push in the original order, then the new ones */
            vects.push_back(exped_reverse); // 1
            vects.push_back(exped_fill); // 2
            vects.push_back(exped_dup); // 3
            vects.push_back(exped_ror); // 4
            vects.push_back(exped_rol); // 5
            vects.push_back(exped_rol2); // 6

            vects.push_back(exped_dup_imag); // 7
            vects.push_back(exped_swap_ri); // 8
            vects.push_back(exped_identity); // 9
            vects.push_back(exped_dup_low); // 10
            vects.push_back(exped_dup_high); // 11
            vects.push_back(exped_generic); // 12

            return vects;
        }
    };
    template <class B>
    struct insert_test
    {
        using batch_type = B;
        using value_type = typename B::value_type;
        static constexpr std::size_t size = B::size;

        void insert_first()
        {
            value_type fill_value = 0;
            value_type sentinel_value = 1;
            batch_type v(fill_value);
            batch_type w = insert(v, sentinel_value, ::xsimd::index<0>());
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
            batch_type w = insert(v, sentinel_value, ::xsimd::index<batch_type::size - 1>());
            std::array<value_type, batch_type::size> data {};
            w.store_unaligned(data.data());
            for (std::size_t i = 0; i < batch_type::size - 1; ++i)
                CHECK_SCALAR_EQ(data[i], fill_value);
            CHECK_SCALAR_EQ(data.back(), sentinel_value);
        }
    };
} // namespace xsimd

TEST_CASE_TEMPLATE("[insert_test]", B, BATCH_TYPES)
{
    xsimd::insert_test<B> Test;
    SUBCASE("insert_first") { Test.insert_first(); }
    SUBCASE("insert_last") { Test.insert_last(); }
}

template <class B>
struct swizzle_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    using arch_type = typename B::arch_type;
    static constexpr std::size_t size = B::size;

    template <template <class> class Pattern, std::size_t VectIndex>
    void check_swizzle_pattern()
    {
        xsimd::init_swizzle_base<value_type, size> swb;
        auto vecs = swb.create_swizzle_vectors();
        auto v_lhs = vecs[0];
        auto v_exped = vecs[VectIndex];

        batch_type b_lhs = batch_type::load_unaligned(v_lhs.data());
        batch_type b_exped = batch_type::load_unaligned(v_exped.data());

        using idx_t = typename xsimd::as_index<value_type>::type;
        auto idx_batch = xsimd::make_batch_constant<idx_t, Pattern<idx_t>, arch_type>();

        CHECK_BATCH_EQ(xsimd::swizzle(b_lhs, idx_batch), b_exped);
        CHECK_BATCH_EQ(xsimd::swizzle(b_lhs, static_cast<xsimd::batch<idx_t, arch_type>>(idx_batch)), b_exped);
    }

    void rotate_right()
    {
        xsimd::init_swizzle_base<value_type, size> sb;
        auto vv = sb.create_swizzle_vectors();
        batch_type b_lhs = batch_type::load_unaligned(vv[0].data());
        batch_type b_exped = batch_type::load_unaligned(vv[4].data());
        CHECK_BATCH_EQ(xsimd::rotate_right<1>(b_lhs), b_exped);
    }

    void rotate_left()
    {
        xsimd::init_swizzle_base<value_type, size> sb;
        auto vv = sb.create_swizzle_vectors();
        batch_type b_lhs = batch_type::load_unaligned(vv[0].data());
        batch_type b_exped = batch_type::load_unaligned(vv[5].data());
        CHECK_BATCH_EQ(xsimd::rotate_left<1>(b_lhs), b_exped);
    }

    void rotate_left_inv()
    {
        xsimd::init_swizzle_base<value_type, size> sb;
        auto vv = sb.create_swizzle_vectors();
        batch_type b_lhs = batch_type::load_unaligned(vv[0].data());
        batch_type b_exped = batch_type::load_unaligned(vv[6].data());
        CHECK_BATCH_EQ(xsimd::rotate_left<size - 1>(b_lhs), b_exped);
    }

    void swizzle_reverse() { check_swizzle_pattern<xsimd::Reversor, 1>(); }
    void swizzle_fill() { check_swizzle_pattern<xsimd::Last, 2>(); }
    void swizzle_dup() { check_swizzle_pattern<xsimd::DupReal, 3>(); }
    void dup_imag() { check_swizzle_pattern<xsimd::DupImag, 7>(); }
    void swap_ri() { check_swizzle_pattern<xsimd::SwapRI, 8>(); }
    void identity() { check_swizzle_pattern<xsimd::Identity, 9>(); }
    void dup_low_pair() { check_swizzle_pattern<xsimd::DupLowPair, 10>(); }
    void dup_high_pair() { check_swizzle_pattern<xsimd::DupHighPair, 11>(); }
};

TEST_CASE_TEMPLATE("[swizzle]", B, BATCH_SWIZZLE_TYPES)
{
    swizzle_test<B> t;

    SUBCASE("reverse") { t.swizzle_reverse(); }

    SUBCASE("rotate")
    {
        t.rotate_left();
        t.rotate_left_inv();
        t.rotate_right();
    }

    SUBCASE("fill") { t.swizzle_fill(); }
    SUBCASE("dup real") { t.swizzle_dup(); }
    SUBCASE("dup imag") { t.dup_imag(); }
    SUBCASE("swap R/I") { t.swap_ri(); }
    SUBCASE("identity") { t.identity(); }
    SUBCASE("dup low pair") { t.dup_low_pair(); }
    SUBCASE("dup high pair") { t.dup_high_pair(); }
}

#endif /* XSIMD_NO_SUPPORTED_ARCHITECTURE */
