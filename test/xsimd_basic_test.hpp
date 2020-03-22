/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_BASIC_TEST_HPP
#define XSIMD_BASIC_TEST_HPP

#include <algorithm>
#include <numeric>
#include <random>
#include <limits>
#include <climits>

#include "xsimd/xsimd.hpp"
#include "xsimd_test_utils.hpp"
#include "xsimd_tester.hpp"

namespace xsimd
{

    /****************
     * basic tester *
     ****************/

    template <class T, std::size_t N, std::size_t A>
    struct simd_basic_tester : simd_tester<T, N, A>
    {
        using base_type = simd_tester<T, N, A>;
        using vector_type = typename base_type::vector_type;
        using value_type = typename base_type::value_type;
        using res_type = typename base_type::res_type;

        std::string name;

        value_type s;
        res_type lhs;
        res_type rhs;
        res_type mix_lhs_rhs;

        value_type extract_res;
        res_type minus_res;
        res_type plus_res;
        res_type add_vv_res;
        res_type add_vs_res;
        res_type add_sv_res;
        res_type sub_vv_res;
        res_type sub_vs_res;
        res_type sub_sv_res;
        res_type mul_vv_res;
        res_type mul_vs_res;
        res_type mul_sv_res;
        res_type div_vv_res;
        res_type div_vs_res;
        res_type div_sv_res;
        res_type and_res;
        res_type or_res;
        res_type xor_res;
        res_type not_res;
        res_type lnot_res;
        res_type land_res;
        res_type lor_res;
        res_type min_res;
        res_type max_res;
        res_type abs_res;
        res_type fabs_res;
        res_type fma_res;
        res_type fms_res;
        res_type fnma_res;
        res_type fnms_res;
        res_type sqrt_res;
        value_type hadd_res;
        res_type haddp_res;
        res_type false_res;
        res_type true_res;

        simd_basic_tester(const std::string& name);
    };


    template <class T, size_t N, size_t A>
    simd_basic_tester<T, N, A>::simd_basic_tester(const std::string& n)
        : name(n)
    {
        using std::min;
        using std::max;
        using std::abs;
        using std::fabs;
        using std::sqrt;
        using std::fma;

        lhs.resize(N);
        rhs.resize(N);
        mix_lhs_rhs.resize(N);
        minus_res.resize(N);
        plus_res.resize(N);
        add_vv_res.resize(N);
        add_vs_res.resize(N);
        add_sv_res.resize(N);
        sub_vv_res.resize(N);
        sub_vs_res.resize(N);
        sub_sv_res.resize(N);
        mul_vv_res.resize(N);
        mul_vs_res.resize(N);
        mul_sv_res.resize(N);
        div_vv_res.resize(N);
        div_vs_res.resize(N);
        div_sv_res.resize(N);
        and_res.resize(N);
        or_res.resize(N);
        xor_res.resize(N);
        not_res.resize(N);
        lnot_res.resize(N);
        land_res.resize(N);
        lor_res.resize(N);
        min_res.resize(N);
        max_res.resize(N);
        abs_res.resize(N);
        fabs_res.resize(N);
        fma_res.resize(N);
        fms_res.resize(N);
        fnma_res.resize(N);
        fnms_res.resize(N);
        sqrt_res.resize(N);
        haddp_res.resize(N);
        false_res.resize(N);
        true_res.resize(N);

        s = value_type(1.4);
        hadd_res = value_type(0);
        for (size_t i = 0; i < N; ++i)
        {
            lhs[i] = value_type(i) / 4 + value_type(1.2) * std::sqrt(value_type(i + 0.25));
            rhs[i] = value_type(10.2) / (i + 2) + value_type(0.25);
            extract_res = lhs[1];
            minus_res[i] = -lhs[i];
            plus_res[i] = +lhs[i];
            add_vv_res[i] = lhs[i] + rhs[i];
            add_vs_res[i] = lhs[i] + s;
            add_sv_res[i] = s + rhs[i];
            sub_vv_res[i] = lhs[i] - rhs[i];
            sub_vs_res[i] = lhs[i] - s;
            sub_sv_res[i] = s - rhs[i];
            mul_vv_res[i] = lhs[i] * rhs[i];
            mul_vs_res[i] = lhs[i] * s;
            mul_sv_res[i] = s * rhs[i];
            div_vv_res[i] = lhs[i] / rhs[i];
            div_vs_res[i] = lhs[i] / s;
            div_sv_res[i] = s / rhs[i];
            // and_res[i] = lhs[i] & rhs[i];
            // or_res[i] = lhs[i] | rhs[i];
            // xor_res[i] = lhs[i] ^ rhs[i];
            // not_res[i] = ~lhs[i];
            // lnot_res[i] = !lhs[i];
            // land_res[i] = lhs[i] && rhs[i];
            // lor_res[i] = lhs[i] || rhs[i];
            min_res[i] = min(lhs[i], rhs[i]);
            max_res[i] = max(lhs[i], rhs[i]);
            abs_res[i] = abs(lhs[i]);
            fabs_res[i] = fabs(lhs[i]);
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_FMA4_VERSION
            fma_res[i] = fma(lhs[i], rhs[i], rhs[i]);
#else
            fma_res[i] = lhs[i] * rhs[i] + rhs[i];
#endif
            fms_res[i] = lhs[i] * rhs[i] - rhs[i];
            fnma_res[i] = -lhs[i] * rhs[i] + rhs[i];
            fnms_res[i] = -lhs[i] * rhs[i] - rhs[i];
            sqrt_res[i] = sqrt(lhs[i]);
            hadd_res += lhs[i];
            for(size_t j = 0; j < base_type::size; j += 2)
            {
                haddp_res[j] += lhs[i];
                if(j + 1 < base_type::size)
                {
                    haddp_res[j + 1] += rhs[i];
                }
            }
            false_res[i] = value_type(0);
            true_res[i] = value_type(1);
        }
        for (size_t i = 0; i < N / 2; ++i)
        {
            mix_lhs_rhs[2 * i] = lhs[2 * i];
            mix_lhs_rhs[2 * i + 1] = rhs[2 * i + 1];
        }
    }

    template <class T, std::size_t N, std::size_t A>
    struct simd_int_basic_tester : simd_tester<T, N, A>
    {
        using base_type = simd_tester<T, N, A>;
        using vector_type = typename base_type::vector_type;
        using value_type = typename base_type::value_type;
        using res_type = typename base_type::res_type;
        using signed_value_type = typename std::make_signed<value_type>::type;
        using signed_vector_type = batch<signed_value_type, N>;
        using signed_res_type = std::vector<signed_value_type, aligned_allocator<signed_value_type, A>>;

        std::string name;

        value_type s;
        int32_t sh_nb;
        res_type lhs;
        res_type rhs;
        res_type mix_lhs_rhs;
        signed_res_type shift;

        value_type extract_res;
        res_type minus_res;
        res_type plus_res;
        res_type add_vv_res;
        res_type add_vs_res;
        res_type add_sv_res;
        res_type sub_vv_res;
        res_type sub_vs_res;
        res_type sub_sv_res;
        res_type mul_vv_res;
        res_type mul_vs_res;
        res_type mul_sv_res;
        res_type div_vv_res;
        res_type mod_vv_res;
        res_type div_vs_res;
        res_type div_sv_res;
        res_type and_res;
        res_type or_res;
        res_type xor_res;
        res_type not_res;
        res_type lnot_res;
        res_type land_res;
        res_type lor_res;
        res_type min_res;
        res_type max_res;
        res_type abs_res;
        res_type fma_res;
        res_type fms_res;
        res_type fnma_res;
        res_type fnms_res;
        value_type hadd_res;
        res_type sl_s_res;
        res_type sl_v_res;
        res_type sr_s_res;
        res_type sr_v_res;
        res_type false_res;
        res_type true_res;

        simd_int_basic_tester(const std::string& name);
    };


    template <class T, size_t N, size_t A>
    simd_int_basic_tester<T, N, A>::simd_int_basic_tester(const std::string& n)
        : name(n)
    {
        using std::min;
        using std::max;
        using std::abs;

        lhs.resize(N);
        rhs.resize(N);
        mix_lhs_rhs.resize(N);
        shift.resize(N);
        minus_res.resize(N);
        plus_res.resize(N);
        add_vv_res.resize(N);
        add_vs_res.resize(N);
        add_sv_res.resize(N);
        sub_vv_res.resize(N);
        sub_vs_res.resize(N);
        sub_sv_res.resize(N);
        mul_vv_res.resize(N);
        mul_vs_res.resize(N);
        mul_sv_res.resize(N);
        div_vv_res.resize(N);
        div_vs_res.resize(N);
        div_sv_res.resize(N);
        mod_vv_res.resize(N);
        and_res.resize(N);
        or_res.resize(N);
        xor_res.resize(N);
        not_res.resize(N);
        lnot_res.resize(N);
        land_res.resize(N);
        lor_res.resize(N);
        min_res.resize(N);
        max_res.resize(N);
        abs_res.resize(N);
        fma_res.resize(N);
        fms_res.resize(N);
        fnma_res.resize(N);
        fnms_res.resize(N);
        sl_s_res.resize(N);
        sl_v_res.resize(N);
        sr_s_res.resize(N);
        sr_v_res.resize(N);
        false_res.resize(N);
        true_res.resize(N);

        s = value_type(1.4);
        sh_nb = 3;
        hadd_res = value_type(0);
        for (size_t i = 0; i < N; ++i)
        {
            bool negative_lhs = std::is_signed<T>::value && (i % 2 == 1);
            lhs[i] = value_type(i) * (negative_lhs ? -10 : 10);
            rhs[i] = value_type(i) + value_type(4);
            shift[i] = signed_value_type(i) % (CHAR_BIT * sizeof(value_type));
            extract_res = lhs[1];
            minus_res[i] = -lhs[i];
            plus_res[i] = +lhs[i];
            add_vv_res[i] = lhs[i] + rhs[i];
            add_vs_res[i] = lhs[i] + s;
            add_sv_res[i] = s + rhs[i];
            sub_vv_res[i] = lhs[i] - rhs[i];
            sub_vs_res[i] = lhs[i] - s;
            sub_sv_res[i] = s - rhs[i];
            mul_vv_res[i] = lhs[i] * rhs[i];
            mul_vs_res[i] = lhs[i] * s;
            mul_sv_res[i] = s * rhs[i];
            div_vv_res[i] = lhs[i] / rhs[i];
            div_vs_res[i] = lhs[i] / s;
            div_sv_res[i] = s / rhs[i];
            mod_vv_res[i] = lhs[i] % rhs[i];
            and_res[i] = lhs[i] & rhs[i];
            or_res[i] = lhs[i] | rhs[i];
            xor_res[i] = lhs[i] ^ rhs[i];
            not_res[i] = ~lhs[i];
            lnot_res[i] = !lhs[i];
            land_res[i] = lhs[i] && rhs[i];
            lor_res[i] = lhs[i] || rhs[i];
            min_res[i] = min(lhs[i], rhs[i]);
            max_res[i] = max(lhs[i], rhs[i]);
            abs_res[i] = uabs(lhs[i]);
            fma_res[i] = lhs[i] * rhs[i] + rhs[i];
            fms_res[i] = lhs[i] * rhs[i] - rhs[i];
            fnma_res[i] = -lhs[i] * rhs[i] + rhs[i];
            fnms_res[i] = -lhs[i] * rhs[i] - rhs[i];
            hadd_res += lhs[i];
            sl_s_res[i] = lhs[i] << sh_nb;
            sl_v_res[i] = lhs[i] << shift[i];
            sr_s_res[i] = lhs[i] >> sh_nb;
            sr_v_res[i] = lhs[i] >> shift[i];
            false_res[i] = value_type(0);
            true_res[i] = value_type(1);
        }

        for (size_t i = 0; i < N / 2; ++i)
        {
            mix_lhs_rhs[2 * i] = lhs[2 * i];
            mix_lhs_rhs[2 * i + 1] = rhs[2 * i + 1];
        }
    }

    template <class T>
    struct get_bool;

    template <class T>
    struct get_bool<batch_bool<T, 2>>
    {
        using type = batch_bool<T, 2>;
        type all_true = type(true);
        type all_false = type(false);
        type half = type(0, 1);
        type ihalf = type(1, 0);
        type interspersed = type(0, 1);

        using vector_type = std::array<bool, 2>;
        vector_type vhalf = {{ false, true }};
        vector_type vihalf = {{ true, false }};
    };

    template <class T, std::size_t N>
    struct get_bool<batch_bool<T, N>>
    {
        // Expect this to be the fallback
        using type = batch_bool<T, 10>;
        type all_true = type(true);
        type all_false = type(false);
        type half = type(0, 0, 0, 0, 0, 1, 1, 1, 1, 1);
        type ihalf = type(1, 1, 1, 1, 1, 0, 0, 0, 0, 0);
        type interspersed = type(0, 1, 0, 1, 0, 1, 0, 1, 0, 1);

        using vector_type = std::array<bool, 10>;
        vector_type vhalf = {{ false, false, false, false, false, true, true, true, true, true }};
        vector_type vihalf = {{ true, true, true, true, true, false, false, false, false, false }};
    };

    template <class T>
    struct get_bool<batch_bool<T, 4>>
    {
        using type = batch_bool<T, 4>;

        type all_true = type(1);
        type all_false = type(0);
        type half = type(0, 0, 1, 1);
        type ihalf = type(1, 1, 0, 0);
        type interspersed = type(0, 1, 0, 1);

        using vector_type = std::array<bool, 4>;
        vector_type vhalf = {{ false, false, true, true }};
        vector_type vihalf = {{ true, true, false, false }};
    };

    template <class T>
    struct get_bool<batch_bool<T, 8>>
    {
        using type = batch_bool<T, 8>;
        type all_true = type(true);
        type all_false = type(false);
        type half = type(0, 0, 0, 0, 1, 1, 1, 1);
        type ihalf = type(1, 1, 1, 1, 0, 0, 0, 0);
        type interspersed = type(0, 1, 0, 1, 0, 1, 0, 1);

        using vector_type = std::array<bool, 8>;
        vector_type vhalf = {{ false, false, false, false, true, true, true, true }};
        vector_type vihalf = {{ true, true, true, true, false, false, false, false }};
    };

    template <class T>
    struct get_bool<batch_bool<T, 16>>
    {
        using type = batch_bool<T, 16>;
        type all_true = type(true);
        type all_false = type(false);
        type half = type(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
        type ihalf = type(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
        type interspersed = type(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);

        using vector_type = std::array<bool, 16>;
        vector_type vhalf = {{ false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true }};
        vector_type vihalf = {{ true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false }};
    };

    template <class T>
    struct get_bool<batch_bool<T, 32>>
    {
        using type = batch_bool<T, 32>;
        type all_true = type(true);
        type all_false = type(false);
        type half = type(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        type ihalf = type(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        type interspersed = type(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);

        using vector_type = std::array<bool, 32>;
        vector_type vhalf = {{ false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
                              true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true }};
        vector_type vihalf = {{ true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
                               false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false }};
    };

    template <class T>
    struct get_bool<batch_bool<T, 64>>
    {
        using type = batch_bool<T, 64>;
        type all_true = type(true);
        type all_false = type(false);
        type half = type (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        type ihalf = type(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        type interspersed = type(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);

        using vector_type = std::array<bool, 64>;
        vector_type vhalf = {{ false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
                              false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
                              true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
                              true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true }};
        vector_type vihalf = {{ true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
                               true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
                               false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
                               false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false }};
    };

    template <class T>
    struct test_more_int
    {
        bool run()
        {
            return true;
        }
    };

    template <class T, std::size_t N>
    bool stored_equal(batch<T, N>& b, const std::array<T, N>& arr)
    {
        std::array<T, N> stored;
        b.store_unaligned(stored.data());
        bool result = true;

        for (std::size_t i = 0; i < N; ++i)
        {
            result = result && (stored[i] == arr[i]);
        }
        return result;
    }

    template <class T>
    struct test_more_int<batch<T, 2>>
    {
        void run()
        {
            using B = batch<T, 2>;
            using BB = batch_bool<T, 2>;

            T max = std::numeric_limits<T>::max();
            T min = std::numeric_limits<T>::min();
            std::array<T, 2> maxmin_cmp{max, min};
            B maxmin(max, min);
            EXPECT_TRUE(stored_equal(maxmin, maxmin_cmp));

            B a(1, 3);
            B b(2);
            B c(2, 3);

            auto r1 = xsimd::max(a, c);
            auto r2 = xsimd::abs(a);
            auto r3 = xsimd::min(a, c);

            EXPECT_TRUE(stored_equal(r1, {2, 3}));
            EXPECT_TRUE(stored_equal(r3, {1, 3}));

            auto r4 = a < b; // test lt
            BB e4(1, 0);
            EXPECT_TRUE(xsimd::all(r4 == e4));
        }
    };

    template <class T>
    struct test_more_int<batch<T, 4>>
    {
        void run()
        {
            using B = batch<T, 4>;
            using BB = batch_bool<T, 4>;

            B a(1,3,1,1);
            B b(2);
            B c(2,3,2,3);

            auto r1 = xsimd::max(a, c);
            auto r2 = xsimd::abs(a);
            auto r3 = xsimd::min(a, c);

            EXPECT_TRUE(stored_equal(r1, {2, 3, 2, 3}));
            EXPECT_TRUE(stored_equal(r3, {1, 3, 1, 1}));

            auto r4 = a < b; // test lt
            BB e4(1,0,1,1);
            EXPECT_TRUE(xsimd::all(r4 == e4));
        }
    };

    template <class T>
    struct test_more_int<batch<T, 8>>
    {
        void run()
        {
            using B = batch<T, 8>;
            using BB = batch_bool<T, 8>;

            T max = std::numeric_limits<T>::max();
            T min = std::numeric_limits<T>::min();
            std::array<T, 8> maxmin_cmp{0, 0, max, 0, min, 0, 0, 0};
            B maxmin(0, 0, max, 0, min, 0, 0, 0);
            stored_equal(maxmin, maxmin_cmp);

            B a(1,3,1,3, 1,1,3,3);
            B b(2);
            B c(2,3,2,3, 2,3,2,3);

            auto r1 = xsimd::max(a, c);
            auto r2 = xsimd::abs(a);
            auto r3 = xsimd::min(a, c);
            auto r4 = a < b; // test lt

            BB e4(1,0,1,0, 1,1,0,0);
            EXPECT_TRUE(xsimd::all(r4 == e4));
        }
    };

    template <class T>
    struct test_more_int<batch<T, 16>>
    {
        void run()
        {
            using B = batch<T, 16>;
            using BB = batch_bool<T, 16>;

            T max = std::numeric_limits<T>::max();
            T min = std::numeric_limits<T>::min();
            std::array<T, 16> maxmin_cmp{0, 0, max, 0, min, 0, 0, 0, 0, 0, max, 0, min, 0, 0, 0};
            B maxmin(0, 0, max, 0, min, 0, 0, 0, 0, 0, max, 0, min, 0, 0, 0);
            stored_equal(maxmin, maxmin_cmp);

            B a(1,3,1,3, 1,3,1,3, 3,3,3,3, min,max,max,min);
            B b(2);
            B c(2,3,2,3, 2,3,2,3, 2,3,2,3, 2,3,2,3);
            auto r1 = xsimd::max(a, b);
            auto r2 = xsimd::abs(a);
            auto r3 = xsimd::min(a, b);
            auto r4 = a < b; // test lt
            auto r5 = a == c;
            auto r6 = a != c;

            BB e4(1,0,1,0, 1,0,1,0, 0,0,0,0, 1,0,0,1);
            EXPECT_TRUE(xsimd::all(r4 == e4));

            BB e5(0,1,0,1, 0,1,0,1, 0,1,0,1, 0,0,0,0);
            EXPECT_TRUE(xsimd::all(r5 == e5));
            EXPECT_TRUE(xsimd::all(r6 == !e5));
        }
    };

    template <class T>
    struct test_more_int<batch<T, 32>>
    {
        void run()
        {
            using B = batch<T, 32>;
            using BB = batch_bool<T, 32>;
            T max = std::numeric_limits<T>::max();
            T min = std::numeric_limits<T>::min();

            B a(1,3,1,3, 1,3,1,3, 1,3,1,3, 1,3,1,3, 1,3,1,3, 1,3,1,3, 3,3,3,3, min,max,max,min);
            B b(2);
            B c(2,3,2,3, 2,3,2,3, 2,3,2,3, 2,3,2,3, 2,3,2,3, 2,3,2,3, 2,3,2,3, 2,3,2,3);

            auto r1 = xsimd::max(a, b);
            auto r2 = xsimd::abs(a);
            auto r3 = xsimd::min(a, b);
            auto r4 = a < b; // test lt

            BB e4(1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0, 0,0,0,0, 1,0,0,1);
            EXPECT_TRUE(xsimd::all(r4 == e4));
        }
    };


#if defined(XSIMD_ENABLE_FALLBACK)
    template <class I, class S>
    bool test_simd_bool(const batch<I, 7>& /* empty */, S& /*stream*/)
    {
        using type = batch_bool<I, 7>;
        type all_true = type(true);
        type all_false = type(false);
        type half = type(std::array<bool, 7>{0, 0, 0, 0, 1, 1, 1});
        type ihalf = type(std::array<bool, 7>{1, 1, 1, 1, 0, 0, 0});

        bool success = true;
        success = success && all((half | ihalf) == all_true);
        success = success && all(half == half);
        return success;
    }

    template <class I, class S>
    bool test_simd_bool(const batch<I, 3>& /* empty */, S& /*stream*/)
    {
        using type = batch_bool<I, 3>;
        type all_true = type(true);
        type all_false = type(false);
        type half = type(std::array<bool, 3>{0, 0, 1});
        type ihalf = type(std::array<bool, 3>{1, 1, 0});

        bool success = true;
        success = success && all((half | ihalf) == all_true);
        success = success && all(half == half);
        return success;
    }
#endif

    template <class I, std::size_t N, class S>
    bool test_simd_bool(const batch<I, N>& /*empty*/, S& stream)
    {
        bool success = true;
        auto bool_g = get_bool<typename simd_batch_traits<batch<I, N>>::batch_bool_type>{};
        success = success && all(bool_g.half != bool_g.ihalf);
        if (!success)
            stream  << "test_simd_bool != failed." << std::endl;
        success = success && all(bool_g.half == !bool_g.ihalf);
        if (!success)
            stream  << "test_simd_bool ! failed." << std::endl;
        success = success && all(bool_g.half == ~bool_g.ihalf);
        if (!success)
            stream  << "test_simd_bool ~ failed." << std::endl;
        success = success && all((bool_g.half | bool_g.ihalf) == bool_g.all_true);
        if (!success)
            stream  << "test_simd_bool | failed." << std::endl;
        success = success && all((bool_g.half & bool_g.ihalf) == bool_g.all_false);
        if (!success)
            stream  << "test_simd_bool & failed." << std::endl;
        return success;
    }

    namespace detail
    {
        template <class I, std::size_t N, class S>
        bool test_simd_bool_buffer_impl(const std::array<I, N>& vhalf, const std::array<I, N>& vihalf, S& stream)
        {
            bool success = true;
            batch_bool<I, N> lhs, rhs, res;
            std::array<I, N> vres;

            detail::load_vec(lhs, vhalf);
            detail::load_vec(rhs, vihalf);

            res = lhs && rhs;
            detail::store_vec(res, vres);
            success = success && (std::count(vres.cbegin(), vres.cend(), false) ==  N);
            if(!success)
                stream << "test_simd_bool_buffer && failed." << std::endl;

            res = lhs || rhs;
            detail::store_vec(res, vres);
            success = success && (std::count(vres.cbegin(), vres.cend(), true) == N);
            if(!success)
                stream << "test_simd_bool_buffer || failed." << std::endl;

            return success;
        }
    }

#if defined(XSIMD_ENABLE_FALLBACK)
    template <class I, class S>
    bool test_simd_bool_buffer(const batch<I, 7>&, S& stream)
    {
        bool success = true;
        std::array<bool, 7> vhalf = {{ false, false, false, false, true, true, true }};
        std::array<bool, 7> vihalf = {{ true, true, true, true, false, false, false }};

        return detail::test_simd_bool_buffer_impl(vhalf, vihalf, stream);
    }

    template <class I, class S>
    bool test_simd_bool_buffer(const batch<I, 3>&, S& stream)
    {
        bool success = true;
        std::array<bool, 3> vhalf = {{ false, false, true }};
        std::array<bool, 3> vihalf = {{ true, true, false }};

        return detail::test_simd_bool_buffer_impl(vhalf, vihalf, stream);
    }
#endif

    template <class I, std::size_t N, class S>
    bool test_simd_bool_buffer(const batch<I, N>&, S& stream)
    {
        bool success = true;
        using batch_bool_type = typename simd_batch_traits<batch<I, N>>::batch_bool_type;
        using bool_getter = get_bool<batch_bool_type>;
        using vector_type = typename bool_getter::vector_type;

        bool_getter bool_g;
        batch_bool<I, N> lhs, rhs, res;
        vector_type vres;

        detail::load_vec(lhs, bool_g.vhalf);
        detail::load_vec(rhs, bool_g.vihalf);

        res = lhs && rhs;
        detail::store_vec(res, vres);
        success = success && (std::count(vres.cbegin(), vres.cend(), false) ==  N);
        if(!success)
            stream << "test_simd_bool_buffer && failed." << std::endl;

        res = lhs || rhs;
        detail::store_vec(res, vres);
        success = success && (std::count(vres.cbegin(), vres.cend(), true) == N);
        if(!success)
            stream << "test_simd_bool_buffer || failed." << std::endl;

        return success;
    }

    /***************
     * basic tests *
     ***************/

    template <class T>
    bool test_simd_basic(std::ostream& out, T& tester)
    {
        using tester_type = T;
        using vector_type = typename tester_type::vector_type;
        using value_type = typename tester_type::value_type;
        using res_type = typename tester_type::res_type;

        using vector_bool_type = typename simd_batch_traits<vector_type>::batch_bool_type;
        using bool_traits = simd_batch_traits<vector_bool_type>;

        static_assert(std::is_same<typename bool_traits::batch_type, vector_type>::value, "vector_bool type mismatch");
        static_assert(bool_traits::size == vector_type::size, "vector bool size mismatch");

        vector_type lhs;
        vector_type rhs;
        vector_type mix_lhs_rhs;
        vector_type vres;
        vector_bool_type bres;
        res_type res(tester_type::size);
        value_type s = tester.s;
        vector_type haddp_input[tester_type::size];
        bool success = true;
        bool tmp_success = true;

        std::string val_type = value_type_name<vector_type>();
        std::string shift = std::string(val_type.size(), '-');
        std::string name = tester.name;
        std::string name_shift = std::string(name.size(), '-');
        std::string dash(8, '-');
        std::string space(8, ' ');

        out << dash << name_shift << '-' << shift << dash << std::endl;
        out << space << name << " " << val_type << std::endl;
        out << dash << name_shift << '-' << shift << dash << std::endl
            << std::endl;

        using return_type1 = simd_return_type<value_type, value_type, vector_type::size>;
        using return_type2 = simd_return_type<bool, value_type, vector_type::size>;
        success = success && std::is_same<return_type1, vector_type>::value;
        success = success && std::is_same<return_type2, vector_bool_type>::value;

        std::string topic = "operator[]               : ";
        detail::load_vec(lhs, tester.lhs);
        value_type es = lhs[1];
        tmp_success = check_almost_equal(topic, es, tester.extract_res, out);
        success = success && tmp_success;
        value_type nvalue = value_type(2);
        lhs[1] = nvalue;
        tmp_success = check_almost_equal(topic, lhs[1], nvalue, out);
        lhs[1] = es;

        topic = "load/store aligned       : ";
        detail::load_vec(lhs, tester.lhs);
        detail::store_vec(lhs, res);
        tmp_success = check_almost_equal(topic, res, tester.lhs, out);
        success = success && tmp_success;

        topic = "load/store unaligned     : ";
        lhs.load_unaligned(&tester.lhs[0]);
        lhs.store_unaligned(&res[0]);
        tmp_success = check_almost_equal(topic, res, tester.lhs, out);
        success = success && tmp_success;

        detail::load_vec(lhs, tester.lhs);
        detail::load_vec(rhs, tester.rhs);
        detail::load_vec(mix_lhs_rhs, tester.mix_lhs_rhs);

        topic = "unary operator-          : ";
        vres = -lhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.minus_res, out);
        success = success && tmp_success;

        topic = "unary operator+          : ";
        vres = +lhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.plus_res, out);
        success = success && tmp_success;

        topic = "operator+=(simd, simd)   : ";
        vres = lhs;
        vres += rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_vv_res, out);
        success = success && tmp_success;

        topic = "operator+=(simd, scalar) : ";
        vres = lhs;
        vres += s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_vs_res, out);
        success = success && tmp_success;

        topic = "operator-=(simd, simd)   : ";
        vres = lhs;
        vres -= rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_vv_res, out);
        success = success && tmp_success;

        topic = "operator-=(simd, scalar) : ";
        vres = lhs;
        vres -= s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_vs_res, out);
        success = success && tmp_success;

        topic = "operator*=(simd, simd)   : ";
        vres = lhs;
        vres *= rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_vv_res, out);
        success = success && tmp_success;

        topic = "operator*=(simd, scalar) : ";
        vres = lhs;
        vres *= s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_vs_res, out);
        success = success && tmp_success;

        topic = "operator/=(simd, simd)   : ";
        vres = lhs;
        vres /= rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_vv_res, out);
        success = success && tmp_success;

        topic = "operator/=(simd, scalar) : ";
        vres = lhs;
        vres /= s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_vs_res, out);
        success = success && tmp_success;

        topic = "operator+(simd, simd)    : ";
        vres = lhs + rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_vv_res, out);
        success = success && tmp_success;

        topic = "operator+(simd, scalar)  : ";
        vres = lhs + s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_vs_res, out);
        success = success && tmp_success;

        topic = "operator+(scalar, simd)  : ";
        vres = s + rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_sv_res, out);
        success = success && tmp_success;

        topic = "operator-(simd, simd)    : ";
        vres = lhs - rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_vv_res, out);
        success = success && tmp_success;

        topic = "operator-(simd, scalar)  : ";
        vres = lhs - s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_vs_res, out);
        success = success && tmp_success;

        topic = "operator-(scalar, simd)  : ";
        vres = s - rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_sv_res, out);
        success = success && tmp_success;

        topic = "operator*(simd, simd)    : ";
        vres = lhs * rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_vv_res, out);
        success = success && tmp_success;

        topic = "operator*(simd, scalar)  : ";
        vres = lhs * s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_vs_res, out);
        success = success && tmp_success;

        topic = "operator*(scalar, simd)  : ";
        vres = s * rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_sv_res, out);
        success = success && tmp_success;

        topic = "operator/(simd, simd)    : ";
        vres = lhs / rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_vv_res, out);
        success = success && tmp_success;

        topic = "operator/(simd, scalar)  : ";
        vres = lhs / s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_vs_res, out);
        success = success && tmp_success;

        topic = "operator/(scalar, simd)  : ";
        vres = s / rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_sv_res, out);
        success = success && tmp_success;

        topic = "min(simd, simd)          : ";
        vres = min(lhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.min_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::min(tester.lhs[0], tester.rhs[0]), tester.min_res[0], out);

        topic = "max(simd, simd)          : ";
        vres = max(lhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.max_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::max(tester.lhs[0], tester.rhs[0]), tester.max_res[0], out);

        topic = "fmin(simd, simd)         : ";
        vres = fmin(lhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.min_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::fmin(tester.lhs[0], tester.rhs[0]), tester.min_res[0], out);

        topic = "fmax(simd, simd)         : ";
        vres = fmax(lhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.max_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::fmax(tester.lhs[0], tester.rhs[0]), tester.max_res[0], out);

        topic = "abs(simd)                : ";
        vres = abs(lhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.abs_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::abs(tester.lhs[0]), tester.abs_res[0], out);

        topic = "fabs(simd)                : ";
        vres = fabs(lhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.fabs_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::fabs(tester.lhs[0]), tester.fabs_res[0], out);

        topic = "sqrt(simd)               : ";
        vres = sqrt(lhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sqrt_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::sqrt(tester.lhs[0]), tester.sqrt_res[0], out);

        topic = "fma(simd, simd, simd)    : ";
        vres = fma(lhs, rhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.fma_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::fma(tester.lhs[0], tester.rhs[0], tester.rhs[0]),  tester.fma_res[0], out);

        topic = "fms(simd, simd, simd)    : ";
        vres = fms(lhs, rhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.fms_res, out);
        success = success && tmp_success;

        topic = "fnma(simd, simd, simd)   : ";
        vres = fnma(lhs, rhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.fnma_res, out);
        success = success && tmp_success;

        topic = "fnms(simd, simd, simd)   : ";
        vres = fnms(lhs, rhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.fnms_res, out);
        success = success && tmp_success;

        topic = "hadd(simd)               : ";
        value_type sres = hadd(lhs);
        tmp_success = check_almost_equal(topic, sres, tester.hadd_res, out);
        success = success && tmp_success;

        topic = "haddp(simd)              : ";
        for(size_t i = 0; i < tester_type::size; i += 2)
        {
            haddp_input[i] = lhs;
            if(i + 1 < tester_type::size)
            {
                haddp_input[i+1] = rhs;
            }
        }
        vres = haddp(haddp_input);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.haddp_res, out);
        success = success && tmp_success;

        topic = "conversion from true     : ";
        vector_bool_type tbt(true);
        vres = tbt;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.true_res, out);
        success = success && tmp_success;

        topic = "conversion from false    : ";
        vector_bool_type fbt(false);
        vres = fbt;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.false_res, out);
        success = success && tmp_success;

        topic = "any                      : ";
        auto any_check_false = (lhs != lhs);
        bool any_res_false = any(any_check_false);
        auto any_check_true = (lhs == mix_lhs_rhs);
        bool any_res_true = any(any_check_true);
        tmp_success = !any_res_false && any_res_true;
        success = success && tmp_success;

        topic = "all                      : ";
        auto all_check_false = (lhs == mix_lhs_rhs);
        bool all_res_false = all(all_check_false);
        auto all_check_true = (lhs == lhs);
        bool all_res_true = all(all_check_true);
        tmp_success = !all_res_false && all_res_true;
        success = success && tmp_success;
        success = success && test_simd_bool(vector_type(0.), out);
        success = success && test_simd_bool_buffer(vector_type(0.), out);

        #define XSIMD_TEST_COMPARISON( OPERATOR ) \
            topic = "operator" #OPERATOR "(simd, simd)  : "; \
            bres = lhs OPERATOR rhs; \
            for(std::size_t i=0;i<T::size;++i ) \
                success &= ( lhs[ i ] OPERATOR rhs[ i ] ) == bres[ i ]; \
            topic = "operator" #OPERATOR "(simd, scalar): "; \
            bres = lhs OPERATOR s; \
            for(std::size_t i=0;i<T::size;++i ) \
                success &= ( lhs[ i ] OPERATOR s ) == bres[ i ] \

        XSIMD_TEST_COMPARISON( <  );
        XSIMD_TEST_COMPARISON( >  );
        XSIMD_TEST_COMPARISON( <= );
        XSIMD_TEST_COMPARISON( <= );
        XSIMD_TEST_COMPARISON( == );
        XSIMD_TEST_COMPARISON( != );
        #undef XSIMD_TEST_COMPARISON

        bres[0] = false;
        tmp_success = !bool(bres[0]);
        success = success && tmp_success;
        bres[0] = true;
        tmp_success = bool(bres[0]);
        success = success && tmp_success;

        bres = vector_bool_type(false);
        vres = bitwise_cast(bres);
        bres = vres == vector_type(value_type(0));
        tmp_success = all(bres);
        success = success && tmp_success;

        bres = vector_bool_type(true);
        vres = bitwise_cast(bres);
        bres = (~vres) == vector_type(value_type(0));
        tmp_success = all(bres);
        success = success && tmp_success;
        
        topic = "not                      : ";
        vres = !lhs;
        for(std::size_t i = 0; i < T::size; ++i)
        {
            success = success && !lhs[i] == vres[i];
        }

        topic = "iterator                 : ";
        auto lhs_iter = lhs.begin();
        auto lhs_citer = lhs.cbegin();
        std::size_t idx = 0;
        while(lhs_iter != lhs.end() || lhs_citer != lhs.cend() && tmp_success)
        {
            tmp_success = check_almost_equal(topic, *lhs_iter, lhs[idx], out);
            tmp_success = check_almost_equal(topic, *lhs_citer, lhs[idx], out) && tmp_success;
            ++lhs_iter, ++lhs_citer, ++idx;
        }
        success = success && tmp_success;

        topic = "reverse iterator         : ";
        auto lhs_riter = lhs.rbegin();
        auto lhs_criter = lhs.crbegin();
        size_t ridx = vector_type::size;
        while(lhs_riter != lhs.rend() || lhs_criter != lhs.crend() && tmp_success)
        {
            --ridx;
            tmp_success = check_almost_equal(topic, *lhs_riter, lhs[ridx], out);
            tmp_success = check_almost_equal(topic, *lhs_criter, lhs[ridx], out) && tmp_success;
            ++lhs_riter, ++lhs_criter;
        }
        success = success && tmp_success;
        return success;
    }

    template <class I, std::size_t N, class S>
    bool test_simd_int_shift(const batch<I, N>& /*empty*/, S& stream)
    {
        int32_t size = static_cast<int32_t>(sizeof(I) * 8);
        bool success = true;

        batch<I, N> lhs, res;
        lhs = batch<I, N>(I(1));

        for (int32_t i = 0; i < size; ++i)
        {
            res = lhs << i;
            I expected = I(1) << i;
            for (std::size_t j = 0; j < N; ++j)
            {
                success = success && (res[j] == expected);
            }
        }
        lhs = batch<I, N>(std::numeric_limits<I>::max());
        for (int32_t i = 0; i < size; ++i)
        {
            res = lhs >> I(i);
            I expected = std::numeric_limits<I>::max() >> i;
            for (std::size_t j = 0; j < N; ++j)
            {
                success = success && (res[j] == expected);
            }
        }
        if (!success)
        {
            stream << "Failed test simd int shift!" << std::endl;
        }

        {
            // Compilation check
            batch<I, N> sh(1);
            res = lhs << sh;
            res = lhs >> sh;
        }
        return success;
    }

    template <std::size_t N, class T, class S>
    bool test_char_loading(T /**/, S& /*stream*/)
    {
        return true;
    }

    template <class vector_type, class value_type>
    bool test_lt_underflow()
    {
        // underflow for unsigned integers
        vector_type test_negative_compare = vector_type(5) - 6;
        if (std::is_unsigned<value_type>::value)
        {
            return !xsimd::any(test_negative_compare < 1);
        }
        else
        {
            return xsimd::all(test_negative_compare < 1);
        }
    }

    template <std::size_t N, class S>
    bool test_char_loading(int8_t /**/, S& stream)
    {
        bool success = true;
        char non_algn[64];
        alignas(64) char algn[64];

        for (std::size_t i = 0; i < 64; ++i)
        {
            non_algn[i] = static_cast<char>(i);
            algn[i] = static_cast<char>(i);
        }

        batch<int8_t, N> bx, by, bz;
        bx.load_aligned(algn);
        by.load_unaligned(non_algn);

        success = success && all(bx == by);
        success = success && (bx[5] == 5);

        bz = bx + by;
        bz.store_aligned(algn);
        bz.store_unaligned(non_algn);

        success = success && std::equal(std::begin(non_algn), std::end(non_algn), std::begin(algn));
        success = success && (algn[5] == 10);

        if (!success)
        {
            stream << "Saving/Loading of chars into int8_t batch did not work!" << std::endl;
        }
        return success;
    }

    template <class T>
    bool test_simd_int_basic(std::ostream& out, T& tester)
    {
        using tester_type = T;
        using vector_type = typename tester_type::vector_type;
        using value_type = typename tester_type::value_type;
        using res_type = typename tester_type::res_type;
        using signed_vector_type = typename tester_type::signed_vector_type;
        using signed_value_type = typename tester_type::signed_value_type;
        using signed_res_type = typename tester_type::signed_res_type;

        using vector_bool_type = typename simd_batch_traits<vector_type>::batch_bool_type;
        using bool_traits = simd_batch_traits<vector_bool_type>;

        static_assert(std::is_same<typename bool_traits::batch_type, vector_type>::value, "vector_bool type mismatch");
        static_assert(bool_traits::size == vector_type::size, "vector bool size mismatch");

        vector_type lhs;
        vector_type rhs;
        vector_type mix_lhs_rhs;
        signed_vector_type shift;
        vector_type vres;
        res_type res(tester_type::size);
        value_type s = tester.s;
        bool success = true;
        bool tmp_success = true;

        std::string val_type = value_type_name<vector_type>();
        std::string val_type_shift = std::string(val_type.size(), '-');
        std::string name = tester.name;
        std::string name_shift = std::string(name.size(), '-');
        std::string dash(8, '-');
        std::string space(8, ' ');

        out << dash << name_shift << '-' << val_type_shift << dash << std::endl;
        out << space << name << " " << val_type << std::endl;
        out << dash << name_shift << '-' << val_type_shift << dash << std::endl
            << std::endl;

        std::string topic = "operator[]               : ";
        detail::load_vec(lhs, tester.lhs);
        value_type es = lhs[1];
        tmp_success = check_almost_equal(topic, es, tester.extract_res, out);
        success = success && tmp_success;
        value_type nvalue = value_type(2);
        lhs[1] = nvalue;
        tmp_success = check_almost_equal(topic, lhs[1], nvalue, out);
        lhs[1] = es;

        topic = "load/store aligned       : ";
        detail::load_vec(lhs, tester.lhs);
        detail::store_vec(lhs, res);
        tmp_success = check_almost_equal(topic, res, tester.lhs, out);
        success = success && tmp_success;

        topic = "load/store unaligned     : ";
        lhs.load_unaligned(&tester.lhs[0]);
        lhs.store_unaligned(&res[0]);
        tmp_success = check_almost_equal(topic, res, tester.lhs, out);
        success = success && tmp_success;

        detail::load_vec(lhs, tester.lhs);
        detail::load_vec(rhs, tester.rhs);
        detail::load_vec(mix_lhs_rhs, tester.mix_lhs_rhs);
        detail::load_vec(shift, tester.shift);

        topic = "unary operator-          : ";
        vres = -lhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.minus_res, out);
        success = success && tmp_success;

        topic = "unary operator+          : ";
        vres = +lhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.plus_res, out);
        success = success && tmp_success;

        topic = "operator+=(simd, simd)   : ";
        vres = lhs;
        vres += rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_vv_res, out);
        success = success && tmp_success;

        topic = "operator+=(simd, scalar) : ";
        vres = lhs;
        vres += s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_vs_res, out);
        success = success && tmp_success;

        topic = "operator-=(simd, simd)   : ";
        vres = lhs;
        vres -= rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_vv_res, out);
        success = success && tmp_success;

        topic = "operator-=(simd, scalar) : ";
        vres = lhs;
        vres -= s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_vs_res, out);
        success = success && tmp_success;

        topic = "operator*=(simd, simd)   : ";
        vres = lhs;
        vres *= rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_vv_res, out);
        success = success && tmp_success;

        topic = "operator*=(simd, scalar) : ";
        vres = lhs;
        vres *= s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_vs_res, out);
        success = success && tmp_success;

        topic = "operator/=(simd, simd)   : ";
        vres = lhs;
        vres /= rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_vv_res, out);
        success = success && tmp_success;

        topic = "operator/=(simd, scalar) : ";
        vres = lhs;
        vres /= s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_vs_res, out);
        success = success && tmp_success;

        topic = "operator+(simd, simd)    : ";
        vres = lhs + rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_vv_res, out);
        success = success && tmp_success;

        topic = "operator+(simd, scalar)  : ";
        vres = lhs + s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_vs_res, out);
        success = success && tmp_success;

        topic = "operator+(scalar, simd)  : ";
        vres = s + rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_sv_res, out);
        success = success && tmp_success;

        topic = "operator-(simd, simd)    : ";
        vres = lhs - rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_vv_res, out);
        success = success && tmp_success;

        topic = "operator-(simd, scalar)  : ";
        vres = lhs - s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_vs_res, out);
        success = success && tmp_success;

        topic = "operator-(scalar, simd)  : ";
        vres = s - rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_sv_res, out);
        success = success && tmp_success;

        topic = "operator*(simd, simd)    : ";
        vres = lhs * rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_vv_res, out);
        success = success && tmp_success;

        topic = "operator*(simd, scalar)  : ";
        vres = lhs * s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_vs_res, out);
        success = success && tmp_success;

        topic = "operator*(scalar, simd)  : ";
        vres = s * rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_sv_res, out);
        success = success && tmp_success;

        topic = "operator/(simd, simd)    : ";
        vres = lhs / rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_vv_res, out);
        success = success && tmp_success;

        topic = "operator/(simd, scalar)  : ";
        vres = lhs / s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_vs_res, out);
        success = success && tmp_success;

        topic = "operator/(scalar, simd)  : ";
        vres = s / rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_sv_res, out);
        success = success && tmp_success;

        topic = "operator%(simd, simd)    : ";
        vres = lhs % rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mod_vv_res, out);
        success = success && tmp_success;

        topic = "min(simd, simd)          : ";
        vres = min(lhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.min_res, out);
        success = success && tmp_success;

        topic = "max(simd, simd)          : ";
        vres = max(lhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.max_res, out);
        success = success && tmp_success;

        topic = "abs(simd)                : ";
        vres = abs(lhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.abs_res, out);
        success = success && tmp_success;

        topic = "hadd(simd)               : ";
        value_type sres = hadd(lhs);
        tmp_success = check_almost_equal(topic, sres, tester.hadd_res, out);
        success = success && tmp_success;

        topic = "fma(simd, simd, simd)    : ";
        vres = fma(lhs, rhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.fma_res, out);
        success = success && tmp_success;

        topic = "fms(simd, simd, simd)    : ";
        vres = fms(lhs, rhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.fms_res, out);
        success = success && tmp_success;

        topic = "fnma(simd, simd, simd)   : ";
        vres = fnma(lhs, rhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.fnma_res, out);
        success = success && tmp_success;

        topic = "fnms(simd, simd, simd)   : ";
        vres = fnms(lhs, rhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.fnms_res, out);
        success = success && tmp_success;

        topic = "shift left(simd, int)    : ";
        vres = lhs << tester.sh_nb;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sl_s_res, out);
        success = success && tmp_success;

        topic = "shift left(simd, simd)    : ";
        vres = lhs << shift;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sl_v_res, out);
        success = success && tmp_success;

        topic = "shift right(simd, int)   : ";
        vres = lhs >> tester.sh_nb;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sr_s_res, out);
        success = success && tmp_success;

        topic = "shift right(simd, simd)   : ";
        vres = lhs >> shift;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sr_v_res, out);
        success = success && tmp_success;

        topic = "conversion from true     : ";
        vector_bool_type tbt(true);
        vres = tbt;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.true_res, out);
        success = success && tmp_success;

        topic = "conversion from false    : ";
        vector_bool_type fbt(false);
        vres = fbt;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.false_res, out);
        success = success && tmp_success;

        topic = "any                      : ";
        auto any_check_false = (lhs != lhs);
        bool any_res_false = any(any_check_false);
        auto any_check_true = (lhs == mix_lhs_rhs);
        bool any_res_true = any(any_check_true);
        tmp_success = !any_res_false && any_res_true;
        success = success && tmp_success;

        topic = "all                      : ";
        auto all_check_false = (lhs == mix_lhs_rhs);
        bool all_res_false = all(all_check_false);
        auto all_check_true = (lhs == lhs);
        bool all_res_true = all(all_check_true);
        tmp_success = !all_res_false && all_res_true;
        success = success && tmp_success;
        
        auto bres = (lhs == lhs);
        bres[0] = false;
        tmp_success = !bool(bres[0]);
        success = success && tmp_success;
        bres[0] = true;
        tmp_success = bool(bres[0]);
        success = success && tmp_success;

        bres = vector_bool_type(false);
        vres = bitwise_cast(bres);
        bres = vres == vector_type(value_type(0));
        tmp_success = all(bres);
        success = success && tmp_success;

        bres = vector_bool_type(true);
        vres = bitwise_cast(bres);
        bres = ~vres == vector_type(value_type(0));
        tmp_success = all(bres);
        success = success && tmp_success;
        
        success = success && test_simd_int_shift(vector_type(value_type(0)), out);
        success = success && test_simd_bool(vector_type(value_type(0)), out);
        success = success && test_simd_bool_buffer(vector_type(value_type(0)), out);
        success = success && test_char_loading<vector_type::size>(value_type(), out);
        success = success && test_lt_underflow<vector_type, value_type>();

        test_more_int<vector_type>{}.run();
        return success;
    }

    /*********************
     * conversion tester *
     *********************/

    template <std::size_t N, std::size_t A>
    struct simd_convert_tester
    {
        using int32_batch = batch<int32_t, N * 2>;
        using int64_batch = batch<int64_t, N>;
        using float_batch = batch<float, N * 2>;
        using double_batch = batch<double, N>;

        using int32_vector = std::vector<int32_t, aligned_allocator<int32_t, A>>;
        using int64_vector = std::vector<int64_t, aligned_allocator<int64_t, A>>;
        using float_vector = std::vector<float, aligned_allocator<float, A>>;
        using double_vector = std::vector<double, aligned_allocator<double, A>>;

        std::string name;

        int32_batch i32pos;
        int32_batch i32neg;
        int64_batch i64pos;
        int64_batch i64neg;
        float_batch fpos;
        float_batch fneg;
        double_batch dpos;
        double_batch dneg;

        int32_vector fposres;
        int32_vector fnegres;
        int64_vector dposres;
        int64_vector dnegres;
        float_vector i32posres;
        float_vector i32negres;
        double_vector i64posres;
        double_vector i64negres;

        simd_convert_tester(const std::string& name);
    };

    template <std::size_t N, std::size_t A>
    inline simd_convert_tester<N, A>::simd_convert_tester(const std::string& n)
        : name(n), i32pos(2), i32neg(-3), i64pos(2), i64neg(-3),
          fpos(float(7.4)), fneg(float(-6.2)), dpos(double(5.4)), dneg(double(-1.2)),
          fposres(2 * N, 7), fnegres(2 * N, -6), dposres(N, 5), dnegres(N, -1),
          i32posres(2 * N, float(2)), i32negres(2 * N, float(-3)),
          i64posres(N, double(2)), i64negres(N, double(-3))
    {
    }

    /*******************
     * conversion test *
     *******************/

    template <class T>
    inline bool test_simd_conversion(std::ostream& out, T& tester)
    {
        using int32_batch = typename T::int32_batch;
        using int64_batch = typename T::int64_batch;
        using float_batch = typename T::float_batch;
        using double_batch = typename T::double_batch;
        using int32_vector = typename T::int32_vector;
        using int64_vector = typename T::int64_vector;
        using float_vector = typename T::float_vector;
        using double_vector = typename T::double_vector;

        int32_batch fbres;
        int64_batch dbres;
        float_batch i32bres;
        double_batch i64bres;
        int32_vector fvres(int32_batch::size);
        int64_vector dvres(int64_batch::size);
        float_vector i32vres(float_batch::size);
        double_vector i64vres(double_batch::size);

        bool success = true;
        bool tmp_success = true;

        std::string name = tester.name;
        std::string name_shift = std::string(name.size(), '-');
        std::string dash(8, '-');
        std::string space(8, ' ');

        out << dash << name_shift << dash << std::endl;
        out << space << name << space << std::endl;
        out << dash << name_shift << dash << std::endl
            << std::endl;

        std::string topic = "positive float  -> int32  : ";
        fbres = to_int(tester.fpos);
        detail::store_vec(fbres, fvres);
        tmp_success = check_almost_equal(topic, fvres, tester.fposres, out);
        success = success && tmp_success;

        topic = "negative float  -> int32  : ";
        fbres = to_int(tester.fneg);
        detail::store_vec(fbres, fvres);
        tmp_success = check_almost_equal(topic, fvres, tester.fnegres, out);
        success = success && tmp_success;

        topic = "positive double -> int64  : ";
        dbres = to_int(tester.dpos);
        detail::store_vec(dbres, dvres);
        tmp_success = check_almost_equal(topic, dvres, tester.dposres, out);
        success = success && tmp_success;

        topic = "negative double -> int64  : ";
        dbres = to_int(tester.dneg);
        detail::store_vec(dbres, dvres);
        tmp_success = check_almost_equal(topic, dvres, tester.dnegres, out);
        success = success && tmp_success;

        topic = "positive int32  -> float  : ";
        i32bres = to_float(tester.i32pos);
        detail::store_vec(i32bres, i32vres);
        tmp_success = check_almost_equal(topic, i32vres, tester.i32posres, out);
        success = success && tmp_success;

        topic = "negative int32  -> float  : ";
        i32bres = to_float(tester.i32neg);
        detail::store_vec(i32bres, i32vres);
        tmp_success = check_almost_equal(topic, i32vres, tester.i32negres, out);
        success = success && tmp_success;

        topic = "positive int64  -> double : ";
        i64bres = to_float(tester.i64pos);
        detail::store_vec(i64bres, i64vres);
        tmp_success = check_almost_equal(topic, i64vres, tester.i64posres, out);
        success = success && tmp_success;

        topic = "negative int64  -> double : ";
        i64bres = to_float(tester.i64neg);
        detail::store_vec(i64bres, i64vres);
        tmp_success = check_almost_equal(topic, i64vres, tester.i64negres, out);
        success = success && tmp_success;

        return success;
    }

    /*********************
     * batch cast tester *
     *********************/

    template <std::size_t N, std::size_t A>
    struct simd_batch_cast_tester
    {
        using int8_batch = batch<int8_t, N * 8>;
        using uint8_batch = batch<uint8_t, N * 8>;
        using int16_batch = batch<int16_t, N * 4>;
        using uint16_batch = batch<uint16_t, N * 4>;
        using int32_batch = batch<int32_t, N * 2>;
        using uint32_batch = batch<uint32_t, N * 2>;
        using int64_batch = batch<int64_t, N>;
        using uint64_batch = batch<uint64_t, N>;
        using float_batch = batch<float, N * 2>;
        using double_batch = batch<double, N>;

        static constexpr std::size_t Alignment = A;

        std::string name;

        std::vector<uint64_t> int_test_values;
        std::vector<float> float_test_values;
        std::vector<double> double_test_values;

        simd_batch_cast_tester(const std::string& n);

        template <class B_in, class B_out, class T>
        bool run_test(std::ostream& out, const std::string& topic, T test_value);
    };

    template <std::size_t N, std::size_t A>
    inline simd_batch_cast_tester<N, A>::simd_batch_cast_tester(const std::string& n)
        : name(n), int_test_values(), float_test_values(), double_test_values()
    {
        int_test_values = {
            0,
            0x01,
            0x7f,
            0x80,
            0xff,
            0x0100,
            0x7fff,
            0x8000,
            0xffff,
            0x00010000,
            0x7fffffff,
            0x80000000,
            0xffffffff,
            0x0000000100000000,
            0x7fffffffffffffff,
            0x8000000000000000,
            0xffffffffffffffff
        };

        float_test_values = {
            0.0f,
            1.0f,
            -1.0f,
            127.0f,
            128.0f,
            -128.0f,
            255.0f,
            256.0f,
            -256.0f,
            32767.0f,
            32768.0f,
            -32768.0f,
            65535.0f,
            65536.0f,
            -65536.0f,
            2147483647.0f,
            2147483648.0f,
            -2147483648.0f,
            4294967167.0f
        };

        double_test_values = {
            0.0,
            1.0,
            -1.0,
            127.0,
            128.0,
            -128.0,
            255.0,
            256.0,
            -256.0,
            32767.0,
            32768.0,
            -32768.0,
            65535.0,
            65536.0,
            -65536.0,
            2147483647.0,
            2147483648.0,
            -2147483648.0,
            4294967295.0,
            4294967296.0,
            -4294967296.0,
            9223372036854775807.0,
            9223372036854775808.0,
            -9223372036854775808.0,
            18446744073709550591.0
        };
    }

    template <std::size_t N, std::size_t A>
    template <class B_in, class B_out, class T>
    inline bool simd_batch_cast_tester<N, A>::run_test(std::ostream& out, const std::string& topic, T test_value)
    {
        using T_in = typename B_in::value_type;
        using T_out = typename B_out::value_type;
        static constexpr std::size_t N_common = B_in::size < B_out::size ? B_in::size : B_out::size;
        using B_common_in = batch<T_in, N_common>;
        using B_common_out = batch<T_out, N_common>;

        T_in in_test_value = static_cast<T_in>(test_value);
        if (is_convertible<T_out>(in_test_value))
        {
            B_common_out res = batch_cast<T_out>(B_common_in(in_test_value));
            return check_almost_equal(topic, res[0], static_cast<T_out>(in_test_value), out);
        }
        return true;
    }

    /*********************
     * bitwise cast test *
     *********************/

    template <class T>
    inline bool test_simd_batch_cast(std::ostream& out, T& tester)
    {
        using int8_batch = typename T::int8_batch;
        using uint8_batch = typename T::uint8_batch;
        using int16_batch = typename T::int16_batch;
        using uint16_batch = typename T::uint16_batch;
        using int32_batch = typename T::int32_batch;
        using uint32_batch = typename T::uint32_batch;
        using int64_batch = typename T::int64_batch;
        using uint64_batch = typename T::uint64_batch;
        using float_batch = typename T::float_batch;
        using double_batch = typename T::double_batch;

        bool success = true;

        std::string name = tester.name;
        std::string name_shift = std::string(name.size(), '-');
        std::string dash(8, '-');
        std::string space(8, ' ');

        out << dash << name_shift << dash << std::endl;
        out << space << name << space << std::endl;
        out << dash << name_shift << dash << std::endl
            << std::endl;

        for (const auto& test_value : tester.int_test_values)
        {
            std::string topic = "batch cast int8   -> int8   : ";
            success &= tester.template run_test<int8_batch, int8_batch>(out, topic, test_value);

            topic = "batch cast int8   -> uint8  : ";
            success &= tester.template run_test<int8_batch, uint8_batch>(out, topic, test_value);

            topic = "batch cast uint8  -> int8   : ";
            success &= tester.template run_test<uint8_batch, int8_batch>(out, topic, test_value);

            topic = "batch cast uint8  -> uint8  : ";
            success &= tester.template run_test<uint8_batch, uint8_batch>(out, topic, test_value);

            topic = "batch cast int16  -> int16  : ";
            success &= tester.template run_test<int16_batch, int16_batch>(out, topic, test_value);

            topic = "batch cast int16  -> uint16 : ";
            success &= tester.template run_test<int16_batch, uint16_batch>(out, topic, test_value);

            topic = "batch cast uint16 -> int16  : ";
            success &= tester.template run_test<uint16_batch, int16_batch>(out, topic, test_value);

            topic = "batch cast uint16 -> uint16 : ";
            success &= tester.template run_test<uint16_batch, uint16_batch>(out, topic, test_value);

            topic = "batch cast int32  -> int32  : ";
            success &= tester.template run_test<int32_batch, int32_batch>(out, topic, test_value);

            topic = "batch cast int32  -> uint32 : ";
            success &= tester.template run_test<int32_batch, uint32_batch>(out, topic, test_value);

            topic = "batch cast int32  -> float  : ";
            success &= tester.template run_test<int32_batch, float_batch>(out, topic, test_value);

            topic = "batch cast uint32 -> int32  : ";
            success &= tester.template run_test<uint32_batch, int32_batch>(out, topic, test_value);

            topic = "batch cast uint32 -> uint32 : ";
            success &= tester.template run_test<uint32_batch, uint32_batch>(out, topic, test_value);

            topic = "batch cast uint32 -> float  : ";
            success &= tester.template run_test<uint32_batch, float_batch>(out, topic, test_value);

            topic = "batch cast int64  -> int64  : ";
            success &= tester.template run_test<int64_batch, int64_batch>(out, topic, test_value);

            topic = "batch cast int64  -> uint64 : ";
            success &= tester.template run_test<int64_batch, uint64_batch>(out, topic, test_value);

            topic = "batch cast int64  -> double : ";
            success &= tester.template run_test<int64_batch, double_batch>(out, topic, test_value);

            topic = "batch cast uint64 -> int64  : ";
            success &= tester.template run_test<uint64_batch, int64_batch>(out, topic, test_value);

            topic = "batch cast uint64 -> uint64 : ";
            success &= tester.template run_test<uint64_batch, uint64_batch>(out, topic, test_value);

            topic = "batch cast uint64 -> double : ";
            success &= tester.template run_test<uint64_batch, double_batch>(out, topic, test_value);
        }

        for (const auto& test_value : tester.float_test_values)
        {
            std::string topic = "batch cast float  -> int32  : ";
            success &= tester.template run_test<float_batch, int32_batch>(out, topic, test_value);

            topic = "batch cast float  -> uint32 : ";
            success &= tester.template run_test<float_batch, uint32_batch>(out, topic, test_value);

            topic = "batch cast float  -> float  : ";
            success &= tester.template run_test<float_batch, float_batch>(out, topic, test_value);
        }

        for (const auto& test_value : tester.double_test_values)
        {
            std::string topic = "batch cast double -> int64  : ";
            success &= tester.template run_test<double_batch, int64_batch>(out, topic, test_value);

            topic = "batch cast double -> uint64 : ";
            success &= tester.template run_test<double_batch, uint64_batch>(out, topic, test_value);

            topic = "batch cast double -> double : ";
            success &= tester.template run_test<double_batch, double_batch>(out, topic, test_value);
        }

        return success;
    }

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
    template <class T>
    inline typename std::enable_if<T::Alignment >= 32, bool>::type test_simd_batch_cast_sizeshift1(std::ostream& out, T& tester)
    {
        using int8_batch = typename T::int8_batch;
        using uint8_batch = typename T::uint8_batch;
        using int16_batch = typename T::int16_batch;
        using uint16_batch = typename T::uint16_batch;
        using int32_batch = typename T::int32_batch;
        using uint32_batch = typename T::uint32_batch;
        using int64_batch = typename T::int64_batch;
        using uint64_batch = typename T::uint64_batch;
        using float_batch = typename T::float_batch;
        using double_batch = typename T::double_batch;

        bool success = true;

        std::string name = tester.name;
        std::string name_shift = std::string(name.size(), '-');
        std::string dash(8, '-');
        std::string space(8, ' ');

        out << dash << name_shift << dash << std::endl;
        out << space << name << space << std::endl;
        out << dash << name_shift << dash << std::endl
            << std::endl;

        for (const auto& test_value : tester.int_test_values)
        {
            std::string topic = "batch cast int8   -> int16  : ";
            success &= tester.template run_test<int8_batch, int16_batch>(out, topic, test_value);

            topic = "batch cast int8   -> uint16 : ";
            success &= tester.template run_test<int8_batch, uint16_batch>(out, topic, test_value);

            topic = "batch cast uint8  -> int16  : ";
            success &= tester.template run_test<uint8_batch, int16_batch>(out, topic, test_value);

            topic = "batch cast uint8  -> uint16 : ";
            success &= tester.template run_test<uint8_batch, uint16_batch>(out, topic, test_value);

            topic = "batch cast int16  -> int8   : ";
            success &= tester.template run_test<int16_batch, int8_batch>(out, topic, test_value);

            topic = "batch cast int16  -> uint8  : ";
            success &= tester.template run_test<int16_batch, uint8_batch>(out, topic, test_value);

            topic = "batch cast int16  -> int32  : ";
            success &= tester.template run_test<int16_batch, int32_batch>(out, topic, test_value);

            topic = "batch cast int16  -> uint32 : ";
            success &= tester.template run_test<int16_batch, uint32_batch>(out, topic, test_value);

            topic = "batch cast int16  -> float  : ";
            success &= tester.template run_test<int16_batch, float_batch>(out, topic, test_value);

            topic = "batch cast uint16 -> int8   : ";
            success &= tester.template run_test<uint16_batch, int8_batch>(out, topic, test_value);

            topic = "batch cast uint16 -> uint8  : ";
            success &= tester.template run_test<uint16_batch, uint8_batch>(out, topic, test_value);

            topic = "batch cast uint16 -> int32  : ";
            success &= tester.template run_test<uint16_batch, int32_batch>(out, topic, test_value);

            topic = "batch cast uint16 -> uint32 : ";
            success &= tester.template run_test<uint16_batch, uint32_batch>(out, topic, test_value);

            topic = "batch cast uint16 -> float  : ";
            success &= tester.template run_test<uint16_batch, float_batch>(out, topic, test_value);

            topic = "batch cast int32  -> int16  : ";
            success &= tester.template run_test<int32_batch, int16_batch>(out, topic, test_value);

            topic = "batch cast int32  -> uint16 : ";
            success &= tester.template run_test<int32_batch, uint16_batch>(out, topic, test_value);

            topic = "batch cast int32  -> int64  : ";
            success &= tester.template run_test<int32_batch, int64_batch>(out, topic, test_value);

            topic = "batch cast int32  -> uint64 : ";
            success &= tester.template run_test<int32_batch, uint64_batch>(out, topic, test_value);

            topic = "batch cast int32  -> double : ";
            success &= tester.template run_test<int32_batch, double_batch>(out, topic, test_value);

            topic = "batch cast uint32 -> int8   : ";
            success &= tester.template run_test<uint32_batch, int16_batch>(out, topic, test_value);

            topic = "batch cast uint32 -> uint8  : ";
            success &= tester.template run_test<uint32_batch, uint16_batch>(out, topic, test_value);

            topic = "batch cast uint32 -> int32  : ";
            success &= tester.template run_test<uint32_batch, int64_batch>(out, topic, test_value);

            topic = "batch cast uint32 -> uint32 : ";
            success &= tester.template run_test<uint32_batch, uint64_batch>(out, topic, test_value);

            topic = "batch cast uint32 -> double : ";
            success &= tester.template run_test<uint32_batch, double_batch>(out, topic, test_value);

            topic = "batch cast int64  -> int32  : ";
            success &= tester.template run_test<int64_batch, int32_batch>(out, topic, test_value);

            topic = "batch cast int64  -> uint32 : ";
            success &= tester.template run_test<int64_batch, uint32_batch>(out, topic, test_value);

            topic = "batch cast int64  -> float  : ";
            success &= tester.template run_test<int64_batch, float_batch>(out, topic, test_value);

            topic = "batch cast uint64 -> int32  : ";
            success &= tester.template run_test<uint64_batch, int32_batch>(out, topic, test_value);

            topic = "batch cast uint64 -> uint32 : ";
            success &= tester.template run_test<uint64_batch, uint32_batch>(out, topic, test_value);

            topic = "batch cast uint64 -> float  : ";
            success &= tester.template run_test<uint64_batch, float_batch>(out, topic, test_value);
        }

        for (const auto& test_value : tester.float_test_values)
        {
            std::string topic = "batch cast float  -> int16  : ";
            success &= tester.template run_test<float_batch, int16_batch>(out, topic, test_value);

            topic = "batch cast float  -> uint16 : ";
            success &= tester.template run_test<float_batch, uint16_batch>(out, topic, test_value);

            topic = "batch cast float  -> int64  : ";
            success &= tester.template run_test<float_batch, int64_batch>(out, topic, test_value);

            topic = "batch cast float  -> uint64 : ";
            success &= tester.template run_test<float_batch, uint64_batch>(out, topic, test_value);

            topic = "batch cast float  -> double : ";
            success &= tester.template run_test<float_batch, double_batch>(out, topic, test_value);
        }

        for (const auto& test_value : tester.double_test_values)
        {
            std::string topic = "batch cast double -> int32  : ";
            success &= tester.template run_test<double_batch, int32_batch>(out, topic, test_value);

            topic = "batch cast double -> uint32 : ";
            success &= tester.template run_test<double_batch, uint32_batch>(out, topic, test_value);

            topic = "batch cast double -> float  : ";
            success &= tester.template run_test<double_batch, float_batch>(out, topic, test_value);
        }

        return success;
    }

    template <class T>
    inline typename std::enable_if<T::Alignment < 32, bool>::type test_simd_batch_cast_sizeshift1(std::ostream&, T&)
    {
        return true;
    }
#endif

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX512_VERSION
    template <class T>
    inline typename std::enable_if<T::Alignment >= 64, bool>::type test_simd_batch_cast_sizeshift2(std::ostream& out, T& tester)
    {
        using int8_batch = typename T::int8_batch;
        using uint8_batch = typename T::uint8_batch;
        using int16_batch = typename T::int16_batch;
        using uint16_batch = typename T::uint16_batch;
        using int32_batch = typename T::int32_batch;
        using uint32_batch = typename T::uint32_batch;
        using int64_batch = typename T::int64_batch;
        using uint64_batch = typename T::uint64_batch;
        using float_batch = typename T::float_batch;
        using double_batch = typename T::double_batch;

        bool success = true;

        std::string name = tester.name;
        std::string name_shift = std::string(name.size(), '-');
        std::string dash(8, '-');
        std::string space(8, ' ');

        out << dash << name_shift << dash << std::endl;
        out << space << name << space << std::endl;
        out << dash << name_shift << dash << std::endl
            << std::endl;

        for (const auto& test_value : tester.int_test_values)
        {
            std::string topic = "batch cast int8   -> int32  : ";
            success &= tester.template run_test<int8_batch, int32_batch>(out, topic, test_value);

            topic = "batch cast int8   -> uint32 : ";
            success &= tester.template run_test<int8_batch, uint32_batch>(out, topic, test_value);

            topic = "batch cast int8   -> float  : ";
            success &= tester.template run_test<int8_batch, float_batch>(out, topic, test_value);

            topic = "batch cast uint8  -> int32  : ";
            success &= tester.template run_test<uint8_batch, int32_batch>(out, topic, test_value);

            topic = "batch cast uint8  -> uint32 : ";
            success &= tester.template run_test<uint8_batch, uint32_batch>(out, topic, test_value);

            topic = "batch cast uint8  -> float  : ";
            success &= tester.template run_test<uint8_batch, float_batch>(out, topic, test_value);

            topic = "batch cast int16  -> int64  : ";
            success &= tester.template run_test<int16_batch, int64_batch>(out, topic, test_value);

            topic = "batch cast int16  -> uint64 : ";
            success &= tester.template run_test<int16_batch, uint64_batch>(out, topic, test_value);

            topic = "batch cast int16  -> double : ";
            success &= tester.template run_test<int16_batch, double_batch>(out, topic, test_value);

            topic = "batch cast uint16 -> int64  : ";
            success &= tester.template run_test<uint16_batch, int64_batch>(out, topic, test_value);

            topic = "batch cast uint16 -> uint64 : ";
            success &= tester.template run_test<uint16_batch, uint64_batch>(out, topic, test_value);

            topic = "batch cast uint16 -> double  : ";
            success &= tester.template run_test<uint16_batch, double_batch>(out, topic, test_value);

            topic = "batch cast int32  -> int8   : ";
            success &= tester.template run_test<int32_batch, int8_batch>(out, topic, test_value);

            topic = "batch cast int32  -> uint8  : ";
            success &= tester.template run_test<int32_batch, uint8_batch>(out, topic, test_value);

            topic = "batch cast uint32 -> int8   : ";
            success &= tester.template run_test<uint32_batch, int8_batch>(out, topic, test_value);

            topic = "batch cast uint32 -> uint8  : ";
            success &= tester.template run_test<uint32_batch, uint8_batch>(out, topic, test_value);

            topic = "batch cast int64  -> int16  : ";
            success &= tester.template run_test<int64_batch, int16_batch>(out, topic, test_value);

            topic = "batch cast int64  -> uint16 : ";
            success &= tester.template run_test<int64_batch, uint16_batch>(out, topic, test_value);

            topic = "batch cast uint64 -> int16  : ";
            success &= tester.template run_test<uint64_batch, int16_batch>(out, topic, test_value);

            topic = "batch cast uint64 -> uint16 : ";
            success &= tester.template run_test<uint64_batch, uint16_batch>(out, topic, test_value);
        }

        for (const auto& test_value : tester.float_test_values)
        {
            std::string topic = "batch cast float  -> int8   : ";
            success &= tester.template run_test<float_batch, int8_batch>(out, topic, test_value);

            topic = "batch cast float  -> uint8  : ";
            success &= tester.template run_test<float_batch, uint8_batch>(out, topic, test_value);
        }

        for (const auto& test_value : tester.double_test_values)
        {
            std::string topic = "batch cast double -> int16  : ";
            success &= tester.template run_test<double_batch, int16_batch>(out, topic, test_value);

            topic = "batch cast double -> uint16 : ";
            success &= tester.template run_test<double_batch, uint16_batch>(out, topic, test_value);
        }

        return success;
    }

    template <class T>
    inline typename std::enable_if<T::Alignment < 64, bool>::type test_simd_batch_cast_sizeshift2(std::ostream&, T&)
    {
        return true;
    }
#endif

    /***********************
     * bitwise cast tester *
     ***********************/

    template <std::size_t N, std::size_t A>
    struct simd_bitwise_cast_tester
    {
        using int32_batch = batch<int32_t, N * 2>;
        using int64_batch = batch<int64_t, N>;
        using float_batch = batch<float, N * 2>;
        using double_batch = batch<double, N>;

        using int32_vector = std::vector<int32_t, aligned_allocator<int32_t, A>>;
        using int64_vector = std::vector<int64_t, aligned_allocator<int64_t, A>>;
        using float_vector = std::vector<float, aligned_allocator<float, A>>;
        using double_vector = std::vector<double, aligned_allocator<double, A>>;

        std::string name;

        int32_batch i32_input;
        int64_batch i64_input;
        float_batch f_input;
        double_batch d_input;

        int32_vector ftoi32_res;
        int32_vector dtoi32_res;
        int64_vector ftoi64_res;
        int64_vector dtoi64_res;
        float_vector i32tof_res;
        float_vector i64tof_res;
        float_vector dtof_res;
        double_vector i32tod_res;
        double_vector i64tod_res;
        double_vector ftod_res;

        simd_bitwise_cast_tester(const std::string& n);
    };

    namespace detail
    {
        union bitcast {
            float f[2];
            int32_t i32[2];
            int64_t i64;
            double d;
        };
    }

    template <std::size_t N, std::size_t A>
    inline simd_bitwise_cast_tester<N, A>::simd_bitwise_cast_tester(const std::string& n)
        : name(n), i32_input(2), i64_input(2), f_input(3.), d_input(2.5e17),
          ftoi32_res(2 * N), dtoi32_res(2 * N), ftoi64_res(N), dtoi64_res(N),
          i32tof_res(2 * N), i64tof_res(2 * N), dtof_res(2 * N),
          i32tod_res(N), i64tod_res(N), ftod_res(N)
    {
        detail::bitcast b1;
        b1.i32[0] = i32_input[0];
        b1.i32[1] = i32_input[1];
        std::fill(i32tof_res.begin(), i32tof_res.end(), b1.f[0]);
        std::fill(i32tod_res.begin(), i32tod_res.end(), b1.d);

        detail::bitcast b2;
        b2.i64 = i64_input[0];
        std::fill(i64tod_res.begin(), i64tod_res.end(), b1.d);
        for (size_t i = 0; i < N; ++i)
        {
            i64tof_res[2 * i] = b2.f[0];
            i64tof_res[2 * i + 1] = b2.f[1];
        }

        detail::bitcast b3;
        b3.f[0] = f_input[0];
        b3.f[1] = f_input[1];
        std::fill(ftoi32_res.begin(), ftoi32_res.end(), b3.i32[0]);
        std::fill(ftoi64_res.begin(), ftoi64_res.end(), b3.i64);
        std::fill(ftod_res.begin(), ftod_res.end(), b3.d);

        detail::bitcast b4;
        b4.d = d_input[0];
        std::fill(dtoi32_res.begin(), dtoi32_res.end(), b4.i32[0]);
        std::fill(dtoi64_res.begin(), dtoi64_res.end(), b4.i64);
        for (size_t i = 0; i < N; ++i)
        {
            dtof_res[2 * i] = b4.f[0];
            dtof_res[2 * i + 1] = b4.f[1];
        }
    }

    /*********************
     * bitwise cast test *
     *********************/

    template <class T>
    inline bool test_simd_bitwise_cast(std::ostream& out, T& tester)
    {
        using int32_batch = typename T::int32_batch;
        using int64_batch = typename T::int64_batch;
        using float_batch = typename T::float_batch;
        using double_batch = typename T::double_batch;
        using int32_vector = typename T::int32_vector;
        using int64_vector = typename T::int64_vector;
        using float_vector = typename T::float_vector;
        using double_vector = typename T::double_vector;

        int32_batch i32bres;
        int64_batch i64bres;
        float_batch fbres;
        double_batch dbres;
        int32_vector i32vres(int32_batch::size);
        int64_vector i64vres(int64_batch::size);
        float_vector fvres(float_batch::size);
        double_vector dvres(double_batch::size);

        bool success = true;
        bool tmp_success = true;

        std::string name = tester.name;
        std::string name_shift = std::string(name.size(), '-');
        std::string dash(8, '-');
        std::string space(8, ' ');

        out << dash << name_shift << dash << std::endl;
        out << space << name << space << std::endl;
        out << dash << name_shift << dash << std::endl
            << std::endl;

        std::string topic = "bitwise cast int32  -> float  : ";
        fbres = bitwise_cast<float_batch>(tester.i32_input);
        detail::store_vec(fbres, fvres);
        tmp_success = check_almost_equal(topic, fvres, tester.i32tof_res, out);
        success = success && tmp_success;

        topic = "bitwise cast int32  -> double : ";
        dbres = bitwise_cast<double_batch>(tester.i32_input);
        detail::store_vec(dbres, dvres);
        tmp_success = check_almost_equal(topic, dvres, tester.i32tod_res, out);
        success = success && tmp_success;

        topic = "bitwise cast int64  -> float  : ";
        fbres = bitwise_cast<float_batch>(tester.i64_input);
        detail::store_vec(fbres, fvres);
        tmp_success = check_almost_equal(topic, fvres, tester.i64tof_res, out);
        success = success && tmp_success;

        topic = "bitwise cast int64  -> double : ";
        dbres = bitwise_cast<double_batch>(tester.i64_input);
        detail::store_vec(dbres, dvres);
        tmp_success = check_almost_equal(topic, dvres, tester.i64tod_res, out);
        success = success && tmp_success;

        topic = "bitwise cast float  -> int32  : ";
        i32bres = bitwise_cast<int32_batch>(tester.f_input);
        detail::store_vec(i32bres, i32vres);
        tmp_success = check_almost_equal(topic, i32vres, tester.ftoi32_res, out);
        success = success && tmp_success;

        topic = "bitwise cast float  -> int64  : ";
        i64bres = bitwise_cast<int64_batch>(tester.f_input);
        detail::store_vec(i64bres, i64vres);
        tmp_success = check_almost_equal(topic, i64vres, tester.ftoi64_res, out);
        success = success && tmp_success;

        topic = "bitwise cast float  -> double : ";
        dbres = bitwise_cast<double_batch>(tester.f_input);
        detail::store_vec(dbres, dvres);
        tmp_success = check_almost_equal(topic, dvres, tester.ftod_res, out);
        success = success && tmp_success;

        topic = "bitwise cast double -> int32  : ";
        i32bres = bitwise_cast<int32_batch>(tester.d_input);
        detail::store_vec(i32bres, i32vres);
        tmp_success = check_almost_equal(topic, i32vres, tester.dtoi32_res, out);
        success = success && tmp_success;

        topic = "bitwise cast double -> int64  : ";
        i64bres = bitwise_cast<int64_batch>(tester.d_input);
        detail::store_vec(i64bres, i64vres);
        tmp_success = check_almost_equal(topic, i64vres, tester.dtoi64_res, out);
        success = success && tmp_success;

        topic = "bitwise cast double -> float  : ";
        fbres = bitwise_cast<float_batch>(tester.d_input);
        detail::store_vec(fbres, fvres);
        tmp_success = check_almost_equal(topic, fvres, tester.dtof_res, out);
        success = success && tmp_success;

        return success;
    }

    /*********************
     * load_store tester *
     *********************/

    template <class T, std::size_t N, std::size_t A>
    struct simd_load_store_tester
    {
        using batch_type = batch<T, N>;
        using res_vector = std::vector<T, aligned_allocator<T, A>>;

        using int8_vector = std::vector<int8_t, aligned_allocator<int8_t, A>>;
        using uint8_vector = std::vector<uint8_t, aligned_allocator<uint8_t, A>>;
        using int16_vector = std::vector<int16_t, aligned_allocator<int16_t, A>>;
        using uint16_vector = std::vector<uint16_t, aligned_allocator<uint16_t, A>>;
        using int32_vector = std::vector<int32_t, aligned_allocator<int32_t, A>>;
        using uint32_vector = std::vector<uint32_t, aligned_allocator<uint32_t, A>>;
        using int64_vector = std::vector<int64_t, aligned_allocator<int64_t, A>>;
        using uint64_vector = std::vector<uint64_t, aligned_allocator<uint64_t, A>>;
#ifdef XSIMD_32_BIT_ABI
        using long_vector = std::vector<long, aligned_allocator<long, A>>;
        using ulong_vector = std::vector<unsigned long, aligned_allocator<unsigned long, A>>;
#endif
        using float_vector = std::vector<float, aligned_allocator<float, A>>;
        using double_vector = std::vector<double, aligned_allocator<double, A>>;

        std::string name;

        int8_vector i8_vec;
        uint8_vector ui8_vec;
        int16_vector i16_vec;
        uint16_vector ui16_vec;
        int32_vector i32_vec;
        uint32_vector ui32_vec;
        int64_vector i64_vec;
        uint64_vector ui64_vec;
        float_vector f_vec;
        double_vector d_vec;

        res_vector exp_vec;

#ifdef XSIMD_32_BIT_ABI
        long_vector long_vec;
        ulong_vector ulong_vec;
#endif

        simd_load_store_tester(const std::string& n);
    };

    namespace detail
    {
        template <class T>
        inline void init_test_vector(T& vec)
        {
            using value_type = typename T::value_type;

            value_type min = value_type(0);
            value_type max = value_type(100);

            std::default_random_engine generator;
            std::uniform_int_distribution<int> distribution(min, max);

            auto gen = [&distribution, &generator](){
                return static_cast<value_type>(distribution(generator));
            };

            std::generate(vec.begin(), vec.end(), gen);
        }
    }

    template <class T, std::size_t N, std::size_t A>
    inline simd_load_store_tester<T, N, A>::simd_load_store_tester(const std::string& n)
        : name(n),
          i8_vec(N), ui8_vec(N), i16_vec(N), ui16_vec(N),
          i32_vec(N), ui32_vec(N), i64_vec(N), ui64_vec(N), f_vec(N), d_vec(N),
          exp_vec(N, T(0))
    {
        detail::init_test_vector(i8_vec);
        detail::init_test_vector(ui8_vec);
        detail::init_test_vector(i16_vec);
        detail::init_test_vector(ui16_vec);
        detail::init_test_vector(i32_vec);
        detail::init_test_vector(ui32_vec);
        detail::init_test_vector(i64_vec);
        detail::init_test_vector(ui64_vec);
        detail::init_test_vector(f_vec);
        detail::init_test_vector(d_vec);

#ifdef XSIMD_32_BIT_ABI
        using ulong = unsigned long;
        long_vec.resize(N);
        ulong_vec.resize(N);
        detail::init_test_vector(long_vec);
        detail::init_test_vector(ulong_vec);
#endif
    }

    /*******************
     * load/store test *
     *******************/

    template <class T>
    inline bool test_simd_load_store(std::ostream& out, T& tester)
    {
        using batch_type = typename T::batch_type;

        using int8_vector = typename T::int8_vector;
        using uint8_vector = typename T::uint8_vector;
        using int16_vector = typename T::int16_vector;
        using uint16_vector = typename T::uint16_vector;
        using int32_vector = typename T::int32_vector;
        using uint32_vector = typename T::uint32_vector;
        using int64_vector = typename T::int64_vector;
        using uint64_vector = typename T::uint64_vector;
        using float_vector = typename T::float_vector;
        using double_vector = typename T::double_vector;
        using res_vector = typename T::res_vector;

        constexpr std::size_t bsize = batch_type::size;

        batch_type bres;
        res_vector vres(bsize);

        int8_vector i8vres(bsize);
        uint8_vector ui8vres(bsize);
        int16_vector i16vres(bsize);
        uint16_vector ui16vres(bsize);
        int32_vector i32vres(bsize);
        uint32_vector ui32vres(bsize);
        int64_vector i64vres(bsize);
        uint64_vector ui64vres(bsize);
        float_vector fvres(bsize);
        double_vector dvres(bsize);

#ifdef XSIMD_32_BIT_ABI
        using long_vector = typename T::long_vector;
        using ulong_vector = typename T::ulong_vector;

        long_vector longvres(bsize);
        ulong_vector ulongvres(bsize);
#endif

        bool success = true;
        bool tmp_success = true;

        std::string name = tester.name;
        std::string name_shift = std::string(name.size(), '-');
        std::string dash(8, '-');
        std::string space(8, ' ');

        out << dash << name_shift << dash << std::endl;
        out << space << name << space << std::endl;
        out << dash << name_shift << dash << std::endl
            << std::endl;

        /*************
         * load test *
         *************/

        std::string topic = "load int8    -> " + name + "  : ";
        detail::load_vec(bres, tester.i8_vec);
        detail::store_vec(bres, vres);
        std::copy(tester.i8_vec.cbegin(), tester.i8_vec.cend(), tester.exp_vec.begin());
        tmp_success = check_almost_equal(topic, vres, tester.exp_vec, out);
        success = tmp_success && success;

        topic = "load uint8   -> " + name + "  : ";
        detail::load_vec(bres, tester.ui8_vec);
        detail::store_vec(bres, vres);
        std::copy(tester.ui8_vec.cbegin(), tester.ui8_vec.cend(), tester.exp_vec.begin());
        tmp_success = check_almost_equal(topic, vres, tester.exp_vec, out);
        success = tmp_success && success;

        topic = "load int16   -> " + name + "  : ";
        detail::load_vec(bres, tester.i16_vec);
        detail::store_vec(bres, vres);
        std::copy(tester.i16_vec.cbegin(), tester.i16_vec.cend(), tester.exp_vec.begin());
        tmp_success = check_almost_equal(topic, vres, tester.exp_vec, out);
        success = tmp_success && success;

        topic = "load uint16  -> " + name + "  : ";
        detail::load_vec(bres, tester.ui16_vec);
        detail::store_vec(bres, vres);
        std::copy(tester.ui16_vec.cbegin(), tester.ui16_vec.cend(), tester.exp_vec.begin());
        tmp_success = check_almost_equal(topic, vres, tester.exp_vec, out);
        success = tmp_success && success;

        topic = "load int32   -> " + name + "  : ";
        detail::load_vec(bres, tester.i32_vec);
        detail::store_vec(bres, vres);
        std::copy(tester.i32_vec.cbegin(), tester.i32_vec.cend(), tester.exp_vec.begin());
        tmp_success = check_almost_equal(topic, vres, tester.exp_vec, out);
        success = tmp_success && success;

        topic = "load uint32  -> " + name + "  : ";
        detail::load_vec(bres, tester.ui32_vec);
        detail::store_vec(bres, vres);
        std::copy(tester.ui32_vec.cbegin(), tester.ui32_vec.cend(), tester.exp_vec.begin());
        tmp_success = check_almost_equal(topic, vres, tester.exp_vec, out);
        success = tmp_success && success;

        topic = "load int64   -> " + name + "  : ";
        detail::load_vec(bres, tester.i64_vec);
        detail::store_vec(bres, vres);
        std::copy(tester.i64_vec.cbegin(), tester.i64_vec.cend(), tester.exp_vec.begin());
        tmp_success = check_almost_equal(topic, vres, tester.exp_vec, out);
        success = tmp_success && success;

        topic = "load uint64  -> " + name + "  : ";
        detail::load_vec(bres, tester.ui64_vec);
        detail::store_vec(bres, vres);
        std::copy(tester.ui64_vec.cbegin(), tester.ui64_vec.cend(), tester.exp_vec.begin());
        tmp_success = check_almost_equal(topic, vres, tester.exp_vec, out);
        success = tmp_success && success;

        topic = "load float  -> " + name + "  : ";
        detail::load_vec(bres, tester.f_vec);
        detail::store_vec(bres, vres);
        std::copy(tester.f_vec.cbegin(), tester.f_vec.cend(), tester.exp_vec.begin());
        tmp_success = check_almost_equal(topic, vres, tester.exp_vec, out);
        success = tmp_success && success;

        topic = "load double  -> " + name + "  : ";
        detail::load_vec(bres, tester.d_vec);
        detail::store_vec(bres, vres);
        std::copy(tester.d_vec.cbegin(), tester.d_vec.cend(), tester.exp_vec.begin());
        tmp_success = check_almost_equal(topic, vres, tester.exp_vec, out);
        success = tmp_success && success;

#ifdef XSIMD_32_BIT_ABI
        topic = "load long    -> " + name + "  : ";
        detail::load_vec(bres, tester.long_vec);
        detail::store_vec(bres, vres);
        std::copy(tester.long_vec.cbegin(), tester.long_vec.cend(), tester.exp_vec.begin());
        tmp_success = check_almost_equal(topic, vres, tester.exp_vec, out);
        success = tmp_success && success;

        topic = "load ulong   -> " + name + "  : ";
        detail::load_vec(bres, tester.ulong_vec);
        detail::store_vec(bres, vres);
        std::copy(tester.ulong_vec.cbegin(), tester.ulong_vec.cend(), tester.exp_vec.begin());
        tmp_success = check_almost_equal(topic, vres, tester.exp_vec, out);
        success = tmp_success && success;
#endif

        /**************
         * store test *
         **************/

        topic = "store " + name + "  -> int8   : ";
        detail::load_vec(bres, tester.i8_vec);
        detail::store_vec(bres, i8vres);
        std::copy(tester.i8_vec.cbegin(), tester.i8_vec.cend(), i8vres.begin());
        tmp_success = check_almost_equal(topic, i8vres, tester.i8_vec, out);
        success = tmp_success && success;

        topic = "store " + name + "  -> uint8  : ";
        detail::load_vec(bres, tester.ui8_vec);
        detail::store_vec(bres, ui8vres);
        std::copy(tester.ui8_vec.cbegin(), tester.ui8_vec.cend(), ui8vres.begin());
        tmp_success = check_almost_equal(topic, ui8vres, tester.ui8_vec, out);
        success = tmp_success && success;

        topic = "store " + name + "  -> int16  : ";
        detail::load_vec(bres, tester.i16_vec);
        detail::store_vec(bres, i16vres);
        std::copy(tester.i16_vec.cbegin(), tester.i16_vec.cend(), i16vres.begin());
        tmp_success = check_almost_equal(topic, i16vres, tester.i16_vec, out);
        success = tmp_success && success;

        topic = "store " + name + "  -> uint16 : ";
        detail::load_vec(bres, tester.ui16_vec);
        detail::store_vec(bres, ui16vres);
        std::copy(tester.ui16_vec.cbegin(), tester.ui16_vec.cend(), ui16vres.begin());
        tmp_success = check_almost_equal(topic, i16vres, tester.i16_vec, out);
        success = tmp_success && success;

        topic = "store " + name + "  -> int32  : ";
        detail::load_vec(bres, tester.i32_vec);
        detail::store_vec(bres, i32vres);
        std::copy(tester.i32_vec.cbegin(), tester.i32_vec.cend(), i32vres.begin());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "store " + name + "  -> uint32 : ";
        detail::load_vec(bres, tester.ui32_vec);
        detail::store_vec(bres, ui32vres);
        std::copy(tester.ui32_vec.cbegin(), tester.ui32_vec.cend(), ui32vres.begin());
        tmp_success = check_almost_equal(topic, ui32vres, tester.ui32_vec, out);
        success = tmp_success && success;

        topic = "store " + name + "  -> int64  : ";
        detail::load_vec(bres, tester.i64_vec);
        detail::store_vec(bres, i64vres);
        std::copy(tester.i64_vec.cbegin(), tester.i64_vec.cend(), i64vres.begin());
        tmp_success = check_almost_equal(topic, i64vres, tester.i64_vec, out);
        success = tmp_success && success;

        topic = "store " + name + "  -> uint64 : ";
        detail::load_vec(bres, tester.ui64_vec);
        detail::store_vec(bres, ui64vres);
        std::copy(tester.ui64_vec.cbegin(), tester.ui64_vec.cend(), ui64vres.begin());
        tmp_success = check_almost_equal(topic, ui64vres, tester.ui64_vec, out);
        success = tmp_success && success;

#ifdef XSIMD_32_BIT_ABI
        topic = "store " + name + "  -> long   : ";
        detail::load_vec(bres, tester.long_vec);
        detail::store_vec(bres, longvres);
        std::copy(tester.long_vec.cbegin(), tester.long_vec.cend(), longvres.begin());
        tmp_success = check_almost_equal(topic, longvres, tester.long_vec, out);
        success = tmp_success && success;

        topic = "store " + name + "  -> ulong  : ";
        detail::load_vec(bres, tester.ulong_vec);
        detail::store_vec(bres, ulongvres);
        std::copy(tester.ulong_vec.cbegin(), tester.ulong_vec.cend(), ulongvres.begin());
        tmp_success = check_almost_equal(topic, ulongvres, tester.ulong_vec, out);
        success = tmp_success && success;
#endif

        topic = "store " + name + "  -> double : ";
        detail::load_vec(bres, tester.d_vec);
        detail::store_vec(bres, dvres);
        std::copy(tester.d_vec.cbegin(), tester.d_vec.cend(), dvres.begin());
        tmp_success = check_almost_equal(topic, dvres, tester.d_vec, out);
        success = tmp_success && success;

        return success;
    }
}

#endif
