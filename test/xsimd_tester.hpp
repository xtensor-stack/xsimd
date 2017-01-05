/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_TESTER_HPP
#define XSIMD_TESTER_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

#include "xsimd/memory/xaligned_allocator.hpp"

namespace xsimd
{

    template <class V, size_t N, size_t A>
    struct simd_basic_tester
    {
        using vector_type = V;
        using value_type = typename simd_vector_traits<vector_type>::value_type;
        using res_type = std::vector<value_type, aligned_allocator<value_type, A>>;

        constexpr static const size_t size = N;

        std::string name;

        value_type s;
        res_type lhs;
        res_type rhs;

        res_type minus_res;
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
        res_type min_res;
        res_type max_res;
        res_type abs_res;
        res_type fma_res;
        res_type sqrt_res;
        value_type hadd_res;
        
        simd_basic_tester(const std::string& name);
    };


    template <class V, size_t N, size_t A>
    simd_basic_tester<V, N, A>::simd_basic_tester(const std::string& n)
        : name(n)
    {
        using std::min;
        using std::max;
        using std::abs;
        using std::sqrt;
        using std::fma;

        lhs.resize(N);
        rhs.resize(N);
        minus_res.resize(N);
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
        min_res.resize(N);
        max_res.resize(N);
        abs_res.resize(N);
        fma_res.resize(N);
        sqrt_res.resize(N);

        s = value_type(1.4);
        hadd_res = value_type(0);
        for(size_t i = 0; i < N; ++i)
        {
            lhs[i] = value_type(i) / 4 + value_type(1.2) * std::sqrt(value_type(i + 0.25));
            rhs[i] = value_type(10.2) / (i+2) + value_type(0.25);
            minus_res[i] = -lhs[i];
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
            //and_res[i] = lhs[i] & rhs[i];
            //or_res[i] = lhs[i] | rhs[i];
            //xor_res[i] = lhs[i] ^ rhs[i];
            //not_res[i] = ~lhs[i];
            //lnot_res[i] = !lhs[i];
            min_res[i] = min(lhs[i], rhs[i]);
            max_res[i] = max(lhs[i], rhs[i]);
            abs_res[i] = abs(lhs[i]);
#ifdef __FMA__
            fma_res[i] = fma(lhs[i], rhs[i], rhs[i]);
#else
            fma_res[i] = lhs[i] * rhs[i] + rhs[i];
#endif
            sqrt_res[i] = sqrt(lhs[i]);
            hadd_res += lhs[i];
        }
    }
}

#endif

