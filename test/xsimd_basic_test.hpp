/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_COMMON_TEST_HPP
#define XSIMD_COMMON_TEST_HPP

#include "xsimd_tester.hpp"
#include "xsimd_test_utils.hpp"

namespace xsimd
{

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
        res_type fms_res;
        res_type fnma_res;
        res_type fnms_res;
        res_type sqrt_res;
        value_type hadd_res;
        
        simd_basic_tester(const std::string& name);
    };


    template <class T, size_t N, size_t A>
    simd_basic_tester<T, N, A>::simd_basic_tester(const std::string& n)
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
        fms_res.resize(N);
        fnma_res.resize(N);
        fnms_res.resize(N);
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
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_FMA4_VERSION
            fma_res[i] = fma(lhs[i], rhs[i], rhs[i]);
#else
            fma_res[i] = lhs[i] * rhs[i] + rhs[i];
#endif
            fms_res[i] = lhs[i] * rhs[i] - rhs[i];
            fnma_res[i] = - lhs[i] * rhs[i] + rhs[i];
            fnms_res[i] = - lhs[i] * rhs[i] - rhs[i];
            sqrt_res[i] = sqrt(lhs[i]);
            hadd_res += lhs[i];
        }
    }

    template <class T, std::size_t N, std::size_t A>
    struct simd_int_basic_tester : simd_tester<T, N, A>
    {
        using base_type = simd_tester<T, N, A>;
        using vector_type = typename base_type::vector_type;
        using value_type = typename base_type::value_type;
        using res_type = typename base_type::res_type;

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
        res_type and_res;
        res_type or_res;
        res_type xor_res;
        res_type not_res;
        res_type lnot_res;
        res_type min_res;
        res_type max_res;
        res_type abs_res;
        res_type fma_res;
        res_type fms_res;
        res_type fnma_res;
        res_type fnms_res;
        value_type hadd_res;

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
        and_res.resize(N);
        or_res.resize(N);
        xor_res.resize(N);
        not_res.resize(N);
        lnot_res.resize(N);
        min_res.resize(N);
        max_res.resize(N);
        abs_res.resize(N);
        fma_res.resize(N);
        fms_res.resize(N);
        fnma_res.resize(N);
        fnms_res.resize(N);

        s = value_type(1.4);
        hadd_res = value_type(0);
        for (size_t i = 0; i < N; ++i)
        {
            lhs[i] = value_type(i) * 10;
            rhs[i] = value_type(4) + value_type(i);
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
            //and_res[i] = lhs[i] & rhs[i];
            //or_res[i] = lhs[i] | rhs[i];
            //xor_res[i] = lhs[i] ^ rhs[i];
            //not_res[i] = ~lhs[i];
            //lnot_res[i] = !lhs[i];
            min_res[i] = min(lhs[i], rhs[i]);
            max_res[i] = max(lhs[i], rhs[i]);
            abs_res[i] = abs(lhs[i]);
            fma_res[i] = lhs[i] * rhs[i] + rhs[i];
            fms_res[i] = lhs[i] * rhs[i] - rhs[i];
            fnma_res[i] = - lhs[i] * rhs[i] + rhs[i];
            fnms_res[i] = - lhs[i] * rhs[i] - rhs[i];
            hadd_res += lhs[i];
        }
    }
    template <class T>
    bool test_simd_basic(std::ostream& out, T& tester)
    {
        using tester_type = T;
        using vector_type = typename tester_type::vector_type;
        using value_type = typename tester_type::value_type;
        using res_type = typename tester_type::res_type;

        vector_type lhs;
        vector_type rhs;
        vector_type vres;
        res_type res(tester_type::size);
        value_type s = tester.s;
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
        out << dash << name_shift << '-' << shift << dash << std::endl << std::endl;

        out << "load/store aligned       : ";
        detail::load_vec(lhs, tester.lhs);
        detail::store_vec(lhs, res);
        tmp_success = check_almost_equal(res, tester.lhs, out);
        success = success && tmp_success;
        
        out << "load/store unaligned     : ";
        lhs.load_unaligned(&tester.lhs[0]);
        lhs.store_unaligned(&res[0]);
        tmp_success = check_almost_equal(res, tester.lhs, out);
        success = success && tmp_success;

        detail::load_vec(lhs, tester.lhs);
        detail::load_vec(rhs, tester.rhs);

        out << "unary operator-          : ";
        vres = -lhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.minus_res, out);
        success = success && tmp_success;

        out << "operator+=(simd, simd)   : ";
        vres = lhs;
        vres += rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.add_vv_res, out);
        success = success && tmp_success;

        out << "operator+=(simd, scalar) : ";
        vres = lhs;
        vres += s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.add_vs_res, out);
        success = success && tmp_success;

        out << "operator-=(simd, simd)   : ";
        vres = lhs;
        vres -= rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.sub_vv_res, out);
        success = success && tmp_success;

        out << "operator-=(simd, scalar) : ";
        vres = lhs;
        vres -= s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.sub_vs_res, out);
        success = success && tmp_success;

        out << "operator*=(simd, simd)   : ";
        vres = lhs;
        vres *= rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.mul_vv_res, out);
        success = success && tmp_success;

        out << "operator*=(simd, scalar) : ";
        vres = lhs;
        vres *= s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.mul_vs_res, out);
        success = success && tmp_success;

        out << "operator/=(simd, simd)   : ";
        vres = lhs;
        vres /= rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.div_vv_res, out);
        success = success && tmp_success;

        out << "operator/=(simd, scalar) : ";
        vres = lhs;
        vres /= s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.div_vs_res, out);
        success = success && tmp_success;

        out << "operator+(simd, simd)    : ";
        vres = lhs + rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.add_vv_res, out);
        success = success && tmp_success;

        out << "operator+(simd, scalar)  : ";
        vres = lhs + s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.add_vs_res, out);
        success = success && tmp_success;

        out << "operator+(scalar, simd)  : ";
        vres = s + rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.add_sv_res, out);
        success = success && tmp_success;
        
        out << "operator-(simd, simd)    : ";
        vres = lhs - rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.sub_vv_res, out);
        success = success && tmp_success;

        out << "operator-(simd, scalar)  : ";
        vres = lhs - s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.sub_vs_res, out);
        success = success && tmp_success;

        out << "operator-(scalar, simd)  : ";
        vres = s - rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.sub_sv_res, out);
        success = success && tmp_success;

        out << "operator*(simd, simd)    : ";
        vres = lhs * rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.mul_vv_res, out);
        success = success && tmp_success;

        out << "operator*(simd, scalar)  : ";
        vres = lhs * s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.mul_vs_res, out);
        success = success && tmp_success;

        out << "operator*(scalar, simd)  : ";
        vres = s * rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.mul_sv_res, out);
        success = success && tmp_success;

        out << "operator/(simd, simd)    : ";
        vres = lhs / rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.div_vv_res, out);
        success = success && tmp_success;

        out << "operator/(simd, scalar)  : ";
        vres = lhs / s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.div_vs_res, out);
        success = success && tmp_success;

        out << "operator/(scalar, simd)  : ";
        vres = s / rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.div_sv_res, out);
        success = success && tmp_success;

        out << "min(simd, simd)          : ";
        vres = min(lhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.min_res, out);
        success = success && tmp_success;

        out << "max(simd, simd)          : ";
        vres = max(lhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.max_res, out);
        success = success && tmp_success;

        out << "abs(simd)                : ";
        vres = abs(lhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.abs_res, out);
        success = success && tmp_success;

        out << "sqrt(simd)               : ";
        vres = sqrt(lhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.sqrt_res, out);
        success = success && tmp_success;

        out << "fma(simd, simd, simd)    : ";
        vres = fma(lhs, rhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.fma_res, out);
        success = success && tmp_success;

        out << "fms(simd, simd, simd)    : ";
        vres = fms(lhs, rhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.fms_res, out);
        success = success && tmp_success;

        out << "fnma(simd, simd, simd)    : ";
        vres = fnma(lhs, rhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.fnma_res, out);
        success = success && tmp_success;

        out << "fnms(simd, simd, simd)    : ";
        vres = fnms(lhs, rhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.fnms_res, out);
        success = success && tmp_success;

        out << "hadd(simd)               : ";
        value_type sres = hadd(lhs);
        tmp_success = check_almost_equal(sres, tester.hadd_res, out);
        success = success && tmp_success;

        return success;
    }

    template <class T>
    bool test_simd_int_basic(std::ostream& out, T& tester)
    {
        using tester_type = T;
        using vector_type = typename tester_type::vector_type;
        using value_type = typename tester_type::value_type;
        using res_type = typename tester_type::res_type;

        vector_type lhs;
        vector_type rhs;
        vector_type vres;
        res_type res(tester_type::size);
        value_type s = tester.s;
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
        out << dash << name_shift << '-' << shift << dash << std::endl << std::endl;

        out << "load/store aligned       : ";
        detail::load_vec(lhs, tester.lhs);
        detail::store_vec(lhs, res);
        tmp_success = check_almost_equal(res, tester.lhs, out);
        success = success && tmp_success;

        out << "load/store unaligned     : ";
        lhs.load_unaligned(&tester.lhs[0]);
        lhs.store_unaligned(&res[0]);
        tmp_success = check_almost_equal(res, tester.lhs, out);
        success = success && tmp_success;

        detail::load_vec(lhs, tester.lhs);
        detail::load_vec(rhs, tester.rhs);

        out << "unary operator-          : ";
        vres = -lhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.minus_res, out);
        success = success && tmp_success;

        out << "operator+=(simd, simd)   : ";
        vres = lhs;
        vres += rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.add_vv_res, out);
        success = success && tmp_success;

        out << "operator+=(simd, scalar) : ";
        vres = lhs;
        vres += s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.add_vs_res, out);
        success = success && tmp_success;

        out << "operator-=(simd, simd)   : ";
        vres = lhs;
        vres -= rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.sub_vv_res, out);
        success = success && tmp_success;

        out << "operator-=(simd, scalar) : ";
        vres = lhs;
        vres -= s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.sub_vs_res, out);
        success = success && tmp_success;

        out << "operator*=(simd, simd)   : ";
        vres = lhs;
        vres *= rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.mul_vv_res, out);
        success = success && tmp_success;

        out << "operator*=(simd, scalar) : ";
        vres = lhs;
        vres *= s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.mul_vs_res, out);
        success = success && tmp_success;

        out << "operator+(simd, simd)    : ";
        vres = lhs + rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.add_vv_res, out);
        success = success && tmp_success;

        out << "operator+(simd, scalar)  : ";
        vres = lhs + s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.add_vs_res, out);
        success = success && tmp_success;

        out << "operator+(scalar, simd)  : ";
        vres = s + rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.add_sv_res, out);
        success = success && tmp_success;

        out << "operator-(simd, simd)    : ";
        vres = lhs - rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.sub_vv_res, out);
        success = success && tmp_success;

        out << "operator-(simd, scalar)  : ";
        vres = lhs - s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.sub_vs_res, out);
        success = success && tmp_success;

        out << "operator-(scalar, simd)  : ";
        vres = s - rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.sub_sv_res, out);
        success = success && tmp_success;

        out << "operator*(simd, simd)    : ";
        vres = lhs * rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.mul_vv_res, out);
        success = success && tmp_success;

        out << "operator*(simd, scalar)  : ";
        vres = lhs * s;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.mul_vs_res, out);
        success = success && tmp_success;

        out << "operator*(scalar, simd)  : ";
        vres = s * rhs;
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.mul_sv_res, out);
        success = success && tmp_success;

        out << "min(simd, simd)          : ";
        vres = min(lhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.min_res, out);
        success = success && tmp_success;

        out << "max(simd, simd)          : ";
        vres = max(lhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.max_res, out);
        success = success && tmp_success;

        out << "abs(simd)                : ";
        vres = abs(lhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.abs_res, out);
        success = success && tmp_success;

        out << "hadd(simd)               : ";
        value_type sres = hadd(lhs);
        tmp_success = check_almost_equal(sres, tester.hadd_res, out);
        success = success && tmp_success;

        out << "fma(simd, simd, simd)    : ";
        vres = fma(lhs, rhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.fma_res, out);
        success = success && tmp_success;

        out << "fms(simd, simd, simd)    : ";
        vres = fms(lhs, rhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.fms_res, out);
        success = success && tmp_success;

        out << "fnma(simd, simd, simd)    : ";
        vres = fnma(lhs, rhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.fnma_res, out);
        success = success && tmp_success;

        out << "fnms(simd, simd, simd)    : ";
        vres = fnms(lhs, rhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.fnms_res, out);
        success = success && tmp_success;

        return success;
    }
}

#endif

