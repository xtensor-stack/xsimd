/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_COMPLEX_BASIC_TEST_HPP
#define XSIMD_COMPLEX_BASIC_TEST_HPP

#include "xsimd_test_utils.hpp"
#include "xsimd_complex_tester.hpp"

#include "xsimd/types/xsimd_traits.hpp"

namespace xsimd
{
    template <class T, std::size_t N, std::size_t A>
    struct simd_complex_basic_tester : simd_complex_tester<T, N, A>
    {
        using base_type = simd_complex_tester<T, N, A>;
        using vector_type = typename base_type::vector_type;
        using value_type = typename base_type::value_type;
        using res_type = typename base_type::res_type;
        using real_value_type = typename base_type::real_value_type;

        std::string name;

        value_type s;
        res_type lhs;
        res_type rhs;
        res_type mix_lhs_rhs;

        value_type extract_res;
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
        value_type hadd_res;

        simd_complex_basic_tester(const std::string& name);
    };

    template <class T, size_t N, size_t A>
    simd_complex_basic_tester<T, N, A>::simd_complex_basic_tester(const std::string& n)
        : name(n)
    {
        lhs.resize(N);
        rhs.resize(N);
        mix_lhs_rhs.resize(N);
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

        s = value_type(real_value_type(1.4), real_value_type(2.3));
        hadd_res = real_value_type(0);
        for (size_t i = 0; i < N; ++i)
        {
            lhs[i] = value_type(real_value_type(i) / real_value_type(4) + real_value_type(1.2) * std::sqrt(real_value_type(i + 0.25)),
                                real_value_type(i) / real_value_type(5));
            rhs[i] = value_type(real_value_type(10.2) / real_value_type(i + 2) + real_value_type(0.25), real_value_type(i) / real_value_type(3.2));
            extract_res = lhs[1];
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
            hadd_res += lhs[i];
        }

        for (size_t i = 0; i < N / 2; ++i)
        {
            mix_lhs_rhs[2 * i] = lhs[2 * i];
            mix_lhs_rhs[2 * i + 1] = rhs[2 * i + 1];
        }
    }

    /***************
     * basic tests *
     ***************/

    template <class T>
    bool test_simd_complex_basic(std::ostream& out, T& tester)
    {
        using tester_type = T;
        using vector_type = typename tester_type::vector_type;
        using value_type = typename tester_type::value_type;
        using real_value_type = typename value_type::value_type;
        using res_type = typename tester_type::res_type;

        vector_type lhs;
        vector_type rhs;
        vector_type mix_lhs_rhs;
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
        out << dash << name_shift << '-' << shift << dash << std::endl
            << std::endl;

        std::string topic = "operator[]               : ";
        tester.load_vec(lhs, tester.lhs);
        value_type es = lhs[1];
        tmp_success = check_almost_equal(topic, es, tester.extract_res, out);
        success = success && tmp_success;

        topic = "load/store aligned       : ";
        tester.load_vec(lhs, tester.lhs);
        tester.store_vec(lhs, res);
        tmp_success = check_almost_equal(topic, res, tester.lhs, out);
        success = success && tmp_success;

        topic = "load/store unaligned     : ";
        tester.load_vec_unaligned(lhs, tester.lhs);
        tester.store_vec_unaligned(lhs, res);
        tmp_success = check_almost_equal(topic, res, tester.lhs, out);
        success = success && tmp_success;

        tester.load_vec(lhs, tester.lhs);
        tester.load_vec(rhs, tester.rhs);
        tester.load_vec(mix_lhs_rhs, tester.mix_lhs_rhs);

        topic = "unary operator-          : ";
        vres = -lhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.minus_res, out);
        success = success && tmp_success;

        topic = "operator+=(simd, simd)   : ";
        vres = lhs;
        vres += rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_vv_res, out);
        success = success && tmp_success;

        topic = "operator+=(simd, scalar) : ";
        vres = lhs;
        vres += s;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_vs_res, out);
        success = success && tmp_success;

        topic = "operator-=(simd, simd)   : ";
        vres = lhs;
        vres -= rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_vv_res, out);
        success = success && tmp_success;

        topic = "operator-=(simd, scalar) : ";
        vres = lhs;
        vres -= s;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_vs_res, out);
        success = success && tmp_success;

        topic = "operator*=(simd, simd)   : ";
        vres = lhs;
        vres *= rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_vv_res, out);
        success = success && tmp_success;

        topic = "operator*=(simd, scalar) : ";
        vres = lhs;
        vres *= s;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_vs_res, out);
        success = success && tmp_success;

        topic = "operator/=(simd, simd)   : ";
        vres = lhs;
        vres /= rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_vv_res, out);
        success = success && tmp_success;

        topic = "operator/=(simd, scalar) : ";
        vres = lhs;
        vres /= s;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_vs_res, out);
        success = success && tmp_success;

        topic = "operator+(simd, simd)    : ";
        vres = lhs + rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_vv_res, out);
        success = success && tmp_success;

        topic = "operator+(simd, scalar)  : ";
        vres = lhs + s;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_vs_res, out);
        success = success && tmp_success;

        topic = "operator+(scalar, simd)  : ";
        vres = s + rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_sv_res, out);
        success = success && tmp_success;

        topic = "operator-(simd, simd)    : ";
        vres = lhs - rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_vv_res, out);
        success = success && tmp_success;

        topic = "operator-(simd, scalar)  : ";
        vres = lhs - s;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_vs_res, out);
        success = success && tmp_success;

        topic = "operator-(scalar, simd)  : ";
        vres = s - rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_sv_res, out);
        success = success && tmp_success;

        topic = "operator*(simd, simd)    : ";
        vres = lhs * rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_vv_res, out);
        success = success && tmp_success;

        topic = "operator*(simd, scalar)  : ";
        vres = lhs * s;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_vs_res, out);
        success = success && tmp_success;

        topic = "operator*(scalar, simd)  : ";
        vres = s * rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_sv_res, out);
        success = success && tmp_success;

        topic = "operator/(simd, simd)    : ";
        vres = lhs / rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_vv_res, out);
        success = success && tmp_success;

        topic = "operator/(simd, scalar)  : ";
        vres = lhs / s;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_vs_res, out);
        success = success && tmp_success;

        topic = "operator/(scalar, simd)  : ";
        vres = s / rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_sv_res, out);
        success = success && tmp_success;

        topic = "hadd(simd)               : ";
        value_type sres = hadd(lhs);
        tmp_success = check_almost_equal(topic, sres, tester.hadd_res, out);
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
        success = success && test_simd_bool(vector_type(real_value_type(0)), out);

        return success;
    }
}

#endif
