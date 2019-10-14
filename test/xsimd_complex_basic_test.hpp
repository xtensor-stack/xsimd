/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_COMPLEX_BASIC_TEST_HPP
#define XSIMD_COMPLEX_BASIC_TEST_HPP

#include "xsimd_test_utils.hpp"
#include "xsimd_complex_tester.hpp"

#include "xsimd/math/xsimd_math_complex.hpp"
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
        using real_res_type = typename base_type::real_res_type;
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
        res_type add_vrv_res;
        res_type add_rvv_res;
        res_type add_vrs_res;
        res_type add_rsv_res;
        res_type sub_vv_res;
        res_type sub_vs_res;
        res_type sub_sv_res;
        res_type sub_vrv_res;
        res_type sub_rvv_res;
        res_type sub_vrs_res;
        res_type sub_rsv_res;
        res_type mul_vv_res;
        res_type mul_vs_res;
        res_type mul_sv_res;
        res_type mul_vrv_res;
        res_type mul_rvv_res;
        res_type mul_vrs_res;
        res_type mul_rsv_res;
        res_type div_vv_res;
        res_type div_vs_res;
        res_type div_sv_res;
        res_type div_vrv_res;
        res_type div_rvv_res;
        res_type div_vrs_res;
        res_type div_rsv_res;
        res_type conj_res;
        real_res_type norm_res;
        res_type proj_res;
        value_type hadd_res;

        simd_complex_basic_tester(const std::string& name);
    };

    template <class T, size_t N, size_t A>
    simd_complex_basic_tester<T, N, A>::simd_complex_basic_tester(const std::string& n)
        : name(n)
    {
        using std::norm;
        using std::proj;
        using std::conj;

        lhs.resize(N);
        rhs.resize(N);
        mix_lhs_rhs.resize(N);
        minus_res.resize(N);
        add_vv_res.resize(N);
        add_vs_res.resize(N);
        add_sv_res.resize(N);
        add_vrv_res.resize(N);
        add_rvv_res.resize(N);
        add_vrs_res.resize(N);
        add_rsv_res.resize(N);
        sub_vv_res.resize(N);
        sub_vs_res.resize(N);
        sub_sv_res.resize(N);
        sub_vrv_res.resize(N);
        sub_rvv_res.resize(N);
        sub_vrs_res.resize(N);
        sub_rsv_res.resize(N);
        mul_vv_res.resize(N);
        mul_vs_res.resize(N);
        mul_sv_res.resize(N);
        mul_vrv_res.resize(N);
        mul_rvv_res.resize(N);
        mul_vrs_res.resize(N);
        mul_rsv_res.resize(N);
        div_vv_res.resize(N);
        div_vs_res.resize(N);
        div_sv_res.resize(N);
        div_vrv_res.resize(N);
        div_rvv_res.resize(N);
        div_vrs_res.resize(N);
        div_rsv_res.resize(N);
        conj_res.resize(N);
        norm_res.resize(N);
        proj_res.resize(N);

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
            add_vrv_res[i] = lhs[i] + rhs[i].real();
            add_rvv_res[i] = lhs[i].real() + rhs[i];
            add_vrs_res[i] = lhs[i] + s.real();
            add_rsv_res[i] = s.real() + rhs[i];
            sub_vv_res[i] = lhs[i] - rhs[i];
            sub_vs_res[i] = lhs[i] - s;
            sub_sv_res[i] = s - rhs[i];
            sub_vrv_res[i] = lhs[i] - rhs[i].real();
            sub_rvv_res[i] = lhs[i].real() - rhs[i];
            sub_vrs_res[i] = lhs[i] - s.real();
            sub_rsv_res[i] = s.real() - rhs[i];
            mul_vv_res[i] = lhs[i] * rhs[i];
            mul_vs_res[i] = lhs[i] * s;
            mul_sv_res[i] = s * rhs[i];
            mul_vrv_res[i] = lhs[i] * rhs[i].real();
            mul_rvv_res[i] = lhs[i].real() * rhs[i];
            mul_vrs_res[i] = lhs[i] * s.real();
            mul_rsv_res[i] = s.real() * rhs[i];
            div_vv_res[i] = lhs[i] / rhs[i];
            div_vs_res[i] = lhs[i] / s;
            div_sv_res[i] = s / rhs[i];
            div_vrv_res[i] = lhs[i] / rhs[i].real();
            div_rvv_res[i] = lhs[i].real() / rhs[i];
            div_vrs_res[i] = lhs[i] / s.real();
            div_rsv_res[i] = s.real() / rhs[i];
            conj_res[i] = conj(lhs[i]);
            norm_res[i] = norm(lhs[i]);
            proj_res[i] = proj(lhs[i]);
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
        using real_res_type = typename tester_type::real_res_type;

        vector_type lhs;
        vector_type rhs;
        vector_type mix_lhs_rhs;
        vector_type vres;
        res_type res(tester_type::size);
        real_res_type rres(tester_type::size);
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

        topic = "operator+=(simd, real)   : ";
        vres = lhs;
        vres += rhs.real();
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_vrv_res, out);
        success = success && tmp_success;

        topic = "operator+=(simd, reals)  : ";
        vres = lhs;
        vres += s.real();
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_vrs_res, out);
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

        topic = "operator-=(simd, real)   : ";
        vres = lhs;
        vres -= rhs.real();
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_vrv_res, out);
        success = success && tmp_success;

        topic = "operator-=(simd, reals)  : ";
        vres = lhs;
        vres -= s.real();
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_vrs_res, out);
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

        topic = "operator*=(simd, real)   : ";
        vres = lhs;
        vres *= rhs.real();
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_vrv_res, out);
        success = success && tmp_success;

        topic = "operator*=(simd, reals)  : ";
        vres = lhs;
        vres *= s.real();
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_vrs_res, out);
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

        topic = "operator/=(simd, real)   : ";
        vres = lhs;
        vres /= rhs.real();
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_vrv_res, out);
        success = success && tmp_success;

        topic = "operator/=(simd, reals)  : ";
        vres = lhs;
        vres /= s.real();
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_vrs_res, out);
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

        topic = "operator+(simd, real)    : ";
        vres = lhs + rhs.real();
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_vrv_res, out);
        success = success && tmp_success;

        topic = "operator+(real, simd)    : ";
        vres = lhs.real() + rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_rvv_res, out);
        success = success && tmp_success;

        topic = "operator+(simd, reals)   : ";
        vres = lhs + s.real();
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_vrs_res, out);
        success = success && tmp_success;

        topic = "operator+(reals, simd)   : ";
        vres = s.real() + rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.add_rsv_res, out);
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

        topic = "operator-(simd, real)    : ";
        vres = lhs - rhs.real();
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_vrv_res, out);
        success = success && tmp_success;

        topic = "operator-(real, simd)    : ";
        vres = lhs.real() - rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_rvv_res, out);
        success = success && tmp_success;

        topic = "operator-(simd, reals)   : ";
        vres = lhs - s.real();
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_vrs_res, out);
        success = success && tmp_success;

        topic = "operator-(reals, simd)   : ";
        vres = s.real() - rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.sub_rsv_res, out);
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

        topic = "operator*(simd, real)    : ";
        vres = lhs * rhs.real();
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_vrv_res, out);
        success = success && tmp_success;

        topic = "operator*(real, simd)    : ";
        vres = lhs.real() * rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_rvv_res, out);
        success = success && tmp_success;

        topic = "operator*(simd, reals)   : ";
        vres = lhs * s.real();
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_vrs_res, out);
        success = success && tmp_success;

        topic = "operator*(reals, simd)   : ";
        vres = s.real() * rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.mul_rsv_res, out);
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

        topic = "operator/(simd, real)    : ";
        vres = lhs / rhs.real();
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_vrv_res, out);
        success = success && tmp_success;

        topic = "operator/(real, simd)    : ";
        vres = lhs.real() / rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_rvv_res, out);
        success = success && tmp_success;

        topic = "operator/(simd, real)    : ";
        vres = lhs / s.real();
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_vrs_res, out);
        success = success && tmp_success;

        topic = "operator/(reals, simd)   : ";
        vres = s.real() / rhs;
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.div_rsv_res, out);
        success = success && tmp_success;

        topic = "conj(simd)               : ";
        vres = xsimd::conj(lhs);
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.conj_res, out);
        success = success && tmp_success;

        topic = "norm(simd)               : ";
        typename vector_type::real_batch rvres = norm(lhs);
        detail::store_vec(rvres, rres);
        tmp_success = check_almost_equal(topic, rres, tester.norm_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::norm(tester.lhs[0]), tester.norm_res[0], out);

        topic = "proj(simd)               : ";
        vres = proj(lhs);
        tester.store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.proj_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::proj(tester.lhs[0]), tester.proj_res[0], out);

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

    /**********************************
     * simd_complex_load_store_tester *
     **********************************/

    template <class T, class R>
    struct complex_rebind;

    template <class T, class R>
    struct complex_rebind<std::complex<T>, R>
    {
        using type = std::complex<R>;
    };

#ifdef XSIMD_ENABLE_XTL_COMPLEX
    template <class T, bool i3ec, class R>
    struct complex_rebind<xtl::xcomplex<T, T, i3ec>, R>
    {
        using type = xtl::xcomplex<R, R, i3ec>;
    };
#endif

    template <class T, class R>
    using complex_rebind_t = typename complex_rebind<T, R>::type;

    template <class T, std::size_t N, std::size_t A>
    struct simd_complex_ls_tester
        : simd_complex_tester<T, N, A>
    {
        using base_type = simd_complex_tester<T, N, A>;
        using batch_type = batch<T, N>;
        using value_type = typename T::value_type;
        using real_batch_type = typename batch_type::real_batch;

        using float_complex = complex_rebind_t<T, float>;
        using double_complex = complex_rebind_t<T, double>;

        using float_vector = std::vector<float, aligned_allocator<float, A>>;
        using double_vector = std::vector<double, aligned_allocator<double, A>>;
        using float_complex_vector = std::vector<float_complex, aligned_allocator<float_complex, A>>;
        using double_complex_vector = std::vector<double_complex, aligned_allocator<double_complex, A>>;

        std::string name;

        float_vector f_vec_real;
        float_vector f_vec_imag;
        float_vector f_vec_zero;
        double_vector d_vec_real;
        double_vector d_vec_imag;
        double_vector d_vec_zero;
        float_complex_vector fc_vec;
        double_complex_vector dc_vec;

        simd_complex_ls_tester(const std::string& n);
    };


    template <class T, std::size_t N, std::size_t A>
    inline simd_complex_ls_tester<T, N, A>::simd_complex_ls_tester(const std::string& n)
        : name(n)
    {
        f_vec_real.resize(N);
        f_vec_imag.resize(N);
        f_vec_zero.resize(N);
        d_vec_real.resize(N);
        d_vec_imag.resize(N);
        d_vec_zero.resize(N);
        fc_vec.resize(N);
        dc_vec.resize(N);
        for (std::size_t i = 0; i < N; ++i)
        {
            f_vec_real[i] = float(2 * i);
            f_vec_imag[i] = float(2 * i + 1);
            f_vec_zero[i] = float(0);
            d_vec_real[i] = double(2 * i);
            d_vec_imag[i] = double(2 * i + 1);
            d_vec_zero[i] = double(0);
            fc_vec[i] = float_complex(f_vec_real[i], f_vec_imag[i]);
            dc_vec[i] = double_complex(d_vec_real[i], d_vec_imag[i]);
        }
    }

    template <class T>
    inline bool test_complex_simd_load_store(std::ostream& out, T& tester)
    {
        using batch_type = typename T::batch_type;
        using real_batch_type = typename T::real_batch_type;
        using float_vector = typename T::float_vector;
        using double_vector = typename T::double_vector;
        using float_complex_vector = typename T::float_complex_vector;
        using double_complex_vector = typename T::double_complex_vector;

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

        std::string topic = "load float complex   : ";
        batch_type fref, fres;
        fref.load_aligned(tester.f_vec_real.data(), tester.f_vec_imag.data());
        fres.load_aligned(tester.fc_vec.data());
        tmp_success = all(fres.real() == fref.real()) && all(fres.imag() == fref.imag());
        success = tmp_success && success;

        topic = "load double complex  : ";
        batch_type dref, dres;
        dref.load_aligned(tester.d_vec_real.data(), tester.d_vec_imag.data());
        dres.load_aligned(tester.dc_vec.data());
        tmp_success = all(dres.real() == dref.real()) && all(dres.imag() == dref.imag());
        success = tmp_success && success;

        topic = "store float complex  : ";
        float_complex_vector fc_res(tester.fc_vec.size());
        fres.store_aligned(fc_res.data());
        tmp_success = check_almost_equal(topic, fc_res, tester.fc_vec, out);
        success = tmp_success && success;

        topic = "store double complex : ";
        double_complex_vector dc_res(tester.dc_vec.size());
        dres.store_aligned(dc_res.data());
        tmp_success = check_almost_equal(topic, dc_res, tester.dc_vec, out);
        success = tmp_success && success;

        topic = "load float complex r : ";
        fref.load_aligned(tester.f_vec_real.data(), tester.f_vec_zero.data());
        fres.load_aligned(tester.f_vec_real.data());
        tmp_success = all(fres.real() == fref.real()) && all(fres.imag() == fref.imag());
        success = tmp_success && success;

        topic = "load double complex r: ";
        dref.load_aligned(tester.d_vec_real.data(), tester.d_vec_zero.data());
        dres.load_aligned(tester.d_vec_real.data());
        tmp_success = all(dres.real() == dref.real()) && all(dres.imag() == dref.imag());
        success = tmp_success && success;

        return success;
    }
}

#endif
