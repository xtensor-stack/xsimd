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

    namespace detail
    {
        template <class V, class S>
        void load_vec(V& vec, const S& src)
        {
            vec.load_aligned(&src[0]);
        }

        template <class V, class R>
        void store_vec(V& vec, R& res)
        {
            vec.store_aligned(&res[0]);
        }
    }

    template <class T>
    bool test_simd_common(std::ostream& out, T& tester)
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

        out << "fma(simd, simd, simd)    : ";
        vres = fma(lhs, rhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.fma_res, out);
        success = success && tmp_success;

        out << "sqrt(simd)               : ";
        vres = sqrt(lhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.sqrt_res, out);
        success = success && tmp_success;

        out << "hadd(simd)               : ";
        value_type sres = hadd(lhs);
        tmp_success = check_almost_equal(sres, tester.hadd_res, out);
        success = success && tmp_success;

        return success;
    }
}

#endif

