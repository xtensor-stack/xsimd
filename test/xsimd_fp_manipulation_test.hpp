/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_FP_MANIPULATION_TEST_HPP
#define XSIMD_FP_MANIPULATION_TEST_HPP

#include "xsimd_tester.hpp"
#include "xsimd_test_utils.hpp"

namespace xsimd
{

    template <class T, std::size_t N, std::size_t A>
    struct simd_fpmanip_tester : simd_tester<T, N, A>
    {
        using base_type = simd_tester<T, N, A>;
        using vector_type = typename base_type::vector_type;
        using ivector_type = batch<as_integer_t<T>, N>;
        using value_type = typename base_type::value_type;
        using res_type = typename base_type::res_type;

        std::string name;

        res_type input;
        int exponent;
        res_type ldexp_res;
        res_type inf_res;
        res_type finite_res;

        simd_fpmanip_tester(const std::string& n);
    };

    template <class T, std::size_t N, std::size_t A>
    inline simd_fpmanip_tester<T, N, A>::simd_fpmanip_tester(const std::string& n)
        : name(n)
    {
        input.resize(N);
        ldexp_res.resize(N);
        inf_res.resize(N);
        finite_res.resize(N);
        exponent = 5;
        for (size_t i = 0; i < N; ++i)
        {
            input[i] = value_type(i) / 4 + value_type(1.2) * std::sqrt(value_type(i + 0.25));
            ldexp_res[i] = std::ldexp(input[i], exponent);
            inf_res[i] = T(0.);
            finite_res[i] = T(1.);
        }
    }

    template <class T>
    bool test_simd_fp_manipulation(std::ostream& out, T& tester)
    {
        using tester_type = T;

        using vector_type = typename tester_type::vector_type;
        using ivector_type = typename tester_type::ivector_type;
        using value_type = typename tester_type::value_type;
        using res_type = typename tester_type::res_type;

        vector_type input;
        ivector_type exponent = ivector_type(tester.exponent);
        vector_type vres;
        res_type res(tester_type::size);
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

        out << "ldexp    : ";
        detail::load_vec(input, tester.input);
        vres = ldexp(input, exponent);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.ldexp_res, out);
        success = success && tmp_success;

        out << "isfinite : ";
        input = vector_type(12.);
        vres = select(isfinite(input), vector_type(1.), vector_type(0.));
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.finite_res, out);
        success = success && tmp_success;
        input = infinity<vector_type>();
        vres = select(isfinite(input), vector_type(1.), vector_type(0.));
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.inf_res, out);
        success = success && tmp_success;

        out << "isinf    : ";
        input = vector_type(12.);
        vres = select(isinf(input), vector_type(0.), vector_type(1.));
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.finite_res, out);
        success = success && tmp_success;
        input = infinity<vector_type>();
        vres = select(isinf(input), vector_type(0.), vector_type(1.));
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.inf_res, out);
        success = success && tmp_success;

        return success;
    }

}

#endif
