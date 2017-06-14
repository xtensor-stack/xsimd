/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_BASIC_MATH_TEST_HPP
#define XSIMD_BASIC_MATH_TEST_HPP

#include "xsimd_tester.hpp"
#include "xsimd_test_utils.hpp"

namespace xsimd
{
    template <class T, std::size_t N, std::size_t A>
    struct simd_basic_math_tester : simd_tester<T, N, A>
    {
        using base_type = simd_tester<T, N, A>;
        using vector_type = typename base_type::vector_type;
        using int_type = as_integer_t<T>;
        using ivector_type = batch<int_type, N>;
        using value_type = typename base_type::value_type;
        using res_type = typename base_type::res_type;
        using ires_type = std::vector<int_type, aligned_allocator<int_type, A>>;

        std::string name;

        res_type lhs_input;
        res_type rhs_input;
        res_type fdim_res;
        res_type inf_res;
        res_type finite_res;

        simd_basic_math_tester(const std::string& n);
    };

    template <class T, std::size_t N, std::size_t A>
    inline simd_basic_math_tester<T, N, A>::simd_basic_math_tester(const std::string& n)
        : name(n)
    {
        lhs_input.resize(N);
        rhs_input.resize(N);
        fdim_res.resize(N);
        inf_res.resize(N);
        finite_res.resize(N);
        for (size_t i = 0; i < N; ++i)
        {
            lhs_input[i] = value_type(i) / 4 + value_type(1.2) * std::sqrt(value_type(i + 0.25));
            rhs_input[i] = value_type(10.2) / (i + 2) + value_type(0.25);
            fdim_res[i] = std::fdim(lhs_input[i], rhs_input[i]);
            inf_res[i] = T(0.);
            finite_res[i] = T(1.);
        }
    }

    template <class T>
    bool test_simd_basic_math(std::ostream& out, T& tester)
    {
        using tester_type = T;

        using vector_type = typename tester_type::vector_type;
        using ivector_type = typename tester_type::ivector_type;
        using value_type = typename tester_type::value_type;
        using res_type = typename tester_type::res_type;

        vector_type lhs;
        vector_type rhs;
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

        out << "fdim     : ";
        detail::load_vec(lhs, tester.lhs_input);
        detail::load_vec(rhs, tester.rhs_input);
        vres = fdim(lhs, rhs);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.fdim_res, out);
        success = success && tmp_success;

        out << "isfinite : ";
        lhs = vector_type(12.);
        vres = select(isfinite(lhs), vector_type(1.), vector_type(0.));
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.finite_res, out);
        success = success && tmp_success;
        lhs = infinity<vector_type>();
        vres = select(isfinite(lhs), vector_type(1.), vector_type(0.));
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.inf_res, out);
        success = success && tmp_success;

        out << "isinf    : ";
        lhs = vector_type(12.);
        vres = select(isinf(lhs), vector_type(0.), vector_type(1.));
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.finite_res, out);
        success = success && tmp_success;
        lhs = infinity<vector_type>();
        vres = select(isinf(lhs), vector_type(0.), vector_type(1.));
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(res, tester.inf_res, out);
        success = success && tmp_success;

        return success;
    }
}

#endif
