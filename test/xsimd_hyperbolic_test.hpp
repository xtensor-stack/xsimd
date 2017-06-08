/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_HYPERBOLIC_TEST_HPP
#define XSIMD_HYPERBOLIC_TEST_HPP

#include "xsimd_tester.hpp"
#include "xsimd_test_utils.hpp"

namespace xsimd
{

    template <class T, std::size_t N, std::size_t A>
    struct simd_hyperbolic_tester : simd_tester<T, N, A>
    {
        using base_type = simd_tester<T, N, A>;
        using vector_type = typename base_type::vector_type;
        using value_type = typename base_type::value_type;
        using res_type = typename base_type::res_type;

        std::string name;

        res_type input;
        res_type sinh_res;
        res_type cosh_res;
        res_type tanh_res;
        res_type asinh_res;

        simd_hyperbolic_tester(const std::string& n);
    };

    template <class T, std::size_t N, std::size_t A>
    simd_hyperbolic_tester<T, N, A>::simd_hyperbolic_tester(const std::string& n)
        : name(n)
    {
        size_t nb_input = N * 10000;
        input.resize(nb_input);
        sinh_res.resize(nb_input);
        cosh_res.resize(nb_input);
        tanh_res.resize(nb_input);
        asinh_res.resize(nb_input);

        for (size_t i = 0; i < nb_input; ++i)
        {
            input[i] = value_type(-1.5) + i * value_type(3) / nb_input;
            sinh_res[i] = std::sinh(input[i]);
            cosh_res[i] = std::cosh(input[i]);
            tanh_res[i] = std::tanh(input[i]);
            asinh_res[i] = std::asinh(input[i]);
        }
    }

    template <class T>
    bool test_simd_hyperbolic(std::ostream& out, T& tester)
    {
        using tester_type = T;
        using vector_type = typename tester_type::vector_type;
        using value_type = typename tester_type::value_type;
        using res_type = typename tester_type::res_type;

        vector_type input;
        vector_type vres;
        res_type res(tester.input.size());

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

        out << "sinh  : ";
        for (size_t i = 0; i < tester.input.size(); i += tester.size)
        {
            detail::load_vec(input, tester.input, i);
            vres = sinh(input);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(res, tester.sinh_res, out);
        success = success && tmp_success;

        out << "cosh  : ";
        for (size_t i = 0; i < tester.input.size(); i += tester.size)
        {
            detail::load_vec(input, tester.input, i);
            vres = cosh(input);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(res, tester.cosh_res, out);
        success = success && tmp_success;

        out << "tanh  : ";
        for (size_t i = 0; i < tester.input.size(); i += tester.size)
        {
            detail::load_vec(input, tester.input, i);
            vres = tanh(input);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(res, tester.tanh_res, out);
        success = success && tmp_success;

        out << "asinh : ";
        for (size_t i = 0; i < tester.input.size(); i += tester.size)
        {
            detail::load_vec(input, tester.input, i);
            vres = asinh(input);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(res, tester.asinh_res, out);
        success = success && tmp_success;

        return success;
    }
}

#endif
