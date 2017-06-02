/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_EXPONENTIAL_TEST_HPP
#define XSIMD_EXPONENTIAL_TEST_HPP

#include "xsimd_tester.hpp"
#include "xsimd_test_utils.hpp"

namespace xsimd
{

    template <class T, std::size_t N, std::size_t A>
    struct simd_exponential_tester : simd_tester<T, N, A>
    {
        using base_type = simd_tester<T, N, A>;
        using vector_type = typename base_type::vector_type;
        using value_type = typename base_type::value_type;
        using res_type = typename base_type::res_type;

        std::string name;


        res_type input_exp;
        res_type exp_res;
        res_type exp2_res;
        res_type expm1_res;

        simd_exponential_tester(const std::string& n);
    };

    template <class T, std::size_t N, std::size_t A>
    simd_exponential_tester<T, N, A>::simd_exponential_tester(const std::string& n)
        : name(n)
    {
        size_t nb_input = N * 10000;
        input_exp.resize(nb_input);
        exp_res.resize(nb_input);
        exp2_res.resize(nb_input);
        expm1_res.resize(nb_input);

        for (size_t i = 0; i < nb_input; ++i)
        {
            input_exp[i] = value_type(-1.5) + i * value_type(3) / nb_input;
            exp_res[i] = std::exp(input_exp[i]);
            exp2_res[i] = std::exp2(input_exp[i]);
            expm1_res[i] = std::expm1(input_exp[i]);
        }
    }

    template <class T>
    bool test_simd_exponential(std::ostream& out, T& tester)
    {
        using tester_type = T;
        using vector_type = typename tester_type::vector_type;
        using value_type = typename tester_type::value_type;
        using res_type = typename tester_type::res_type;

        vector_type input_exp;
        vector_type vres;
        res_type res(tester.input_exp.size());

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

        out << "exp   : ";
        for (size_t i = 0; i < tester.input_exp.size(); i += tester.size)
        {
            detail::load_vec(input_exp, tester.input_exp, i);
            vres = exp(input_exp);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(res, tester.exp_res, out);
        success = success && tmp_success;

        out << "exp2  : ";
        for (size_t i = 0; i < tester.input_exp.size(); i += tester.size)
        {
            detail::load_vec(input_exp, tester.input_exp, i);
            vres = exp2(input_exp);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(res, tester.exp2_res, out);
        success = success && tmp_success;

        out << "expm1 : ";
        for (size_t i = 0; i < tester.input_exp.size(); i += tester.size)
        {
            detail::load_vec(input_exp, tester.input_exp, i);
            vres = expm1(input_exp);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(res, tester.expm1_res, out);
        success = success && tmp_success;

        return success;
    }
}

#endif
