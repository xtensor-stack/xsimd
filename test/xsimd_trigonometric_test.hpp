/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_TRIGONOMETRIC_TEST_HPP
#define XSIMD_TRIGONOMETRIC_TEST_HPP

#include "xsimd_tester.hpp"
#include "xsimd_test_utils.hpp"

namespace xsimd
{

    template <class T, std::size_t N, std::size_t A>
    struct simd_trigonometric_tester : simd_tester<T, N, A>
    {
        using base_type = simd_tester<T, N, A>;
        using vector_type = typename base_type::vector_type;
        using value_type = typename base_type::value_type;
        using res_type = typename base_type::res_type;

        std::string name;

        res_type input;
        res_type sin_res;
        res_type cos_res;
        res_type tan_res;
        res_type ainput;
        res_type asin_res;
        res_type acos_res;

        simd_trigonometric_tester(const std::string& n);
    };

    template <class T, std::size_t N, std::size_t A>
    simd_trigonometric_tester<T, N, A>::simd_trigonometric_tester(const std::string& n)
        : name(n)
    {
        size_t nb_input = N * 10000;
        input.resize(nb_input);
        sin_res.resize(nb_input);
        cos_res.resize(nb_input);
        tan_res.resize(nb_input);
        ainput.resize(nb_input);
        asin_res.resize(nb_input);
        acos_res.resize(nb_input);

        for (size_t i = 0; i < nb_input; ++i)
        {
            input[i] = value_type(0.) + i * value_type(80.) / nb_input;
            sin_res[i] = std::sin(input[i]);
            cos_res[i] = std::cos(input[i]);
            tan_res[i] = std::tan(input[i]);
            ainput[i] = value_type(-1.) + value_type(2.) * i / nb_input;
            asin_res[i] = std::asin(ainput[i]);
            acos_res[i] = std::acos(ainput[i]);
        }
    }

    template <class T>
    bool test_simd_trigonometric(std::ostream& out, T& tester)
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

        out << "sin   : ";
        for (size_t i = 0; i < tester.input.size(); i += tester.size)
        {
            detail::load_vec(input, tester.input, i);
            vres = sin(input);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(res, tester.sin_res, out);
        success = success && tmp_success;

        out << "cos   : ";
        for (size_t i = 0; i < tester.input.size(); i += tester.size)
        {
            detail::load_vec(input, tester.input, i);
            vres = cos(input);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(res, tester.cos_res, out);
        success = success && tmp_success;

        out << "tan   : ";
        for (size_t i = 0; i < tester.input.size(); i += tester.size)
        {
            detail::load_vec(input, tester.input, i);
            vres = tan(input);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(res, tester.tan_res, out);
        success = success && tmp_success;

        out << "asin  : ";
        for (size_t i = 0; i < tester.ainput.size(); i += tester.size)
        {
            detail::load_vec(input, tester.ainput, i);
            vres = asin(input);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(res, tester.asin_res, out);

        out << "acos  : ";
        for (size_t i = 0; i < tester.ainput.size(); i += tester.size)
        {
            detail::load_vec(input, tester.ainput, i);
            vres = acos(input);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(res, tester.acos_res, out);

        success = success && tmp_success;
        return success;
    }
}

#endif
