/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_TRIGONOMETRIC_TEST_HPP
#define XSIMD_TRIGONOMETRIC_TEST_HPP

#include "xsimd_test_utils.hpp"
#include "xsimd_tester.hpp"
#include "xsimd/xsimd.hpp"

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
        res_type atan_input;
        res_type atan_res;
        value_type atan2_lhs;
        res_type atan2_res;

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
        atan_input.resize(nb_input);
        atan_res.resize(nb_input);
        atan2_lhs = 1.5;
        atan2_res.resize(nb_input);

        for (size_t i = 0; i < nb_input; ++i)
        {
            input[i] = value_type(0.) + i * value_type(80.) / nb_input;
            sin_res[i] = std::sin(input[i]);
            cos_res[i] = std::cos(input[i]);
            tan_res[i] = std::tan(input[i]);
            ainput[i] = value_type(-1.) + value_type(2.) * i / nb_input;
            asin_res[i] = std::asin(ainput[i]);
            acos_res[i] = std::acos(ainput[i]);
            atan_input[i] = value_type(-10.) + i * value_type(20.) / nb_input;
            atan_res[i] = std::atan(atan_input[i]);
            atan2_res[i] = std::atan2(atan2_lhs, atan_input[i]);
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
        vector_type vres2;
        res_type res(tester.input.size());
        res_type res2(tester.input.size());

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

        std::string topic = "sin   : ";
        for (size_t i = 0; i < tester.input.size(); i += tester.size)
        {
            detail::load_vec(input, tester.input, i);
            vres = sin(input);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.sin_res, out);
        success = success && tmp_success;
        (void)xsimd::sin(tester.input[0]);

        topic = "cos   : ";
        for (size_t i = 0; i < tester.input.size(); i += tester.size)
        {
            detail::load_vec(input, tester.input, i);
            vres = cos(input);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.cos_res, out);
        success = success && tmp_success;
        (void)xsimd::cos(tester.input[0]);

        topic = "sincos: ";
        for (size_t i = 0; i < tester.input.size(); i += tester.size)
        {
            detail::load_vec(input, tester.input, i);
            sincos(input, vres, vres2);
            detail::store_vec(vres, res, i);
            detail::store_vec(vres2, res2, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.sin_res, out);
        tmp_success = check_almost_equal(topic, res2, tester.cos_res, out);
        success = success && tmp_success;
        (void)xsimd::sincos(tester.input[0], res[0], res2[0]);

        topic = "tan   : ";
        for (size_t i = 0; i < tester.input.size(); i += tester.size)
        {
            detail::load_vec(input, tester.input, i);
            vres = tan(input);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.tan_res, out);
        success = success && tmp_success;
        (void)xsimd::tan(tester.input[0]);

        topic = "asin  : ";
        for (size_t i = 0; i < tester.ainput.size(); i += tester.size)
        {
            detail::load_vec(input, tester.ainput, i);
            vres = asin(input);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.asin_res, out);
        (void)xsimd::asin(tester.input[0]);

        topic = "acos  : ";
        for (size_t i = 0; i < tester.ainput.size(); i += tester.size)
        {
            detail::load_vec(input, tester.ainput, i);
            vres = acos(input);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.acos_res, out);
        (void)xsimd::acos(tester.input[0]);

        topic = "atan  : ";
        for (size_t i = 0; i < tester.atan_input.size(); i += tester.size)
        {
            detail::load_vec(input, tester.atan_input, i);
            vres = atan(input);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.atan_res, out);
        (void)xsimd::atan(tester.input[0]);

        topic = "atan2 : ";
        vector_type atan2_lhs(tester.atan2_lhs);
        for (size_t i = 0; i < tester.atan_input.size(); i += tester.size)
        {
            detail::load_vec(input, tester.atan_input, i);
            vres = atan2(atan2_lhs, input);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.atan2_res, out);
        (void)xsimd::atan2(tester.atan2_lhs, tester.input[0]);

        success = success && tmp_success;
        return success;
    }
}

#endif
