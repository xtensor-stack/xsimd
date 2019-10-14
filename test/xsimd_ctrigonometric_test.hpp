/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_CTRIGONOMETRIC_TEST_HPP
#define XSIMD_CTRIGONOMETRIC_TEST_HPP

#include "xsimd_test_utils.hpp"
#include "xsimd_complex_tester.hpp"
#include "xsimd/xsimd.hpp"

namespace xsimd
{
    template <class T, std::size_t N, std::size_t A>
    struct simd_ctrigonometric_tester : simd_complex_tester<T, N, A>
    {
        using base_type = simd_complex_tester<T, N, A>;
        using vector_type = typename base_type::vector_type;
        using value_type = typename base_type::value_type;
        using res_type = typename base_type::res_type;
        using real_vector_type = typename vector_type::real_batch;
        using real_value_type = typename vector_type::real_value_type;

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

        simd_ctrigonometric_tester(const std::string& n);
    };

    template <class T, std::size_t N, std::size_t A>
    simd_ctrigonometric_tester<T, N, A>::simd_ctrigonometric_tester(const std::string& n)
        : name(n)
    {
        using std::sin;
        using std::cos;
        using std::tan;
        using std::asin;
        using std::acos;
        using std::atan;

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

        for (size_t i = 0; i < nb_input; ++i)
        {
            input[i] = value_type(real_value_type(0.) + i * real_value_type(80.) / nb_input,
                                  real_value_type(0.1) + i * real_value_type(56.) / nb_input);
            sin_res[i] = sin(input[i]);
            cos_res[i] = cos(input[i]);
            tan_res[i] = tan(input[i]);
            ainput[i] = value_type(real_value_type(-1.) + real_value_type(2.) * i / nb_input,
                                   real_value_type(-1.1) + real_value_type(2.1) * i / nb_input);
            asin_res[i] = asin(ainput[i]);
            acos_res[i] = acos(ainput[i]);
            atan_input[i] = value_type(real_value_type(-10.) + i * real_value_type(20.) / nb_input,
                                       real_value_type(-9.) + i * real_value_type(21.) / nb_input);
            atan_res[i] = atan(atan_input[i]);
        }
    }

    template <class T>
    bool test_simd_ctrigonometric(std::ostream& out, T& tester)
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

        std::string topic = "csin  : ";
        for (size_t i = 0; i < tester.input.size(); i += tester.size)
        {
            tester.load_vec(input, tester.input, i);
            vres = sin(input);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.sin_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::sin(tester.input[0]), tester.sin_res[0], out);

        topic = "ccos  : ";
        for (size_t i = 0; i < tester.input.size(); i += tester.size)
        {
            tester.load_vec(input, tester.input, i);
            vres = cos(input);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.cos_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::cos(tester.input[0]), tester.cos_res[0], out);

        topic = "csincos: ";
        for (size_t i = 0; i < tester.input.size(); i += tester.size)
        {
            tester.load_vec(input, tester.input, i);
            sincos(input, vres, vres2);
            tester.store_vec(vres, res, i);
            tester.store_vec(vres2, res2, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.sin_res, out);
        tmp_success = check_almost_equal(topic, res2, tester.cos_res, out);
        success = success && tmp_success;

        xsimd::sincos(tester.input[0], res[0], res2[0]);
        success &= check_almost_equal(topic, res[0], tester.sin_res[0], out);
        success &= check_almost_equal(topic, res2[0], tester.cos_res[0], out);

        topic = "ctan  : ";
        for (size_t i = 0; i < tester.input.size(); i += tester.size)
        {
            tester.load_vec(input, tester.input, i);
            vres = tan(input);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.tan_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::tan(tester.input[0]), tester.tan_res[0], out);

        topic = "casin : ";
        for (size_t i = 0; i < tester.ainput.size(); i += tester.size)
        {
            tester.load_vec(input, tester.ainput, i);
            vres = asin(input);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.asin_res, out);
        success &= check_almost_equal(topic, xsimd::asin(tester.ainput[0]), tester.asin_res[0], out);

        topic = "cacos : ";
        for (size_t i = 0; i < tester.ainput.size(); i += tester.size)
        {
            tester.load_vec(input, tester.ainput, i);
            vres = acos(input);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.acos_res, out);
        success &= check_almost_equal(topic, xsimd::acos(tester.ainput[0]), tester.acos_res[0], out);

        topic = "catan : ";
        for (size_t i = 0; i < tester.atan_input.size(); i += tester.size)
        {
            tester.load_vec(input, tester.atan_input, i);
            vres = atan(input);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.atan_res, out);
        success &= check_almost_equal(topic, xsimd::atan(tester.atan_input[0]), tester.atan_res[0], out);

        success = success && tmp_success;
        return success;
    }
}

#endif
