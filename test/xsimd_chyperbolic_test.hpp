/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_CHYPERBOLIC_TEST_HPP
#define XSIMD_CHYPERBOLIC_TEST_HPP

#include "xsimd_test_utils.hpp"
#include "xsimd_complex_tester.hpp"
#include "xsimd/xsimd.hpp"

namespace xsimd
{

    template <class T, std::size_t N, std::size_t A>
    struct simd_chyperbolic_tester : simd_complex_tester<T, N, A>
    {
        using base_type = simd_complex_tester<T, N, A>;
        using vector_type = typename base_type::vector_type;
        using value_type = typename base_type::value_type;
        using res_type = typename base_type::res_type;
        using real_vector_type = typename vector_type::real_batch;
        using real_value_type = typename vector_type::real_value_type;

        std::string name;

        res_type input;
        res_type sinh_res;
        res_type cosh_res;
        res_type tanh_res;
        res_type asinh_res;
        res_type acosh_input;
        res_type acosh_res;
        res_type atanh_input;
        res_type atanh_res;

        simd_chyperbolic_tester(const std::string& n);
    };

    template <class T, std::size_t N, std::size_t A>
    simd_chyperbolic_tester<T, N, A>::simd_chyperbolic_tester(const std::string& n)
        : name(n)
    {
        using std::sinh;
        using std::cosh;
        using std::tanh;
        using std::asinh;
        using std::acosh;
        using std::atanh;
        size_t nb_input = N * 10000;
        input.resize(nb_input);
        sinh_res.resize(nb_input);
        cosh_res.resize(nb_input);
        tanh_res.resize(nb_input);
        asinh_res.resize(nb_input);
        acosh_input.resize(nb_input);
        acosh_res.resize(nb_input);
        atanh_input.resize(nb_input);
        atanh_res.resize(nb_input);

        for (size_t i = 0; i < nb_input; ++i)
        {
            input[i] = value_type(real_value_type(-1.5) + i * real_value_type(3) / nb_input,
                                  real_value_type(-1.3) + i * real_value_type(2.5) / nb_input);
            sinh_res[i] = sinh(input[i]);
            cosh_res[i] = cosh(input[i]);
            tanh_res[i] = tanh(input[i]);
            asinh_res[i] = asinh(input[i]);
            acosh_input[i] = value_type(real_value_type(1.) + i * real_value_type(3) / nb_input,
                                        real_value_type(1.2) + i * real_value_type(2.7) / nb_input);
            acosh_res[i] = acosh(acosh_input[i]);
            atanh_input[i] = value_type(real_value_type(-0.95) + i * real_value_type(1.9) / nb_input,
                                        real_value_type(-0.94) + i * real_value_type(1.8) / nb_input);
            atanh_res[i] = atanh(atanh_input[i]);
        }
    }

    template <class T>
    bool test_simd_chyperbolic(std::ostream& out, T& tester)
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
        out << dash << name_shift << '-' << shift << dash << std::endl
            << std::endl;

        std::string topic = "sinh  : ";
        for (size_t i = 0; i < tester.input.size(); i += tester.size)
        {
            tester.load_vec(input, tester.input, i);
            vres = sinh(input);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.sinh_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::sinh(tester.input[0]), tester.sinh_res[0], out);

        topic = "cosh  : ";
        for (size_t i = 0; i < tester.input.size(); i += tester.size)
        {
            tester.load_vec(input, tester.input, i);
            vres = cosh(input);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.cosh_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::cosh(tester.input[0]), tester.cosh_res[0], out);

        topic = "tanh  : ";
        for (size_t i = 0; i < tester.input.size(); i += tester.size)
        {
            tester.load_vec(input, tester.input, i);
            vres = tanh(input);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.tanh_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::tanh(tester.input[0]), tester.tanh_res[0], out);

        topic = "asinh : ";
        for (size_t i = 0; i < tester.input.size(); i += tester.size)
        {
            tester.load_vec(input, tester.input, i);
            vres = asinh(input);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.asinh_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::asinh(tester.input[0]), tester.asinh_res[0], out);

        topic = "acosh : ";
        for (size_t i = 0; i < tester.acosh_input.size(); i += tester.size)
        {
            tester.load_vec(input, tester.acosh_input, i);
            vres = acosh(input);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.acosh_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::acosh(tester.acosh_input[0]), tester.acosh_res[0], out);

        topic = "atanh : ";
        for (size_t i = 0; i < tester.atanh_input.size(); i += tester.size)
        {
            tester.load_vec(input, tester.atanh_input, i);
            vres = atanh(input);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.atanh_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::atanh(tester.atanh_input[0]), tester.atanh_res[0], out);

        return success;
    }
}

#endif
