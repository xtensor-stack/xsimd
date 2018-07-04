/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_CEXPONENTIAL_TESTER_HPP
#define XSIMD_CEXPONENTIAL_TESTER_HPP

#include "xsimd_test_utils.hpp"
#include "xsimd_complex_tester.hpp"

namespace xsimd
{
    template <class T, std::size_t N, std::size_t A>
    struct simd_cexponential_tester : simd_complex_tester<T, N, A>
    {
        using base_type = simd_complex_tester<T, N, A>;
        using vector_type = typename base_type::vector_type;
        using value_type = typename base_type::value_type;
        using res_type = typename base_type::res_type;
        using real_vector_type = typename vector_type::real_batch;
        using real_value_type = typename vector_type::real_value_type;

        std::string name;

        res_type exp_input;
        res_type exp_res;

        res_type log_input;
        res_type log_res;
        res_type log10_res;

        simd_cexponential_tester(const std::string& n);
    };

    template <class T, std::size_t N, std::size_t A>
    simd_cexponential_tester<T, N, A>::simd_cexponential_tester(const std::string& n)
        : name(n)
    {
        using std::exp;

        size_t nb_input = N * 10000;
        exp_input.resize(nb_input);
        exp_res.resize(nb_input);
        log_input.resize(nb_input);
        log_res.resize(nb_input);
        log10_res.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            exp_input[i] = value_type(real_value_type(-1.5) + i * real_value_type(3) / nb_input,
                                      real_value_type(-1.3) + i * real_value_type(2) / nb_input);
            exp_res[i] = exp(exp_input[i]);
            log_input[i] = value_type(real_value_type(0.001 + i * 100 / nb_input),
                                      real_value_type(0.002 + i * 110 / nb_input));
            log_res[i] = log(log_input[i]);
            log10_res[i] = log10(log_input[i]);
        }
    }

    template <class T>
    bool test_simd_cexponential(std::ostream& out, T& tester)
    {
        using tester_type = T;
        using vector_type = typename tester_type::vector_type;
        using value_type = typename tester_type::value_type;
        using res_type = typename tester_type::res_type;

        vector_type exp_input;
        vector_type log_input;
        vector_type vres;
        res_type res(tester.exp_input.size());

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

        std::string topic = "cexp   : ";
        for (size_t i = 0; i < tester.exp_input.size(); i += tester.size)
        {
            tester.load_vec(exp_input, tester.exp_input, i);
            vres = exp(exp_input);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.exp_res, out);
        success = success && tmp_success;


        topic = "clog   : ";
        for (size_t i = 0; i < tester.log_input.size(); i += tester.size)
        {
            tester.load_vec(log_input, tester.log_input, i);
            vres = log(log_input);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.log_res, out);
        success = success && tmp_success;

        topic = "clog10 : ";
        for (size_t i = 0; i < tester.log_input.size(); i += tester.size)
        {
            tester.load_vec(log_input, tester.log_input, i);
            vres = log10(log_input);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.log10_res, out);
        success = success && tmp_success;

        return success;
    }
}

#endif
