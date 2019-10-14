/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_CPOWER_TEST_HPP
#define XSIMD_CPOWER_TEST_HPP

#include "xsimd_test_utils.hpp"
#include "xsimd_complex_tester.hpp"
#include "xsimd/xsimd.hpp"

namespace xsimd
{
    template <class T, std::size_t N, std::size_t A>
    struct simd_cpower_tester : simd_complex_tester<T, N, A>
    {
        using base_type = simd_complex_tester<T, N, A>;
        using vector_type = typename base_type::vector_type;
        using value_type = typename base_type::value_type;
        using res_type = typename base_type::res_type;
        using real_res_type = typename base_type::real_res_type;
        using real_vector_type = typename vector_type::real_batch;
        using real_value_type = typename vector_type::real_value_type;

        std::string name;

        res_type lhs_nn;
        res_type lhs_pn;
        res_type lhs_np;
        res_type lhs_pp;
        res_type rhs;
        real_res_type abs_res;
        real_res_type arg_res;
        res_type pow_res;
        res_type sqrt_nn_res;
        res_type sqrt_pn_res;
        res_type sqrt_np_res;
        res_type sqrt_pp_res;

        simd_cpower_tester(const std::string& n);
    };


    template <class T, std::size_t N, std::size_t A>
    simd_cpower_tester<T, N, A>::simd_cpower_tester(const std::string& n)
        : name(n)
    {
        using std::abs;
        using std::arg;
        using std::pow;
        using std::sqrt;
        size_t nb_input = N * 10000;
        lhs_nn.resize(nb_input);
        lhs_pn.resize(nb_input);
        lhs_np.resize(nb_input);
        lhs_pp.resize(nb_input);
        rhs.resize(nb_input);
        abs_res.resize(nb_input);
        arg_res.resize(nb_input);
        pow_res.resize(nb_input);
        sqrt_nn_res.resize(nb_input);
        sqrt_pn_res.resize(nb_input);
        sqrt_np_res.resize(nb_input);
        sqrt_pp_res.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            real_value_type real = (real_value_type(i) / 4 + real_value_type(1.2) * std::sqrt(real_value_type(i + 0.25)))/ 100;
            real_value_type imag = (real_value_type(i) / 7 + real_value_type(1.7) * std::sqrt(real_value_type(i + 0.37))) / 100;
            lhs_nn[i] = value_type(-real, -imag);
            lhs_pn[i] = value_type(real, -imag);
            lhs_np[i] = value_type(-real, imag);
            lhs_pp[i] = value_type(real, imag);
            rhs[i] = value_type(real_value_type(10.2) / (i + 2) + real_value_type(0.25),
                                real_value_type(9.1) / (i + 3) + real_value_type(0.45));
            abs_res[i] = abs(lhs_np[i]);
            arg_res[i] = arg(lhs_np[i]);
            pow_res[i] = pow(lhs_np[i], rhs[i]);
            sqrt_nn_res[i] = sqrt(lhs_nn[i]);
            sqrt_pn_res[i] = sqrt(lhs_pn[i]);
            sqrt_np_res[i] = sqrt(lhs_np[i]);
            sqrt_pp_res[i] = sqrt(lhs_pp[i]);
        }
    }

    template <class T>
    bool test_simd_cpower(std::ostream& out, T& tester)
    {
        using tester_type = T;
        using vector_type = typename tester_type::vector_type;
        using real_vector_type = typename tester_type::real_vector_type;
        using value_type = typename tester_type::value_type;
        using res_type = typename tester_type::res_type;
        using real_res_type = typename tester_type::real_res_type;

        vector_type lhs;
        vector_type rhs;
        vector_type vres;
        real_vector_type rvres;
        res_type res(tester.lhs_nn.size());
        real_res_type rres(res.size());
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

        std::string topic = "abs     : ";
        for (size_t i = 0; i < tester.lhs_np.size(); i += tester.size)
        {
            tester.load_vec(lhs, tester.lhs_np, i);
            rvres = abs(lhs);
            rvres.store_aligned(&rres[i]);
        }
        tmp_success = check_almost_equal(topic, rres, tester.abs_res, out);
        success = success && tmp_success;

        topic = "arg     : ";
        for (size_t i = 0; i < tester.lhs_np.size(); i += tester.size)
        {
            tester.load_vec(lhs, tester.lhs_np, i);
            rvres = arg(lhs);
            rvres.store_aligned(&rres[i]);
        }
        tmp_success = check_almost_equal(topic, rres, tester.arg_res, out);
        success = success && tmp_success;

        topic = "pow     : ";
        for (size_t i = 0; i < tester.lhs_np.size(); i += tester.size)
        {
            tester.load_vec(lhs, tester.lhs_np, i);
            tester.load_vec(rhs, tester.rhs, i);
            vres = pow(lhs, rhs);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.pow_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::pow(tester.lhs_np[0], tester.rhs[0]), tester.pow_res[0], out);

        topic = "sqrt_nn : ";
        for (size_t i = 0; i < tester.lhs_nn.size(); i += tester.size)
        {
            tester.load_vec(lhs, tester.lhs_nn, i);
            vres = sqrt(lhs);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.sqrt_nn_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::sqrt(tester.lhs_nn[0]), tester.sqrt_nn_res[0], out);

        topic = "sqrt_pn : ";
        for (size_t i = 0; i < tester.lhs_pn.size(); i += tester.size)
        {
            tester.load_vec(lhs, tester.lhs_pn, i);
            vres = sqrt(lhs);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.sqrt_pn_res, out);
        success = success && tmp_success;

        topic = "sqrt_np : ";
        for (size_t i = 0; i < tester.lhs_np.size(); i += tester.size)
        {
            tester.load_vec(lhs, tester.lhs_np, i);
            vres = sqrt(lhs);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.sqrt_np_res, out);
        success = success && tmp_success;

        topic = "sqrt_pp : ";
        for (size_t i = 0; i < tester.lhs_pp.size(); i += tester.size)
        {
            tester.load_vec(lhs, tester.lhs_pp, i);
            vres = sqrt(lhs);
            tester.store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.sqrt_pp_res, out);
        success = success && tmp_success;

        return success;
    }
}

#endif
