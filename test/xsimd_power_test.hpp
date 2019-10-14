/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_POWER_TEST_HPP
#define XSIMD_POWER_TEST_HPP

#include "xsimd_test_utils.hpp"
#include "xsimd_tester.hpp"
#include "xsimd/xsimd.hpp"

namespace xsimd
{
    template <class T, std::size_t N, std::size_t A>
    struct simd_power_tester : simd_tester<T, N, A>
    {
        using base_type = simd_tester<T, N, A>;
        using vector_type = typename base_type::vector_type;
        using value_type = typename base_type::value_type;
        using res_type = typename base_type::res_type;

        std::string name;

        res_type lhs;
        res_type rhs;
        res_type pow_res;
        res_type ipow_res;
        res_type cbrt_res;
        res_type hypot_res;

        simd_power_tester(const std::string& name);
    };

    template <class T, std::size_t N, std::size_t A>
    simd_power_tester<T, N, A>::simd_power_tester(const std::string& n)
        : name(n)
    {
        size_t nb_input = N * 10000;
        lhs.resize(nb_input);
        rhs.resize(nb_input);
        pow_res.resize(nb_input);
        ipow_res.resize(nb_input);
        cbrt_res.resize(nb_input);
        hypot_res.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            lhs[i] = value_type(i) / 4 + value_type(1.2) * std::sqrt(value_type(i + 0.25));
            rhs[i] = value_type(10.2) / (i + 2) + value_type(0.25);
            pow_res[i] = std::pow(lhs[i], rhs[i]);
            ipow_res[i] = std::pow(lhs[i], (long)i/N - nb_input / N / 2);
            cbrt_res[i] = std::cbrt(lhs[i]);
            hypot_res[i] = std::hypot(lhs[i], rhs[i]);
        }
    }

    template <class T>
    bool test_simd_power(std::ostream& out, T& tester)
    {
        using tester_type = T;
        using vector_type = typename tester_type::vector_type;
        using value_type = typename tester_type::value_type;
        using res_type = typename tester_type::res_type;

        vector_type lhs;
        vector_type rhs;
        vector_type vres;
        res_type res(tester.lhs.size());
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

        std::string topic = "pow     : ";
        for (size_t i = 0; i < tester.lhs.size(); i += tester.size)
        {
            detail::load_vec(lhs, tester.lhs, i);
            detail::load_vec(rhs, tester.rhs, i);
            vres = pow(lhs, rhs);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.pow_res, out);
        success = success && tmp_success;

        topic = "ipow     : ";

        for (size_t i = 0; i < tester.lhs.size(); i += tester.size)
        {
            detail::load_vec(lhs, tester.lhs, i);
            vres = pow(lhs, i/tester.size - tester.lhs.size() / tester.size / 2);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.ipow_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::pow(tester.lhs[0], - tester.lhs.size() / tester.size / 2),
                                             tester.ipow_res[0], out);

        topic = "hypot   : ";
        for (size_t i = 0; i < tester.lhs.size(); i += tester.size)
        {
            detail::load_vec(lhs, tester.lhs, i);
            detail::load_vec(rhs, tester.rhs, i);
            vres = hypot(lhs, rhs);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.hypot_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::hypot(tester.lhs[0], tester.rhs[0]), tester.hypot_res[0], out);

        topic = "cbrt    : ";
        for (size_t i = 0; i < tester.lhs.size(); i += tester.size)
        {
            detail::load_vec(lhs, tester.lhs, i);
            vres = cbrt(lhs);
            detail::store_vec(vres, res, i);
        }
        tmp_success = check_almost_equal(topic, res, tester.cbrt_res, out);
        success &= check_almost_equal(topic, xsimd::cbrt(tester.lhs[0]), res[0], out);

        success = success && tmp_success;
        return success;
    }
}

#endif
