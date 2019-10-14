/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_ROUNDING_TEST_HPP
#define XSIMD_ROUNDING_TEST_HPP

#include "xsimd_test_utils.hpp"
#include "xsimd_tester.hpp"
#include "xsimd/xsimd.hpp"

namespace xsimd
{

    template <class T, std::size_t N, std::size_t A>
    struct simd_rounding_tester : simd_tester<T, N, A>
    {
        using base_type = simd_tester<T, N, A>;
        using vector_type = typename base_type::vector_type;
        using value_type = typename base_type::value_type;
        using res_type = typename base_type::res_type;

        std::string name;

        res_type input;
        res_type ceil_res;
        res_type floor_res;
        res_type trunc_res;
        res_type round_res;
        res_type nearbyint_res;
        res_type rint_res;

        simd_rounding_tester(const std::string& n);
    };

    template <class T, std::size_t N, std::size_t A>
    simd_rounding_tester<T, N, A>::simd_rounding_tester(const std::string& n)
        : name(n)
    {
        size_t size = 8;
        input.resize(size);
        ceil_res.resize(size);
        floor_res.resize(size);
        trunc_res.resize(size);
        round_res.resize(size);
        nearbyint_res.resize(size);
        rint_res.resize(size);

        input[0] = value_type(-3.5);
        input[1] = value_type(-2.7);
        input[2] = value_type(-2.5);
        input[3] = value_type(-2.3);
        input[4] = value_type(2.3);
        input[5] = value_type(2.5);
        input[6] = value_type(2.7);
        input[7] = value_type(3.5);

        for (size_t i = 0; i < size; ++i)
        {
            ceil_res[i] = std::ceil(input[i]);
            floor_res[i] = std::floor(input[i]);
            trunc_res[i] = std::trunc(input[i]);
            round_res[i] = std::round(input[i]);
            nearbyint_res[i] = std::nearbyint(input[i]);
            rint_res[i] = std::rint(input[i]);
        }
    }

    template <class T>
    bool test_simd_rounding(std::ostream& out, T& tester)
    {
        using tester_type = T;
        using vector_type = typename tester_type::vector_type;
        using value_type = typename tester_type::value_type;
        using res_type = typename tester_type::res_type;

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

        std::string topic = "ceil      : ";
        for (size_t i = 0; i < res.size(); ++i)
        {
            res[i] = ceil(vector_type(tester.input[i]))[0];
        }
        tmp_success = check_almost_equal(topic, res, tester.ceil_res, out);
        success = success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::ceil(tester.input[0]), tester.ceil_res[0], out);

        topic = "floor     : ";
        for (size_t i = 0; i < res.size(); ++i)
        {
            res[i] = floor(vector_type(tester.input[i]))[0];
        }
        tmp_success = check_almost_equal(topic, res, tester.floor_res, out);
        success = success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::floor(tester.input[0]), tester.floor_res[0], out);

        topic = "trunc     : ";
        for (size_t i = 0; i < res.size(); ++i)
        {
            res[i] = trunc(vector_type(tester.input[i]))[0];
        }
        tmp_success = check_almost_equal(topic, res, tester.trunc_res, out);
        success = success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::trunc(tester.input[0]), tester.trunc_res[0], out);

        topic = "round     : ";
        for (size_t i = 0; i < res.size(); ++i)
        {
            res[i] = round(vector_type(tester.input[i]))[0];
        }
        tmp_success = check_almost_equal(topic, res, tester.round_res, out);
        success = success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::round(tester.input[0]), tester.round_res[0], out);

        topic = "nearbyint : ";
        for (size_t i = 0; i < res.size(); ++i)
        {
            res[i] = nearbyint(vector_type(tester.input[i]))[0];
        }
        tmp_success = check_almost_equal(topic, res, tester.nearbyint_res, out);
        success = success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::nearbyint(tester.input[0]), tester.nearbyint_res[0], out);

        topic = "rint      : ";
        for (size_t i = 0; i < res.size(); ++i)
        {
            res[i] = rint(vector_type(tester.input[i]))[0];
        }
        tmp_success = check_almost_equal(topic, res, tester.rint_res, out);
        success = success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::rint(tester.input[0]), tester.rint_res[0], out);

        return success;
    }
}

#endif
