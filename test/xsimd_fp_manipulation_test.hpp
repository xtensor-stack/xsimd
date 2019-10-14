/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_FP_MANIPULATION_TEST_HPP
#define XSIMD_FP_MANIPULATION_TEST_HPP

#include "xsimd_test_utils.hpp"
#include "xsimd_tester.hpp"
#include "xsimd/xsimd.hpp"

namespace xsimd
{

    template <class T, std::size_t N, std::size_t A>
    struct simd_fpmanip_tester : simd_tester<T, N, A>
    {
        using base_type = simd_tester<T, N, A>;
        using vector_type = typename base_type::vector_type;
        using int_type = as_integer_t<T>;
        using ivector_type = batch<int_type, N>;
        using value_type = typename base_type::value_type;
        using res_type = typename base_type::res_type;
        using ires_type = std::vector<int_type, aligned_allocator<int_type, A>>;

        std::string name;

        res_type input;
        int exponent;
        res_type ldexp_res;
        res_type frexp_res;
        ires_type exp_frexp_res;

        simd_fpmanip_tester(const std::string& n);
    };

    template <class T, std::size_t N, std::size_t A>
    inline simd_fpmanip_tester<T, N, A>::simd_fpmanip_tester(const std::string& n)
        : name(n)
    {
        input.resize(N);
        ldexp_res.resize(N);
        frexp_res.resize(N);
        exp_frexp_res.resize(N);
        exponent = 5;
        for (size_t i = 0; i < N; ++i)
        {
            input[i] = value_type(i) / 4 + value_type(1.2) * std::sqrt(value_type(i + 0.25));
            ldexp_res[i] = std::ldexp(input[i], exponent);
            int tmp;
            frexp_res[i] = std::frexp(input[i], &tmp);
            exp_frexp_res[i] = static_cast<int_type>(tmp);
        }
    }

    template <class T>
    bool test_simd_fp_manipulation(std::ostream& out, T& tester)
    {
        using tester_type = T;

        using vector_type = typename tester_type::vector_type;
        using ivector_type = typename tester_type::ivector_type;
        using value_type = typename tester_type::value_type;
        using res_type = typename tester_type::res_type;

        vector_type input;
        ivector_type exponent = ivector_type(tester.exponent);
        vector_type vres;
        res_type res(tester_type::size);
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

        std::string topic = "ldexp    : ";
        detail::load_vec(input, tester.input);
        vres = ldexp(input, exponent);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.ldexp_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::ldexp(tester.input[0], tester.exponent), tester.ldexp_res[0], out);

        topic = "frexp    : ";
        detail::load_vec(input, tester.input);
        vres = frexp(input, exponent);
        detail::store_vec(vres, res);
        tmp_success = check_almost_equal(topic, res, tester.frexp_res, out);
        success = success && tmp_success;
        success &= check_almost_equal(topic, xsimd::frexp(tester.input[0], tester.exponent), tester.frexp_res[0], out);

        return success;
    }
}

#endif
