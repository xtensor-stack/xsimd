/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_POLY_EVALUATION_TEST
#define XSIMD_POLY_EVALUATION_TEST

#include "xsimd_test_utils.hpp"
#include "xsimd_tester.hpp"
#include "xsimd/xsimd.hpp"

namespace xsimd
{
    template <class T, std::size_t N, std::size_t A>
    struct simd_poly_evaluation_tester : simd_tester<T, N, A>
    {
        using base_type = simd_tester<T, N, A>;
        using vector_type = typename base_type::vector_type;
        using value_type = typename base_type::value_type;
        using res_type = typename base_type::res_type;

        std::string name;

        res_type lhs;

        simd_poly_evaluation_tester(const std::string& name);
    };

    template <class T, std::size_t N, std::size_t A>
    simd_poly_evaluation_tester<T, N, A>::simd_poly_evaluation_tester(const std::string& n)
        : name(n)
    {
        size_t nb_input = N * 10000;
        lhs.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            lhs[i] = value_type(i) / 4 + value_type(1.2) * std::sqrt(value_type(i + 0.25));
        }
    }

    template <class T>
    bool test_simd_poly_evaluation(std::ostream& out, T& tester)
    {
        using tester_type = T;
        using vector_type = typename tester_type::vector_type;
        using value_type = typename tester_type::value_type;
        using res_type = typename tester_type::res_type;

        vector_type lhs;
        vector_type vres;
        res_type horner_res(tester.lhs.size());
        res_type estrin_res(tester.lhs.size());

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

        std::string topic = "estrin     : ";
        for (size_t i = 0; i < tester.lhs.size(); i += tester.size)
        {
            detail::load_vec(lhs, tester.lhs, i);
            vres = horner<vector_type, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(lhs);
            detail::store_vec(vres, horner_res, i);
            vres = estrin<vector_type, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(lhs);
            detail::store_vec(vres, estrin_res, i);
        }
        return check_almost_equal(topic, horner_res, estrin_res, out);
    }
}

#endif // XSIMD_POLY_EVALUATION_TEST

