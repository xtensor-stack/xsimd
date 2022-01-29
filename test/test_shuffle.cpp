/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "xsimd/xsimd.hpp"
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE

#include "test_utils.hpp"

namespace
{
    template <typename T, std::size_t N>
    struct init_shuffle_base
    {
        using shuffle_vector_type = std::array<T, N>;
        shuffle_vector_type lhs_in, rhs_in, exp_lo, exp_hi;

        std::vector<shuffle_vector_type> create_vectors()
        {
            std::vector<shuffle_vector_type> vects;
            vects.reserve(4);

            constexpr size_t K = 128 / (sizeof(T) * 8);
            constexpr size_t P = N / K;

            /* Generate input data: lhs, rhs */
            for (size_t p = 0; p < P; ++p)
            {
                for (size_t i = 0; i < K; ++i)
                {
                    lhs_in[i + p * K] = 2 * i + 1;
                    rhs_in[i + p * K] = 2 * i + 2;
                }
            }
            vects.push_back(std::move(lhs_in));
            vects.push_back(std::move(rhs_in));

            /* Expected shuffle data */
            for (size_t p = 0; p < P; ++p)
            {
                for (size_t i = 0, j = 0; i < K / 2; ++i, j = j + 2)
                {
                    exp_lo[j + p * K] = lhs_in[i];
                    exp_hi[j + p * K] = lhs_in[i + K / 2];

                    exp_lo[j + 1 + p * K] = rhs_in[i];
                    exp_hi[j + 1 + p * K] = rhs_in[i + K / 2];
                }
            }
            vects.push_back(std::move(exp_lo));
            vects.push_back(std::move(exp_hi));

            return vects;
        }
    };
}

template <class B>
class shuffle_test : public testing::Test
{
protected:
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;

    shuffle_test()
    {
        std::cout << "shuffle-128 test" << std::endl;
    }

    void shuffle_low_high()
    {
        init_shuffle_base<value_type, size> shuffle_base;
        auto shuffle_base_vecs = shuffle_base.create_vectors();
        auto v_lhs = shuffle_base_vecs[0];
        auto v_rhs = shuffle_base_vecs[1];
        auto v_exp_lo = shuffle_base_vecs[2];
        auto v_exp_hi = shuffle_base_vecs[3];

        B b_lhs = B::load_unaligned(v_lhs.data());
        B b_rhs = B::load_unaligned(v_rhs.data());
        B b_exp_lo = B::load_unaligned(v_exp_lo.data());
        B b_exp_hi = B::load_unaligned(v_exp_hi.data());

        B b_res_lo = xsimd::zip_lo(b_lhs, b_rhs);
        EXPECT_BATCH_EQ(b_res_lo, b_exp_lo) << print_function_name("zip low test");

        B b_res_hi = xsimd::zip_hi(b_lhs, b_rhs);
        EXPECT_BATCH_EQ(b_res_hi, b_exp_hi) << print_function_name("zip high test");
    }
};

TYPED_TEST_SUITE(shuffle_test, batch_types, simd_test_names);

TYPED_TEST(shuffle_test, shuffle_low_high)
{
    this->shuffle_low_high();
}
#endif
