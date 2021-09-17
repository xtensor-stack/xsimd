/***************************************************************************
 *                                                                          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/
#include "test_utils.hpp"

namespace
{
    template <typename T, std::size_t N>
    struct init_shuffle_128_base
    {
        using shuffle_vector_type = std::array<T, N>;
        shuffle_vector_type lhs_in, rhs_in, exp_lo, exp_hi;

        std::vector<shuffle_vector_type> create_vectors()
        {
            std::vector<shuffle_vector_type> vects;
            vects.reserve(4);

            /* Generate input data: lhs, rhs */
            for (size_t i = 0; i < N; ++i)
            {
                lhs_in[i] = 2*i + 1;
                rhs_in[i] = 2*i + 2;
            }
            vects.push_back(std::move(lhs_in));
            vects.push_back(std::move(rhs_in));

            /* Expected shuffle data */
            for (size_t i = 0, j= 0; i < N/2; ++i, j=j+2)
            {
                exp_lo[j] = lhs_in[i];
                exp_hi[j] = lhs_in[i + N/2];

                exp_lo[j + 1] = rhs_in[i];
                exp_hi[j + 1] = rhs_in[i + N/2];
            }
            vects.push_back(std::move(exp_lo));
            vects.push_back(std::move(exp_hi));

            return vects;
        }
    };
}

template <class B>
class shuffle_128_test : public testing::Test
{
  protected:
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;

    shuffle_128_test()
    {
        std::cout << "shuffle-128 test" << std::endl;
    }

    void shuffle_128_low_high()
    {
        init_shuffle_128_base<value_type, size> shuffle_base;
        auto shuffle_base_vecs = shuffle_base.create_vectors();
        auto v_lhs = shuffle_base_vecs[0];
        auto v_rhs = shuffle_base_vecs[1];
        auto v_exp_lo = shuffle_base_vecs[2];
        auto v_exp_hi = shuffle_base_vecs[3];

        B b_lhs = B::load_unaligned(v_lhs.data());
        B b_rhs = B::load_unaligned(v_rhs.data());
        B b_exp_lo = B::load_unaligned(v_exp_lo.data());
        B b_exp_hi = B::load_unaligned(v_exp_hi.data());

        /* Only Test 128bit */
        if ((sizeof(value_type) * size) == 16)
        {
            B b_res_lo = xsimd::zip_lo(b_lhs, b_rhs);
            EXPECT_BATCH_EQ(b_res_lo, b_exp_lo) << print_function_name("shuffle-128 low test");

            B b_res_hi = xsimd::zip_hi(b_lhs, b_rhs);
            EXPECT_BATCH_EQ(b_res_hi, b_exp_hi) << print_function_name("shuffle-128 high test");
        }
    }
};

TYPED_TEST_SUITE(shuffle_128_test, batch_types, simd_test_names);

TYPED_TEST(shuffle_128_test, shuffle_128_low_high)
{
    this->shuffle_128_low_high();
}


#if XSIMD_WITH_AVX2 || XSIMD_WITH_AVX512
template <class B>
class shuffle_nbit_test : public testing::Test
{
  protected:
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t b_size = B::size;

    using int32_batch = xsimd::batch<int32_t>;
    using int32_vector = std::vector<int32_t, xsimd::default_allocator<int32_t>>;

    shuffle_nbit_test()
    {
        std::cout << "shuffle-nbit test" << std::endl;
    }

    void shuffle_32bit()
    {
        if((sizeof(value_type) * b_size) == 64) {
            int input[16] = {0x01020304, 0x05060708, 0x09101112, 0x13141516,
                             0x17181920, 0x21222324, 0x25262728, 0x29303132,
                             0x33343536, 0x37383940, 0x41262728, 0x29303148,
                             0x49505152, 0x53545556, 0x57585960, 0x61626364,};
            int mask[16] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
            int expected[16] = {0x05060708, 0x01020304, 0x01020304, 0x01020304,
                                0x01020304, 0x01020304, 0x01020304, 0x01020304,
                                0x01020304, 0x01020304, 0x01020304, 0x01020304,
                                0x01020304, 0x01020304, 0x01020304, 0x01020304,};

            int32_vector v_input(input, input + int32_batch::size);
            int32_vector v_mask(mask, mask + int32_batch::size);
            int32_vector v_expected(expected, expected + int32_batch::size);

            int32_batch b_input = int32_batch::load_unaligned(v_input.data());
            int32_batch b_mask = int32_batch::load_unaligned(v_mask.data());
            int32_batch b_expected = int32_batch::load_unaligned(v_expected.data());

            int32_batch b_op = xsimd::shuffle_nbit(b_mask, b_input);
            EXPECT_BATCH_EQ(b_op, b_expected) << print_function_name("shuffle_nbit: 32bit");

        } else if((sizeof(value_type) * b_size) == 32) {
            int input[8] = {0x01020304, 0x05060708, 0x09101112, 0x13141516,
                            0x17181920, 0x21222324, 0x25262728, 0x29303132,};
            int mask[8] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,};
            int expected[8] = {0x05060708, 0x01020304, 0x01020304, 0x01020304,
                               0x01020304, 0x01020304, 0x01020304, 0x01020304,};

            int32_vector v_input(input, input + int32_batch::size);
            int32_vector v_mask(mask, mask + int32_batch::size);
            int32_vector v_expected(expected, expected + int32_batch::size);

            int32_batch b_input = int32_batch::load_unaligned(v_input.data());
            int32_batch b_mask = int32_batch::load_unaligned(v_mask.data());
            int32_batch b_expected = int32_batch::load_unaligned(v_expected.data());

            int32_batch b_op = xsimd::shuffle_nbit(b_mask, b_input);
            EXPECT_BATCH_EQ(b_op, b_expected) << print_function_name("shuffle_nbit: 32bit");

        } else {
            return;
        }
    }
};

TYPED_TEST_SUITE(shuffle_nbit_test, batch_types, simd_test_names);

TYPED_TEST(shuffle_nbit_test, shuffle_32bit)
{
    this->shuffle_32bit();
}
#endif // XSIMD_WITH_AVX2 || XSIMD_WITH_AVX512

