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

#include "test_utils.hpp"

#if !XSIMD_WITH_NEON && !XSIMD_WITH_NEON64

namespace xsimd
{
    template <typename T, std::size_t N>
    struct init_swizzle_base
    {
        using swizzle_vector_type = std::array<T, N>;
        swizzle_vector_type lhs_in, exped_reverse, exped_fill, exped_dup;

        template <int... Indices>
        std::vector<swizzle_vector_type> create_swizzle_vectors()
        {
            std::vector<swizzle_vector_type> vects;

            /* Generate input data */
            for (std::size_t i = 0; i < N; ++i)
            {
                lhs_in[i] = 2 * i + 1;
            }
            vects.push_back(std::move(lhs_in));

            /* Expected reversed data */
            for (std::size_t i = 0; i < N; ++i)
            {
                exped_reverse[i] = lhs_in[N - 1 - i];
                exped_fill[i] = lhs_in[N - 1];
                exped_dup[i] = lhs_in[2 * (i / 2)];
            }
            vects.push_back(std::move(exped_reverse));
            vects.push_back(std::move(exped_fill));
            vects.push_back(std::move(exped_dup));

            return vects;
        }
    };
}

struct Reversor
{
    static constexpr unsigned get(unsigned i, unsigned n)
    {
        return n - 1 - i;
    }
};

struct Last
{
    static constexpr unsigned get(unsigned, unsigned n)
    {
        return n - 1;
    }
};

struct Dup
{
    static constexpr unsigned get(unsigned i, unsigned)
    {
        return 2 * (i / 2);
    }
};

template <class T>
struct as_index
{
    using type = xsimd::as_unsigned_integer_t<T>;
};

template <class T, class A>
struct as_index<xsimd::batch<std::complex<T>, A>> : as_index<xsimd::batch<T, A>>
{
};

template <class B>
class swizzle_test : public testing::Test
{
protected:
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;

    swizzle_test()
    {
        std::cout << "swizzle tests" << std::endl;
    }

    void swizzle_reverse()
    {
        xsimd::init_swizzle_base<value_type, size> swizzle_base;
        auto swizzle_vecs = swizzle_base.create_swizzle_vectors();
        auto v_lhs = swizzle_vecs[0];
        auto v_exped = swizzle_vecs[1];

        B b_lhs = B::load_unaligned(v_lhs.data());
        B b_exped = B::load_unaligned(v_exped.data());

        B b_res = xsimd::swizzle(b_lhs, xsimd::make_batch_constant<typename as_index<batch_type>::type, Reversor>());
        EXPECT_BATCH_EQ(b_res, b_exped) << print_function_name("swizzle reverse test");
    }

    void swizzle_fill()
    {
        xsimd::init_swizzle_base<value_type, size> swizzle_base;
        auto swizzle_vecs = swizzle_base.create_swizzle_vectors();
        auto v_lhs = swizzle_vecs[0];
        auto v_exped = swizzle_vecs[2];

        B b_lhs = B::load_unaligned(v_lhs.data());
        B b_exped = B::load_unaligned(v_exped.data());

        B b_res = xsimd::swizzle(b_lhs, xsimd::make_batch_constant<typename as_index<batch_type>::type, Last>());
        EXPECT_BATCH_EQ(b_res, b_exped) << print_function_name("swizzle fill test");
    }

    void swizzle_dup()
    {
        xsimd::init_swizzle_base<value_type, size> swizzle_base;
        auto swizzle_vecs = swizzle_base.create_swizzle_vectors();
        auto v_lhs = swizzle_vecs[0];
        auto v_exped = swizzle_vecs[3];

        B b_lhs = B::load_unaligned(v_lhs.data());
        B b_exped = B::load_unaligned(v_exped.data());

        B b_res = xsimd::swizzle(b_lhs, xsimd::make_batch_constant<typename as_index<batch_type>::type, Dup>());
        EXPECT_BATCH_EQ(b_res, b_exped) << print_function_name("swizzle dup test");
    }
};

TYPED_TEST_SUITE(swizzle_test, batch_swizzle_types, simd_test_names);

TYPED_TEST(swizzle_test, swizzle_reverse)
{
    this->swizzle_reverse();
}

TYPED_TEST(swizzle_test, swizzle_fill)
{
    this->swizzle_fill();
}

TYPED_TEST(swizzle_test, swizzle_dup)
{
    this->swizzle_dup();
}

#endif
