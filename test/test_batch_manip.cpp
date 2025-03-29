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

namespace xsimd
{
    template <typename T, std::size_t N>
    struct init_swizzle_base
    {
        using swizzle_vector_type = std::array<T, N>;
        swizzle_vector_type lhs_in, exped_reverse, exped_fill, exped_dup, exped_ror, exped_rol;

        template <int... Indices>
        std::vector<swizzle_vector_type> create_swizzle_vectors()
        {
            std::vector<swizzle_vector_type> vects;

            /* Generate input data */
            for (std::size_t i = 0; i < N; ++i)
            {
                lhs_in[i] = static_cast<T>(2 * i + 1);
            }
            vects.push_back(std::move(lhs_in));

            /* Expected reversed data */
            for (std::size_t i = 0; i < N; ++i)
            {
                exped_reverse[i] = lhs_in[N - 1 - i];
                exped_fill[i] = lhs_in[N - 1];
                exped_dup[i] = lhs_in[2 * (i / 2)];
                exped_ror[i] = lhs_in[(i - 1) % N];
                exped_rol[i] = lhs_in[(i + 1) % N];
            }
            vects.push_back(std::move(exped_reverse));
            vects.push_back(std::move(exped_fill));
            vects.push_back(std::move(exped_dup));
            vects.push_back(std::move(exped_ror));
            vects.push_back(std::move(exped_rol));

            return vects;
        }
    };
}

template <class T>
struct Reversor
{
    static constexpr T get(T i, T n)
    {
        return n - 1 - i;
    }
};

template <class T>
struct Last
{
    static constexpr T get(T, T n)
    {
        return n - 1;
    }
};

template <class T>
struct Dup
{
    static constexpr T get(T i, T)
    {
        return 2 * (i / 2);
    }
};

template <class T>
struct as_index
{
    using type = xsimd::as_unsigned_integer_t<T>;
};

template <class T>
struct as_index<std::complex<T>> : as_index<T>
{
};

template <class B>
struct insert_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;

    void insert_first()
    {
        value_type fill_value = 0;
        value_type sentinel_value = 1;
        batch_type v(fill_value);
        batch_type w = insert(v, sentinel_value, ::xsimd::index<0>());
        std::array<value_type, batch_type::size> data;
        w.store_unaligned(data.data());
        CHECK_SCALAR_EQ(data.front(), sentinel_value);
        for (size_t i = 1; i < batch_type::size; ++i)
            CHECK_SCALAR_EQ(data[i], fill_value);
    }

    void insert_last()
    {
        value_type fill_value = 0;
        value_type sentinel_value = 1;
        batch_type v(fill_value);
        batch_type w = insert(v, sentinel_value, ::xsimd::index<batch_type::size - 1>());
        std::array<value_type, batch_type::size> data;
        w.store_unaligned(data.data());
        for (size_t i = 0; i < batch_type::size - 1; ++i)
            CHECK_SCALAR_EQ(data[i], fill_value);
        CHECK_SCALAR_EQ(data.back(), sentinel_value);
    }
};

TEST_CASE_TEMPLATE("[insert_test]", B, BATCH_TYPES)
{
    insert_test<B> Test;
    SUBCASE("insert_first")
    {
        Test.insert_first();
    }

    SUBCASE("insert_last")
    {
        Test.insert_last();
    }
}

template <class B>
struct swizzle_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    using arch_type = typename B::arch_type;
    static constexpr size_t size = B::size;

    void rotate_right()
    {
        xsimd::init_swizzle_base<value_type, size> swizzle_base;
        auto swizzle_vecs = swizzle_base.create_swizzle_vectors();
        auto v_lhs = swizzle_vecs[0];
        auto v_exped = swizzle_vecs[4];

        B b_lhs = B::load_unaligned(v_lhs.data());
        B b_exped = B::load_unaligned(v_exped.data());

        B b_res = xsimd::rotate_right<1>(b_lhs);
        CHECK_BATCH_EQ(b_res, b_exped);
    }

    void rotate_left()
    {
        xsimd::init_swizzle_base<value_type, size> swizzle_base;
        auto swizzle_vecs = swizzle_base.create_swizzle_vectors();
        auto v_lhs = swizzle_vecs[0];
        auto v_exped = swizzle_vecs[5];

        B b_lhs = B::load_unaligned(v_lhs.data());
        B b_exped = B::load_unaligned(v_exped.data());

        B b_res = xsimd::rotate_left<1>(b_lhs);
        CHECK_BATCH_EQ(b_res, b_exped);
    }

    void swizzle_reverse()
    {
        xsimd::init_swizzle_base<value_type, size> swizzle_base;
        auto swizzle_vecs = swizzle_base.create_swizzle_vectors();
        auto v_lhs = swizzle_vecs[0];
        auto v_exped = swizzle_vecs[1];

        B b_lhs = B::load_unaligned(v_lhs.data());
        B b_exped = B::load_unaligned(v_exped.data());

        using index_type = typename as_index<value_type>::type;
        auto index_batch = xsimd::make_batch_constant<index_type, Reversor<index_type>, arch_type>();

        B b_res = xsimd::swizzle(b_lhs, index_batch);
        CHECK_BATCH_EQ(b_res, b_exped);

        B b_dyres = xsimd::swizzle(b_lhs, (xsimd::batch<index_type, arch_type>)index_batch);
        CHECK_BATCH_EQ(b_dyres, b_exped);
    }

    void swizzle_fill()
    {
        xsimd::init_swizzle_base<value_type, size> swizzle_base;
        auto swizzle_vecs = swizzle_base.create_swizzle_vectors();
        auto v_lhs = swizzle_vecs[0];
        auto v_exped = swizzle_vecs[2];

        B b_lhs = B::load_unaligned(v_lhs.data());
        B b_exped = B::load_unaligned(v_exped.data());

        using index_type = typename as_index<value_type>::type;
        auto index_batch = xsimd::make_batch_constant<index_type, Last<index_type>, arch_type>();

        B b_res = xsimd::swizzle(b_lhs, index_batch);
        CHECK_BATCH_EQ(b_res, b_exped);

        B b_dyres = xsimd::swizzle(b_lhs, (xsimd::batch<index_type, arch_type>)index_batch);
        CHECK_BATCH_EQ(b_dyres, b_exped);
    }

    void swizzle_dup()
    {
        xsimd::init_swizzle_base<value_type, size> swizzle_base;
        auto swizzle_vecs = swizzle_base.create_swizzle_vectors();
        auto v_lhs = swizzle_vecs[0];
        auto v_exped = swizzle_vecs[3];

        B b_lhs = B::load_unaligned(v_lhs.data());
        B b_exped = B::load_unaligned(v_exped.data());

        using index_type = typename as_index<value_type>::type;
        auto index_batch = xsimd::make_batch_constant<index_type, Dup<index_type>, arch_type>();

        B b_res = xsimd::swizzle(b_lhs, index_batch);
        CHECK_BATCH_EQ(b_res, b_exped);

        B b_dyres = xsimd::swizzle(b_lhs, (xsimd::batch<index_type, arch_type>)index_batch);
        CHECK_BATCH_EQ(b_dyres, b_exped);
    }
};

TEST_CASE_TEMPLATE("[swizzle]", B, BATCH_SWIZZLE_TYPES)
{
    swizzle_test<B> Test;
    SUBCASE("reverse")
    {
        Test.swizzle_reverse();
    }

    SUBCASE("rotate")
    {
        Test.rotate_left();
        Test.rotate_right();
    }

    SUBCASE("fill")
    {
        Test.swizzle_fill();
    }

    SUBCASE("dup")
    {
        Test.swizzle_dup();
    }
}

#endif
