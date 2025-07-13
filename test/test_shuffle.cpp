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

#ifdef __linux__
#include "endian.h"
#if BYTE_ORDER == BIG_ENDIAN
#define XSIMD_NO_SLIDE
#endif
#endif

#include <numeric>

namespace
{
    template <typename T, std::size_t N>
    struct zip_base
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
                lhs_in[i] = static_cast<T>('A' + 2 * i + 1);
                rhs_in[i] = static_cast<T>('A' + 2 * i);
            }
            vects.push_back(std::move(lhs_in));
            vects.push_back(std::move(rhs_in));

            /* Expected zipped data */
            for (size_t i = 0; i < N / 2; ++i)
            {
                exp_lo[2 * i] = lhs_in[i];
                exp_lo[2 * i + 1] = rhs_in[i];
                exp_hi[2 * i] = lhs_in[i + N / 2];
                exp_hi[2 * i + 1] = rhs_in[i + N / 2];
            }
            vects.push_back(std::move(exp_lo));
            vects.push_back(std::move(exp_hi));

            return vects;
        }
    };
}

template <class B>
struct zip_test : zip_base<typename B::value_type, B::size>
{
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using zip_base<value_type, size>::create_vectors;

    void zip_low()
    {
        auto zipped_vecs = create_vectors();
        auto v_lhs = zipped_vecs[0];
        auto v_rhs = zipped_vecs[1];
        auto v_exp_lo = zipped_vecs[2];

        B b_lhs = B::load_unaligned(v_lhs.data());
        B b_rhs = B::load_unaligned(v_rhs.data());
        B b_exp_lo = B::load_unaligned(v_exp_lo.data());

        B b_res_lo = xsimd::zip_lo(b_lhs, b_rhs);
        CHECK_BATCH_EQ(b_res_lo, b_exp_lo);
    }
    void zip_hi()
    {
        auto zipped_vecs = create_vectors();
        auto v_lhs = zipped_vecs[0];
        auto v_rhs = zipped_vecs[1];
        auto v_exp_hi = zipped_vecs[3];

        B b_lhs = B::load_unaligned(v_lhs.data());
        B b_rhs = B::load_unaligned(v_rhs.data());
        B b_exp_hi = B::load_unaligned(v_exp_hi.data());

        B b_res_hi = xsimd::zip_hi(b_lhs, b_rhs);
        CHECK_BATCH_EQ(b_res_hi, b_exp_hi);
    }
};

#if !XSIMD_WITH_AVX512F || XSIMD_WITH_AVX512BW
#define ZIP_BATCH_TYPES BATCH_TYPES
#else
#define ZIP_BATCH_TYPES xsimd::batch<float>, xsimd::batch<double>, xsimd::batch<int32_t>, xsimd::batch<int64_t>
#endif

TEST_CASE_TEMPLATE("[zip]", B, ZIP_BATCH_TYPES)

{
    zip_test<B> Test;
    SUBCASE("zip low") { Test.zip_low(); }
    SUBCASE("zip high") { Test.zip_hi(); }
}

namespace
{
    template <typename T, std::size_t N>
    struct init_slide_base
    {
        using slide_vector_type = std::array<T, N>;
        slide_vector_type v_in,
            v_left0, v_left_full, v_left_half, v_left_above_half, v_left_below_half, v_left_one,
            v_right0, v_right_full, v_right_half, v_right_above_half, v_right_below_half, v_right_one;
        static constexpr unsigned full_slide = N * sizeof(T);
        static constexpr unsigned half_slide = full_slide / 2;
        static constexpr unsigned above_half_slide = half_slide + half_slide / 2;
        static constexpr unsigned below_half_slide = half_slide / 2;
        static constexpr bool activate_above_below_checks = above_half_slide / sizeof(T) * sizeof(T) == above_half_slide;

        init_slide_base()
        {
            std::iota(v_in.begin(), v_in.end(), 1);

            v_left0 = v_in;

            std::fill(v_left_full.begin(), v_left_full.end(), 0);

            std::fill(v_left_half.begin(), v_left_half.end(), 0);
            std::iota(v_left_half.begin() + half_slide / sizeof(T), v_left_half.end(), 1);

            std::fill(v_left_one.begin(), v_left_one.end(), 0);
            std::iota(v_left_one.begin() + 1, v_left_one.end(), 1);

            if (activate_above_below_checks)
            {
                std::fill(v_left_above_half.begin(), v_left_above_half.end(), 0);
                std::iota(v_left_above_half.begin() + above_half_slide / sizeof(T), v_left_above_half.end(), 1);

                std::fill(v_left_below_half.begin(), v_left_below_half.end(), 0);
                std::iota(v_left_below_half.begin() + below_half_slide / sizeof(T), v_left_below_half.end(), 1);
            }

            v_right0 = v_in;

            std::fill(v_right_full.begin(), v_right_full.end(), 0);

            std::fill(v_right_half.begin(), v_right_half.end(), 0);
            std::iota(v_right_half.begin(), v_right_half.begin() + half_slide / sizeof(T), v_in[half_slide / sizeof(T)]);

            std::fill(v_right_one.begin(), v_right_one.end(), 0);
            std::iota(v_right_one.begin(), v_right_one.begin() + full_slide / sizeof(T) - 1, v_in[1]);

            if (activate_above_below_checks)
            {
                std::fill(v_right_above_half.begin(), v_right_above_half.end(), 0);
                std::iota(v_right_above_half.begin(), v_right_above_half.begin() + (full_slide - above_half_slide) / sizeof(T), v_in[above_half_slide / sizeof(T)]);

                std::fill(v_right_below_half.begin(), v_right_below_half.end(), 0);
                std::iota(v_right_below_half.begin(), v_right_below_half.begin() + (full_slide - below_half_slide) / sizeof(T), v_in[below_half_slide / sizeof(T)]);
            }
        }
    };
}

template <class B>
struct slide_test : public init_slide_base<typename B::value_type, B::size>
{
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using base = init_slide_base<typename B::value_type, B::size>;
    using base::above_half_slide;
    using base::activate_above_below_checks;
    using base::below_half_slide;
    using base::full_slide;
    using base::half_slide;

    void slide_left()
    {
        B b_in = B::load_unaligned(this->v_in.data());
        B b_left0 = B::load_unaligned(this->v_left0.data());
        B b_left_full = B::load_unaligned(this->v_left_full.data());
        B b_left_half = B::load_unaligned(this->v_left_half.data());
        B b_left_one = B::load_unaligned(this->v_left_one.data());
        B b_left_above_half = B::load_unaligned(this->v_left_above_half.data());
        B b_left_below_half = B::load_unaligned(this->v_left_below_half.data());

        B b_res_left0 = xsimd::slide_left<0>(b_in);
        INFO("slide_left 0");
        CHECK_BATCH_EQ(b_res_left0, b_left0);

        B b_res_left_full = xsimd::slide_left<full_slide>(b_in);
        INFO("slide_left full");
        CHECK_BATCH_EQ(b_res_left_full, b_left_full);

#ifndef XSIMD_NO_SLIDE
        B b_res_left_half = xsimd::slide_left<half_slide>(b_in);
        INFO("slide_left half_slide");
        CHECK_BATCH_EQ(b_res_left_half, b_left_half);

        B b_res_left_one = xsimd::slide_left<sizeof(value_type)>(b_in);
        INFO("slide_left one_slide");
        CHECK_BATCH_EQ(b_res_left_one, b_left_one);

        if (activate_above_below_checks)
        {
            B b_res_left_above_half = xsimd::slide_left<above_half_slide>(b_in);
            INFO("slide_left above_half_slide");
            CHECK_BATCH_EQ(b_res_left_above_half, b_left_above_half);

            B b_res_left_below_half = xsimd::slide_left<below_half_slide>(b_in);
            INFO("slide_left below_half_slide");
            CHECK_BATCH_EQ(b_res_left_below_half, b_left_below_half);
        }
#endif
    }

    void slide_right()
    {
        B b_in = B::load_unaligned(this->v_in.data());
        B b_right0 = B::load_unaligned(this->v_right0.data());
        B b_right_full = B::load_unaligned(this->v_right_full.data());
        B b_right_half = B::load_unaligned(this->v_right_half.data());
        B b_right_one = B::load_unaligned(this->v_right_one.data());
        B b_right_above_half = B::load_unaligned(this->v_right_above_half.data());
        B b_right_below_half = B::load_unaligned(this->v_right_below_half.data());

        B b_res_right0 = xsimd::slide_right<0>(b_in);
        INFO("slide_right 0");
        CHECK_BATCH_EQ(b_res_right0, b_right0);

        B b_res_right_full = xsimd::slide_right<full_slide>(b_in);
        INFO("slide_right full");
        CHECK_BATCH_EQ(b_res_right_full, b_right_full);

#ifndef XSIMD_NO_SLIDE
        B b_res_right_half = xsimd::slide_right<half_slide>(b_in);
        INFO("slide_right half_slide");
        CHECK_BATCH_EQ(b_res_right_half, b_right_half);

        B b_res_right_one = xsimd::slide_right<sizeof(value_type)>(b_in);
        INFO("slide_right one_slide");
        CHECK_BATCH_EQ(b_res_right_one, b_right_one);

        if (activate_above_below_checks)
        {
            B b_res_right_above_half = xsimd::slide_right<above_half_slide>(b_in);
            INFO("slide_right above_half_slide");
            CHECK_BATCH_EQ(b_res_right_above_half, b_right_above_half);

            B b_res_right_below_half = xsimd::slide_right<below_half_slide>(b_in);
            INFO("slide_right below_half_slide");
            CHECK_BATCH_EQ(b_res_right_below_half, b_right_below_half);
        }
#endif
    }
};

TEST_CASE_TEMPLATE("[slide]", B, BATCH_INT_TYPES)
{
    slide_test<B> Test;
    SUBCASE("slide_left")
    {
        Test.slide_left();
    }
    SUBCASE("slide_right")
    {
        Test.slide_right();
    }
}

template <class B>
struct compress_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    using mask_batch_type = typename B::batch_bool_type;

    static constexpr size_t size = B::size;
    std::array<value_type, size> input;
    std::array<bool, size> mask;
    std::array<value_type, size> expected;

    compress_test()
    {
        for (size_t i = 0; i < size; ++i)
        {
            input[i] = static_cast<value_type>(i);
        }
    }

    void full()
    {
        std::fill(mask.begin(), mask.end(), true);

        for (size_t i = 0; i < size; ++i)
            expected[i] = input[i];

        auto b = xsimd::compress(
            batch_type::load_unaligned(input.data()),
            mask_batch_type::load_unaligned(mask.data()));
        CHECK_BATCH_EQ(b, expected);
    }

    void empty()
    {
        std::fill(mask.begin(), mask.end(), false);

        for (size_t i = 0; i < size; ++i)
            expected[i] = 0;

        auto b = xsimd::compress(
            batch_type::load_unaligned(input.data()),
            mask_batch_type::load_unaligned(mask.data()));
        CHECK_BATCH_EQ(b, expected);
    }

    void interleave()
    {
        for (size_t i = 0; i < size; ++i)
            mask[i] = i % 2 == 0;

        for (size_t i = 0; i < size; ++i)
            expected[i] = i < size / 2 ? input[2 * i] : 0;

        auto b = xsimd::compress(
            batch_type::load_unaligned(input.data()),
            mask_batch_type::load_unaligned(mask.data()));
        CHECK_BATCH_EQ(b, expected);
    }

    void common()
    {
        for (size_t i = 0; i < size; ++i)
            mask[i] = i % 3 == 0;

        for (size_t i = 0; i < size; ++i)
            expected[i] = i < size / 3 ? input[3 * i] : 0;

        auto b = xsimd::compress(
            batch_type::load_unaligned(input.data()),
            mask_batch_type::load_unaligned(mask.data()));
        CHECK_BATCH_EQ(b, expected);
    }
};

#define XSIMD_COMPRESS_TYPES BATCH_FLOAT_TYPES, xsimd::batch<uint8_t>, xsimd::batch<int8_t>, xsimd::batch<uint16_t>, xsimd::batch<int16_t>, xsimd::batch<uint32_t>, xsimd::batch<int32_t>, xsimd::batch<uint64_t>, xsimd::batch<int64_t>

TEST_CASE_TEMPLATE("[compress]", B, XSIMD_COMPRESS_TYPES)
{
    compress_test<B> Test;
    SUBCASE("empty")
    {
        Test.empty();
    }
    SUBCASE("full")
    {
        Test.full();
    }
    // SUBCASE("interleave")
    //{
    //      Test.interleave();
    // }
    // SUBCASE("common")
    //{
    //      Test.common();
    // }
}

template <class B>
struct expand_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    using mask_batch_type = typename B::batch_bool_type;

    static constexpr size_t size = B::size;
    std::array<value_type, size> input;
    std::array<bool, size> mask;
    std::array<value_type, size> expected;

    expand_test()
    {
        for (size_t i = 0; i < size; ++i)
        {
            input[i] = static_cast<value_type>(i);
        }
    }

    void full()
    {
        std::fill(mask.begin(), mask.end(), true);

        for (size_t i = 0; i < size; ++i)
            expected[i] = input[i];

        auto b = xsimd::expand(
            batch_type::load_unaligned(input.data()),
            mask_batch_type::load_unaligned(mask.data()));
        CHECK_BATCH_EQ(b, expected);
    }

    void empty()
    {
        std::fill(mask.begin(), mask.end(), false);

        for (size_t i = 0; i < size; ++i)
            expected[i] = 0;

        auto b = xsimd::expand(
            batch_type::load_unaligned(input.data()),
            mask_batch_type::load_unaligned(mask.data()));
        CHECK_BATCH_EQ(b, expected);
    }

    void interleave()
    {
        for (size_t i = 0; i < size; ++i)
            mask[i] = i % 2 == 0;

        for (size_t i = 0, j = 0; i < size; ++i)
            expected[i] = mask[i] ? input[j++] : 0;

        auto b = xsimd::expand(
            batch_type::load_unaligned(input.data()),
            mask_batch_type::load_unaligned(mask.data()));
        CHECK_BATCH_EQ(b, expected);
    }

    void common()
    {
        for (size_t i = 0; i < size; ++i)
            mask[i] = i % 3 == 0;

        for (size_t i = 0, j = 0; i < size; ++i)
            expected[i] = mask[i] ? input[j++] : 0;

        auto b = xsimd::expand(
            batch_type::load_unaligned(input.data()),
            mask_batch_type::load_unaligned(mask.data()));
        CHECK_BATCH_EQ(b, expected);
    }
};

#define XSIMD_EXPAND_TYPES XSIMD_COMPRESS_TYPES

TEST_CASE_TEMPLATE("[expand]", B, XSIMD_EXPAND_TYPES)
{
    expand_test<B> Test;
    SUBCASE("empty")
    {
        Test.empty();
    }
    SUBCASE("full")
    {
        Test.full();
    }
    SUBCASE("interleave")
    {
        Test.interleave();
    }
    SUBCASE("common")
    {
        Test.common();
    }
}

template <class B>
struct shuffle_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    using arch_type = typename B::arch_type;
    using mask_type = xsimd::as_unsigned_integer_t<value_type>;
    static constexpr size_t size = B::size;
    std::array<value_type, size> lhs;
    std::array<value_type, size> rhs;

    shuffle_test()
    {
        for (size_t i = 0; i < size; ++i)
        {
            lhs[i] = static_cast<value_type>(i);
            rhs[i] = static_cast<value_type>(i + size);
        }
    }

    void no_op()
    {
        B b_lhs = B::load_unaligned(lhs.data());
        B b_rhs = B::load_unaligned(rhs.data());

        struct no_op_lhs_generator
        {
            static constexpr size_t get(size_t index, size_t /*size*/)
            {
                return index;
            }
        };

        INFO("no op lhs");
        B b_res_lhs = xsimd::shuffle(b_lhs, b_rhs, xsimd::make_batch_constant<mask_type, no_op_lhs_generator, arch_type>());
        CHECK_BATCH_EQ(b_res_lhs, b_lhs);

        struct no_op_rhs_generator
        {
            static constexpr size_t get(size_t index, size_t size)
            {
                return index + size;
            }
        };

        INFO("no op rhs");
        B b_res_rhs = xsimd::shuffle(b_lhs, b_rhs, xsimd::make_batch_constant<mask_type, no_op_rhs_generator, arch_type>());
        CHECK_BATCH_EQ(b_res_rhs, b_rhs);
    }

    void common()
    {
        B b_lhs = B::load_unaligned(lhs.data());
        B b_rhs = B::load_unaligned(rhs.data());

        struct common_generator
        {
            static constexpr size_t get(size_t index, size_t size)
            {
                return (index & 1) ? (size - index - 1) : (size + (size - index - 1));
            }
        };

        std::array<value_type, size> ref;
        for (size_t i = 0; i < size; ++i)
        {
            size_t ri = size - i - 1;
            ref[i] = (i & 1) ? lhs[ri] : rhs[ri];
        }
        B b_ref = B::load_unaligned(ref.data());

        B b_res = xsimd::shuffle(b_lhs, b_rhs, xsimd::make_batch_constant<mask_type, common_generator, arch_type>());
        CHECK_BATCH_EQ(b_res, b_ref);
    }

    void pick()
    {
        B b_lhs = B::load_unaligned(lhs.data());
        B b_rhs = B::load_unaligned(rhs.data());

        struct pick_generator
        {
            static constexpr size_t get(size_t index, size_t size)
            {
                return index > 2 ? 0 : size;
            }
        };

        std::array<value_type, size> ref;
        for (size_t i = 0; i < size; ++i)
            ref[i] = (i > 2) ? lhs[0] : rhs[0];
        B b_ref = B::load_unaligned(ref.data());

        B b_res = xsimd::shuffle(b_lhs, b_rhs, xsimd::make_batch_constant<mask_type, pick_generator, arch_type>());
        CHECK_BATCH_EQ(b_res, b_ref);
    }

    void shuffle()
    {
        B b_lhs = B::load_unaligned(lhs.data());
        B b_rhs = B::load_unaligned(rhs.data());

        {
            struct shuffle_lo_generator
            {
                static constexpr size_t get(size_t index, size_t size)
                {
                    return size - index - 1;
                }
            };

            std::array<value_type, size> ref;
            for (size_t i = 0; i < size; ++i)
                ref[i] = lhs[size - i - 1];
            B b_ref = B::load_unaligned(ref.data());

            INFO("shuffle first batch");
            B b_res = xsimd::shuffle(b_lhs, b_rhs, xsimd::make_batch_constant<mask_type, shuffle_lo_generator, arch_type>());
            CHECK_BATCH_EQ(b_res, b_ref);
        }

        {
            struct shuffle_hi_generator
            {
                static constexpr size_t get(size_t index, size_t size)
                {
                    return size + size - index - 1;
                }
            };

            std::array<value_type, size> ref;
            for (size_t i = 0; i < size; ++i)
                ref[i] = rhs[size - i - 1];
            B b_ref = B::load_unaligned(ref.data());

            INFO("shuffle second batch");
            B b_res = xsimd::shuffle(b_lhs, b_rhs, xsimd::make_batch_constant<mask_type, shuffle_hi_generator, arch_type>());
            CHECK_BATCH_EQ(b_res, b_ref);
        }
    }

    void transpose()
    {
        B b_lhs = B::load_unaligned(lhs.data());
        std::array<B, size> b_matrix;
        for (size_t i = 0; i < size; ++i)
            b_matrix[i] = b_lhs;
        std::array<value_type, size * size> ref_matrix;
        for (size_t i = 0; i < size; ++i)
            for (size_t j = 0; j < size; ++j)
                ref_matrix[i * size + j] = lhs[i];

        INFO("transpose");
        xsimd::transpose(b_matrix.data(), b_matrix.data() + b_matrix.size());
        for (size_t i = 0; i < size; ++i)
        {
            CHECK_BATCH_EQ(b_matrix[i], B::load_unaligned(&ref_matrix[i * size]));
        }
    }

    void select()
    {
        B b_lhs = B::load_unaligned(lhs.data());
        B b_rhs = B::load_unaligned(rhs.data());

        struct select_generator
        {
            static constexpr size_t get(size_t index, size_t size)
            {
                return (index % 3) ? (index + size) : index;
            }
        };

        std::array<value_type, size> ref;
        for (size_t i = 0; i < size; ++i)
            ref[i] = (i % 3) ? rhs[i] : lhs[i];
        B b_ref = B::load_unaligned(ref.data());

        INFO("select");
        B b_res = xsimd::shuffle(b_lhs, b_rhs, xsimd::make_batch_constant<mask_type, select_generator, arch_type>());
        CHECK_BATCH_EQ(b_res, b_ref);
    }

    void zip()
    {
        B b_lhs = B::load_unaligned(lhs.data());
        B b_rhs = B::load_unaligned(rhs.data());

        struct zip_lo_generator
        {
            static constexpr size_t get(size_t index, size_t size)
            {
                return (index & 1) ? (index / 2 + size) : index / 2;
            }
        };

        std::array<value_type, size> ref_lo;
        for (size_t i = 0; i < size; ++i)
            ref_lo[i] = (i & 1) ? rhs[i / 2] : lhs[i / 2];
        B b_ref_lo = B::load_unaligned(ref_lo.data());

        INFO("zip_lo");
        B b_res_lo = xsimd::shuffle(b_lhs, b_rhs, xsimd::make_batch_constant<mask_type, zip_lo_generator, arch_type>());
        CHECK_BATCH_EQ(b_res_lo, b_ref_lo);

        struct zip_hi_generator
        {
            static constexpr size_t get(size_t index, size_t size)
            {
                return (index & 1) ? (size / 2 + index / 2 + size) : (size / 2 + index / 2);
            }
        };

        std::array<value_type, size> ref_hi;
        for (size_t i = 0; i < size; ++i)
        {
            ref_hi[i] = (i & 1) ? rhs[size / 2 + i / 2] : lhs[size / 2 + i / 2];
        }
        B b_ref_hi = B::load_unaligned(ref_hi.data());

        INFO("zip_hi");
        B b_res_hi = xsimd::shuffle(b_lhs, b_rhs, xsimd::make_batch_constant<mask_type, zip_hi_generator, arch_type>());
        CHECK_BATCH_EQ(b_res_hi, b_ref_hi);
    }
};

#define XSIMD_SHUFFLE_TYPES BATCH_FLOAT_TYPES, xsimd::batch<uint32_t>, xsimd::batch<int32_t>, xsimd::batch<uint64_t>, xsimd::batch<int64_t>

TEST_CASE_TEMPLATE("[shuffle]", B, XSIMD_SHUFFLE_TYPES)
{
    shuffle_test<B> Test;
    SUBCASE("no-op")
    {
        Test.no_op();
    }
    SUBCASE("common")
    {
        Test.common();
    }
    SUBCASE("pick")
    {
        Test.pick();
    }
    SUBCASE("select")
    {
        Test.select();
    }
    SUBCASE("shuffle")
    {
        Test.shuffle();
    }
    SUBCASE("transpose")
    {
        Test.transpose();
    }
    SUBCASE("zip")
    {
        Test.zip();
    }
}

TEST_CASE_TEMPLATE("[small integer transpose]", B, xsimd::batch<uint16_t>, xsimd::batch<int16_t>, xsimd::batch<uint8_t>, xsimd::batch<int8_t>)
{
    shuffle_test<B> Test;
    SUBCASE("transpose")
    {
        Test.transpose();
    }
}

#if (XSIMD_WITH_SSE2 && !XSIMD_WITH_AVX)
TEST_CASE_TEMPLATE("[small integer shuffle]", B, xsimd::batch<uint16_t>, xsimd::batch<int16_t>)
{
    shuffle_test<B> Test;
    SUBCASE("shuffle")
    {
        Test.shuffle();
    }
}
#endif

#endif
