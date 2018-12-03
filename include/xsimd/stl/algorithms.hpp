/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xsimd/memory/xsimd_load_store.hpp"

namespace xsimd
{
    template <class I1, class I2, class O1, class UF>
    void transform(I1 first, I2 last, O1 out_first, UF&& f)
    {
        using value_type = typename std::decay<decltype(*first)>::type;
        using traits = simd_traits<value_type>;
        using batch_type = typename traits::type;

        std::size_t size = static_cast<std::size_t>(std::distance(first, last));
        std::size_t simd_size = traits::size;

        const auto* ptr_begin = &(*first);
        const auto* ptr_end = &(*last);
        auto* ptr_out = &(*out_first);

        std::size_t align_begin = xsimd::get_alignment_offset(ptr_begin, size, simd_size);
        std::size_t out_align = xsimd::get_alignment_offset(ptr_out, size, simd_size);
        std::size_t align_end = align_begin + ((size - align_begin) & ~(simd_size - 1));

        if (align_begin == out_align)
        {
            for (std::size_t i = 0; i < align_begin; ++i)
            {
                out_first[i] = f(first[i]);
            }

            batch_type batch;
            for (std::size_t i = align_begin; i < align_end; i += simd_size)
            {
                xsimd::load_aligned(&first[i], batch);
                xsimd::store_aligned(&out_first[i], f(batch));
            }

            for (std::size_t i = align_end; i < size; ++i)
            {
                out_first[i] = f(first[i]);
            }
        }
        else
        {
            for (std::size_t i = 0; i < align_begin; ++i)
            {
                out_first[i] = f(first[i]);
            }

            batch_type batch;
            for (std::size_t i = align_begin; i < align_end; i += simd_size)
            {
                xsimd::load_aligned(&first[i], batch);
                xsimd::store_unaligned(&out_first[i], f(batch));
            }

            for (std::size_t i = align_end; i < size; ++i)
            {
                out_first[i] = f(first[i]);
            }
        }
    }

    template <class I1, class I2, class I3, class O1, class UF>
    void transform(I1 first_1, I2 last_1, I3 first_2, O1 out_first, UF&& f)
    {
        using value_type = typename std::decay<decltype(*first_1)>::type;
        using traits = simd_traits<value_type>;
        using batch_type = typename traits::type;

        std::size_t size = static_cast<std::size_t>(std::distance(first_1, last_1));
        std::size_t simd_size = traits::size;

        const auto* ptr_begin_1 = &(*first_1);
        const auto* ptr_begin_2 = &(*first_2);
        const auto* ptr_end = &(*last_1);
        auto* ptr_out = &(*out_first);

        std::size_t align_begin_1 = xsimd::get_alignment_offset(ptr_begin_1, size, simd_size);
        std::size_t align_begin_2 = xsimd::get_alignment_offset(ptr_begin_2, size, simd_size);
        std::size_t out_align = xsimd::get_alignment_offset(ptr_out, size, simd_size);
        std::size_t align_end = align_begin_1 + ((size - align_begin_1) & ~(simd_size - 1));

        #define XSIMD_LOOP_MACRO(A1, A2, A3)                                    \
            for (std::size_t i = 0; i < align_begin_1; ++i)                     \
            {                                                                   \
                out_first[i] = f(first_1[i], first_2[i]);                       \
            }                                                                   \
                                                                                \
            batch_type batch_1, batch_2;                                        \
            for (std::size_t i = align_begin_1; i < align_end; i += simd_size)  \
            {                                                                   \
                xsimd::A1(&first_1[i], batch_1);                                \
                xsimd::A2(&first_2[i], batch_2);                                \
                xsimd::A3(&out_first[i], f(batch_1, batch_2));                  \
            }                                                                   \
                                                                                \
            for (std::size_t i = align_end; i < size; ++i)                      \
            {                                                                   \
                out_first[i] = f(first_1[i], first_2[i]);                       \
            }                                                                   \

        if (align_begin_1 == out_align && align_begin_1 == align_begin_2)
        {
            XSIMD_LOOP_MACRO(load_aligned, load_aligned, store_aligned);
        }
        else if (align_begin_1 == out_align && align_begin_1 != align_begin_2)
        {
            XSIMD_LOOP_MACRO(load_aligned, load_unaligned, store_aligned);
        }
        else if (align_begin_1 != out_align && align_begin_1 == align_begin_2)
        {
            XSIMD_LOOP_MACRO(load_aligned, load_aligned, store_unaligned);
        }
        else if (align_begin_1 != out_align && align_begin_1 != align_begin_2)
        {
            XSIMD_LOOP_MACRO(load_aligned, load_unaligned, store_unaligned);
        }

        #undef XSIMD_LOOP_MACRO
    }
}