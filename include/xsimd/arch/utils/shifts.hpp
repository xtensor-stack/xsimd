/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 * Copyright (c) Marco Barbone                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_UTILS_SHIFTS_HPP
#define XSIMD_UTILS_SHIFTS_HPP

#include "xsimd/config/xsimd_inline.hpp"
#include "xsimd/types/xsimd_batch.hpp"

namespace xsimd
{
    namespace kernel
    {
        namespace utils
        {
            template <typename I, I offset, I length, I... Vs>
            struct select_stride
            {
                static constexpr I values_array[] = { Vs... };

                template <typename K>
                static constexpr K get(K i, K)
                {
                    return static_cast<K>(values_array[length * i + offset]);
                }
            };

            template <class T, class T2, class A, class R, T... Vs>
            XSIMD_INLINE batch<T, A> bitwise_lshift_as_twice_larger(
                batch<T, A> const& self, batch_constant<T, A, Vs...>, R req) noexcept
            {
                static_assert(sizeof(T2) == 2 * sizeof(T), "One size must be twice the other");

                const auto self2 = bitwise_cast<T2>(self);

                // Lower byte: shift as twice the size and mask bits flowing to higher byte.
                constexpr auto shifts_lo = make_batch_constant<T2, select_stride<T, 0, 2, Vs...>, A>();
                const auto shifted_lo = bitwise_lshift<A>(self2, shifts_lo, req);
                const batch<T2, A> mask_lo { T2 { 0x00FF } };
                const auto masked_lo = bitwise_and<A>(shifted_lo, mask_lo, req);

                // Higher byte: mask bits that would flow from lower byte and shift as twice the size.
                constexpr auto shifts_hi = make_batch_constant<T2, select_stride<T, 1, 2, Vs...>, A>();
                const batch<T2, A> mask_hi { T2 { 0xFF00 } };
                const auto masked_hi = bitwise_and<A>(self2, mask_hi, req);
                const auto shifted_hi = bitwise_lshift<A>(masked_hi, shifts_hi, req);

                return bitwise_cast<T>(bitwise_or(masked_lo, shifted_hi, req));
            }
        }
    }
}

#endif
