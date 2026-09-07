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

#ifndef XSIMD_COMMON_BITWISE_HPP
#define XSIMD_COMMON_BITWISE_HPP

#include "./xsimd_common_details.hpp"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace xsimd
{

    namespace kernel
    {

        using namespace types;

        namespace detail
        {
            // Pattern P repeated across U from bit 0 up. The type of P is the
            // repeat unit, so repeat_pattern<uint64_t, uint8_t(0x55)>() is
            // 0x5555555555555555 and repeat_pattern<uint32_t, uint16_t(0x0f0f)>()
            // is 0x0f0f0f0f.
            template <class U, auto P>
            constexpr U repeat_pattern() noexcept
            {
                using unit = decltype(P);
                static_assert(std::is_unsigned<unit>::value, "the repeat unit must be unsigned");
                static_assert(P != 0, "the pattern must be non-zero");
                static_assert(sizeof(U) % sizeof(unit) == 0, "the repeat unit must divide U");
                U result = 0;
                for (std::size_t i = 0; i < sizeof(U) / sizeof(unit); ++i)
                    result = U(result | U(U(P) << (i * sizeof(unit) * CHAR_BIT)));
                return result;
            }
        }

        // popcount
        // SWAR fold, popcount64b from Hacker's Delight, listed in
        // https://en.wikipedia.org/wiki/Hamming_weight#Efficient_implementation
        template <class A, class T, class /*=std::enable_if_t<std::is_integral_v<T>>*/>
        XSIMD_INLINE batch<T, A> popcount(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            using U = as_unsigned_integer_t<T>;
            using w_type = batch<uint64_t, A>;
            constexpr std::size_t bits = sizeof(T) * CHAR_BIT;

            // Every step runs on 64-bit lanes whatever T is. Each step confines
            // its own carries, so a bit that crosses a T boundary is always
            // masked off again, and targets with no narrow shift do not pay for
            // one to be emulated.
            w_type x = bitwise_cast<uint64_t>(self);
            x = x - ((x >> 1) & w_type(detail::repeat_pattern<uint64_t, uint8_t(0x55)>()));
            w_type const m2(detail::repeat_pattern<uint64_t, uint8_t(0x33)>());
            x = (x & m2) + ((x >> 2) & m2);
            x = (x + (x >> 4)) & w_type(detail::repeat_pattern<uint64_t, uint8_t(0x0f)>());
            if constexpr (bits == 8)
                return bitwise_cast<T>(x);

            // Byte counts are at most 8, so a per-lane sum is at most 64 and no
            // byte carries into the next. The final mask keeps the one byte per
            // lane that holds the whole count and drops the bytes that summed
            // across a lane boundary.
            x = x + (x >> 8);
            if constexpr (bits >= 32)
                x = x + (x >> 16);
            if constexpr (bits >= 64)
                x = x + (x >> 32);
            return bitwise_cast<T>(x) & batch<T, A>(T(U(0xff)));
        }
    }
}

#endif
