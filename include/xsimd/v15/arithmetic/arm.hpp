/****************************************************************************
 * Copyright (c) xsimd contributors                                         *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_ARITHMETIC_ARM_HPP
#define XSIMD_ARITHMETIC_ARM_HPP

#include "../../arch/utils/sve.hpp"
#include "../../config/xsimd_macros.hpp"
#include "../../types/xsimd_arm_registers.hpp"
#include "../../types/xsimd_batch.hpp"
#include "../../utils/xsimd_type_traits.hpp"
#include "../kernel_fwd.hpp"

#include <type_traits>

namespace xsimd::kernel
{
    namespace detail
    {
        template <class T, class A>
        XSIMD_INLINE batch<T, A> neon_vaddq(batch<T, A> a, batch<T, A> b) noexcept
        {
            using TN = map_to_sized_type_t<T>;
            if constexpr (std::is_same_v<TN, uint8_t>)
            {
                return { vaddq_u8(a.to_native(), b.to_native()) };
            }
            else if constexpr (std::is_same_v<TN, int8_t>)
            {
                return { vaddq_s8(a.to_native(), b.to_native()) };
            }
            else if constexpr (std::is_same_v<TN, uint16_t>)
            {
                return { vaddq_u16(a.to_native(), b.to_native()) };
            }
            else if constexpr (std::is_same_v<TN, int16_t>)
            {
                return { vaddq_s16(a.to_native(), b.to_native()) };
            }
            else if constexpr (std::is_same_v<TN, uint32_t>)
            {
                return { vaddq_u32(a.to_native(), b.to_native()) };
            }
            else if constexpr (std::is_same_v<TN, int32_t>)
            {
                return { vaddq_s32(a.to_native(), b.to_native()) };
            }
            else if constexpr (std::is_same_v<TN, uint64_t>)
            {
                return { vaddq_u64(a.to_native(), b.to_native()) };
            }
            else if constexpr (std::is_same_v<TN, int64_t>)
            {
                return { vaddq_s64(a.to_native(), b.to_native()) };
            }
            else if constexpr (std::is_same_v<TN, float>)
            {
                return { vaddq_f32(a.to_native(), b.to_native()) };
            }
            else if constexpr (std::is_same_v<TN, double> && !std::is_same_v<A, neon>)
            {
                // Unavailable in neon neon arm v7
                return { vaddq_f64(a.to_native(), b.to_native()) };
            }
            else
            {
                static_assert(false, "unsupported data type");
            }
        }
    }

    template <class T, class A>
    XSIMD_INLINE batch<T, A> add(batch<T, A> lhs, batch<T, A> rhs) noexcept
    {
        if constexpr (is_sve_v<A>)
        {
            return svadd_x(detail::svptrue<T>(), lhs, rhs);
        }
        else if constexpr (std::is_same_v<A, neon64> || std::is_same_v<A, neon>)
        {
            return detail::neon_vaddq(lhs, rhs);
        }
    }
}

#endif
