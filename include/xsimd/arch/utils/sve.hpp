/****************************************************************************
 * Copyright (c) xsimd contributors                                         *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_ARCH_UTILS_SVE_HPP
#define XSIMD_ARCH_UTILS_SVE_HPP

#include "../../config/xsimd_macros.hpp"
#include "../../types/xsimd_sve_register.hpp"

#include <type_traits>

// Define a inline namespace with the explicit SVE vector size to avoid ODR violation
// When dynamically dispatching between different SVE sizes.
// While most code is safe from ODR violation as the size is already encoded in the
// register (and hence batch) types, utilities can quickly fall prone to this issue.
#define XSIMD_SVE_NAMESPACE XSIMD_CONCAT(sve, XSIMD_SVE_BITS)

namespace xsimd::kernel::detail
{
    inline namespace XSIMD_SVE_NAMESPACE
    {
        template <class T>
        XSIMD_INLINE auto svptrue() noexcept
        {
#if XSIMD_WITH_SVE
            if constexpr (sizeof(T) == 1)
            {
                return svptrue_b8();
            }
            else if constexpr (sizeof(T) == 2)
            {
                return svptrue_b16();
            }
            else if constexpr (sizeof(T) == 4)
            {
                return svptrue_b32();
            }
            else if constexpr (sizeof(T) == 8)
            {
                return svptrue_b64();
            }
#endif
        }
    }
}
#endif
