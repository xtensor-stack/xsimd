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

#include <type_traits>
#include <vector>

#include "doctest/doctest.h"

#include "xsimd/memory/xsimd_aligned_allocator.hpp"
#include "xsimd/memory/xsimd_alignment.hpp"

struct mock_container
{
};

TEST_CASE("[alignment]")
{
    using u_vector_type = std::vector<double>;
    using a_vector_type = std::vector<double, xsimd::default_allocator<double>>;

    using u_vector_align = xsimd::container_alignment_t<u_vector_type>;
    using a_vector_align = xsimd::container_alignment_t<a_vector_type>;
    using mock_align = xsimd::container_alignment_t<mock_container>;

    if (xsimd::default_arch::requires_alignment())
    {
        CHECK_UNARY((std::is_same<u_vector_align, xsimd::unaligned_mode>::value));
        CHECK_UNARY((std::is_same<a_vector_align, xsimd::aligned_mode>::value));
        CHECK_UNARY((std::is_same<mock_align, xsimd::unaligned_mode>::value));
    }
}

TEST_CASE("[is_aligned]")
{
    float f[100];
    void* unaligned_f = static_cast<void*>(&f[0]);
    constexpr std::size_t alignment = xsimd::default_arch::alignment();
    std::size_t aligned_f_size = sizeof(f);
    void* aligned_f = std::align(alignment, sizeof(f), unaligned_f, aligned_f_size);
    CHECK_UNARY(xsimd::is_aligned(aligned_f));

    // GCC does not generate correct alignment on ARM
    // (see https://godbolt.org/z/obv1n8bWq)
#if !(XSIMD_WITH_NEON && defined(__GNUC__) && !defined(__clang__))
    alignas(alignment) char aligned[8];
    CHECK_UNARY(xsimd::is_aligned(&aligned[0]));
    CHECK_UNARY(!xsimd::is_aligned(&aligned[3]));
#endif
}
#endif
