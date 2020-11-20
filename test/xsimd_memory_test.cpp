/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <vector>
#include <type_traits>

#include "gtest/gtest.h"

#include "xsimd/config/xsimd_instruction_set.hpp"

#ifdef XSIMD_INSTR_SET_AVAILABLE

#include "xsimd/memory/xsimd_alignment.hpp"

namespace xsimd
{
    struct mock_container {};

    TEST(xsimd, alignment)
    {
        using u_vector_type = std::vector<double>;
        using a_vector_type = std::vector<double, aligned_allocator<double, XSIMD_DEFAULT_ALIGNMENT>>;

        using u_vector_align = container_alignment_t<u_vector_type>;
        using a_vector_align = container_alignment_t<a_vector_type>;
        using mock_align = container_alignment_t<mock_container>;

        EXPECT_TRUE((std::is_same<u_vector_align, unaligned_mode>::value));
        EXPECT_TRUE((std::is_same<a_vector_align, aligned_mode>::value));
        EXPECT_TRUE((std::is_same<mock_align, unaligned_mode>::value));
    }
}
#endif // XSIMD_INSTR_SET_AVAILABLE
