cmake_minimum_required(VERSION 3.1)
project(nxsimd)

include_directories(include)
set(NXSIMD_TARGET simd_test)

set(NXSIMD_HEADERS
    include/nxsimd/config/nx_platform_config.hpp
    include/nxsimd/config/nx_simd_config.hpp
    include/nxsimd/memory/nx_aligned_allocator.hpp
    include/nxsimd/memory/nx_aligned_stack_buffer.hpp
    include/nxsimd/types/nx_avx_double.hpp
    include/nxsimd/types/nx_avx_float.hpp
    include/nxsimd/types/nx_simd_base.hpp
    include/nxsimd/types/nx_simd_traits.hpp
    include/nxsimd/types/nx_sse_double.hpp
    include/nxsimd/types/nx_sse_float.hpp
    include/nxsimd/nx_simd.hpp
)

set(NXSIMD_TESTS
    test/main.cpp
)

add_executable(${NXSIMD_TARGET} ${NXSIMD_TESTS} ${NXSIMD_HEADERS})
set_property(TARGET ${NXSIMD_TARGET} PROPERTY CXX_STANDARD 11)

