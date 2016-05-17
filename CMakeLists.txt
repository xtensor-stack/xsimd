cmake_minimum_required(VERSION 3.1)
project(nxsimd)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CPP14_FLAG)
    CHECK_CXX_COMPILER_FLAG("-std=c++11" HAS_CPP11_FLAG)

    if (HAS_CPP14_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    elseif (HAS_CPP11_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    else()
        message(FATAL_ERROR "Unsupported compiler -- pybind11 requires C++11 support!")
    endif()

    # Enable link time optimization and set the default symbol
    # visibility to hidden (very important to obtain small binaries)
    if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
        # Default symbol visibility
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

        # Check for Link Time Optimization support
        # (GCC/Clang)
        CHECK_CXX_COMPILER_FLAG("-flto" HAS_LTO_FLAG)
        if (HAS_LTO_FLAG)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
        endif()

        # Intel equivalent to LTO is called IPO
        if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            CHECK_CXX_COMPILER_FLAG("-ipo" HAS_IPO_FLAG)
            if (HAS_IPO_FLAG)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ipo")
            endif()
        endif()
    endif()
endif()

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

