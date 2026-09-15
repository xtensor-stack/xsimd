############################################################################
# Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         #
# Martin Renou                                                             #
# Copyright (c) QuantStack                                                 #
# Copyright (c) Serge Guelton                                              #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.13)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(xsimd-test)

    enable_testing()

    find_package(xsimd REQUIRED CONFIG)
    include("${xsimd_DIR}/Architecture.cmake")
endif ()

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting tests build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
else()
    message(STATUS "Tests build type is ${CMAKE_BUILD_TYPE}")
endif()

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

OPTION(XSIMD_ENABLE_WERROR "Turn on -Werror" OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    if (NOT WIN32 AND NOT ANDROID)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-parameter -Wextra -Wreorder")
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
          set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshorten-64-to-32")
        endif()

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC")
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4267 /wd4005 /wd4146 /wd4800")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /MANIFEST:NO")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES Clang AND MSVC AND WIN32) # We are using clang-cl
    add_compile_options(/EHsc /bigobj)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /MANIFEST:NO")
endif()

set(XSIMD_TESTS
    main.cpp
    test_api.cpp
    test_arch.cpp
    test_basic_math.cpp
    test_batch.cpp
    test_batch_bool.cpp
    test_batch_cast.cpp
    test_batch_complex.cpp
    test_batch_float.cpp
    test_batch_int.cpp
    test_bit.cpp
    test_bitwise_cast.cpp
    test_batch_constant.cpp
    test_batch_manip.cpp
    test_complex_exponential.cpp
    test_complex_hyperbolic.cpp
    test_complex_power.cpp
    test_complex_trigonometric.cpp
    test_conversion.cpp
    test_cpu_features.cpp
    test_custom_default_arch.cpp
    test_error_gamma.cpp
    test_explicit_batch_instantiation.cpp
    test_exponential.cpp
    test_extract_pair.cpp
    test_fp_manipulation.cpp
    test_hyperbolic.cpp
    test_load_store.cpp
    test_memory.cpp
    test_poly_evaluation.cpp
    test_power.cpp
    test_rounding.cpp
    test_select.cpp
    test_shuffle.cpp
    test_sum.cpp
    test_traits.cpp
    test_trigonometric.cpp
    test_utils_bits.cpp
    test_xsimd_api.cpp
    test_utils.hpp
)

if(NOT MSVC)
    list(APPEND XSIMD_TESTS test_gnu_source.cpp)
endif()

set(TARGET_ARCH "native" CACHE STRING "Target architecture arguments")

add_executable(test_xsimd ${XSIMD_TESTS})
target_link_libraries(test_xsimd PRIVATE xsimd)
# Enable SIMD flag and set low baseline to ensure we do not accidentally
# rely on other features.
if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(XSIMD_ARM64_BASELINE "armv8.0")
else()
    set(XSIMD_ARM64_BASELINE "armv8-a")
endif()
xsimd_target_set_arch(
    test_xsimd
    PRIVATE
    ARCH "${TARGET_ARCH}"
    ARM32_BASELINE "armv7-a"
    ARM64_BASELINE "${XSIMD_ARM64_BASELINE}"
    LOONGARCH64_BASELINE "loongarch64"
    PPC64_BASELINE "power8"
    RISCV64_BASELINE "rv64gc"
    S390X_BASELINE "z14"
    X86_32_BASELINE "i386"
    X86_64_BASELINE "x86-64"
)

option(DOWNLOAD_DOCTEST OFF)
find_package(doctest QUIET)
if (doctest_FOUND)
    set(DOCTEST_MINIMAL_VERSION 2.4.9)
    if (doctest_VERSION VERSION_LESS DOCTEST_MINIMAL_VERSION)
        message(FATAL_ERROR "Requires doctest >= ${DOCTEST_MINIMAL_VERSION}")
    endif()
    target_link_libraries(test_xsimd PRIVATE doctest::doctest)
elseif(DOWNLOAD_DOCTEST)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doctest")
    file(DOWNLOAD
        "https://github.com/doctest/doctest/releases/download/v2.4.9/doctest.h"
        "${CMAKE_CURRENT_BINARY_DIR}/doctest/doctest.h"
        STATUS DOWNLOAD_DOCTEST_STATUS)
    list(GET DOWNLOAD_DOCTEST_STATUS 0 DOWNLOAD_DOCTEST_STATUS_CODE)
    list(GET DOWNLOAD_DOCTEST_STATUS 1 DOWNLOAD_DOCTEST_ERROR_MESSAGE)
    if(${DOWNLOAD_DOCTEST_STATUS_CODE} EQUAL 0)
        message(STATUS "Successfully downloaded doctest.h")
    else()
        message(FATAL_ERROR "Error occurred during download of doctest: ${DOWNLOAD_DOCTEST_ERROR_MESSAGE}")
    endif()
    target_include_directories(test_xsimd PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
else()
    message(FATAL_ERROR "
    Cannot find required doctest component.
    Please either set CMAKE_PREFIX_PATH to the location of doctestConfig.cmake,
    or set DOWNLOAD_DOCTEST=ON")
endif()

if(ENABLE_XTL_COMPLEX)
  add_compile_definitions(XSIMD_ENABLE_XTL_COMPLEX=1)
  target_include_directories(test_xsimd PRIVATE ${xtl_INCLUDE_DIRS})
endif()

add_test(NAME test_xsimd COMMAND test_xsimd)

add_custom_target(xtest COMMAND test_xsimd DEPENDS test_xsimd)

if (XSIMD_ENABLE_WERROR)
    target_compile_options(test_xsimd PRIVATE -Werror -Wall -DXSIMD_SKIP_ON_WERROR)
endif()

add_subdirectory(doc)
OPTION(BUILD_ARCHITECTURE_TESTS "xsimd compilation tests for various -march targets" ON)
if(BUILD_ARCHITECTURE_TESTS)
    add_subdirectory(architectures)
endif()

if(EMULATED_COMPILE_FLAGS)
    message(STATUS ${EMULATED_COMPILE_FLAGS})
    target_compile_options(test_xsimd PRIVATE ${EMULATED_COMPILE_FLAGS})
endif()

if(EMSCRIPTEN)
    set_target_properties(test_xsimd PROPERTIES LINK_FLAGS "-s MODULARIZE=1 -s EXPORT_NAME=test_xsimd_wasm -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -lembind")
    target_compile_options(test_xsimd
        PUBLIC --std=c++17
        PUBLIC "SHELL: -msimd128"
        PUBLIC "SHELL: -msse2"
    )
endif()
