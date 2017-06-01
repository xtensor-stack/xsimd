############################################################################
# Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.1)
project(xsimd)

set(XSIMD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Versionning
# ===========

file(STRINGS "${XSIMD_INCLUDE_DIR}/xsimd/config/xsimd_config.hpp" xsimd_version_defines
     REGEX "#define XSIMD_VERSION_(MAJOR|MINOR|PATCH)")
foreach(ver ${xsimd_version_defines})
    if(ver MATCHES "#define XSIMD_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(XSIMD_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(${PROJECT_NAME}_VERSION 
    ${XSIMD_VERSION_MAJOR}.${XSIMD_VERSION_MINOR}.${XSIMD_VERSION_PATCH})
message(STATUS "xsimd v${${PROJECT_NAME}_VERSION}")


# Build
# =====

OPTION(BUILD_TESTS "xsimd test suite" ON)

set(XSIMD_HEADERS
    ${XSIMD_INCLUDE_DIR}/xsimd/xsimd.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/config/xsimd_align.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/config/xsimd_config.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/config/xsimd_include.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/config/xsimd_instruction_set.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_exp_reduction.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_exponential.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_fp_manipulation.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_fp_sign.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_horner.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_numerical_constant.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_rounding.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/memory/xsimd_aligned_allocator.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/memory/xsimd_aligned_stack_buffer.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx_conversion.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx_double.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx_float.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx_int32.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx_int64.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_sse_conversion.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_sse_double.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_sse_float.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_sse_int32.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_sse_int64.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_base.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_traits.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_types_include.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_utils.hpp
)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()

# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(DIRECTORY ${XSIMD_INCLUDE_DIR}/xsimd
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# GNUInstallDirs "DATADIR" wrong here; CMake search path wants "share".
set(XSIMD_CMAKECONFIG_INSTALL_DIR "share/cmake/${PROJECT_NAME}" CACHE STRING "install path for xsimdConfig.cmake")

configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION ${XSIMD_CMAKECONFIG_INSTALL_DIR})
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${XSIMD_CMAKECONFIG_INSTALL_DIR})

