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

file(STRINGS "${XSIMD_INCLUDE_DIR}/xsimd/xsimd_config.hpp" xsimd_version_defines
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

set(XSIMD_HEADERS
    ${XSIMD_INCLUDE_DIR}/xsimd/xsimd.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/config/xplatform_config.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/config/xsimd_include.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/config/xsimd_config.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/memory/xaligned_allocator.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/memory/xaligned_stack_buffer.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xavx_double.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xavx_float.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsse_double.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsse_float.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_base.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_traits.hpp
)

add_subdirectory(test)

# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(FILES ${XSIMD_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xsimd)

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

