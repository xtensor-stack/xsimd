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

project(xsimd)

# Versioning
# ==========

file(STRINGS "include/xsimd/config/xsimd_config.hpp" xsimd_version_defines
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

add_library(xsimd INTERFACE)
add_library(xsimd::xsimd ALIAS xsimd)

target_include_directories(xsimd INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

OPTION(ENABLE_XTL_COMPLEX "enables support for xcomplex defined in xtl" OFF)
OPTION(BUILD_TESTS "xsimd test suite" OFF)

target_compile_features(xsimd INTERFACE cxx_std_14)
if(ENABLE_XTL_COMPLEX)
    find_package(xtl 0.8.0 REQUIRED)
    target_compile_definitions(xsimd INTERFACE XSIMD_ENABLE_XTL_COMPLEX=1)
    target_link_libraries(xsimd INTERFACE xtl)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

OPTION(BUILD_BENCHMARK "xsimd benchmarks" OFF)
if(BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()

OPTION(BUILD_EXAMPLES "xsimd examples" OFF)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
# ============

OPTION(XSIMD_SKIP_INSTALL "Skip installation or not. By default it is OFF" OFF)
if(${XSIMD_SKIP_INSTALL})
    return() # skip installation
endif ()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS xsimd
        EXPORT ${PROJECT_NAME}-targets)

# Makes the project importable from the build directory
export(EXPORT ${PROJECT_NAME}-targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

install(DIRECTORY include/xsimd
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# GNUInstallDirs "DATADIR" wrong here; CMake search path wants "share".
set(XSIMD_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}" CACHE STRING "install path for xsimdConfig.cmake")

configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION ${XSIMD_CMAKECONFIG_INSTALL_DIR})

# xsimd is header-only and does not depend on the architecture.
# Remove CMAKE_SIZEOF_VOID_P from xtensorConfigVersion.cmake so that an xtensorConfig.cmake
# generated for a 64 bit target can be used for 32 bit targets and vice versa.
set(_XTENSOR_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${XSIMD_CMAKECONFIG_INSTALL_DIR})
install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION ${XSIMD_CMAKECONFIG_INSTALL_DIR})

configure_file(${PROJECT_NAME}.pc.in
               "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
                @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/pkgconfig/")
