############################################################################
# Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         #
# Martin Renou                                                             #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.1)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(xsimd-examples)

    find_package(xsimd REQUIRED CONFIG)
    set(XSIMD_INCLUDE_DIR ${xsimd_INCLUDE_DIR})
endif ()

include_directories(${XSIMD_INCLUDE_DIR})

message(STATUS "Forcing examples build type to Release")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

set(CMAKE_CXX_FLAGS "-O3 -march=native -mtune=native -fopenmp -std=c++14")

add_executable(mandelbrot EXCLUDE_FROM_ALL mandelbrot.cpp ${XSIMD_HEADERS})

add_custom_target(xmandelbrot COMMAND mandelbrot DEPENDS mandelbrot)
