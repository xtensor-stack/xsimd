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

if(WIN32)
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
        set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
        if(MSVC_VERSION GREATER_EQUAL MSVC14)
            add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
            #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_SCL_SECURE_NO_WARNINGS /D\"_SECURE_SCL=0\" /D\"_HAS_ITERATOR_DEBUGGING=0\"")
        endif()
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES Clang AND WIN32) # We are using clang-cl
        add_compile_options(/EHsc /MP /bigobj /arch:AVX2)
        set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
    endif()
else()
    set(CMAKE_CXX_FLAGS "-O3 -march=native -mtune=native -fopenmp -std=c++14")
endif()

add_executable(mandelbrot EXCLUDE_FROM_ALL mandelbrot.cpp ${XSIMD_HEADERS})

add_custom_target(xmandelbrot COMMAND mandelbrot DEPENDS mandelbrot)
