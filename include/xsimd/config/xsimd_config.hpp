/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_CONFIG_HPP
#define XSIMD_CONFIG_HPP

#include "xsimd_platform_config.hpp"

#ifdef XUSE_AVX
    #define XDEFAULT_ALIGNMENT 32
#else
    #define XDEFAULT_ALIGNMENT 16
#endif

#ifndef XDEFAULT_ALLOCATOR
    #ifdef XUSE_SSE_OR_AVX
        #define XDEFAULT_ALLOCATOR(T) xsimd::aligned_allocator<T, XDEFAULT_ALIGNMENT>
    #else
        #define XDEFAULT_ALLOCATOR(T) std::allocator<T>
    #endif
#endif

#ifndef XSTACK_ALLOCATION_LIMIT
    #define XSTACK_ALLOCATION_LIMIT 20000
#endif

#endif

