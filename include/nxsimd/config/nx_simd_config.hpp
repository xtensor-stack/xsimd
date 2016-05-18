#ifndef NX_SIMD_CONFIG_HPP
#define NX_SIMD_CONFIG_HPP

#include "nx_platform_config.hpp"

#ifdef NX_USE_AVX
    #define NX_DEFAULT_ALIGNMENT 32
#else
    #define NX_DEFAULT_ALIGNMENT 16
#endif

#ifndef NX_DEFAULT_ALLOCATOR
    #ifdef NX_USE_SSE_OR_AVX
        #define NX_DEFAULT_ALLOCATOR(T) nxsimd::aligned_allocator<T, NX_DEFAULT_ALIGNMENT>
    #else
        #define NX_DEFAULT_ALLOCATOR(T) std::allocator<T>
    #endif
#endif

#ifndef NX_STACK_ALLOCATION_LIMIT
    #define NX_STACK_ALLOCATION_LIMIT 20000
#endif

#endif

