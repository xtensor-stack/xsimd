//
// Copyright (c) 2016 Johan Mabille
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
//

#ifndef NX_SIMD_CONFIG_HPP
#define NX_SIMD_CONFIG_HPP

#include "nx_platform_config.hpp"

namespace nxsimd
{

// Include the appropriate header file for intrinsic functions
#if SSE_INSTR_SET > 7                  // AVX2 and later
    #ifdef __GNUC__
        #include <x86intrin.h>         // x86intrin.h includes header files for whatever instruction
                                       // sets are specified on the compiler command line, such as:
                                       // xopintrin.h, fma4intrin.h
    #else
        #include <immintrin.h>         // MS version of immintrin.h covers AVX, AVX2 and FMA3
    #endif // __GNUC__
#elif SSE_INSTR_SET == 7
    #include <immintrin.h>             // AVX
#elif SSE_INSTR_SET == 6
    #include <nmmintrin.h>             // SSE4.2
#elif SSE_INSTR_SET == 5
    #include <smmintrin.h>             // SSE4.1
#elif SSE_INSTR_SET == 4
    #include <tmmintrin.h>             // SSSE3
#elif SSE_INSTR_SET == 3
    #include <pmmintrin.h>             // SSE3
#elif SSE_INSTR_SET == 2
    #include <emmintrin.h>             // SSE2
#elif SSE_INSTR_SET == 1
    #include <xmmintrin.h>             // SSE
#endif // SSE_INSTR_SET

// AMD  instruction sets
#if defined (__XOP__) || defined (__FMA4__)
    #ifdef __GNUC__
        #include <x86intrin.h>         // AMD XOP (Gnu)
    #else
        #include <ammintrin.h>         // AMD XOP (Microsoft)
    #endif //  __GNUC__
#elif defined (__SSE4A__)              // AMD SSE4A
    #include <ammintrin.h>
#endif // __XOP__

// FMA3 instruction set
#if defined (__FMA__)
#include <fmaintrin.h>
#endif // __FMA__

// FMA4 instruction set
#if defined (__FMA4__)
    #include <fma4intrin.h>
#endif // __FMA4__


// TODO: add ALTIVEC instruction setET > 7
}

#endif
