/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_ALIGN_HPP
#define XSIMD_ALIGN_HPP

#include "xsimd_instruction_set.hpp"

/**************************************************
 * Platform checks for aligned malloc functions
 **************************************************/

#if ((defined __QNXNTO__) || (defined _GNU_SOURCE) || ((defined _XOPEN_SOURCE) && (_XOPEN_SOURCE >= 600))) \
 && (defined _POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO > 0)
  #define XSIMD_HAS_POSIX_MEMALIGN 1
#else
  #define XSIMD_HAS_POSIX_MEMALIGN 0
#endif

#if defined(XSIMD_X86_INSTR_SET_AVAILABLE)
    #define XSIMD_HAS_MM_MALLOC 1
#else
    #define XSIMD_HAS_MM_MALLOC 0
#endif

/************************************
 * Stack allocation and alignment
 ************************************/

#ifndef XSIMD_ALLOCA
    #if defined(__linux__)
        #define XSIMD_ALLOCA alloca
    #elif defined(_MSC_VER)
        #define XSIMD_ALLOCA _alloca
    #endif
#endif

/***********
 * headers *
 ***********/

#if defined(_MSC_VER) || defined(__MINGW64__) || defined(__MINGW32__)
    #include <malloc.h>
#elif defined(__GNUC__)
    #include <mm_malloc.h>
    #if defined(XSIMD_ALLOCA)
        #include <alloca.h>
    #endif
#else
    #include <stdlib.h>
#endif

#endif

