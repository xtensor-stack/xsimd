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

// GNU world

// According to http://www.gnu.org/s/libc/manual/html_node/Aligned-Memory-Blocks.html,
// "The address of a block returned by malloc or realloc in GNU systems is always a multiple of eight
// (or sixteen on 64-bit systems)"
// According to this document, http://gcc.fyxm.net/summit/2003/Porting%20to%2064%20bit.pdf
// page 114, "[The] LP64 model [...] is used by all 64-bit UNIX ports"
// Therefore, we use this predefined macro instead of __x86_64__ (this last one won't work on
// PowerPC or SPARC)
#if defined(__GLIBC__) && ((__GLIBC__>=2 && __GLIBC_MINOR__ >= 8) || __GLIBC__>2) \
 && defined(__LP64__)
  #define XSIMD_GLIBC_MALLOC_ALREADY_16ALIGNED 1
#else
  #define XSIMD_GLIBC_MALLOC_ALREADY_16ALIGNED 0
#endif

// FreeBSD world

// FreeBSD 6 seems to have 16-byte aligned malloc
//   See http://svn.freebsd.org/viewvc/base/stable/6/lib/libc/stdlib/malloc.c?view=markup
// FreeBSD 7 seems to have 16-byte aligned malloc except on ARM and MIPS architectures
//   See http://svn.freebsd.org/viewvc/base/stable/7/lib/libc/stdlib/malloc.c?view=markup
#if defined(__FreeBSD__) && !defined(__arm__) && !defined(__mips__)
  #define XSIMD_FREEBSD_MALLOC_ALREADY_16ALIGNED 1
#else
  #define XSIMD_FREEBSD_MALLOC_ALREADY_16ALIGNED 0
#endif

#if (defined(__APPLE__) \
 || defined(_WIN64) \
 || XSIMD_GLIBC_MALLOC_ALREADY_16ALIGNED \
 || XSIMD_FREEBSD_MALLOC_ALREADY_16ALIGNED)
  #define XSIMD_MALLOC_ALREADY_16ALIGNED 1
#else
  #define XSIMD_MALLOC_ALREADY_16ALIGNED 0
#endif

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

#ifdef XSIMD_USE_SSE
    #define XSIMD_MALLOC_ALREADY_ALIGNED XSIMD_MALLOC_ALREADY_16ALIGNED
#else
    #define XSIMD_MALLOC_ALREADY_ALIGNED 0
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

// TODO : remove this, use alignas specifier
#if (defined __GNUC__)
    #define XSIMD_STACK_ALIGN(N) __attribute__((aligned(N)))
#elif (defined _MSC_VER)
    #define XSIMD_STACK_ALIGN(N) __declspec(align(N))
#else
    #error Equivalent of __attribute__((aligned(N))) unknown
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

