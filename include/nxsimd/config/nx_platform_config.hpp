//
// Copyright (c) 2016 Johan Mabille
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
//

#ifndef NX_PLATFORM_CONFIG_HPP
#define NX_PLATFORM_CONFIG_HPP

// ************************************************************************* //
// ***********                 SSE instruction set               *********** //
// ************************************************************************* //

#if (defined(_M_AMD64) || defined(_M_X64) || defined(__amd64)) && ! defined(__x86_64__)
    #define __x86_64__ 1
#endif

// Find sse instruction set from compiler macros if SSE_INSTR_SET not defined
// Note: Not all compilers define these macros automatically
#ifndef SSE_INSTR_SET
    #if defined ( __AVX2__ )
        #define SSE_INSTR_SET 8
    #elif defined ( __AVX__ )
        #define SSE_INSTR_SET 7
    #elif defined ( __SSE4_2__ )
        #define SSE_INSTR_SET 6
    #elif defined ( __SSE4_1__ )
        #define SSE_INSTR_SET 5
    #elif defined ( __SSSE3__ )
        #define SSE_INSTR_SET 4
    #elif defined ( __SSE3__ )
        #define SSE_INSTR_SET 3
    #elif defined ( __SSE2__ ) || defined ( __x86_64__ )
        #define SSE_INSTR_SET 2
    #elif defined ( __SSE__ )
        #define SSE_INSTR_SET 1
    #elif defined ( _M_IX86_FP )  // Defined in MS compiler on 32bits system. 1: SSE, 2: SSE2
        #define SSE_INSTR_SET _M_IX86_FP
    #else
        #define SSE_INSTR_SET 0
    #endif // instruction set defines
#endif // SSE_INSTR_SET


// ************************************************************************* //
// *******        Platform checks for aligned malloc functions       ******* //
// ************************************************************************* //

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
  #define NX_GLIBC_MALLOC_ALREADY_16ALIGNED 1
#else
  #define NX_GLIBC_MALLOC_ALREADY_16ALIGNED 0
#endif

// FreeBSD world

// FreeBSD 6 seems to have 16-byte aligned malloc
//   See http://svn.freebsd.org/viewvc/base/stable/6/lib/libc/stdlib/malloc.c?view=markup
// FreeBSD 7 seems to have 16-byte aligned malloc except on ARM and MIPS architectures
//   See http://svn.freebsd.org/viewvc/base/stable/7/lib/libc/stdlib/malloc.c?view=markup
#if defined(__FreeBSD__) && !defined(__arm__) && !defined(__mips__)
  #define NX_FREEBSD_MALLOC_ALREADY_16ALIGNED 1
#else
  #define NX_FREEBSD_MALLOC_ALREADY_16ALIGNED 0
#endif

#if (defined(__APPLE__) \
 || defined(_WIN64) \
 || NX_GLIBC_MALLOC_ALREADY_16ALIGNED \
 || NX_FREEBSD_MALLOC_ALREADY_16ALIGNED)
  #define NX_MALLOC_ALREADY_16ALIGNED 1
#else
  #define NX_MALLOC_ALREADY_16ALIGNED 0
#endif

#if ((defined __QNXNTO__) || (defined _GNU_SOURCE) || ((defined _XOPEN_SOURCE) && (_XOPEN_SOURCE >= 600))) \
 && (defined _POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO > 0)
  #define NX_HAS_POSIX_MEMALIGN 1
#else
  #define NX_HAS_POSIX_MEMALIGN 0
#endif

#if SSE_INSTR_SET > 0
    #define NX_HAS_MM_MALLOC 1
#else
    #define NX_HAS_MM_MALLOC 0
#endif

#if ((SSE_INSTR_SET > 6) && !defined(FORBID_AVX))
    #define NX_USE_AVX
#elif ((SSE_INSTR_SET > 0) && !defined(FORBID_SSE))
    #define NX_USE_SSE
#endif

#ifdef NX_USE_SSE
    #define NX_MALLOC_ALREADY_ALIGNED NX_MALLOC_ALREADY_16ALIGNED
#else
    #define NX_MALLOC_ALREADY_ALIGNED 0
#endif

#ifndef NX_ALLOCA
    #if defined(__linux__)
        #define NX_ALLOCA alloca
    #elif defined(_MSC_VER)
        #define NX_ALLOCA _alloca
    #endif
#endif

#ifndef NX_STACK_ALLOCATION_LIMIT
    #define NX_STACK_ALLOCATION_LIMIT 20000
#endif

#endif

