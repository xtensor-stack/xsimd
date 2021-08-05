#ifndef XSIMD_CONFIG_HPP
#define XSIMD_CONFIG_HPP

#define XSIMD_VERSION_MAJOR 8
#define XSIMD_VERSION_MINOR 0
#define XSIMD_VERSION_PATCH 0

/**
 * high level free functions
 *
 * @defgroup xsimd_config_macro Instruction Set Detection
 */

/**
 * @ingroup xsimd_config_macro
 *
 * Set to 1 if SSE2 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSE2__
#define XSIMD_WITH_SSE2 1
#else
#define XSIMD_WITH_SSE2 0
#endif

/**
 * @ingroup xsimd_config_macro
 *
 * Set to 1 if SSE3 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSE3__
#define XSIMD_WITH_SSE3 1
#else
#define XSIMD_WITH_SSE3 0
#endif

/**
 * @ingroup xsimd_config_macro
 *
 * Set to 1 if SSSE3 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSSE3__
#define XSIMD_WITH_SSSE3 1
#else
#define XSIMD_WITH_SSSE3 0
#endif

/**
 * @ingroup xsimd_config_macro
 *
 * Set to 1 if SSE4.1 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSE4_1__
#define XSIMD_WITH_SSE4_1 1
#else
#define XSIMD_WITH_SSE4_1 0
#endif

/**
 * @ingroup xsimd_config_macro
 *
 * Set to 1 if SSE4.2 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSE4_2__
#define XSIMD_WITH_SSE4_2 1
#else
#define XSIMD_WITH_SSE4_2 0
#endif

/**
 * @ingroup xsimd_config_macro
 *
 * Set to 1 if AVX is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX__
#define XSIMD_WITH_AVX 1
#else
#define XSIMD_WITH_AVX 0
#endif

/**
 * @ingroup xsimd_config_macro
 *
 * Set to 1 if AVX2 is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX2__
#define XSIMD_WITH_AVX2 1
#else
#define XSIMD_WITH_AVX2 0
#endif

/**
 * @ingroup xsimd_config_macro
 *
 * Set to 1 if FMA  for SSE is available at compile-time, to 0 otherwise.
 */
#ifdef __FMA__

#if defined(__SSE__) && ! defined(__AVX__)
#define XSIMD_WITH_FMA3 1
#else
#define XSIMD_WITH_FMA3 0
#endif

#else
#define XSIMD_WITH_FMA3 0
#endif

/**
 * @ingroup xsimd_config_macro
 *
 * Set to 1 if FMA for AVX is available at compile-time, to 0 otherwise.
 */
#ifdef __FMA__

#if defined(__AVX__)
#define XSIMD_WITH_FMA5 1
#else
#define XSIMD_WITH_FMA5 0
#endif

#else
#define XSIMD_WITH_FMA5 0
#endif

/**
 * @ingroup xsimd_config_macro
 *
 * Set to 1 if AVX512F is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512F__
    // AVX512 instructions are supported starting with gcc 6
    // see https://www.gnu.org/software/gcc/gcc-6/changes.html
    #if defined(__GNUC__) && __GNUC__ < 6
        #define XSIMD_WITH_AVX512F 0
    #else
        #define XSIMD_WITH_AVX512F 1
        #if __GNUC__ == 6
            #define XSIMD_AVX512_SHIFT_INTRINSICS_IMM_ONLY 1
        #endif
    #endif
#else
#define XSIMD_WITH_AVX512F 0
#endif

/**
 * @ingroup xsimd_config_macro
 *
 * Set to 1 if AVX512CD is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512CD__
// Avoids repeating the GCC workaround over and over
#define XSIMD_WITH_AVX512CD XSIMD_WITH_AVX512F
#else
#define XSIMD_WITH_AVX512CD 0
#endif

/**
 * @ingroup xsimd_config_macro
 *
 * Set to 1 if AVX512DQ is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512DQ__
#define XSIMD_WITH_AVX512DQ XSIMD_WITH_AVX512F
#else
#define XSIMD_WITH_AVX512DQ 0
#endif

/**
 * @ingroup xsimd_config_macro
 *
 * Set to 1 if AVX512BW is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512BW__
#define XSIMD_WITH_AVX512BW XSIMD_WITH_AVX512F
#else
#define XSIMD_WITH_AVX512BW 0
#endif

#ifdef __ARM_NEON

/**
 * @ingroup xsimd_config_macro
 *
 * Set to 1 if NEON is available at compile-time, to 0 otherwise.
 */
    #if __ARM_ARCH >= 7
        #define XSIMD_WITH_NEON 1
    #else
        #define XSIMD_WITH_NEON 0
    #endif

/**
 * @ingroup xsimd_config_macro
 *
 * Set to 1 if NEON64 is available at compile-time, to 0 otherwise.
 */
    #ifdef __aarch64__
        #define XSIMD_WITH_NEON64 1
    #else
        #define XSIMD_WITH_NEON64 0
    #endif
#else
    #define XSIMD_WITH_NEON 0
    #define XSIMD_WITH_NEON64 0
#endif

// Workaround for MSVC compiler
#ifdef _MSC_VER

#if XSIMD_WITH_AVX512
#define XSIMD_WITH_AVX2 1
#endif

#if XSIMD_WITH_AVX2
#define XSIMD_WITH_AVX 1
#endif

#if XSIMD_WITH_AVX
#define XSIMD_WITH_SSE4_2 1
#endif

#if XSIMD_WITH_SSE4_2
#define XSIMD_WITH_SSE4_1 1
#endif

#if XSIMD_WITH_SSE4_1
#define XSIMD_WITH_SSSE3 1
#endif

#if XSIMD_WITH_SSSE3
#define XSIMD_WITH_SSE3 1
#endif

#if XSIMD_WITH_SSE3
#define XSIMD_WITH_SSE2 1
#endif

#endif

#endif
