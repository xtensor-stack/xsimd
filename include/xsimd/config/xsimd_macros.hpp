/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_MACROS_HPP
#define XSIMD_MACROS_HPP

#include "./xsimd_config.hpp"

#if defined(__VEC__)
#define XSIMD_INLINE inline
#elif defined __has_attribute
#if __has_attribute(always_inline)
#define XSIMD_INLINE inline __attribute__((always_inline))
#else
#define XSIMD_INLINE inline
#endif
#elif defined(_MSC_VER)
#define XSIMD_INLINE inline __forceinline
#else
#define XSIMD_INLINE inline
#endif

#define XSIMD_CONCAT_INNER(a, b) a##b
#define XSIMD_CONCAT(a, b) XSIMD_CONCAT_INNER(a, b)

#if defined(__FAST_MATH__)
#define XSIMD_NO_DENORMALS
#define XSIMD_NO_INFINITIES
#define XSIMD_NO_NANS
#endif

#if defined(__has_cpp_attribute)
// if this check passes, then the compiler supports feature test macros
#if __has_cpp_attribute(nodiscard) >= 201603L
// if this check passes, then the compiler supports [[nodiscard]] without a message
#define XSIMD_NO_DISCARD [[nodiscard]]
#endif
#endif

#if !defined(XSIMD_NO_DISCARD) && XSIMD_CPP_VERSION >= 201703L
// this means that the previous tests failed, but we are using C++17 or higher
#define XSIMD_NO_DISCARD [[nodiscard]]
#endif

#if !defined(XSIMD_NO_DISCARD) && (defined(__GNUC__) || defined(__clang__))
// this means that the previous checks failed, but we are using GCC or Clang
#define XSIMD_NO_DISCARD __attribute__((warn_unused_result))
#endif

#if !defined(XSIMD_NO_DISCARD)
// this means that all the previous checks failed, so we fallback to doing nothing
#define XSIMD_NO_DISCARD
#endif

#ifdef __cpp_if_constexpr
// this means that the compiler supports the `if constexpr` construct
#define XSIMD_IF_CONSTEXPR if constexpr
#endif

#if !defined(XSIMD_IF_CONSTEXPR) && XSIMD_CPP_VERSION >= 201703L
// this means that the previous test failed, but we are using C++17 or higher
#define XSIMD_IF_CONSTEXPR if constexpr
#endif

#if !defined(XSIMD_IF_CONSTEXPR)
// this means that all the previous checks failed, so we fallback to a normal `if`
#define XSIMD_IF_CONSTEXPR if
#endif

#endif
