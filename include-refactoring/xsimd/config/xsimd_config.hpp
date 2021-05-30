#ifndef XSIMD_CONFIG_HPP
#define XSIMD_CONFIG_HPP

#ifdef __SSE__
#define XSIMD_WITH_SSE
#endif

#ifdef __SSE2__
#define XSIMD_WITH_SSE2
#endif

#ifdef __SSE3__
#define XSIMD_WITH_SSE3
#endif

#ifdef __SSE4_1__
#define XSIMD_WITH_SSE4_1
#endif

#ifdef __SSE4_2__
#define XSIMD_WITH_SSE4_2
#endif

#endif
