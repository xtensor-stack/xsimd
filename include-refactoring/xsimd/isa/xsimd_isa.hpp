#ifndef XSIMD_ISA_HPP
#define XSIMD_ISA_HPP

#include "../config/xsimd_arch.hpp"

#include "./xsimd_generic.hpp"

#ifdef XSIMD_WITH_SSE
#include "./xsimd_sse.hpp"
#endif

#ifdef XSIMD_WITH_SSE2
#include "./xsimd_sse2.hpp"
#endif

#ifdef XSIMD_WITH_SSE3
#include "./xsimd_sse3.hpp"
#endif

#ifdef XSIMD_WITH_SSE4_1
#include "./xsimd_sse4_1.hpp"
#endif

#ifdef XSIMD_WITH_SSE4_2
#include "./xsimd_sse4_2.hpp"
#endif

#endif

