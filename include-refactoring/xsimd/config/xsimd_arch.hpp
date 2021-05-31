#ifndef XSIMD_ARCH_HPP
#define XSIMD_ARCH_HPP

#include "./xsimd_config.hpp"

#ifdef XSIMD_WITH_SSE
#include "../types/xsimd_sse_register.hpp"
#endif

#ifdef XSIMD_WITH_SSE2
#include "../types/xsimd_sse2_register.hpp"
#endif

#ifdef XSIMD_WITH_SSE3
#include "../types/xsimd_sse3_register.hpp"
#endif

#ifdef XSIMD_WITH_SSE4_1
#include "../types/xsimd_sse4_1_register.hpp"
#endif

#ifdef XSIMD_WITH_SSE4_2
#include "../types/xsimd_sse4_2_register.hpp"
#endif

#include <cassert>

namespace xsimd {

#if defined(XSIMD_WITH_SSE4_2)
using default_arch = sse4_2;
#elif defined(XSIMD_WITH_SSE4_1)
using default_arch = sse4_1;
#elif defined(XSIMD_WITH_SSE3)
using default_arch = sse3;
#elif defined(XSIMD_WITH_SSE2)
using default_arch = sse2;
#elif defined(XSIMD_WITH_SSE)
using default_arch = sse;
#else
#error no supported isa
#endif

}

#endif

