#ifndef XSIMD_ARCH_HPP
#define XSIMD_ARCH_HPP

#include "./xsimd_config.hpp"

#ifdef XSIMD_WITH_SSE
#include "../types/xsimd_sse_register.hpp"
#endif

#ifdef XSIMD_WITH_SSE3
#include "../types/xsimd_sse3_register.hpp"
#endif

#include <cassert>

namespace xsimd {

#if defined(XSIMD_WITH_SSE3)
using default_arch = sse3;
#elif defined(XSIMD_WITH_SSE)
using default_arch = sse;
#else
#error no supported isa
#endif

}

#endif

