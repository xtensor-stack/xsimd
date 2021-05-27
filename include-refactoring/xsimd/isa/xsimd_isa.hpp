#ifndef XSIMD_ISA_HPP
#define XSIMD_ISA_HPP

#include "../config/xsimd_arch.hpp"

#include "./xsimd_generic.hpp"

#ifdef XSIMD_WITH_SSE
#include "./xsimd_sse.hpp"
#endif

#ifdef XSIMD_WITH_SSE3
#include "./xsimd_sse3.hpp"
#endif

#endif

