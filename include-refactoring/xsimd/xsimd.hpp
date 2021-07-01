#ifndef XSIMD_HPP
#define XSIMD_HPP

#if defined(__GNUC__)
#define XSIMD_NO_DISCARD __attribute__((warn_unused_result))
#else
#define XSIMD_NO_DISCARD
#endif


#include "types/xsimd_batch.hpp"
#include "types/xsimd_batch_constant.hpp"
#include "types/xsimd_api.hpp"
#include "arch/xsimd_scalar.hpp"
#include "memory/xsimd_aligned_allocator.hpp"
#endif
