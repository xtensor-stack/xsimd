#ifndef XSIMD_ARCH_HPP
#define XSIMD_ARCH_HPP

namespace xsimd {

// forward declaration
template<class T, size_t N> class batch;

namespace arch {

struct sse {
  template<class T>
  using batch = xsimd::batch<T, 128 / ( 8 * sizeof(T))>;
  static constexpr size_t alignment = 8;
};

struct sse2 : sse {
  static constexpr size_t alignment = 16;
};

struct sse3 : sse2 {
};

struct sse4_1 : sse3 {
};

struct sse4_2 : sse4_1{
};

struct avx {
  template<class T>
  using batch = xsimd::batch<T, 256 / ( 8 * sizeof(T))>;
  static constexpr size_t alignment = 32;
};

// FIXME: unsure of that one
struct fma3 : avx {
};

struct avx2 {
  template<class T>
  using batch = xsimd::batch<T, 256 / ( 8 * sizeof(T))>;
  static constexpr size_t alignment = 32;
};

struct avx512 {
  template<class T>
  using batch = xsimd::batch<T, 512 / ( 8 * sizeof(T))>;
  static constexpr size_t alignment = 64;
};

struct neon64 {
  template<class T>
  using batch = xsimd::batch<T, 512 / ( 8 * sizeof(T))>;
  static constexpr size_t alignment = 32;
};

struct neon {
  template<class T>
  using batch = xsimd::batch<T, 512 / ( 8 * sizeof(T))>;
  static constexpr size_t alignment = 16;
};

struct scalar {
  template<class T>
  using batch = xsimd::batch<T, 1>;
  static constexpr size_t alignment = sizeof(void*);
};

template <class T, class InstructionSet>
using batch = typename InstructionSet::template batch<T>;

#if XSIMD_X86_INSTR_SET == XSIMD_X86_SSE_VERSION
using x86 = sse;
#elif XSIMD_X86_INSTR_SET == XSIMD_X86_SSE2_VERSION
using x86 = sse2;
#elif XSIMD_X86_INSTR_SET == XSIMD_X86_SSE3_VERSION
using x86 = sse3;
#elif XSIMD_X86_INSTR_SET == XSIMD_X86_SSE4_1_VERSION
using x86 = sse4_1;
#elif XSIMD_X86_INSTR_SET == XSIMD_X86_SSE4_2_VERSION
using x86 = sse4_2;
#elif XSIMD_X86_INSTR_SET == XSIMD_X86_AVX_VERSION
using x86 = avx;
#elif XSIMD_X86_INSTR_SET == XSIMD_X86_FMA3_VERSION
using x86 = fma3;
#elif XSIMD_X86_INSTR_SET == XSIMD_X86_AVX2_VERSION
using x86 = avx2;
#elif XSIMD_X86_INSTR_SET == XSIMD_X86_AVX512_VERSION
using x86 = avx512;
#endif

#if XSIMD_ARM_INSTR_SET == XSIMD_ARM7_NEON_VERSION
using arm = neon;
#elif XSIMD_ARM_INSTR_SET == XSIMD_ARM8_32_NEON_VERSION
using arm = neon;
#elif XSIMD_ARM_INSTR_SET == XSIMD_ARM8_64_NEON_VERSION
using arm = neon64;
#endif

#if XSIMD_X86_INSTR_SET != XSIMD_VERSION_NUMBER_NOT_AVAILABLE
using default_ = x86;
#elif XSIMD_ARM_INSTR_SET != XSIMD_VERSION_NUMBER_NOT_AVAILABLE
using default_ = arm;
#else
using default_ = scalar;
#endif

}


}

#endif
