#ifndef XSIMD_SSE4_2_HPP
#define XSIMD_SSE4_2_HPP

#include "../types/xsimd_sse4_2_register.hpp"

namespace xsimd {

  namespace kernel {
    using namespace types;

    // lt
    template<class A>
    batch_bool<int64_t, A> lt(batch<int64_t, A> const& self, batch<int64_t, A> const& other, requires_arch<sse4_2>) {
      return _mm_cmpgt_epi64(other, self);
    }
    template<class A>
    batch_bool<uint64_t, A> lt(batch<uint64_t, A> const& self, batch<uint64_t, A> const& other, requires_arch<sse4_2>) {
      auto xself = _mm_xor_si128(self, _mm_set1_epi64x(std::numeric_limits<int64_t>::lowest()));
      auto xother = _mm_xor_si128(other, _mm_set1_epi64x(std::numeric_limits<int64_t>::lowest()));
      return _mm_cmpgt_epi64(xother, xself);
    }

  }

}

#endif

