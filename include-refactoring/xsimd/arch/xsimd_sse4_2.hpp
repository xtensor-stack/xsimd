#ifndef XSIMD_SSE4_2_HPP
#define XSIMD_SSE4_2_HPP

#include "../types/xsimd_sse4_2_register.hpp"

namespace xsimd {

  namespace kernel {
    using namespace types;

    // ceil
    template<class A> batch<float, A> ceil(batch<float, A> const& self, requires<sse4_2>) {
      return _mm_ceil_ps(self);
    }
    template<class A> batch<double, A> ceil(batch<double, A> const& self, requires<sse4_2>) {
      return _mm_ceil_pd(self);
    }

    // floor
    template<class A> batch<float, A> floor(batch<float, A> const& self, requires<generic>) {
      return _mm_floor_ps(self);
    }
    template<class A> batch<double, A> floor(batch<double, A> const& self, requires<generic>) {
      return _mm_floor_pd(self);
    }

    // lt
    template<class A>
    batch_bool<int64_t, A> lt(batch<int64_t, A> const& self, batch<int64_t, A> const& other, requires<sse4_2>) {
      return _mm_cmpgt_epi64(other, self);
    }
    template<class A>
    batch_bool<uint64_t, A> lt(batch<uint64_t, A> const& self, batch<uint64_t, A> const& other, requires<sse4_2>) {
      auto xself = _mm_xor_si128(self, _mm_set1_epi64x(std::numeric_limits<int64_t>::lowest()));
      auto xother = _mm_xor_si128(other, _mm_set1_epi64x(std::numeric_limits<int64_t>::lowest()));
      return _mm_cmpgt_epi64(xother, xself);
    }

    // nearbyint
    template<class A> batch<float, A> nearbyint(batch<float, A> const& self, requires<generic>) {
      return _mm_round_ps(self, _MM_FROUND_TO_NEAREST_INT);
    }
    template<class A> batch<double, A> nearbyint(batch<double, A> const& self, requires<generic>) {
      return _mm_round_pd(self, _MM_FROUND_TO_NEAREST_INT);
    }

    // trunc
    template<class A> batch<float, A> trunc(batch<float, A> const& self, requires<generic>) {
      return _mm_round_ps(x, _MM_FROUND_TO_ZERO);
    }
    template<class A> batch<double, A> trunc(batch<double, A> const& self, requires<generic>) {
      return _mm_round_pd(x, _MM_FROUND_TO_ZERO);
    }

  }

}

#endif

