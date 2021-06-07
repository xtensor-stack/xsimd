#ifndef XSIMD_SSE4_1_HPP
#define XSIMD_SSE4_1_HPP

#include "../types/xsimd_sse4_1_register.hpp"

#include <smmintrin.h>

namespace xsimd {

  namespace kernel {
    using namespace types;
    // ceil
    template<class A> batch<float, A> ceil(batch<float, A> const& self, requires<sse4_1>) {
      return _mm_round_ps(self, _MM_FROUND_CEIL);
    }
    template<class A> batch<double, A> ceil(batch<double, A> const& self, requires<sse4_1>) {
      return _mm_round_pd(self, _MM_FROUND_CEIL);
    }

    // floor
    template<class A> batch<float, A> floor(batch<float, A> const& self, requires<sse4_1>) {
      return _mm_round_ps(self, _MM_FROUND_FLOOR);
    }
    template<class A> batch<double, A> floor(batch<double, A> const& self, requires<sse4_1>) {
      return _mm_round_pd(self, _MM_FROUND_FLOOR);
    }

    // nearbyint
    template<class A> batch<float, A> nearbyint(batch<float, A> const& self, requires<sse4_1>) {
      return _mm_round_ps(self, _MM_FROUND_TO_NEAREST_INT);
    }
    template<class A> batch<double, A> nearbyint(batch<double, A> const& self, requires<sse4_1>) {
      return _mm_round_pd(self, _MM_FROUND_TO_NEAREST_INT);
    }

    // select
    template<class A> batch<float, A> select(batch_bool<float, A> const& cond, batch<float, A> const& true_br, batch<float, A> const& false_br, requires<sse4_1>) {
      return _mm_blendv_ps(false_br, true_br, cond);
    }
    template<class A> batch<double, A> select(batch_bool<double, A> const& cond, batch<double, A> const& true_br, batch<double, A> const& false_br, requires<sse4_1>) {
      return _mm_blendv_pd(false_br, true_br, cond);
    }


    // trunc
    template<class A> batch<float, A> trunc(batch<float, A> const& self, requires<sse4_1>) {
      return _mm_round_ps(self, _MM_FROUND_TO_ZERO);
    }
    template<class A> batch<double, A> trunc(batch<double, A> const& self, requires<sse4_1>) {
      return _mm_round_pd(self, _MM_FROUND_TO_ZERO);
    }


  }

}

#endif

