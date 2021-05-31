#ifndef XSIMD_SSE2_HPP
#define XSIMD_SSE2_HPP

#include "../types/xsimd_sse2_register.hpp"

#include <emmintrin.h>


namespace xsimd {

  namespace kernel {
    using namespace types;

    // add
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> add(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      switch(sizeof(T)) {
        case 1: return _mm_add_epi8(self, other);
        case 2: return _mm_add_epi16(self, other);
        case 4: return _mm_add_epi32(self, other);
        case 8: return _mm_add_epi64(self, other);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }
    template<class A> batch<float, A> add(batch<float, A> const& self, batch<float, A> const& other, requires<sse2>) {
      return _mm_add_ps(self, other);
    }
    template<class A> batch<double, A> add(batch<double, A> const& self, batch<double, A> const& other, requires<sse2>) {
      return _mm_add_pd(self, other);
    }

    // broadcast
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> broadcast(T val, requires<sse2>) {
      switch(sizeof(T)) {
        case 1: return _mm_set1_epi8(val);
        case 2: return _mm_set1_epi16(val);
        case 4: return _mm_set1_epi32(val);
        case 8: return _mm_set1_epi64x(val);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }
    template<class A> batch<float, A> broadcast(float val, requires<sse2>) {
      return _mm_set1_ps(val);
    }
    template<class A> batch<double, A> broadcast(double val, requires<sse2>) {
      return _mm_set1_pd(val);
    }

    // sub
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> sub(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      switch(sizeof(T)) {
        case 1: return _mm_sub_epi8(self, other);
        case 2: return _mm_sub_epi16(self, other);
        case 4: return _mm_sub_epi32(self, other);
        case 8: return _mm_sub_epi64(self, other);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }
    template<class A> batch<float, A> sub(batch<float, A> const& self, batch<float, A> const& other, requires<sse2>) {
      return _mm_sub_ps(self, other);
    }
    template<class A> batch<double, A> sub(batch<double, A> const& self, batch<double, A> const& other, requires<sse2>) {
      return _mm_sub_pd(self, other);
    }


  }

}

#endif


