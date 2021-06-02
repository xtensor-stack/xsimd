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

    // batch_cast
    template<class A> batch<float, A> batch_cast(batch<int32_t, A> const& self, batch<float, A> const&, requires<sse2>) {
      return _mm_cvtepi32_ps(self);
    }
    template<class A> batch<int32_t, A> batch_cast(batch<float, A> const& self, batch<int32_t, A> const&, requires<sse2>) {
      return _mm_cvttps_epi32(self);
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

    // load_aligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> load_aligned(T const* mem, convert<T>, requires<sse2>) {
      return _mm_load_si128((__m128i const*)mem);
    }

    // load_unaligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> load_unaligned(T const* mem, convert<T>, requires<sse2>) {
      return _mm_loadu_si128((__m128i const*)mem);
    }

    // store_aligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_aligned(T *mem, batch<T, A> const& self, requires<sse2>) {
      return _mm_store_si128((__m128i *)mem, self);
    }

    // store_unaligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_unaligned(T *mem, batch<T, A> const& self, requires<sse2>) {
      return _mm_storeu_si128((__m128i *)mem, self);
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

    // to_float
    template<class A>
    batch<float, A> to_float(batch<int32_t, A> const& self, requires<sse2>) {
      return _mm_cvtepi32_ps(self);
    }
    template<class A>
    batch<double, A> to_float(batch<int64_t, A> const& self, requires<sse2>) {
      // FIXME: call _mm_cvtepi64_pd
      alignas(A::alignment()) int64_t buffer[batch<int64_t, A>::size];
      self.store_aligned(&buffer[0]);
      return {(double)buffer[0], (double)buffer[1]};
    }

    // to_int
    template<class A>
    batch<int32_t, A> to_int(batch<float, A> const& self, requires<sse2>) {
      return _mm_cvttps_epi32(self);
    }

    template<class A>
    batch<int64_t, A> to_int(batch<double, A> const& self, requires<sse2>) {
      // FIXME: call _mm_cvttpd_epi64
      alignas(A::alignment()) double buffer[batch<double, A>::size];
      self.store_aligned(&buffer[0]);
      return {(int64_t)buffer[0], (int64_t)buffer[1]};
    }


  }

}

#endif


