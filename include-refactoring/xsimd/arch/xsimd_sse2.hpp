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

    // bitwise_and
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_and(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      return _mm_and_si128(self, other);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> bitwise_and(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires<sse2>) {
      return _mm_and_si128(self, other);
    }

    // bitwise_lshift
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_lshift(batch<T, A> const& self, int32_t other, requires<sse2>) {
      switch(sizeof(T)) {
        case 1: return _mm_and_si128(_mm_set1_epi8(0xFF << other), _mm_slli_epi32(self, other));
        case 2: return _mm_slli_epi16(self, other);
        case 4: return _mm_slli_epi32(self, other);
        case 8: return _mm_slli_epi64(self, other);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }

    // bitwise_not
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_not(batch<T, A> const& self, requires<sse2>) {
      return _mm_xor_si128(self, _mm_set1_epi32(-1));
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> bitwise_not(batch_bool<T, A> const& self, requires<sse2>) {
      return _mm_xor_si128(self, _mm_set1_epi32(-1));
    }

    // bitwise_or
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_or(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      return _mm_or_si128(self, other);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> bitwise_or(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires<sse2>) {
      return _mm_or_si128(self, other);
    }

    // bitwise_rshift
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_rshift(batch<T, A> const& self, int32_t other, requires<sse2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: {
            __m128i sign_mask = _mm_set1_epi16((0xFF00 >> other) & 0x00FF);
            __m128i cmp_is_negative = _mm_cmpgt_epi8(_mm_setzero_si128(), self);
            __m128i res = _mm_srai_epi16(self, other);
            return _mm_or_si128(_mm_and_si128(sign_mask, cmp_is_negative), _mm_andnot_si128(sign_mask, res));
          }
          case 2: return _mm_srai_epi16(self, other);
          case 4: return _mm_srai_epi32(self, other);
          case 8: {
            // from https://github.com/samyvilar/vect/blob/master/vect_128.h
            return _mm_or_si128(
                _mm_srli_epi64(self, other),
                _mm_slli_epi64(
                    _mm_srai_epi32(_mm_shuffle_epi32(self, _MM_SHUFFLE(3, 3, 1, 1)), 32),
                    64 - other));
          }
          default: assert(false && "unsupported arch/op combination"); return {};
        }
      }
      else {
        switch(sizeof(T)) {
          case 1: return _mm_and_si128(_mm_set1_epi8(0xFF >> other), _mm_srli_epi32(self, other));
          case 2: return _mm_srli_epi16(self, other);
          case 4: return _mm_srli_epi32(self, other);
          case 8: return _mm_srli_epi64(self, other);
          default: assert(false && "unsupported arch/op combination"); return {};
        }
      }
    }

    // bitwise_xor
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_xor(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      return _mm_xor_si128(self, other);
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

    // convert
    namespace conversion {
    template<class A> batch<float, A> fast(batch<int32_t, A> const& self, batch<float, A> const&, requires<sse2>) {
      return _mm_cvtepi32_ps(self);
    }
    template<class A> batch<int32_t, A> fast(batch<float, A> const& self, batch<int32_t, A> const&, requires<sse2>) {
      return _mm_cvttps_epi32(self);
    }
    }

    // eq
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> eq(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      switch(sizeof(T)) {
        case 1: return _mm_cmpeq_epi8(self, other);
        case 2: return _mm_cmpeq_epi16(self, other);
        case 4: return _mm_cmpeq_epi32(self, other);
        case 8: {
            __m128i tmp1 = _mm_cmpeq_epi32(self, other);
            __m128i tmp2 = _mm_shuffle_epi32(tmp1, 0xB1);
            __m128i tmp3 = _mm_and_si128(tmp1, tmp2);
            __m128i tmp4 = _mm_srai_epi32(tmp3, 31);
            return _mm_shuffle_epi32(tmp4, 0xF5);
        }
        default: assert(false && "unsupported arch/op combination"); return {};
      }
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

    // lt
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> lt(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return _mm_cmplt_epi8(self, other);
          case 2: return _mm_cmplt_epi16(self, other);
          case 4: return _mm_cmplt_epi32(self, other);
          case 8: {
              __m128i tmp1 = _mm_sub_epi64(self, other);
              __m128i tmp2 = _mm_xor_si128(self, other);
              __m128i tmp3 = _mm_andnot_si128(other, self);
              __m128i tmp4 = _mm_andnot_si128(tmp2, tmp1);
              __m128i tmp5 = _mm_or_si128(tmp3, tmp4);
              __m128i tmp6 = _mm_srai_epi32(tmp5, 31);
              return _mm_shuffle_epi32(tmp6, 0xF5);
          }
          default: assert(false && "unsupported arch/op combination"); return {};
        }
      }
      else {
        switch(sizeof(T)) {
          case 1: return _mm_cmplt_epi8(_mm_xor_si128(self, _mm_set1_epi8(std::numeric_limits<int8_t>::lowest())), _mm_xor_si128(other, _mm_set1_epi8(std::numeric_limits<int8_t>::lowest())));
          case 2: return _mm_cmplt_epi16(_mm_xor_si128(self, _mm_set1_epi16(std::numeric_limits<int16_t>::lowest())), _mm_xor_si128(other, _mm_set1_epi16(std::numeric_limits<int16_t>::lowest())));
          case 4: return _mm_cmplt_epi32(_mm_xor_si128(self, _mm_set1_epi32(std::numeric_limits<int32_t>::lowest())), _mm_xor_si128(other, _mm_set1_epi32(std::numeric_limits<int32_t>::lowest())));
          case 8: {
                auto xself = _mm_xor_si128(self, _mm_set1_epi64x(std::numeric_limits<int64_t>::lowest()));
                auto xother = _mm_xor_si128(other, _mm_set1_epi64x(std::numeric_limits<int64_t>::lowest()));
                __m128i tmp1 = _mm_sub_epi64(xself, xother);
                __m128i tmp2 = _mm_xor_si128(xself, xother);
                __m128i tmp3 = _mm_andnot_si128(xother, xself);
                __m128i tmp4 = _mm_andnot_si128(tmp2, tmp1);
                __m128i tmp5 = _mm_or_si128(tmp3, tmp4);
                __m128i tmp6 = _mm_srai_epi32(tmp5, 31);
                return _mm_shuffle_epi32(tmp6, 0xF5);
                  }
          default: assert(false && "unsupported arch/op combination"); return {};
        }
      }
    }
    // select
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> select(batch_bool<T, A> const& cond, batch<T, A> const& true_br, batch<T, A> const& false_br, requires<sse2>) {
      return _mm_or_si128(_mm_and_si128(cond, true_br), _mm_andnot_si128(cond, false_br));
    }

    // store_aligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_aligned(T *mem, batch<T, A> const& self, requires<sse2>) {
      return _mm_store_si128((__m128i *)mem, self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_aligned(T *mem, batch_bool<T, A> const& self, requires<sse2>) {
      return _mm_store_si128((__m128i *)mem, self);
    }

    // store_unaligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_unaligned(T *mem, batch<T, A> const& self, requires<sse2>) {
      return _mm_storeu_si128((__m128i *)mem, self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_unaligned(T *mem, batch_bool<T, A> const& self, requires<sse2>) {
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


