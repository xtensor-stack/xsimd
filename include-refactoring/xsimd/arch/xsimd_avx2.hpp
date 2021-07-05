#ifndef XSIMD_AVX2_HPP
#define XSIMD_AVX2_HPP

#include "../types/xsimd_avx2_register.hpp"


namespace xsimd {

  namespace kernel {
    using namespace types;
    // forward declaration
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> abs(batch<T, A> const& self, requires<generic>);
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_lshift(batch<T, A> const& self, batch<T, A> const& other, requires<generic>);
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_rshift(batch<T, A> const& self, batch<T, A> const& other, requires<generic>);
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> mul(batch<T, A> const& self, batch<T, A> const& other, requires<generic>);

    // abs
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> abs(batch<T, A> const& self, requires<avx2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return _mm256_abs_epi8(self);
          case 2: return _mm256_abs_epi16(self);
          case 4: return _mm256_abs_epi32(self);
          default: return abs(self, avx{});
        }
      }
      return self;
    }

    // add
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> add(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      switch(sizeof(T)) {
        case 1: return _mm256_add_epi8(self, other);
        case 2: return _mm256_add_epi16(self, other);
        case 4: return _mm256_add_epi32(self, other);
        case 8: return _mm256_add_epi64(self, other);
        default: return add(self, other, avx{});
      }
    }

    // bitwise_lshift
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_lshift(batch<T, A> const& self, int32_t other, requires<avx2>) {
      switch(sizeof(T)) {
        case 2: return _mm256_slli_epi16(self, other);
        case 4: return _mm256_slli_epi32(self, other);
        case 8: return _mm256_slli_epi64(self, other);
        default: return bitwise_lshift(self, other, avx{});
      }
    }

    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_lshift(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      switch(sizeof(T)) {
        case 4: return _mm256_sllv_epi32(self, other);
        case 8: return _mm256_sllv_epi64(self, other);
        default: return bitwise_lshift(self, other, avx{});
      }
    }

    // bitwise_rshift
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_rshift(batch<T, A> const& self, int32_t other, requires<avx2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 2: return _mm256_srai_epi16(self, other);
          case 4: return _mm256_srai_epi32(self, other);
          default: return bitwise_rshift(self, other, avx{});
        }
      }
      else {
        switch(sizeof(T)) {
          case 2: return _mm256_srli_epi16(self, other);
          case 4: return _mm256_srli_epi32(self, other);
          case 8: return _mm256_srli_epi64(self, other);
          default: return bitwise_rshift(self, other, avx{});
        }
      }
    }

    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_rshift(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 4: return _mm256_srav_epi32(self, other);
          default: return bitwise_rshift(self, other, avx{});
        }
      }
      else {
        switch(sizeof(T)) {
          case 4: return _mm256_srlv_epi32(self, other);
          default: return bitwise_rshift(self, other, avx{});
        }
      }
    }


    // eq
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> eq(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      switch(sizeof(T)) {
        case 1: return _mm256_cmpeq_epi8(self, other);
        case 2: return _mm256_cmpeq_epi16(self, other);
        case 4: return _mm256_cmpeq_epi32(self, other);
        case 8: return _mm256_cmpeq_epi64(self, other);
        default: eq(self, other, avx{});
      }
    }

    // gt
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return _mm256_cmpgt_epi8(self, other);
          case 2: return _mm256_cmpgt_epi16(self, other);
          case 4: return _mm256_cmpgt_epi32(self, other);
          case 8: return _mm256_cmpgt_epi64(self, other);
          default: return gt(self, other, avx{});
        }
      }
      else {
          return gt(self, other, avx{});
      }
    }

    // hadd
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    T hadd(batch<T, A> const& self, requires<avx2>) {
      switch(sizeof(T)) {
        case 4:
          {
                __m256i tmp1 = _mm256_hadd_epi32(self, self);
                __m256i tmp2 = _mm256_hadd_epi32(tmp1, tmp1);
                __m128i tmp3 = _mm256_extracti128_si256(tmp2, 1);
                __m128i tmp4 = _mm_add_epi32(_mm256_castsi256_si128(tmp2), tmp3);
                return _mm_cvtsi128_si32(tmp4);
          }
        case 8:
          {
                __m256i tmp1 = _mm256_shuffle_epi32(self, 0x0E);
                __m256i tmp2 = _mm256_add_epi64(self, tmp1);
                __m128i tmp3 = _mm256_extracti128_si256(tmp2, 1);
                __m128i res = _mm_add_epi64(_mm256_castsi256_si128(tmp2), tmp3);
                return _mm_cvtsi128_si64(res);
          }
          default: return hadd(self, avx{});
      }
    }

    // max
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> max(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return _mm256_max_epi8(self, other);
          case 2: return _mm256_max_epi16(self, other);
          case 4: return _mm256_max_epi32(self, other);
          default: return max(self, other, avx{});
        }
      }
      else {
        switch(sizeof(T)) {
          case 1: return _mm256_max_epu8(self, other);
          case 2: return _mm256_max_epu16(self, other);
          case 4: return _mm256_max_epu32(self, other);
          default: return max(self, other, avx{});
        }
      }
    }

    // min
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> min(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1:  return _mm256_min_epi8(self, other);
          case 2:  return _mm256_min_epi16(self, other);
          case 4:  return _mm256_min_epi32(self, other);
          default: return min(self, other, avx{});
        }
      }
      else {
        switch(sizeof(T)) {
          case 1:  return _mm256_min_epu8(self, other);
          case 2:  return _mm256_min_epu16(self, other);
          case 4:  return _mm256_min_epu32(self, other);
          default: return min(self, other, avx{});
        }
      }
    }

    // mul
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> mul(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      switch(sizeof(T)) {
        case 2: return _mm256_mullo_epi16(self, other);
        case 4: return _mm256_mullo_epi32(self, other);
        default: return mul(self, other, avx{});
      }
    }

    // sadd
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> sadd(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return _mm256_adds_epi8(self, other);
          case 2: return _mm256_adds_epi16(self, other);
          default: return sadd(self, other, avx{});
        }
      }
      else {
        switch(sizeof(T)) {
          case 1: return _mm256_adds_epu8(self, other);
          case 2: return _mm256_adds_epu16(self, other);
          default: return sadd(self, other, avx{});
        }
      }
    }

    // select
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> select(batch_bool<T, A> const& cond, batch<T, A> const& true_br, batch<T, A> const& false_br, requires<avx2>) {
      switch(sizeof(T)) {
        case 1: return _mm256_blendv_epi8(false_br, true_br, cond);
        case 2: return _mm256_blendv_epi8(false_br, true_br, cond);
        case 4: return _mm256_blendv_epi8(false_br, true_br, cond);
        case 8: return _mm256_blendv_epi8(false_br, true_br, cond);
        default: return select(cond, true_br, false_br, avx{});
      }
    }

    // ssub
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> ssub(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return _mm256_subs_epi8(self, other);
          case 2: return _mm256_subs_epi16(self, other);
          default: return ssub(self, other, avx{});
        }
      }
      else {
        switch(sizeof(T)) {
          case 1: return _mm256_subs_epu8(self, other);
          case 2: return _mm256_subs_epu16(self, other);
          default: return ssub(self, other, avx{});
        }
      }
    }

    // sub
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> sub(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      switch(sizeof(T)) {
        case 1: return _mm256_sub_epi8(self, other);
        case 2: return _mm256_sub_epi16(self, other);
        case 4: return _mm256_sub_epi32(self, other);
        case 8: return _mm256_sub_epi64(self, other);
        default: return sub(self, other, avx{});
      }
    }



  }

}

#endif
