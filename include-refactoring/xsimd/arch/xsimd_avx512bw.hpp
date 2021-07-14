#ifndef XSIMD_AVX512BW_HPP
#define XSIMD_AVX512BW_HPP

#include "../types/xsimd_avx512bw_register.hpp"

namespace xsimd {

  namespace kernel {
    using namespace types;

    namespace detail {
    template<class A, class T,  int Cmp>
    batch_bool<T, A> compare_int_avx512bw(batch<T, A> const& self, batch<T, A> const& other) {
      using register_type = typename batch_bool<T, A>::register_type;
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return (register_type)_mm512_cmp_epi8_mask(self, other, Cmp);
          case 2: return (register_type)_mm512_cmp_epi16_mask(self, other, Cmp);
          case 4: return (register_type)_mm512_cmp_epi32_mask(self, other, Cmp);
          case 8: return (register_type)_mm512_cmp_epi64_mask(self, other, Cmp);
        }
      }
      else {
        switch(sizeof(T)) {
          case 1: return (register_type)_mm512_cmp_epu8_mask(self, other, Cmp);
          case 2: return (register_type)_mm512_cmp_epu16_mask(self, other, Cmp);
          case 4: return (register_type)_mm512_cmp_epu32_mask(self, other, Cmp);
          case 8: return (register_type)_mm512_cmp_epu64_mask(self, other, Cmp);
        }
      }
    }
    }

    // abs
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> abs(batch<T, A> const& self, requires<avx512bw>) {
      if(std::is_unsigned<T>::value)
        return self;

      switch(sizeof(T)) {
        case 1: return _mm512_abs_epi8(self);
        case 2: return _mm512_abs_epi16(self);
        default: return abs(self, avx512f{});
      }
    }

    // add
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> add(batch<T, A> const& self, batch<T, A> const& other, requires<avx512bw>) {
      switch(sizeof(T)) {
        case 1: return _mm512_add_epi8(self, other);
        case 2: return _mm512_add_epi16(self, other);
        default: return add(self, other, avx512f{});
      }
    }

    // bitwise_lshift
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_lshift(batch<T, A> const& self, int32_t other, requires<avx512bw>) {
      switch(sizeof(T)) {
        case 2: return _mm512_slli_epi16(self, other);
        default: return bitwise_lshift(self, other, avx512f{});
      }
    }

    // bitwise_rshift
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_rshift(batch<T, A> const& self, int32_t other, requires<avx512bw>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 2: return _mm512_srai_epi16(self, other);
          default: return bitwise_rshift(self, other, avx512f{});
        }
      }
      else {
        switch(sizeof(T)) {
          case 2: return _mm512_srli_epi8(self, other);
          default: return bitwise_rshift(self, other, avx512f{});
        }
      }
    }

    // eq
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> eq(batch<T, A> const& self, batch<T, A> const& other, requires<avx512bw>) {
      return detail::compare_int_avx512bw<A, T, _MM_CMPINT_EQ>(self, other);
    }

    // ge
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> ge(batch<T, A> const& self, batch<T, A> const& other, requires<avx512bw>) {
      return detail::compare_int_avx512bw<A, T, _MM_CMPINT_GE>(self, other);
    }

    // gt
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other, requires<avx512bw>) {
      return detail::compare_int_avx512bw<A, T, _MM_CMPINT_GT>(self, other);
    }


    // le
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> le(batch<T, A> const& self, batch<T, A> const& other, requires<avx512bw>) {
      return detail::compare_int_avx512bw<A, T, _MM_CMPINT_LE>(self, other);
    }

    // lt
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> lt(batch<T, A> const& self, batch<T, A> const& other, requires<avx512bw>) {
      return detail::compare_int_avx512bw<A, T, _MM_CMPINT_LT>(self, other);
    }

    // max
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> max(batch<T, A> const& self, batch<T, A> const& other, requires<avx512bw>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return _mm512_max_epi8(self, other);
          case 2: return _mm512_max_epi16(self, other);
          default: return max(self, other, avx512f{});
        }
      }
      else {
        switch(sizeof(T)) {
          case 1: return _mm512_max_epu8(self, other);
          case 2: return _mm512_max_epu16(self, other);
          default: return max(self, other, avx512f{});
        }
      }
    }

    // min
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> min(batch<T, A> const& self, batch<T, A> const& other, requires<avx512bw>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return _mm512_min_epi8(self, other);
          case 2: return _mm512_min_epi16(self, other);
          default: return min(self, other, avx512f{});
        }
      }
      else {
        switch(sizeof(T)) {
          case 1: return _mm512_min_epu8(self, other);
          case 2: return _mm512_min_epu16(self, other);
          default: return min(self, other, avx512f{});
        }
      }
    }

    // mul
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> mul(batch<T, A> const& self, batch<T, A> const& other, requires<avx512bw>) {
      switch(sizeof(T)) {
        case 1: {
                __m512i upper = _mm512_and_si512(_mm512_mullo_epi16(self, self), _mm512_srli_epi16(_mm512_set1_epi16(-1), 8));
                __m512i lower = _mm512_slli_epi16(_mm512_mullo_epi16(_mm512_srli_epi16(self, 8), _mm512_srli_epi16(other, 8)), 8);
                return _mm512_or_si512(upper, lower);
        }
        case 2: return _mm512_mullo_epi16(self, other);
        default: return mul(self, other, avx512f{});
      }
    }


    // neq
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> neq(batch<T, A> const& self, batch<T, A> const& other, requires<avx512bw>) {
      return detail::compare_int_avx512bw<A, T, _MM_CMPINT_NE>(self, other);
    }

    // sadd
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> sadd(batch<T, A> const& self, batch<T, A> const& other, requires<avx512bw>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return _mm512_adds_epi8(self, other);
          case 2: return _mm512_adds_epi16(self, other);
          default: return sadd(self, other, avx512f{});
        }
      }
      else {
        switch(sizeof(T)) {
          case 1: return _mm512_adds_epu8(self, other);
          case 2: return _mm512_adds_epu16(self, other);
          default: return sadd(self, other, avx512f{});
        }
      }
    }

    // select
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> select(batch_bool<T, A> const& cond, batch<T, A> const& true_br, batch<T, A> const& false_br, requires<avx512bw>) {
      switch(sizeof(T)) {
        case 1: return _mm512_mask_blend_epi8(cond, false_br, true_br);
        case 2: return _mm512_mask_blend_epi16(cond, false_br, true_br);
        default: return select(cond, true_br, false_br, avx512f{});
      };
    }


    // ssub
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> ssub(batch<T, A> const& self, batch<T, A> const& other, requires<avx512bw>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return _mm512_subs_epi8(self, other);
          case 2: return _mm512_subs_epi16(self, other);
          default: return ssub(self, other, avx512f{});
        }
      }
      else {
        switch(sizeof(T)) {
          case 1: return _mm512_subs_epu8(self, other);
          case 2: return _mm512_subs_epu16(self, other);
          default: return ssub(self, other, avx512f{});
        }
      }
    }


    // sub
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> sub(batch<T, A> const& self, batch<T, A> const& other, requires<avx512bw>) {
      switch(sizeof(T)) {
        case 1: return _mm512_sub_epi8(self, other);
        case 2: return _mm512_sub_epi16(self, other);
          default: return sub(self, other, avx512f{});
      }
    }

  }

}

#endif
