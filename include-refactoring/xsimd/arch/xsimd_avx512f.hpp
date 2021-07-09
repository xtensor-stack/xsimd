#ifndef XSIMD_AVX512F_HPP
#define XSIMD_AVX512F_HPP

#include "../types/xsimd_avx512f_register.hpp"

namespace xsimd {

  namespace kernel {
    using namespace types;

    namespace detail {
      inline void split_avx512(__m512 val, __m256& low, __m256& high) {
        low =_mm512_castps512_ps256(val);
        high =_mm512_extractf32x8_ps(val, 1);
      }
      inline void split_avx512(__m512d val, __m256d& low, __m256d& high) {
        low =_mm512_castpd512_pd256(val);
        high =_mm512_extractf64x4_pd(val, 1);
      }
      inline void split_avx512(__m512i val, __m256i& low, __m256i& high) {
        low =_mm512_castsi512_si256(val);
        high =_mm512_extracti64x4_epi64(val, 1);
      }
      inline __m512i merge_avx(__m256i low, __m256i high) {
        return _mm512_inserti64x4(_mm512_castsi256_si512(low), high, 1);
      }
      inline __m512 merge_avx(__m256 low, __m256 high) {
        return _mm512_insertf32x8(_mm512_castps256_ps512(low), high, 1);
      }
      inline __m512d merge_avx(__m256d low, __m256d high) {
        return _mm512_insertf64x4(_mm512_castpd256_pd512(low), high, 1);
      }
      template<class F>
      __m512i fwd_to_avx(F f, __m512i self, __m512i other) {
        __m256i self_low, self_high, other_low, other_high;
        split_avx512(self, self_low, self_high);
        split_avx512(other, other_low, other_high);
        __m256i res_low = f(self_low, other_low);
        __m256i res_high = f(self_high, other_high);
        return merge_avx(res_low, res_high);
      }
      template<class F>
      __m512i fwd_to_avx(F f, __m512i self, int32_t other) {
        __m256i self_low, self_high;
        split_avx512(self, self_low, self_high);
        __m256i res_low = f(self_low, other);
        __m256i res_high = f(self_high, other);
        return merge_avx(res_low, res_high);
      }
    }

    // abs
    template<class A> batch<float, A> abs(batch<float, A> const& self, requires<avx512f>) {
      __m512 self_asf = (__m512)self;
      __m512i self_asi = *reinterpret_cast<__m512i *>(&self_asf);
      __m512i res_asi = _mm512_and_epi32(_mm512_set1_epi32(0x7FFFFFFF), self_asi);
      return *reinterpret_cast<__m512*>(&res_asi);
    }
    template<class A> batch<double, A> abs(batch<double, A> const& self, requires<avx512f>) {
                __m512d self_asd = (__m512d)self;
                __m512i self_asi = *reinterpret_cast<__m512i*>(&self_asd);
                __m512i res_asi = _mm512_and_epi64(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFF),
                                                   self_asi);
                return *reinterpret_cast<__m512d*>(&res_asi);
    }

    // add
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> add(batch<T, A> const& self, batch<T, A> const& other, requires<avx512f>) {
      switch(sizeof(T)) {
        case 1: return detail::fwd_to_avx([](__m256i s, __m256i o) { return add(batch<T, avx2>(s), batch<T, avx2>(o)); }, self, other);
        case 2: return detail::fwd_to_avx([](__m256i s, __m256i o) { return add(batch<T, avx2>(s), batch<T, avx2>(o)); }, self, other);
        case 4: return _mm512_add_epi32(self, other);
        case 8: return _mm512_add_epi64(self, other);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }
    template<class A> batch<float, A> add(batch<float, A> const& self, batch<float, A> const& other, requires<avx512f>) {
      return _mm512_add_ps(self, other);
    }
    template<class A> batch<double, A> add(batch<double, A> const& self, batch<double, A> const& other, requires<avx512f>) {
      return _mm512_add_pd(self, other);
    }

    // all
    template<class A, class T>
    bool all(batch_bool<T, A> const& self, requires<avx512f>) {
      using register_type = typename batch_bool<T, A>::register_type;
      return self.data == register_type(-1);
    }

    // any
    template<class A, class T>
    bool any(batch_bool<T, A> const& self, requires<avx512f>) {
      using register_type = typename batch_bool<T, A>::register_type;
      return self.data != register_type(0);
    }

    // bitwise_and
    template<class A> batch<float, A> bitwise_and(batch<float, A> const& self, batch<float, A> const& other, requires<avx512f>) {
      return _mm512_and_ps(self, other);
    }
    template<class A> batch<double, A> bitwise_and(batch<double, A> const& self, batch<double, A> const& other, requires<avx512f>) {
      return _mm512_and_pd(self, other);
    }

    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_and(batch<T, A> const& self, batch<T, A> const& other, requires<avx512f>) {
      return _mm512_and_si512(self, other);
    }

    template<class A, class T>
    batch_bool<T, A> bitwise_and(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires<avx512f>) {
      using register_type = typename batch_bool<T, A>::register_type;
      return register_type(self.data & other.data);
    }

    // bitwise_lshift
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_lshift(batch<T, A> const& self, int32_t other, requires<avx512f>) {
      switch(sizeof(T)) {
        case 1: {
        __m512i tmp = _mm512_slli_epi32(self, other);
        return _mm512_and_si512(_mm512_set1_epi8(0xFF << other), tmp);
        }
        case 2: return detail::fwd_to_avx([](__m256i s, int32_t o) { return bitwise_lshift(batch<T, avx2>(s), o, avx2{}); }, self, other);
        case 4: return _mm512_slli_epi32(self, other);
        case 8: return detail::fwd_to_avx([](__m256i s, int32_t o) { return bitwise_lshift(batch<T, avx2>(s), o, avx2{}); }, self, other);

        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }

    // bitwise_not
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_not(batch<T, A> const& self, requires<avx512f>) {
      return _mm512_xor_si512(self, _mm512_set1_epi32(-1));
    }
    template<class A, class T>
    batch_bool<T, A> bitwise_not(batch_bool<T, A> const& self, requires<avx512f>) {
      using register_type = typename batch_bool<T, A>::register_type;
      return register_type(~self.data);
    }

    template<class A> batch<float, A> bitwise_not(batch<float, A> const& self, requires<avx512f>) {
      return _mm256_xor_ps(self, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
    }
    template <class A>
    batch<double, A> bitwise_not(batch<double, A> const &self, requires<avx512f>) {
      return _mm256_xor_pd(self, _mm256_castsi256_pd(_mm256_set1_epi32(-1)));
    }

    // bitwise_or
    template<class A> batch<float, A> bitwise_or(batch<float, A> const& self, batch<float, A> const& other, requires<avx512f>) {
      return _mm512_or_ps(self, other);
    }
    template<class A> batch<double, A> bitwise_or(batch<double, A> const& self, batch<double, A> const& other, requires<avx512f>) {
      return _mm512_or_pd(self, other);
    }

    template<class A, class T> batch_bool<T, A> bitwise_or(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires<avx512f>) {
      using register_type = typename batch_bool<T, A>::register_type;
      return register_type(self.data | other.data);
    }

    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_or(batch<T, A> const& self, batch<T, A> const& other, requires<avx512f>) {
      return _mm512_or_si512(self, other);
    }

    // bitwise_rshift
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_rshift(batch<T, A> const& self, int32_t other, requires<avx512f>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: {__m512i tmp = _mm512_slli_epi32(self, other); return _mm512_and_si512(_mm512_set1_epi8(0xFF << other), tmp);}
          case 2: {__m512i tmp = _mm512_slli_epi32(self, other); return _mm512_and_si512(_mm512_set1_epi16(0xFF << other), tmp);}
          case 4: return _mm512_srai_epi32(self, other);
          case 8: return _mm512_srai_epi64(self, other);
          default: assert(false && "unsupported arch/op combination"); return {};
        }
      }
      else {
        switch(sizeof(T)) {
          case 1: {__m512i tmp = _mm512_srli_epi32(self, other); return _mm512_and_si512(_mm512_set1_epi8(0xFF >> other), tmp);}
          case 2: return _mm512_srli_epi16(self, other);
          case 4: return _mm512_srli_epi32(self, other);
          case 8: return _mm512_srli_epi64(self, other);
          default: assert(false && "unsupported arch/op combination"); return {};
        }
      }
    }

    // bitwise_xor
    template<class A> batch<float, A> bitwise_xor(batch<float, A> const& self, batch<float, A> const& other, requires<avx512f>) {
      return _mm512_xor_ps(self, other);
    }
    template<class A> batch<double, A> bitwise_xor(batch<double, A> const& self, batch<double, A> const& other, requires<avx512f>) {
      return _mm512_xor_pd(self, other);
    }

    template<class A, class T> batch_bool<T, A> bitwise_xor(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires<avx512f>) {
      using register_type = typename batch_bool<T, A>::register_type;
      return register_type(self.data | other.data);
    }

    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_xor(batch<T, A> const& self, batch<T, A> const& other, requires<avx512f>) {
      return _mm512_xor_si512(self, other);
    }

    // bitwise_cast
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<float, A> bitwise_cast(batch<T, A> const& self, batch<float, A> const &, requires<avx512f>) {
      return _mm512_castsi512_ps(self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<double, A> bitwise_cast(batch<T, A> const& self, batch<double, A> const &, requires<avx512f>) {
      return _mm512_castsi512_pd(self);
    }
    template<class A, class T, class Tp, class=typename std::enable_if<std::is_integral<typename std::common_type<T, Tp>::type>::value, void>::type>
    batch<Tp, A> bitwise_cast(batch<T, A> const& self, batch<Tp, A> const &, requires<avx512f>) {
      return batch<Tp, A>(self.data);
    }
    template<class A>
    batch<double, A> bitwise_cast(batch<float, A> const& self, batch<double, A> const &, requires<avx512f>) {
      return _mm512_castps_pd(self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_cast(batch<float, A> const& self, batch<T, A> const &, requires<avx512f>) {
      return _mm512_castps_si512(self);
    }
    template<class A>
    batch<float, A> bitwise_cast(batch<double, A> const& self, batch<float, A> const &, requires<avx512f>) {
      return _mm512_castpd_ps(self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_cast(batch<double, A> const& self, batch<T, A> const &, requires<avx512f>) {
      return _mm512_castpd_si512(self);
    }

    // bool_cast
    template<class A> batch_bool<int32_t, A> bool_cast(batch_bool<float, A> const& self, requires<avx512f>) {
        return self.data;
    }
    template<class A> batch_bool<float, A> bool_cast(batch_bool<int32_t, A> const& self, requires<avx512f>) {
        return self.data;
    }
    template<class A> batch_bool<int64_t, A> bool_cast(batch_bool<double, A> const& self, requires<avx512f>) {
        return self.data;
    }
    template<class A> batch_bool<double, A> bool_cast(batch_bool<int64_t, A> const& self, requires<avx512f>) {
        return self.data;
    }


    // broadcast
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> broadcast(T val, requires<avx512f>) {
      switch(sizeof(T)) {
        case 1: return _mm512_set1_epi8(val);
        case 2: return _mm512_set1_epi16(val);
        case 4: return _mm512_set1_epi32(val);
        case 8: return _mm512_set1_epi64(val);
        default: assert(false && "unsupported"); return {};
      }
    }
    template<class A> batch<float, A> broadcast(float val, requires<avx512f>) {
      return _mm512_set1_ps(val);
    }
    template<class A> batch<double, A> broadcast(double val, requires<avx512f>) {
      return _mm512_set1_pd(val);
    }

    // ceil
    template<class A> batch<float, A> ceil(batch<float, A> const& self, requires<avx512f>) {
      return _mm512_ceil_ps(self);
    }
    template<class A> batch<double, A> ceil(batch<double, A> const& self, requires<avx512f>) {
      return _mm512_ceil_pd(self);
    }


    // complex_low
    template<class A> batch<float, A> complex_low(batch<std::complex<float>, A> const& self, requires<avx512f>) {
        __m512i idx = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
        return _mm512_permutex2var_ps(self.real(), idx, self.imag());

    }
    template<class A> batch<double, A> complex_low(batch<std::complex<double>, A> const& self, requires<avx512f>) {
        __m512i idx = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
        return _mm512_permutex2var_pd(self.real(), idx, self.imag());
    }

    // complex_high
    template<class A> batch<float, A> complex_high(batch<std::complex<float>, A> const& self, requires<avx512f>) {
        __m512i idx = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
        return _mm512_permutex2var_ps(self.real(), idx, self.imag());
    }
    template<class A> batch<double, A> complex_high(batch<std::complex<double>, A> const& self, requires<avx512f>) {
        __m512i idx = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
        return _mm512_permutex2var_pd(self.real(), idx, self.imag());
    }

    // convert
    namespace detail {
    template<class A> batch<float, A> fast_cast(batch<int32_t, A> const& self, batch<float, A> const&, requires<avx512f>) {
      return _mm512_cvtepi32_ps(self);
    }
    template<class A> batch<int32_t, A> fast_cast(batch<float, A> const& self, batch<int32_t, A> const&, requires<avx512f>) {
      return _mm512_cvttps_epi32(self);
    }
    }

    // div
    template<class A> batch<float, A> div(batch<float, A> const& self, batch<float, A> const& other, requires<avx512f>) {
      return _mm512_div_ps(self, other);
    }
    template<class A> batch<double, A> div(batch<double, A> const& self, batch<double, A> const& other, requires<avx512f>) {
      return _mm512_div_pd(self, other);
    }


    // eq
    template<class A> batch_bool<float, A> eq(batch<float, A> const& self, batch<float, A> const& other, requires<avx512f>) {
      return _mm512_cmp_ps_mask(self, other, _CMP_EQ_OQ);
    }
    template<class A> batch_bool<double, A> eq(batch<double, A> const& self, batch<double, A> const& other, requires<avx512f>) {
      return _mm512_cmp_pd_mask(self, other, _CMP_EQ_OQ);
    }

    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> eq(batch<T, A> const& self, batch<T, A> const& other, requires<avx512f>) {
      using register_type = typename batch_bool<T, A>::register_type;
        switch(sizeof(T)) {
          case 4: return (register_type)_mm512_cmpeq_epu32_mask(self, other);
          case 8: return (register_type)_mm512_cmpeq_epu64_mask(self, other);
          default: assert(false && "unsupported vector / arch combination");
        }
    }
    template<class A, class T>
    batch_bool<T, A> eq(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires<avx512f>) {
      using register_type = typename batch_bool<T, A>::register_type;
      return register_type(~self.data ^ other.data);
    }

    // floor
    template<class A> batch<float, A> floor(batch<float, A> const& self, requires<avx512f>) {
      return _mm512_floor_ps(self);
    }
    template<class A> batch<double, A> floor(batch<double, A> const& self, requires<avx512f>) {
      return _mm512_floor_pd(self);
    }

#if 0
    // ge
    template<class A> batch_bool<float, A> ge(batch<float, A> const& self, batch<float, A> const& other, requires<avx512f>) {
      return _mm256_cmp_ps(self, other, _CMP_GE_OQ);
    }
    template<class A> batch_bool<double, A> ge(batch<double, A> const& self, batch<double, A> const& other, requires<avx512f>) {
      return _mm256_cmp_pd(self, other, _CMP_GE_OQ);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> ge(batch<T, A> const& self, batch<T, A> const& other, requires<avx512f>) {
      return detail::fwd_to_sse([](__m128i s, __m128i o) { return ge(batch<T, sse4_2>(s), batch<T, sse4_2>(o)); }, self, other);
    }

    // gt
    template<class A> batch_bool<float, A> gt(batch<float, A> const& self, batch<float, A> const& other, requires<avx512f>) {
      return _mm256_cmp_ps(self, other, _CMP_GT_OQ);
    }
    template<class A> batch_bool<double, A> gt(batch<double, A> const& self, batch<double, A> const& other, requires<avx512f>) {
      return _mm256_cmp_pd(self, other, _CMP_GT_OQ);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other, requires<avx512f>) {
      return detail::fwd_to_sse([](__m128i s, __m128i o) { return gt(batch<T, sse4_2>(s), batch<T, sse4_2>(o)); }, self, other);
    }
#endif


    // hadd
    template<class A> float hadd(batch<float, A> const& rhs, requires<avx512f>) {
                __m256 tmp1 = _mm512_extractf32x8_ps(rhs, 1);
                __m256 tmp2 = _mm512_extractf32x8_ps(rhs, 0);
                __m256 res1 = _mm256_add_ps(tmp1, tmp2);
                return hadd(batch<float, avx2>(res1), avx2{});

    }
    template <class A>
    double hadd(batch<double, A> const &rhs, requires<avx512f>) {
                __m256d tmp1 = _mm512_extractf64x4_pd(rhs, 1);
                __m256d tmp2 = _mm512_extractf64x4_pd(rhs, 0);
                __m256d res1 = _mm256_add_pd(tmp1, tmp2);
                return hadd(batch<double, avx2>(res1), avx2{});
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    T hadd(batch<T, A> const& self, requires<avx512f>) {
      __m256i low, high;
      detail::split_avx512(self, low, high);
      batch<T, avx2> blow(low), bhigh(high);
      return hadd(blow, avx2{}) + hadd(bhigh, avx2{});
    }

    // haddp
    template<class A> batch<float, A> haddp(batch<float, A> const* row, requires<avx512f>) {
                // The following folds over the vector once:
                // tmp1 = [a0..8, b0..8]
                // tmp2 = [a8..f, b8..f]
#define XSIMD_AVX512_HADDP_STEP1(I, a, b)                                                  \
        batch<float, avx512f> res ## I;                                                             \
        {                                                                                      \
            auto tmp1 = _mm512_shuffle_f32x4(a, b, _MM_SHUFFLE(1, 0, 1, 0));                   \
            auto tmp2 = _mm512_shuffle_f32x4(a, b, _MM_SHUFFLE(3, 2, 3, 2));                   \
            res ## I = _mm512_add_ps(tmp1, tmp2);                                              \
        }                                                                                      \

                XSIMD_AVX512_HADDP_STEP1(0, row[0], row[2]);
                XSIMD_AVX512_HADDP_STEP1(1, row[4], row[6]);
                XSIMD_AVX512_HADDP_STEP1(2, row[1], row[3]);
                XSIMD_AVX512_HADDP_STEP1(3, row[5], row[7]);
                XSIMD_AVX512_HADDP_STEP1(4, row[8], row[10]);
                XSIMD_AVX512_HADDP_STEP1(5, row[12], row[14]);
                XSIMD_AVX512_HADDP_STEP1(6, row[9], row[11]);
                XSIMD_AVX512_HADDP_STEP1(7, row[13], row[15]);

#undef XSIMD_AVX512_HADDP_STEP1

                // The following flds the code and shuffles so that hadd_ps produces the correct result
                // tmp1 = [a0..4,  a8..12,  b0..4,  b8..12] (same for tmp3)
                // tmp2 = [a5..8, a12..16, b5..8, b12..16]  (same for tmp4)
                // tmp5 = [r1[0], r1[2], r2[0], r2[2], r1[4], r1[6] ...
#define XSIMD_AVX512_HADDP_STEP2(I, a, b, c, d)                                             \
        batch<float, avx2> halfx ## I;                                                             \
        {                                                                                       \
            auto tmp1 = _mm512_shuffle_f32x4(a, b, _MM_SHUFFLE(2, 0, 2, 0));        \
            auto tmp2 = _mm512_shuffle_f32x4(a, b, _MM_SHUFFLE(3, 1, 3, 1));        \
          \
            auto resx1 = _mm512_add_ps(tmp1, tmp2);                                               \
          \
            auto tmp3 = _mm512_shuffle_f32x4(c, d, _MM_SHUFFLE(2, 0, 2, 0));        \
            auto tmp4 = _mm512_shuffle_f32x4(c, d, _MM_SHUFFLE(3, 1, 3, 1));        \
          \
            auto resx2 = _mm512_add_ps(tmp3, tmp4);                                               \
          \
            auto tmp5 = _mm512_shuffle_ps(resx1, resx2, _MM_SHUFFLE(2, 0, 2, 0));   \
            auto tmp6 = _mm512_shuffle_ps(resx1, resx2, _MM_SHUFFLE(3, 1, 3, 1));   \
          \
            auto resx3 = _mm512_add_ps(tmp5, tmp6);                                               \
                                                                                                \
            halfx ## I  = _mm256_hadd_ps(_mm512_extractf32x8_ps(resx3, 0),                      \
                                         _mm512_extractf32x8_ps(resx3, 1));                     \
        }                                                                                       \

                XSIMD_AVX512_HADDP_STEP2(0, res0, res1, res2, res3);
                XSIMD_AVX512_HADDP_STEP2(1, res4, res5, res6, res7);

#undef XSIMD_AVX512_HADDP_STEP2

                auto concat = _mm512_castps256_ps512(halfx0);
                concat = _mm512_insertf32x8(concat, halfx1, 1);
                return concat;
    }
    template <class A>
    batch<double, A> haddp(batch<double, A> const *row, requires<avx512f>) {
#define step1(I, a, b)                                                   \
        batch<double, avx512f> res ## I;                                           \
        {                                                                    \
            auto tmp1 = _mm512_shuffle_f64x2(a, b, _MM_SHUFFLE(1, 0, 1, 0)); \
            auto tmp2 = _mm512_shuffle_f64x2(a, b, _MM_SHUFFLE(3, 2, 3, 2)); \
            res ## I = _mm512_add_pd(tmp1, tmp2);                                        \
        }                                                                    \

                step1(1, row[0], row[2]);
                step1(2, row[4], row[6]);
                step1(3, row[1], row[3]);
                step1(4, row[5], row[7]);

#undef step1

                auto tmp5 = _mm512_shuffle_f64x2(res1, res2, _MM_SHUFFLE(2, 0, 2, 0));
                auto tmp6 = _mm512_shuffle_f64x2(res1, res2, _MM_SHUFFLE(3, 1, 3, 1));

                auto resx1 = _mm512_add_pd(tmp5, tmp6);

                auto tmp7 = _mm512_shuffle_f64x2(res3, res4, _MM_SHUFFLE(2, 0, 2, 0));
                auto tmp8 = _mm512_shuffle_f64x2(res3, res4, _MM_SHUFFLE(3, 1, 3, 1));

                auto resx2 = _mm512_add_pd(tmp7, tmp8);

                auto tmpx = _mm512_shuffle_pd(resx1, resx2, 0b00000000);
                auto tmpy = _mm512_shuffle_pd(resx1, resx2, 0b11111111);

                return tmpx + tmpy;
    }

    // isnan
    template<class A> batch_bool<float, A> isnan(batch<float, A> const& self, requires<avx512f>) {
                return _mm512_cmp_ps_mask(self, self, _CMP_UNORD_Q);
    }
    template<class A> batch_bool<double, A> isnan(batch<double, A> const& self, requires<avx512f>) {
                return _mm512_cmp_pd_mask(self, self, _CMP_UNORD_Q);
    }

    // le
    template<class A> batch_bool<float, A> le(batch<float, A> const& self, batch<float, A> const& other, requires<avx512f>) {
      return _mm512_cmp_ps_mask(self, other, _CMP_LE_OQ);
    }
    template<class A> batch_bool<double, A> le(batch<double, A> const& self, batch<double, A> const& other, requires<avx512f>) {
      return _mm512_cmp_pd_mask(self, other, _CMP_LE_OQ);
    }

    // load_aligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> load_aligned(T const* mem, convert<T>, requires<avx512f>) {
      return _mm512_load_si512((__m512i const*)mem);
    }
    template<class A> batch<float, A> load_aligned(float const* mem, convert<float>, requires<avx512f>) {
      return _mm512_load_ps(mem);
    }
    template<class A> batch<double, A> load_aligned(double const* mem, convert<double>, requires<avx512f>) {
      return _mm512_load_pd(mem);
    }

    // load_complex
    template<class A> batch<std::complex<float>, A> load_complex(batch<float, A> const& hi, batch<float, A> const& lo, requires<avx512f>) {
        __m512i real_idx = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        __m512i imag_idx = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        auto real = _mm512_permutex2var_ps(hi, real_idx, lo);
        auto imag = _mm512_permutex2var_ps(hi, imag_idx, lo);
        return {real, imag};
    }
    template<class A> batch<std::complex<double>, A> load_complex(batch<double,A> const& hi, batch<double,A> const& lo, requires<avx512f>) {
        __m512i real_idx = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
        __m512i imag_idx = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
        auto real = _mm512_permutex2var_pd(hi, real_idx, lo);
        auto imag = _mm512_permutex2var_pd(hi, imag_idx, lo);
        return {real, imag};
    }

    // load_unaligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> load_unaligned(T const* mem, convert<T>, requires<avx512f>) {
      return _mm512_loadu_si512((__m512i const*)mem);
    }
    template<class A> batch<float, A> load_unaligned(float const* mem, convert<float>, requires<avx512f>){
      return _mm512_loadu_ps(mem);
    }
    template<class A> batch<double, A> load_unaligned(double const* mem, convert<double>, requires<avx512f>){
      return _mm512_loadu_pd(mem);
    }

    // lt
    template<class A> batch_bool<float, A> lt(batch<float, A> const& self, batch<float, A> const& other, requires<avx512f>) {
      return _mm512_cmp_ps_mask(self, other, _CMP_LT_OQ);
    }
    template<class A> batch_bool<double, A> lt(batch<double, A> const& self, batch<double, A> const& other, requires<avx512f>) {
      return _mm512_cmp_pd_mask(self, other, _CMP_LT_OQ);
    }

    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> lt(batch<T, A> const& self, batch<T, A> const& other, requires<avx512f>) {
      using register_type = typename batch_bool<T, A>::register_type;
        switch(sizeof(T)) {
          case 4: return (register_type)_mm512_cmplt_epu32_mask(self, other);
          case 8: return (register_type)_mm512_cmplt_epu64_mask(self, other);
          default: assert(false && "unsupported vector / arch combination");
        }
    }

    // max
    template<class A> batch<float, A> max(batch<float, A> const& self, batch<float, A> const& other, requires<avx512f>) {
      return _mm512_max_ps(self, other);
    }
    template<class A> batch<double, A> max(batch<double, A> const& self, batch<double, A> const& other, requires<avx512f>) {
      return _mm512_max_pd(self, other);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> max(batch<T, A> const& self, batch<T, A> const& other, requires<avx512f>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 4: return _mm512_max_epi32(self, other);
          case 8: return _mm512_max_epi64(self, other);
          default: return detail::fwd_to_avx([](__m256i s, __m256i o) { return max(batch<T, avx2>(s), batch<T, avx2>(o)); }, self, other);
        }
      }
      else {
        switch(sizeof(T)) {
          case 4: return _mm512_max_epu32(self, other);
          case 8: return _mm512_max_epi32(self, other);
          default: return detail::fwd_to_avx([](__m256i s, __m256i o) { return max(batch<T, avx2>(s), batch<T, avx2>(o)); }, self, other);
        }
      }
    }

    // min
    template<class A> batch<float, A> min(batch<float, A> const& self, batch<float, A> const& other, requires<avx512f>) {
      return _mm512_min_ps(self, other);
    }
    template<class A> batch<double, A> min(batch<double, A> const& self, batch<double, A> const& other, requires<avx512f>) {
      return _mm512_min_pd(self, other);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> min(batch<T, A> const& self, batch<T, A> const& other, requires<avx512f>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 4: return _mm512_min_epi32(self, other);
          case 8: return _mm512_min_epi64(self, other);
          default: return detail::fwd_to_avx([](__m256i s, __m256i o) { return min(batch<T, avx2>(s), batch<T, avx2>(o)); }, self, other);
        }
      }
      else {
        switch(sizeof(T)) {
          case 4: return _mm512_min_epu32(self, other);
          case 8: return _mm512_min_epi32(self, other);
          default: return detail::fwd_to_avx([](__m256i s, __m256i o) { return min(batch<T, avx2>(s), batch<T, avx2>(o)); }, self, other);
        }
      }
    }

    // mul
    template<class A> batch<float, A> mul(batch<float, A> const& self, batch<float, A> const& other, requires<avx512f>) {
      return _mm512_mul_ps(self, other);
    }
    template<class A> batch<double, A> mul(batch<double, A> const& self, batch<double, A> const& other, requires<avx512f>) {
      return _mm512_mul_pd(self, other);
    }

    // nearbyint
    template<class A> batch<float, A> nearbyint(batch<float, A> const& self, requires<avx512f>) {
      return _mm512_roundscale_round_ps(self, _MM_FROUND_TO_NEAREST_INT, _MM_FROUND_CUR_DIRECTION);
    }
    template<class A> batch<double, A> nearbyint(batch<double, A> const& self, requires<avx512f>) {
      return _mm512_roundscale_round_pd(self, _MM_FROUND_TO_NEAREST_INT, _MM_FROUND_CUR_DIRECTION);
    }

    // neg
    template<class A, class T>
    batch<T, A> neg(batch<T, A> const& self, requires<avx512f>) {
      return 0 - self;
    }

    // neq
    template<class A> batch_bool<float, A> neq(batch<float, A> const& self, batch<float, A> const& other, requires<avx512f>) {
      return _mm512_cmp_ps_mask(self, other, _CMP_NEQ_OQ);
    }
    template<class A> batch_bool<double, A> neq(batch<double, A> const& self, batch<double, A> const& other, requires<avx512f>) {
      return _mm512_cmp_pd_mask(self, other, _CMP_NEQ_OQ);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> neq(batch<T, A> const& self, batch<T, A> const& other, requires<avx512f>) {
        return ~(self == other);
    }

    template<class A, class T>
    batch_bool<T, A> neq(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires<avx512f>) {
      using register_type = typename batch_bool<T, A>::register_type;
        return register_type(self.data != other.data);
    }

    // sadd
    template<class A> batch<float, A> sadd(batch<float, A> const& self, batch<float, A> const& other, requires<avx512f>) {
      return add(self, other); // no saturated arithmetic on floating point numbers
    }
    template<class A> batch<double, A> sadd(batch<double, A> const& self, batch<double, A> const& other, requires<avx512f>) {
      return add(self, other); // no saturated arithmetic on floating point numbers
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> sadd(batch<T, A> const& self, batch<T, A> const& other, requires<avx512f>) {
      using register_type = typename batch_bool<T, A>::register_type;
      if(std::is_signed<T>::value) {
        auto mask = other < 0;
        auto self_pos_branch = min(std::numeric_limits<T>::max() - other, self);
        auto self_neg_branch = max(std::numeric_limits<T>::min() - other, self);
        return other + select(mask, self_neg_branch, self_pos_branch);
      }
      else {
        const auto diffmax = std::numeric_limits<T>::max() - self;
        const auto mindiff = min(diffmax, other);
        return self + mindiff;
      }
    }

    // select
    template<class A> batch<float, A> select(batch_bool<float, A> const& cond, batch<float, A> const& true_br, batch<float, A> const& false_br, requires<avx512f>) {
                return _mm512_mask_blend_ps(cond, false_br, true_br);
    }
    template<class A> batch<double, A> select(batch_bool<double, A> const& cond, batch<double, A> const& true_br, batch<double, A> const& false_br, requires<avx512f>) {
                return _mm512_mask_blend_pd(cond, false_br, true_br);
    }

    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> select(batch_bool<T, A> const& cond, batch<T, A> const& true_br, batch<T, A> const& false_br, requires<avx512f>) {
      switch(sizeof(T)) {
#if 0
        case 1: {
          __m256i cond_low, cond_hi;
          detail::split_avx512(cond, cond_low, cond_hi);

          __m256i true_low, true_hi;
          detail::split_avx512(true_br, true_low, true_hi);

          __m256i false_low, false_hi;
          detail::split_avx512(false_br, false_low, false_hi);

          __m256i res_low = select(batch_bool<T, avx2>(cond_low), batch<T, avx2>(true_low), batch<T, avx2>(false_low), avx2{});
          __m256i res_hi = select(batch_bool<T, avx2>(cond_hi), batch<T, avx2>(true_hi), batch<T, avx2>(false_hi), avx2{});
          return detail::merge_avx(res_low, res_hi);
        }
#endif
        case 2: {
          __m512i mcond = _mm512_maskz_broadcastw_epi16((__mmask32)cond, _mm_set1_epi16(~0));
          __m256i cond_low, cond_hi;
          detail::split_avx512(mcond, cond_low, cond_hi);

          __m256i true_low, true_hi;
          detail::split_avx512(true_br, true_low, true_hi);

          __m256i false_low, false_hi;
          detail::split_avx512(false_br, false_low, false_hi);

          __m256i res_low = select(batch_bool<T, avx2>(cond_low), batch<T, avx2>(true_low), batch<T, avx2>(false_low), avx2{});
          __m256i res_hi = select(batch_bool<T, avx2>(cond_hi), batch<T, avx2>(true_hi), batch<T, avx2>(false_hi), avx2{});
          return detail::merge_avx(res_low, res_hi);
        }
        case 4: return _mm512_mask_blend_epi32(cond, false_br, true_br);
        case 8: return _mm512_mask_blend_epi64(cond, false_br, true_br);
        default: assert(false && "unsupported arch/type combination");
      };
    }
    template<class A, class T, bool... Values, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> select(batch_bool_constant<batch<T, A>, Values...> const&, batch<T, A> const& true_br, batch<T, A> const& false_br, requires<avx512f>) {
      return select(batch_bool<T, A>{Values...}, true_br, false_br, avx512f{});
    }



    // set
    template<class A>
    batch<float, A> set(batch<float, A> const&, requires<avx512f>, float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15) {
      return _mm512_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
    }

    template<class A>
    batch<double, A> set(batch<double, A> const&, requires<avx512f>, double v0, double v1, double v2, double v3, double v4, double v5, double v6, double v7) {
      return _mm512_setr_pd(v0, v1, v2, v3, v4, v5, v6, v7);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> set(batch<T, A> const&, requires<avx512f>, T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7) {
      return _mm512_set_epi64(v7, v6, v5, v4, v3, v2, v1, v0);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> set(batch<T, A> const&, requires<avx512f>, T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15) {
      return _mm512_setr_epi32(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> set(batch<T, A> const&, requires<avx512f>, T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15,
        T v16, T v17, T v18, T v19, T v20, T v21, T v22, T v23, T v24, T v25, T v26, T v27, T v28, T v29, T v30, T v31) {
      return _mm512_setr_epi32(
          v0  | (v1  << 8*sizeof(T)),
          v2  | (v3  << 8*sizeof(T)),
          v4  | (v5  << 8*sizeof(T)),
          v6  | (v7  << 8*sizeof(T)),
          v8  | (v9  << 8*sizeof(T)),
          v10 | (v11 << 8*sizeof(T)),
          v12 | (v13 << 8*sizeof(T)),
          v14 | (v15 << 8*sizeof(T)),
          v16 | (v17 << 8*sizeof(T)),
          v18 | (v19 << 8*sizeof(T)),
          v20 | (v21 << 8*sizeof(T)),
          v22 | (v23 << 8*sizeof(T)),
          v24 | (v25 << 8*sizeof(T)),
          v26 | (v27 << 8*sizeof(T)),
          v28 | (v29 << 8*sizeof(T)),
          v30 | (v31 << 8*sizeof(T))
          );

    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> set(batch<T, A> const&, requires<avx512f>, T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15,
      T v16, T v17, T v18, T v19, T v20, T v21, T v22, T v23, T v24, T v25, T v26, T v27, T v28, T v29, T v30, T v31,
      T v32, T v33, T v34, T v35, T v36, T v37, T v38, T v39, T v40, T v41, T v42, T v43, T v44, T v45, T v46, T v47,
      T v48, T v49, T v50, T v51, T v52, T v53, T v54, T v55, T v56, T v57, T v58, T v59, T v60, T v61, T v62, T v63
      ) {
      return _mm512_setr_epi32(
          v0  | (v1  << 8*sizeof(T)) | (v2  << 16*sizeof(T)) | (v3  << 24*sizeof(T)),
          v4  | (v5  << 8*sizeof(T)) | (v6  << 16*sizeof(T)) | (v7  << 24*sizeof(T)),
          v8  | (v9  << 8*sizeof(T)) | (v10 << 16*sizeof(T)) | (v11 << 24*sizeof(T)),
          v12 | (v13 << 8*sizeof(T)) | (v14 << 16*sizeof(T)) | (v15 << 24*sizeof(T)),
          v16 | (v17 << 8*sizeof(T)) | (v18 << 16*sizeof(T)) | (v19 << 24*sizeof(T)),
          v20 | (v21 << 8*sizeof(T)) | (v22 << 16*sizeof(T)) | (v23 << 24*sizeof(T)),
          v24 | (v25 << 8*sizeof(T)) | (v26 << 16*sizeof(T)) | (v27 << 24*sizeof(T)),
          v28 | (v29 << 8*sizeof(T)) | (v30 << 16*sizeof(T)) | (v31 << 24*sizeof(T)),
          v32 | (v33 << 8*sizeof(T)) | (v34 << 16*sizeof(T)) | (v35 << 24*sizeof(T)),
          v36 | (v37 << 8*sizeof(T)) | (v38 << 16*sizeof(T)) | (v39 << 24*sizeof(T)),
          v40 | (v41 << 8*sizeof(T)) | (v42 << 16*sizeof(T)) | (v43 << 24*sizeof(T)),
          v44 | (v45 << 8*sizeof(T)) | (v46 << 16*sizeof(T)) | (v47 << 24*sizeof(T)),
          v48 | (v49 << 8*sizeof(T)) | (v50 << 16*sizeof(T)) | (v51 << 24*sizeof(T)),
          v52 | (v53 << 8*sizeof(T)) | (v54 << 16*sizeof(T)) | (v55 << 24*sizeof(T)),
          v56 | (v57 << 8*sizeof(T)) | (v58 << 16*sizeof(T)) | (v59 << 24*sizeof(T)),
          v60 | (v61 << 8*sizeof(T)) | (v62 << 16*sizeof(T)) | (v63 << 24*sizeof(T))
          );
    }

    template<class A, class T, class... Values>
    batch_bool<T, A> set(batch_bool<T, A> const&, requires<avx512f>, Values... values) {
      static_assert(sizeof...(Values) == batch_bool<T, A>::size, "consistent init");
      using register_type = typename batch_bool<T, A>::register_type;
      register_type r = 0;
      unsigned shift = 0;
      (void)std::initializer_list<register_type>{(r|=register_type(values?1:0) << (shift++))...};
      return r;
    }

    // sqrt
    template<class A> batch<float, A> sqrt(batch<float, A> const& val, requires<avx512f>) {
      return _mm512_sqrt_ps(val);
    }
    template<class A> batch<double, A> sqrt(batch<double, A> const& val, requires<avx512f>) {
      return _mm512_sqrt_pd(val);
    }

    // ssub
    template<class A> batch<float, A> ssub(batch<float, A> const& self, batch<float, A> const& other, requires<avx512f>) {
      return _mm512_sub_ps(self, other); // no saturated arithmetic on floating point numbers
    }
    template<class A> batch<double, A> ssub(batch<double, A> const& self, batch<double, A> const& other, requires<avx512f>) {
      return _mm512_sub_pd(self, other); // no saturated arithmetic on floating point numbers
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> ssub(batch<T, A> const& self, batch<T, A> const& other, requires<avx512f>) {
      if(std::is_signed<T>::value) {
         return sadd(self, -other);
      }
      else {
        const auto diff = min(self, other);
        return self - diff;
      }
    }

    // store_aligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_aligned(T *mem, batch<T, A> const& self, requires<avx512f>) {
      return _mm512_store_si512((__m512i *)mem, self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_aligned(T *mem, batch_bool<T, A> const& self, requires<avx512f>) {
      return _mm512_store_si512((__m512i *)mem, self);
    }
    template<class A> void store_aligned(float *mem, batch<float, A> const& self, requires<avx512f>) {
      return _mm512_store_ps(mem, self);
    }
    template<class A> void store_aligned(double *mem, batch<double, A> const& self, requires<avx512f>) {
      return _mm512_store_pd(mem, self);
    }

    // store_unaligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_unaligned(T *mem, batch<T, A> const& self, requires<avx512f>) {
      return _mm512_storeu_si512((__m512i *)mem, self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_unaligned(T *mem, batch_bool<T, A> const& self, requires<avx512f>) {
      return _mm512_storeu_si512((__m512i *)mem, self);
    }
    template<class A> void store_unaligned(float *mem, batch<float, A> const& self, requires<avx512f>) {
      return _mm512_storeu_ps(mem, self);
    }
    template<class A> void store_unaligned(double *mem, batch<double, A> const& self, requires<avx512f>) {
      return _mm512_storeu_pd(mem, self);
    }

    // sub
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> sub(batch<T, A> const& self, batch<T, A> const& other, requires<avx512f>) {
      switch(sizeof(T)) {
        case 1: return detail::fwd_to_avx([](__m256i s, __m256i o) { return sub(batch<T, avx2>(s), batch<T, avx2>(o)); }, self, other);
        case 2: return detail::fwd_to_avx([](__m256i s, __m256i o) { return sub(batch<T, avx2>(s), batch<T, avx2>(o)); }, self, other);
        case 4: return _mm512_sub_epi32(self, other);
        case 8: return _mm512_sub_epi64(self, other);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }
    template<class A> batch<float, A> sub(batch<float, A> const& self, batch<float, A> const& other, requires<avx512f>) {
      return _mm512_sub_ps(self, other);
    }
    template<class A> batch<double, A> sub(batch<double, A> const& self, batch<double, A> const& other, requires<avx512f>) {
      return _mm512_sub_pd(self, other);
    }

    // to_float
    template<class A>
    batch<float, A> to_float(batch<int32_t, A> const& self, requires<avx512f>) {
      return _mm512_cvtepi32_ps(self);
    }
    template<class A>
    batch<double, A> to_float(batch<int64_t, A> const& self, requires<avx512f>) {
      // FIXME: call _mm_cvtepi64_pd
      alignas(A::alignment()) int64_t buffer[batch<int64_t, A>::size];
      self.store_aligned(&buffer[0]);
      return {(double)buffer[0], (double)buffer[1], (double)buffer[2], (double)buffer[3],};
    }

    // to_int
    template<class A>
    batch<int32_t, A> to_int(batch<float, A> const& self, requires<avx512f>) {
      return _mm512_cvttps_epi32(self);
    }

    template<class A>
    batch<int64_t, A> to_int(batch<double, A> const& self, requires<avx512f>) {
      // FIXME: call _mm_cvttpd_epi64
      alignas(A::alignment()) double buffer[batch<double, A>::size];
      self.store_aligned(&buffer[0]);
      return {(int64_t)buffer[0], (int64_t)buffer[1], (int64_t)buffer[2], (int64_t)buffer[3]};
    }

    // trunc
    template<class A> batch<float, A> trunc(batch<float, A> const& self, requires<avx512f>) {
      return _mm512_roundscale_round_ps(self, _MM_FROUND_TO_ZERO, _MM_FROUND_CUR_DIRECTION);
    }
    template<class A> batch<double, A> trunc(batch<double, A> const& self, requires<avx512f>) {
      return _mm512_roundscale_round_pd(self, _MM_FROUND_TO_ZERO, _MM_FROUND_CUR_DIRECTION);
    }

    // zip_hi
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> zip_hi(batch<T, A> const& self, batch<T, A> const& other, requires<avx512f>) {
      switch(sizeof(T)) {
        case 1: return _mm512_unpackhi_epi8(self, other);
        case 2: return _mm512_unpackhi_epi16(self, other);
        case 4: return _mm512_unpackhi_epi32(self, other);
        case 8: return _mm512_unpackhi_epi64(self, other);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }
    template<class A> batch<float, A> zip_hi(batch<float, A> const& self, batch<float, A> const& other, requires<avx512f>) {
      return _mm512_unpackhi_ps(self, other);
    }
    template<class A> batch<double, A> zip_hi(batch<double, A> const& self, batch<double, A> const& other, requires<avx512f>) {
      return _mm512_unpackhi_pd(self, other);
    }

    // zip_lo
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> zip_lo(batch<T, A> const& self, batch<T, A> const& other, requires<avx512f>) {
      switch(sizeof(T)) {
        case 1: return _mm512_unpacklo_epi8(self, other);
        case 2: return _mm512_unpacklo_epi16(self, other);
        case 4: return _mm512_unpacklo_epi32(self, other);
        case 8: return _mm512_unpacklo_epi64(self, other);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }
    template<class A> batch<float, A> zip_lo(batch<float, A> const& self, batch<float, A> const& other, requires<avx512f>) {
      return _mm512_unpacklo_ps(self, other);
    }
    template<class A> batch<double, A> zip_lo(batch<double, A> const& self, batch<double, A> const& other, requires<avx512f>) {
      return _mm512_unpacklo_pd(self, other);
    }

  }

}

#endif
