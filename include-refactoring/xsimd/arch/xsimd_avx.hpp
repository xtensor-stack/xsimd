#ifndef XSIMD_AVX_HPP
#define XSIMD_AVX_HPP

#include "../types/xsimd_avx_register.hpp"


namespace xsimd {

  namespace kernel {
    using namespace types;

    namespace detail {
    void split_avx(__m256i val, __m128i& low, __m128i& high) {
        low =_mm256_castsi256_si128(val);
        high =_mm256_extractf128_si256(val, 1);
      }
      __m256i merge_sse(__m128i low, __m128i high) {
        return _mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1);
      }
    }

    // abs
    template<class A> batch<float, A> abs(batch<float, A> const& self, requires<avx>) {
      __m256 sign_mask = _mm256_set1_ps(-0.f);  // -0.f = 1 << 31
      return _mm256_andnot_ps(sign_mask, self);
    }
    template<class A> batch<double, A> abs(batch<double, A> const& self, requires<avx>) {
      __m256d sign_mask = _mm256_set1_pd(-0.f);  // -0.f = 1 << 31
      return _mm256_andnot_pd(sign_mask, self);
    }

    // add
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> add(batch<T, A> const&, batch<T, A> const&, requires<avx>) {
      static_assert(std::is_same<A, sse>::value, "unsupported arch / op combination");
    }
    template<class A> batch<float, A> add(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_add_ps(self, other);
    }
    template<class A> batch<double, A> add(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_add_pd(self, other);
    }

    // all
    template<class A> bool all(batch<float, A> const& self, requires<avx>) {
      return _mm256_testc_ps(self, batch_bool<float, A>(true)) != 0;
    }

    // any
    template<class A> bool any(batch<float, A> const& self, requires<avx>) {
      return !_mm256_testz_ps(self, self);
    }

    // bitwise_and
    template<class A> batch<float, A> bitwise_and(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_and_ps(self, other);
    }
    template<class A> batch<double, A> bitwise_and(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_and_pd(self, other);
    }

    template<class A> batch_bool<float, A> bitwise_and(batch_bool<float, A> const& self, batch_bool<float, A> const& other, requires<avx>) {
      return _mm256_and_ps(self, other);
    }
    template<class A> batch_bool<double, A> bitwise_and(batch_bool<double, A> const& self, batch_bool<double, A> const& other, requires<avx>) {
      return _mm256_and_pd(self, other);
    }

    // bitwise_or
    template<class A> batch<float, A> bitwise_or(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_or_ps(self, other);
    }
    template<class A> batch<double, A> bitwise_or(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_or_pd(self, other);
    }
    template<class A> batch_bool<float, A> bitwise_or(batch_bool<float, A> const& self, batch_bool<float, A> const& other, requires<avx>) {
      return _mm256_or_ps(self, other);
    }
    template<class A> batch_bool<double, A> bitwise_or(batch_bool<double, A> const& self, batch_bool<double, A> const& other, requires<avx>) {
      return _mm256_or_pd(self, other);
    }

    // bitwise_xor
    template<class A> batch<float, A> bitwise_xor(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_xor_ps(self, other);
    }
    template<class A> batch<double, A> bitwise_xor(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_xor_pd(self, other);
    }
    template<class A> batch_bool<float, A> bitwise_xor(batch_bool<float, A> const& self, batch_bool<float, A> const& other, requires<avx>) {
      return _mm256_xor_ps(self, other);
    }
    template<class A> batch_bool<double, A> bitwise_xor(batch_bool<double, A> const& self, batch_bool<double, A> const& other, requires<avx>) {
      return _mm256_xor_pd(self, other);
    }
    // bitwise_cast
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<float, A> bitwise_cast(batch<T, A> const& self, batch<float, A> const &, requires<avx>) {
      return _mm256_castsi256_ps(self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<double, A> bitwise_cast(batch<T, A> const& self, batch<double, A> const &, requires<avx>) {
      return _mm256_castsi256_pd(self);
    }
    template<class A, class T, class Tp, class=typename std::enable_if<std::is_integral<typename std::common_type<T, Tp>::type>::value, void>::type>
    batch<Tp, A> bitwise_cast(batch<T, A> const& self, batch<Tp, A> const &, requires<avx>) {
      return batch<Tp, A>(self.data);
    }
    template<class A>
    batch<double, A> bitwise_cast(batch<float, A> const& self, batch<double, A> const &, requires<avx>) {
      return _mm256_castps_pd(self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_cast(batch<float, A> const& self, batch<T, A> const &, requires<avx>) {
      return _mm256_castps_si256(self);
    }
    template<class A>
    batch<float, A> bitwise_cast(batch<double, A> const& self, batch<float, A> const &, requires<avx>) {
      return _mm256_castpd_ps(self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_cast(batch<double, A> const& self, batch<T, A> const &, requires<avx>) {
      return _mm256_castpd_si128(self);
    }

    // bitwise_not
    template<class A> batch<float, A> bitwise_not(batch<float, A> const& self, requires<avx>) {
      return _mm256_xor_ps(self, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
    }
    template <class A>
    batch<double, A> bitwise_not(batch<double, A> const &self, requires<avx>) {
      return _mm256_xor_pd(self, _mm256_castsi256_pd(_mm256_set1_epi32(-1)));
    }
    template<class A> batch_bool<float, A> bitwise_not(batch_bool<float, A> const& self, requires<avx>) {
      return _mm256_xor_ps(self, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
    }
    template <class A>
    batch_bool<double, A> bitwise_not(batch_bool<double, A> const &self, requires<avx>) {
      return _mm256_xor_pd(self, _mm256_castsi256_pd(_mm256_set1_epi32(-1)));
    }

    // bool_cast
    template<class A> batch_bool<int32_t, A> bool_cast(batch_bool<float, A> const& self, requires<avx>) {
        return _mm256_castps_si256(self);
    }
    template<class A> batch_bool<float, A> bool_cast(batch_bool<int32_t, A> const& self, requires<avx>) {
        return _mm256_castsi256_ps(self);
    }
    template<class A> batch_bool<int64_t, A> bool_cast(batch_bool<double, A> const& self, requires<avx>) {
        return _mm256_castpd_si256(self);
    }
    template<class A> batch_bool<double, A> bool_cast(batch_bool<int64_t, A> const& self, requires<avx>) {
        return _mm256_castsi256_pd(self);
    }

    // broadcast
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> broadcast(T val, requires<avx>) {
      switch(sizeof(T)) {
        case 1: return _mm256_set1_epi8(val);
        case 2: return _mm256_set1_epi16(val);
        case 4: return _mm256_set1_epi32(val);
        case 8: return _mm256_set1_epi64x(val);
        default: assert(false && "unsupported"); return {};
      }
    }
    template<class A> batch<float, A> broadcast(float val, requires<avx>) {
      return _mm256_set1_ps(val);
    }
    template<class A> batch<double, A> broadcast(double val, requires<avx>) {
      return _mm256_set1_pd(val);
    }

    // complex_low
    template<class A> batch<float, A> complex_low(batch<std::complex<float>, A> const& self, requires<avx>) {
      return _mm256_unpacklo_ps(self.real(), self.imag());
    }
    template<class A> batch<double, A> complex_low(batch<std::complex<double>, A> const& self, requires<avx>) {
      return _mm256_unpacklo_pd(self.real(), self.imag());
    }

    // complex_high
    template<class A> batch<float, A> complex_high(batch<std::complex<float>, A> const& self, requires<avx>) {
      return _mm256_unpackhi_ps(self.real(), self.imag());
    }
    template<class A> batch<double, A> complex_high(batch<std::complex<double>, A> const& self, requires<avx>) {
      return _mm256_unpackhi_pd(self.real(), self.imag());
    }

    // div
    template<class A> batch<float, A> div(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_div_ps(self, other);
    }
    template<class A> batch<double, A> div(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_div_pd(self, other);
    }

    // eq
    template<class A> batch_bool<float, A> eq(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_cmpeq_ps(self, other);
    }
    template<class A> batch_bool<double, A> eq(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_cmpeq_pd(self, other);
    }
    template<class A> batch_bool<float, A> eq(batch_bool<float, A> const& self, batch_bool<float, A> const& other, requires<avx>) {
      return  _mm256_castsi128_ps(_mm256_cmpeq_epi32(_mm256_castps_si128(self), _mm256_castps_si128(other)));
    }
    template<class A> batch_bool<double, A> eq(batch_bool<double, A> const& self, batch_bool<double, A> const& other, requires<avx>) {
      return  _mm256_castsi128_pd(_mm256_cmpeq_epi32(_mm256_castpd_si128(self), _mm256_castpd_si128(other)));
    }

    // ge
    template<class A> batch_bool<float, A> ge(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_cmpge_ps(self, other);
    }
    template<class A> batch_bool<double, A> ge(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_cmpge_pd(self, other);
    }

    // gt
    template<class A> batch_bool<float, A> gt(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_cmpgt_ps(self, other);
    }
    template<class A> batch_bool<double, A> gt(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_cmpgt_pd(self, other);
    }

    // hadd
    template<class A> float hadd(batch<float, A> const& rhs, requires<avx>) {
                // Warning about _mm256_hadd_ps:
                // _mm256_hadd_ps(a,b) gives
                // (a0+a1,a2+a3,b0+b1,b2+b3,a4+a5,a6+a7,b4+b5,b6+b7). Hence we can't
                // rely on a naive use of this method
                // rhs = (x0, x1, x2, x3, x4, x5, x6, x7)
                // tmp = (x4, x5, x6, x7, x0, x1, x2, x3)
                __m256 tmp = _mm256_permute2f128_ps(rhs, rhs, 1);
                // tmp = (x4+x0, x5+x1, x6+x2, x7+x3, x0+x4, x1+x5, x2+x6, x3+x7)
                tmp = _mm256_add_ps(rhs, tmp);
                // tmp = (x4+x0+x5+x1, x6+x2+x7+x3, -, -, -, -, -, -)
                tmp = _mm256_hadd_ps(tmp, tmp);
                // tmp = (x4+x0+x5+x1+x6+x2+x7+x3, -, -, -, -, -, -, -)
                tmp = _mm256_hadd_ps(tmp, tmp);
                return _mm_cvtss_f32(_mm256_extractf128_ps(tmp, 0));

    }
    template <class A>
    double hadd(batch<double, A> const &rhs, requires<avx>) {
                // rhs = (x0, x1, x2, x3)
                // tmp = (x2, x3, x0, x1)
                __m256d tmp = _mm256_permute2f128_pd(rhs, rhs, 1);
                // tmp = (x2+x0, x3+x1, -, -)
                tmp = _mm256_add_pd(rhs, tmp);
                // tmp = (x2+x0+x3+x1, -, -, -)
                tmp = _mm256_hadd_pd(tmp, tmp);
                return _mm_cvtsd_f64(_mm256_extractf128_pd(tmp, 0));
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    T hadd(batch<T, A> const& self, requires<avx>) {
      __m128i low, high;
      detail::split_avx(self, low, high);
      __m128i res_low = _mm_sub_epi32(_mm_setzero_si128(), low);
      __m128i res_high = _mm_sub_epi32(_mm_setzero_si128(), high);
      return detail::merge_sse(res_low, res_high);
    }

    // haddp
    template<class A> batch<float, A> haddp(batch<float, A> const* row, requires<avx>) {
                // row = (a,b,c,d,e,f,g,h)
                // tmp0 = (a0+a1, a2+a3, b0+b1, b2+b3, a4+a5, a6+a7, b4+b5, b6+b7)
                __m256 tmp0 = _mm256_hadd_ps(row[0], row[1]);
                // tmp1 = (c0+c1, c2+c3, d1+d2, d2+d3, c4+c5, c6+c7, d4+d5, d6+d7)
                __m256 tmp1 = _mm256_hadd_ps(row[2], row[3]);
                // tmp1 = (a0+a1+a2+a3, b0+b1+b2+b3, c0+c1+c2+c3, d0+d1+d2+d3,
                // a4+a5+a6+a7, b4+b5+b6+b7, c4+c5+c6+c7, d4+d5+d6+d7)
                tmp1 = _mm256_hadd_ps(tmp0, tmp1);
                // tmp0 = (e0+e1, e2+e3, f0+f1, f2+f3, e4+e5, e6+e7, f4+f5, f6+f7)
                tmp0 = _mm256_hadd_ps(row[4], row[5]);
                // tmp2 = (g0+g1, g2+g3, h0+h1, h2+h3, g4+g5, g6+g7, h4+h5, h6+h7)
                __m256 tmp2 = _mm256_hadd_ps(row[6], row[7]);
                // tmp2 = (e0+e1+e2+e3, f0+f1+f2+f3, g0+g1+g2+g3, h0+h1+h2+h3,
                // e4+e5+e6+e7, f4+f5+f6+f7, g4+g5+g6+g7, h4+h5+h6+h7)
                tmp2 = _mm256_hadd_ps(tmp0, tmp2);
                // tmp0 = (a0+a1+a2+a3, b0+b1+b2+b3, c0+c1+c2+c3, d0+d1+d2+d3,
                // e4+e5+e6+e7, f4+f5+f6+f7, g4+g5+g6+g7, h4+h5+h6+h7)
                tmp0 = _mm256_blend_ps(tmp1, tmp2, 0b11110000);
                // tmp1 = (a4+a5+a6+a7, b4+b5+b6+b7, c4+c5+c6+c7, d4+d5+d6+d7,
                // e0+e1+e2+e3, f0+f1+f2+f3, g0+g1+g2+g3, h0+h1+h2+h3)
                tmp1 = _mm256_permute2f128_ps(tmp1, tmp2, 0x21);
                return _mm256_add_ps(tmp0, tmp1);
    }
    template <class A>
    batch<double, A> haddp(batch<double, A> const *row, requires<avx>) {
                // row = (a,b,c,d)
                // tmp0 = (a0+a1, b0+b1, a2+a3, b2+b3)
                __m256d tmp0 = _mm256_hadd_pd(row[0], row[1]);
                // tmp1 = (c0+c1, d0+d1, c2+c3, d2+d3)
                __m256d tmp1 = _mm256_hadd_pd(row[2], row[3]);
                // tmp2 = (a0+a1, b0+b1, c2+c3, d2+d3)
                __m256d tmp2 = _mm256_blend_pd(tmp0, tmp1, 0b1100);
                // tmp1 = (a2+a3, b2+b3, c2+c3, d2+d3)
                tmp1 = _mm256_permute2f128_pd(tmp0, tmp1, 0x21);
                return _mm256_add_pd(tmp1, tmp2);
    }

    // isnan
    template<class A> batch_bool<float, A> isnan(batch<float, A> const& self, requires<avx>) {
                return _mm256_cmp_ps(self, self, _CMP_UNORD_Q);
    }
    template<class A> batch_bool<double, A> isnan(batch<double, A> const& self, requires<avx>) {
                return _mm256_cmp_pd(self, self, _CMP_UNORD_Q);
    }

    // le
    template<class A> batch_bool<float, A> le(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_cmp_ps(self, other, _CMP_LE_OQ);
    }
    template<class A> batch_bool<double, A> le(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_cmp_pd(self, other, _CMP_LE_OQ);
    }

    // load_aligned
    template<class A> batch<float, A> load_aligned(float const* mem, convert<float>, requires<avx>) {
      return _mm256_load_ps(mem);
    }
    template<class A> batch<double, A> load_aligned(double const* mem, convert<double>, requires<avx>) {
      return _mm256_load_pd(mem);
    }

    // load_complex
    template<class A> batch<std::complex<float>, A> load_complex(batch<float> const& hi, batch<float> const& lo, requires<avx>) {
      return {_mm256_shuffle_ps(hi, lo, _MM_SHUFFLE(2, 0, 2, 0)), _mm256_shuffle_ps(hi, lo, _MM_SHUFFLE(3, 1, 3, 1))};
    }
    template<class A> batch<std::complex<double>, A> load_complex(batch<double> const& hi, batch<double> const& lo, requires<avx>) {
      return {_mm256_shuffle_pd(hi, lo, _MM_SHUFFLE2(0, 0)), _mm256_shuffle_pd(hi, lo, _MM_SHUFFLE2(1, 1))};
    }

    // load_unaligned
    template<class A> batch<float, A> load_unaligned(float const* mem, convert<float>, requires<avx>){
      return _mm256_loadu_ps(mem);
    }
    template<class A> batch<double, A> load_unaligned(double const* mem, convert<double>, requires<avx>){
      return _mm256_loadu_pd(mem);
    }

    // lt
    template<class A> batch_bool<float, A> lt(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_cmp_ps(self, other, _CMP_LT_OQ);
    }
    template<class A> batch_bool<double, A> lt(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_cmp_pd(self, other, _CMP_LT_OQ);
    }

    // max
    template<class A> batch<float, A> max(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_max_ps(self, other);
    }
    template<class A> batch<double, A> max(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_max_pd(self, other);
    }

    // min
    template<class A> batch<float, A> min(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_min_ps(self, other);
    }
    template<class A> batch<double, A> min(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_min_pd(self, other);
    }

    // mul
    template<class A> batch<float, A> mul(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_mul_ps(self, other);
    }
    template<class A> batch<double, A> mul(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_mul_pd(self, other);
    }

    // neg
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> neg(batch<T, A> const& self, requires<avx>) {
      return 0 - self;
    }
    template<class A> batch<float, A> neg(batch<float, A> const& self, requires<avx>) {
      return _mm256_xor_ps(self, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)));
    }
    template <class A>
    batch<double, A> neg(batch<double, A> const &self, requires<avx>) {
      return _mm256_xor_pd(self, _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000)));
    }

    // set
    template<class A, class... Values>
    batch<float, A> set(batch<float, A> const&, requires<avx>, Values... values) {
      static_assert(sizeof...(Values) == batch<float, A>::size, "consistent init");
      return _mm256_setr_ps(values...);
    }

    template<class A, class... Values>
    batch<double, A> set(batch<double, A> const&, requires<avx>, Values... values) {
      static_assert(sizeof...(Values) == batch<double, A>::size, "consistent init");
      return _mm256_setr_pd(values...);
    }

    // store_aligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_aligned(T *mem, batch<T, A> const& self, requires<avx2>) {
      return _mm256_store_si256((__m256i *)mem, self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_aligned(T *mem, batch_bool<T, A> const& self, requires<avx2>) {
      return _mm256_store_si256((__m256i *)mem, self);
    }
    template<class A> void store_aligned(float *mem, batch<float, A> const& self, requires<avx>) {
      return _mm256_store_ps(mem, self);
    }
    template<class A> void store_aligned(double *mem, batch<double, A> const& self, requires<avx>) {
      return _mm256_store_pd(mem, self);
    }

    // store_unaligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_unaligned(T *mem, batch<T, A> const& self, requires<avx2>) {
      return _mm256_storeu_si256((__m256i *)mem, self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_unaligned(T *mem, batch_bool<T, A> const& self, requires<avx2>) {
      return _mm256_storeu_si256((__m256i *)mem, self);
    }
    template<class A> void store_unaligned(float *mem, batch<float, A> const& self, requires<avx>) {
      return _mm256_storeu_ps(mem, self);
    }
    template<class A> void store_unaligned(double *mem, batch<double, A> const& self, requires<avx>) {
      return _mm256_storeu_pd(mem, self);
    }

  }

}

#endif
