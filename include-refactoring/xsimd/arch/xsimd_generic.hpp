#ifndef XSIMD_GENERIC_HPP
#define XSIMD_GENERIC_HPP

#include "../types/xsimd_generic_arch.hpp"
#include "../types/xsimd_utils.hpp"
#include "../math/xsimd_rem_pio2.hpp"
#include "./xsimd_constants.hpp"

#include <limits>
#include <tuple>


namespace xsimd {
  // Forward declaration. Should we put them in a separate file?
  template<class T, class A>
  batch<T, A> abs(batch<T, A> const& self);
  template<class T, class A>
  bool any(batch_bool<T, A> const& self);
  template<class T, class A>
  batch<T, A> bitofsign(batch<T, A> const& self);
  template<class B, class T, class A>
  B bitwise_cast(batch<T, A> const& self);
  template<class A>
  batch_bool<float, A> bool_cast(batch_bool<int32_t, A> const& self);
  template<class A>
  batch_bool<int32_t, A> bool_cast(batch_bool<float, A> const& self);
  template<class A>
  batch_bool<double, A> bool_cast(batch_bool<int64_t, A> const& self);
  template<class A>
  batch_bool<int64_t, A> bool_cast(batch_bool<double, A> const& self);
  template<class T, class A>
  batch<T, A> exp(batch<T, A> const& self);
  template<class T, class A>
  batch<T, A> fma(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z);
  template<class T, class A>
  batch<T, A> frexp(const batch<T, A>& x, const batch<as_integer_t<T>, A>& e);
  template <class T, class A, uint64_t... Coefs>
  batch<T, A> horner(const batch<T, A>& self);
  template<class T, class A>
  batch_bool<T, A> is_even(batch<T, A> const& self);
  template<class T, class A>
  batch_bool<T, A> is_flint(batch<T, A> const& self);
  template<class T, class A>
  batch_bool<T, A> is_odd(batch<T, A> const& self);
  template<class T, class A>
  batch_bool<T, A> isinf(batch<T, A> const& self);
  template<class T, class A>
  batch_bool<T, A> isnan(batch<T, A> const& self);
  template<class T, class A>
  batch<T, A> ldexp(const batch<T, A>& x, const batch<as_integer_t<T>, A>& e);
  template<class T, class A>
  batch<T, A> log(batch<T, A> const& self);
  template<class T, class A>
  batch<T, A> nearbyint(batch<T, A> const& self);
  template<class T, class A>
  batch<T, A> select(batch_bool<T, A> const&, batch<T, A> const& , batch<T, A> const& );
  template<class T, class A>
  batch<T, A> sign(batch<T, A> const& self);
  template<class T, class A>
  batch<T, A> signnz(batch<T, A> const& self);
  template<class T, class A>
  batch<T, A> sqrt(batch<T, A> const& self);
  template<class T, class A>
  batch<as_float_t<T>, A> to_float(batch<T, A> const& self);
  template<class T, class A>
  batch<as_integer_t<T>, A> to_int(batch<T, A> const& self);
  template<class T, class A>
  batch<T, A> trunc(batch<T, A> const& self);


  namespace kernel {

    namespace detail {
      // Generic conversion handling machinery. Each architecture must define
      // conversion function when such conversions exits in the form of
      // intrinsic. Then we use that information to automatically decide whether
      // to use scalar or vector conversion when doing load / store / batch_cast
      struct with_fast_conversion{};
      struct with_slow_conversion{};
        template<class A, class From, class To>
        class has_fast_conversion {
          template<class T0, class T1>
          static std::true_type get(decltype(kernel::conversion::fast(batch<T0, A>{}, batch<T1, A>{}, A{}))*);
          template<class T0, class T1>
          static std::false_type get(...);
          public:
          static constexpr bool value = decltype(get<From, To>(nullptr))::value;
        };
        template<class A, class From, class To>
        using conversion_type = typename std::conditional<has_fast_conversion<A, From, To>::value, with_fast_conversion, with_slow_conversion>::type;
    }

    namespace detail {
        template <class B, uint64_t c>
        inline B coef() noexcept
        {
            using value_type = typename B::value_type;
            return B(bit_cast<value_type>(as_unsigned_integer_t<value_type>(c)));
        }
        template <class B>
        inline B horner(const B&) noexcept
        {
            return B(typename B::value_type(0.));
        }

        template <class B, uint64_t c0>
        inline B horner(const B&) noexcept
        {
            return coef<B, c0>();
        }

        template <class B, uint64_t c0, uint64_t c1, uint64_t... args>
        inline B horner(const B& self) noexcept
        {
            return fma(self, horner<B, c1, args...>(self), coef<B, c0>());
        }

        template <class B>
        inline B horner1(const B&) noexcept
        {
            return B(1.);
        }

        template <class B, uint64_t c0>
        inline B horner1(const B& x) noexcept
        {
            return x + detail::coef<B, c0>();
        }

        template <class B, uint64_t c0, uint64_t c1, uint64_t... args>
        inline B horner1(const B& x) noexcept
        {
            return fma(x, horner1<B, c1, args...>(x), detail::coef<B, c0>());
        }
    }

    using namespace types;
    // abs
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> abs(batch<T, A> const& self, requires<generic>)
    {
      if(std::is_unsigned<T>::value)
        return self;
      else {
        auto sign = bitofsign(self);
        auto inv = self ^ sign;
        return inv - sign;
      }
    }

    // acos
    template<class A, class T> batch<T, A> acos(batch<T, A> const& self, requires<generic>) {
      using batch_type = batch<T, A>;
        batch_type x = abs(self);
        auto x_larger_05 = x > batch_type(0.5);
        x = select(x_larger_05, sqrt(fma(batch_type(-0.5), x, batch_type(0.5))), self);
        x = asin(x);
        x = select(x_larger_05, x + x, x);
        x = select(self < batch_type(-0.5), constants::pi<batch_type>() - x, x);
        return select(x_larger_05, x, constants::pio2<batch_type>() - x);
    }

    // asin
    template<class A> batch<float, A> asin(batch<float, A> const& self, requires<generic>) {
      using batch_type = batch<float, A>;
                batch_type x = abs(self);
                batch_type sign = bitofsign(self);
                auto x_larger_05 = x > batch_type(0.5);
                batch_type z = select(x_larger_05, batch_type(0.5) * (batch_type(1.) - x), x * x);
                x = select(x_larger_05, sqrt(z), x);
                batch_type z1 = detail::horner<batch_type,
                              0x3e2aaae4,
                              0x3d9980f6,
                              0x3d3a3ec7,
                              0x3cc617e3,
                              0x3d2cb352>(z);
                z1 = fma(z1, z * x, x);
                z = select(x_larger_05, constants::pio2<batch_type>() - (z1 + z1), z1);
                return z ^ sign;
    }
    template<class A> batch<double, A> asin(batch<double, A> const& self, requires<generic>) {
      using batch_type = batch<double, A>;
                batch_type x = abs(self);
                auto small_cond = x < constants::sqrteps<batch_type>();
                batch_type ct1 = batch_type(bit_cast<double>(int64_t(0x3fe4000000000000)));
                batch_type zz1 = batch_type(1.) - x;
                batch_type vp = zz1 * detail::horner<batch_type,
                                    0x403c896240f3081dull,
                                    0xc03991aaac01ab68ull,
                                    0x401bdff5baf33e6aull,
                                    0xbfe2079259f9290full,
                                    0x3f684fc3988e9f08ull>(zz1) /
                    detail::horner1<batch_type,
                            0x40756709b0b644beull,
                            0xc077fe08959063eeull,
                            0x40626219af6a7f42ull,
                            0xc035f2a2b6bf5d8cull>(zz1);
                zz1 = sqrt(zz1 + zz1);
                batch_type z = constants::pio4<batch_type>() - zz1;
                zz1 = fms(zz1, vp, constants::pio_2lo<batch_type>());
                z = z - zz1;
                zz1 = z + constants::pio4<batch_type>();
                batch_type zz2 = self * self;
                z = zz2 * detail::horner<batch_type,
                                 0xc020656c06ceafd5ull,
                                 0x40339007da779259ull,
                                 0xc0304331de27907bull,
                                 0x4015c74b178a2dd9ull,
                                 0xbfe34341333e5c16ull,
                                 0x3f716b9b0bd48ad3ull>(zz2) /
                    detail::horner1<batch_type,
                            0xc04898220a3607acull,
                            0x4061705684ffbf9dull,
                            0xc06265bb6d3576d7ull,
                            0x40519fc025fe9054ull,
                            0xc02d7b590b5e0eabull>(zz2);
                zz2 = fma(x, z, x);
                return select(x > batch_type(1.), constants::nan<batch_type>(),
                              select(small_cond, x,
                                     select(x > ct1, zz1, zz2)) ^
                                  bitofsign(self));
    }

    // atan
    namespace detail {
    template<class A>
            static inline batch<float, A> kernel_atan(const batch<float, A>& x, const batch<float, A>& recx)
            {
              using batch_type = batch<float, A>;
                const auto flag1 = x < constants::tan3pio8<batch_type>();
                const auto flag2 = (x >= batch_type(bit_cast<float>((uint32_t)0x3ed413cd))) && flag1;
                batch_type yy = select(flag1, batch_type(0.), constants::pio2<batch_type>());
                yy = select(flag2, constants::pio4<batch_type>(), yy);
                batch_type xx = select(flag1, x, -recx);
                xx = select(flag2, (x - batch_type(1.)) / (x + batch_type(1.)), xx);
                const batch_type z = xx * xx;
                batch_type z1 = detail::horner<batch_type,
                              0xbeaaaa2aul,
                              0x3e4c925ful,
                              0xbe0e1b85ul,
                              0x3da4f0d1ul>(z);
                z1 = fma(xx, z1 * z, xx);
                z1 = select(flag2, z1 + constants::pio_4lo<batch_type>(), z1);
                z1 = select(!flag1, z1 + constants::pio_2lo<batch_type>(), z1);
                return yy + z1;
            }
    template<class A>
            static inline batch<double, A> kernel_atan(const batch<double, A>& x, const batch<double, A>& recx)
            {
              using batch_type = batch<double, A>;
                const auto flag1 = x < constants::tan3pio8<batch_type>();
                const auto flag2 = (x >= constants::tanpio8<batch_type>()) && flag1;
                batch_type yy = select(flag1, batch_type(0.), constants::pio2<batch_type>());
                yy = select(flag2, constants::pio4<batch_type>(), yy);
                batch_type xx = select(flag1, x, -recx);
                xx = select(flag2, (x - batch_type(1.)) / (x + batch_type(1.)), xx);
                batch_type z = xx * xx;
                z *= detail::horner<batch_type,
                            0xc0503669fd28ec8eull,
                            0xc05eb8bf2d05ba25ull,
                            0xc052c08c36880273ull,
                            0xc03028545b6b807aull,
                            0xbfec007fa1f72594ull>(z) /
                    detail::horner1<batch_type,
                            0x4068519efbbd62ecull,
                            0x407e563f13b049eaull,
                            0x407b0e18d2e2be3bull,
                            0x4064a0dd43b8fa25ull,
                            0x4038dbc45b14603cull>(z);
                z = fma(xx, z, xx);
                z = select(flag2, z + constants::pio_4lo<batch_type>(), z);
                z = z + select(flag1, batch_type(0.), constants::pio_2lo<batch_type>());
                return yy + z;
            }
    }
    template<class A, class T> batch<T, A> atan(batch<T, A> const& self, requires<generic>) {
      using batch_type = batch<T, A>;
                const batch_type absa = abs(self);
                const batch_type x = detail::kernel_atan(absa, batch_type(1.) / absa);
                return x ^ bitofsign(self);
    }

    // atan2
    template<class A, class T> batch<T, A> atan2(batch<T, A> const& self, batch<T, A> const& other, requires<generic>) {
      using batch_type = batch<T, A>;
                const batch_type q = abs(self / other);
                const batch_type z = detail::kernel_atan(q, batch_type(1.) / q);
                return select(other > batch_type(0.), z, constants::pi<batch_type>() - z) * signnz(self);
    }

    // batch_cast
    template<class A, class T> batch<T, A> batch_cast(batch<T, A> const& self, batch<T, A> const&, requires<generic>) {
      return self;
    }

    namespace detail {
    template<class A, class T_out, class T_in>
    batch<T_out, A> batch_cast(batch<T_in, A> const& self, batch<T_out, A> const& out, requires<generic>, with_fast_conversion) {
      return conversion::fast<A>(self, out, A{});
    }
    template<class A, class T_out, class T_in>
    batch<T_out, A> batch_cast(batch<T_in, A> const& self, batch<T_out, A> const&, requires<generic>, with_slow_conversion) {
      static_assert(!std::is_same<T_in, T_out>::value, "there should be no conversion for this type combination");
      using batch_type_in = batch<T_in, A>;
      using batch_type_out = batch<T_out, A>;
      static_assert(batch_type_in::size == batch_type_out::size, "compatible sizes");
      alignas(A::alignment()) T_in buffer_in[batch_type_in::size];
      alignas(A::alignment()) T_out buffer_out[batch_type_out::size];
      self.store_aligned(&buffer_in[0]);
      std::copy(std::begin(buffer_in), std::end(buffer_in), std::begin(buffer_out));
      return batch_type_out::load_aligned(buffer_out);
    }

    }

    template<class A, class T_out, class T_in>
    batch<T_out, A> batch_cast(batch<T_in, A> const& self, batch<T_out, A> const& out, requires<generic>) {
      return detail::batch_cast(self, out, A{}, detail::conversion_type<A, T_in, T_out>{});
    }

    // bitofsign
    template<class A, class T> batch<T, A> bitofsign(batch<T, A> const& self, requires<generic>) {
      static_assert(std::is_integral<T>::value, "int type implementation");
      if(std::is_unsigned<T>::value)
        return batch<T, A>(0);
      else
        return self >> (T)(8 * sizeof(T) - 1);
    }

    template<class A> batch<float, A> bitofsign(batch<float, A> const& self, requires<generic>) {
      return self & constants::minuszero<batch<float, A>>();
    }
    template<class A> batch<double, A> bitofsign(batch<double, A> const& self, requires<generic>) {
      return self & constants::minuszero<batch<double, A>>();
    }

    // cbrt
    template<class A> batch<float, A> cbrt(batch<float, A> const& self, requires<generic>) {
      using batch_type = batch<float, A>;
                batch_type z = abs(self);
#ifndef XSIMD_NO_DENORMALS
                auto denormal = z < constants::smallestposval<batch_type>();
                z = select(denormal, z * constants::twotonmb<batch_type>(), z);
                batch_type f = select(denormal, constants::twotonmbo3<batch_type>(), batch_type(1.));
#endif
                const batch_type CBRT2 (bit_cast<float>(0x3fa14518));
                const batch_type CBRT4 (bit_cast<float>(0x3fcb2ff5));
                const batch_type CBRT2I (bit_cast<float>(0x3f4b2ff5));
                const batch_type CBRT4I (bit_cast<float>(0x3f214518));
                using i_type = as_integer_t<batch_type>;
                i_type e;
                batch_type x = frexp(z, e);
                x = detail::horner<batch_type,
                           0x3ece0609,
                           0x3f91eb77,
                           0xbf745265,
                           0x3f0bf0fe,
                           0xbe09e49a>(x);
                auto flag = e >= i_type(0);
                i_type e1 = abs(e);
                i_type rem = e1;
                e1 /= i_type(3);
                rem -= e1 * i_type(3);
                e = e1 * sign(e);
                const batch_type cbrt2 = select(bool_cast(flag), CBRT2, CBRT2I);
                const batch_type cbrt4 = select(bool_cast(flag), CBRT4, CBRT4I);
                batch_type fact = select(bool_cast(rem == i_type(1)), cbrt2, batch_type(1.));
                fact = select(bool_cast(rem == i_type(2)), cbrt4, fact);
                x = ldexp(x * fact, e);
                x -= (x - z / (x * x)) * batch_type(1.f / 3.f);
#ifndef XSIMD_NO_DENORMALS
                x = (x | bitofsign(self)) * f;
#else
                x = x | bitofsign(self);
#endif
#ifndef XSIMD_NO_INFINITIES
                return select(self == batch_type(0.) || isinf(self), self, x);
#else
                return select(self == batch_type(0.), self, x);
#endif
    }
    template<class A> batch<double, A> cbrt(batch<double, A> const& self, requires<generic>) {
      using batch_type = batch<double, A>;
                batch_type z = abs(self);
#ifndef XSIMD_NO_DENORMALS
                auto denormal = z < constants::smallestposval<batch_type>();
                z = select(denormal, z * constants::twotonmb<batch_type>(), z);
                batch_type f = select(denormal, constants::twotonmbo3<batch_type>(), batch_type(1.));
#endif
                const batch_type CBRT2(bit_cast<double>(int64_t(0x3ff428a2f98d728b)));
                const batch_type CBRT4(bit_cast<double>(int64_t(0x3ff965fea53d6e3d)));
                const batch_type CBRT2I(bit_cast<double>(int64_t(0x3fe965fea53d6e3d)));
                const batch_type CBRT4I(bit_cast<double>(int64_t(0x3fe428a2f98d728b)));
                using i_type = as_integer_t<batch_type>;
                i_type e;
                batch_type x = frexp(z, e);
                x = detail::horner<batch_type,
                           0x3fd9c0c12122a4feull,
                           0x3ff23d6ee505873aull,
                           0xbfee8a4ca3ba37b8ull,
                           0x3fe17e1fc7e59d58ull,
                           0xbfc13c93386fdff6ull>(x);
                auto flag = e >= typename i_type::value_type(0);
                i_type e1 = abs(e);
                i_type rem = e1;
                e1 /= i_type(3);
                rem -= e1 * i_type(3);
                e = e1 * sign(e);
                const batch_type cbrt2 = select(bool_cast(flag), CBRT2, CBRT2I);
                const batch_type cbrt4 = select(bool_cast(flag), CBRT4, CBRT4I);
                batch_type fact = select(bool_cast(rem == i_type(1)), cbrt2, batch_type(1.));
                fact = select(bool_cast(rem == i_type(2)), cbrt4, fact);
                x = ldexp(x * fact, e);
                x -= (x - z / (x * x)) * batch_type(1. / 3.);
                x -= (x - z / (x * x)) * batch_type(1. / 3.);
#ifndef XSIMD_NO_DENORMALS
                x = (x | bitofsign(self)) * f;
#else
                x = x | bitofsign(self);
#endif
#ifndef XSIMD_NO_INFINITIES
                return select(self == batch_type(0.) || isinf(self), self, x);
#else
                return select(self == batch_type(0.), self, x);
#endif
    }

    // clip
    template<class A, class T> batch<T, A> clip(batch<T, A> const& self, batch<T, A> const& lo, batch<T, A> const& hi, requires<generic>) {
      return min(hi, max(self, lo));
    }

    // ceil
    template<class A, class T> batch<T, A> ceil(batch<T, A> const& self, requires<generic>) {
      batch<T, A> truncated_self = trunc(self);
      return select(truncated_self < self, truncated_self + 1, truncated_self);
    }

    // copysign
    template<class A, class T> batch<T, A> copysign(batch<T, A> const& self, batch<T, A> const& other, requires<generic>) {
      return abs(self) | bitofsign(other);
    }

    // cos
    namespace detail
    {
        template <class T, class A>
        batch<T, A> quadrant(const batch<T, A>& x) {
          return x & batch<T, A>(3);
        }

        template <class A>
        batch<float, A> quadrant(const batch<float, A>& x) {
          return to_float(quadrant(to_int(x)));
        }

        template <class A>
        batch<double, A> quadrant(const batch<double, A>& x) {
          using batch_type = batch<double, A>;
                batch_type a = x * batch_type(0.25);
                return (a - floor(a)) * batch_type(4.);
        }

        template<class A>
        inline batch<float, A> cos_eval(const batch<float, A>& z)
        {
          using batch_type = batch<float, A>;
            batch_type y = horner<batch_type,
                         0x3d2aaaa5,
                         0xbab60619,
                         0x37ccf5ce>(z);
            return batch_type(1.) + fma(z, batch_type(-0.5), y * z * z);
        }

        template<class A>
        inline batch<float, A> sin_eval(const batch<float, A>& z, const batch<float, A>& x)
            {
          using batch_type = batch<float, A>;
                batch_type y = horner<batch_type,
                             0xbe2aaaa2,
                             0x3c08839d,
                             0xb94ca1f9>(z);
                return fma(y * z, x, x);
            }

        template<class A>
            static inline batch<float, A> base_tancot_eval(const batch<float, A>& z)
            {
          using batch_type = batch<float, A>;
                batch_type zz = z * z;
                batch_type y = horner<batch_type,
                             0x3eaaaa6f,
                             0x3e0896dd,
                             0x3d5ac5c9,
                             0x3cc821b5,
                             0x3b4c779c,
                             0x3c19c53b>(zz);
                return fma(y, zz * z, z);
            }

            template <class A, class BB>
            static inline batch<float, A> tan_eval(const batch<float, A>& z, const BB& test)
            {
          using batch_type = batch<float, A>;
                batch_type y = base_tancot_eval(z);
                return select(test, y, -batch_type(1.) / y);
            }

            template <class A, class BB>
            static inline batch<float, A> cot_eval(const batch<float, A>& z, const BB& test)
            {
          using batch_type = batch<float, A>;
                batch_type y = base_tancot_eval(z);
                return select(test, batch_type(1.) / y, -y);
            }

            template<class A>
            static inline batch<double, A> cos_eval(const batch<double, A>& z)
            {
          using batch_type = batch<double, A>;
                batch_type y = horner<batch_type,
                             0x3fe0000000000000ull,
                             0xbfa5555555555551ull,
                             0x3f56c16c16c15d47ull,
                             0xbefa01a019ddbcd9ull,
                             0x3e927e4f8e06d9a5ull,
                             0xbe21eea7c1e514d4ull,
                             0x3da8ff831ad9b219ull>(z);
                return batch_type(1.) - y * z;
            }

            template<class A>
            static inline batch<double, A> sin_eval(const batch<double, A>& z, const batch<double, A>& x)
            {
          using batch_type = batch<double, A>;
                batch_type y = horner<batch_type,
                             0xbfc5555555555548ull,
                             0x3f8111111110f7d0ull,
                             0xbf2a01a019bfdf03ull,
                             0x3ec71de3567d4896ull,
                             0xbe5ae5e5a9291691ull,
                             0x3de5d8fd1fcf0ec1ull>(z);
                return fma(y * z, x, x);
            }

            template<class A>
            static inline batch<double, A> base_tancot_eval(const batch<double, A>& z)
            {
          using batch_type = batch<double, A>;
                batch_type zz = z * z;
                batch_type num = detail::horner<batch_type,
                               0xc1711fead3299176ull,
                               0x413199eca5fc9dddull,
                               0xc0c992d8d24f3f38ull>(zz);
                batch_type den = detail::horner1<batch_type,
                                0xc189afe03cbe5a31ull,
                                0x4177d98fc2ead8efull,
                                0xc13427bc582abc96ull,
                                0x40cab8a5eeb36572ull>(zz);
                return fma(z, (zz * (num / den)), z);
            }

            template <class A, class BB>
            static inline batch<double, A> tan_eval(const batch<double, A>& z, const BB& test)
            {
          using batch_type = batch<double, A>;
                batch_type y = base_tancot_eval(z);
                return select(test, y, -batch_type(1.) / y);
            }

            template <class A, class BB>
            static inline batch<double, A> cot_eval(const batch<double, A>& z, const BB& test)
            {
          using batch_type = batch<double, A>;
                batch_type y = base_tancot_eval(z);
                return select(test, batch_type(1.) / y, -y);
            }

        struct trigo_radian_tag
        {
        };
        struct trigo_pi_tag
        {
        };

        template <class B, class Tag = trigo_radian_tag>
        struct trigo_reducer
        {
            static inline B reduce(const B& x, B& xr)
            {
                if (all(x <= constants::pio4<B>()))
                {
                    xr = x;
                    return B(0.);
                }
                else if (all(x <= constants::pio2<B>()))
                {
                    auto test = x > constants::pio4<B>();
                    xr = x - constants::pio2_1<B>();
                    xr -= constants::pio2_2<B>();
                    xr -= constants::pio2_3<B>();
                    xr = select(test, xr, x);
                    return select(test, B(1.), B(0.));
                }
                else if (all(x <= constants::twentypi<B>()))
                {
                    B xi = nearbyint(x * constants::twoopi<B>());
                    xr = fnma(xi, constants::pio2_1<B>(), x);
                    xr -= xi * constants::pio2_2<B>();
                    xr -= xi * constants::pio2_3<B>();
                    return quadrant(xi);
                }
                else if (all(x <= constants::mediumpi<B>()))
                {
                    B fn = nearbyint(x * constants::twoopi<B>());
                    B r = x - fn * constants::pio2_1<B>();
                    B w = fn * constants::pio2_1t<B>();
                    B t = r;
                    w = fn * constants::pio2_2<B>();
                    r = t - w;
                    w = fn * constants::pio2_2t<B>() - ((t - r) - w);
                    t = r;
                    w = fn * constants::pio2_3<B>();
                    r = t - w;
                    w = fn * constants::pio2_3t<B>() - ((t - r) - w);
                    xr = r - w;
                    return quadrant(fn);
                }
                else
                {
                    static constexpr std::size_t size = B::size;
                    using value_type = typename B::value_type;
                    alignas(B) std::array<value_type, size> tmp;
                    alignas(B) std::array<value_type, size> txr;
                    alignas(B) std::array<value_type, size> args;
                    x.store_aligned(args.data());

                    for (std::size_t i = 0; i < size; ++i)
                    {
                        double arg = args[i];
                        if (arg == std::numeric_limits<value_type>::infinity())
                        {
                            tmp[i] = 0.;
                            txr[i] = std::numeric_limits<value_type>::quiet_NaN();
                        }
                        else
                        {
                            double y[2];
                            std::int32_t n = ::xsimd::detail::__ieee754_rem_pio2(arg, y);
                            tmp[i] = value_type(n & 3);
                            txr[i] = value_type(y[0]);
                        }
                    }
                    xr.load_aligned(&txr[0]);
                    B res;
                    res.load_aligned(&tmp[0]);
                    return res;
                }
            }
        };

        template <class B>
        struct trigo_reducer<B, trigo_pi_tag>
        {
            static inline B reduce(const B& x, B& xr)
            {
                B xi = nearbyint(x * B(2.));
                B x2 = x - xi * B(0.5);
                xr = x2 * constants::pi<B>();
                return quadrant(xi);
            }
        };

    }
    template<class A, class T> batch<T, A> cos(batch<T, A> const& self, requires<generic>) {
      using batch_type = batch<T, A>;
                const batch_type x = abs(self);
                batch_type xr = constants::nan<batch_type>();
                const batch_type n = detail::trigo_reducer<batch_type>::reduce(x, xr);
                auto tmp = select(n >= batch_type(2.), batch_type(1.), batch_type(0.));
                auto swap_bit = fma(batch_type(-2.), tmp, n);
                auto sign_bit = select((swap_bit ^ tmp) != batch_type(0.), constants::signmask<batch_type>(), batch_type(0.));
                const batch_type z = xr * xr;
                const batch_type se = detail::sin_eval(z, xr);
                const batch_type ce = detail::cos_eval(z);
                const batch_type z1 = select(swap_bit != batch_type(0.), se, ce);
                return z1 ^ sign_bit;
    }

    // cosh

    namespace detail {
        template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
          batch<T, A>
            average(const batch<T, A>& x1, const batch<T, A>& x2)
            {
                return (x1 & x2) + ((x1 ^ x2) >> 1);
            }

        template <class A, class T>
          batch<T, A>
            averagef(const batch<T, A>& x1, const batch<T, A>& x2)
            {
              using batch_type = batch<T, A>;
                return fma(x1, batch_type(0.5), x2 * batch_type(0.5));
            }
        template<class A>
          batch<float, A> average(batch<float, A> const & x1, batch<float, A> const & x2) {
            return averagef(x1, x2);
          }
        template<class A>
          batch<double, A> average(batch<double, A> const & x1, batch<double, A> const & x2) {
            return averagef(x1, x2);
          }
    }
    template<class A, class T> batch<T, A> cosh(batch<T, A> const& self, requires<generic>) {
      using batch_type = batch<T, A>;
                batch_type x = abs(self);
                auto test1 = x > (constants::maxlog<batch_type>() - constants::log_2<batch_type>());
                batch_type fac = select(test1, batch_type(0.5), batch_type(1.));
                batch_type tmp = exp(x * fac);
                batch_type tmp1 = batch_type(0.5) * tmp;
                return select(test1, tmp1 * tmp, detail::average(tmp, batch_type(1.) / tmp));
    }

    // div
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> div(batch<T, A> const& self, batch<T, A> const& other, requires<generic>) {
      constexpr std::size_t size = batch<T, A>::size;
      alignas(A::alignment()) T self_buffer[size];
      alignas(A::alignment()) T other_buffer[size];
      self.store_aligned(&self_buffer[0]);
      other.store_aligned(&other_buffer[0]);
      for(std::size_t i = 0; i < size; ++i)
        self_buffer[i] /= other_buffer[i];
      return batch<T, A>::load_aligned(self_buffer);
    }

    // erf

    namespace detail {
        template <class B>
        struct erf_kernel;

        template <class A>
        struct erf_kernel<batch<float, A>>
        {
            using batch_type = batch<float, A>;
            // computes erf(a0)/a0
            // x is sqr(a0) and 0 <= abs(a0) <= 2/3
            static inline batch_type erf1(const batch_type& x)
            {
                return detail::horner<batch_type,
                              0x3f906eba,  //   1.128379154774254e+00
                              0xbec0937e,  //  -3.761252839094832e-01
                              0x3de70f22,  //   1.128218315189123e-01
                              0xbcdb61f4,  //  -2.678010670585737e-02
                              0x3ba4468d,  //   5.013293006147870e-03
                              0xba1fc83b  //  -6.095205117313012e-04
                              >(x);
            }

            // computes erfc(x)*exp(sqr(x))
            // x >=  2/3
            static inline batch_type erfc2(const batch_type& x)
            {
                return detail::horner<batch_type,
                              0x3f0a0e8b,  //   5.392844046572836e-01
                              0xbf918a62,  //  -1.137035586823118e+00
                              0x3e243828,  //   1.603704761054187e-01
                              0x3ec4ca6e,  //   3.843569094305250e-01
                              0x3e1175c7,  //   1.420508523645926e-01
                              0x3e2006f0,  //   1.562764709849380e-01
                              0xbfaea865,  //  -1.364514006347145e+00
                              0x4050b063,  //   3.260765682222576e+00
                              0xc0cd1a85,  //  -6.409487379234005e+00
                              0x40d67e3b,  //   6.702908785399893e+00
                              0xc0283611  //  -2.628299919293280e+00
                              >(x);
            }

            static inline batch_type erfc3(const batch_type& x)
            {
                return (batch_type(1.) - x) * detail::horner<batch_type,
                                            0x3f7ffffe,  //   9.9999988e-01
                                            0xbe036d7e,  //  -1.2834737e-01
                                            0xbfa11698,  //  -1.2585020e+00
                                            0xbffc9284,  //  -1.9732213e+00
                                            0xc016c985,  //  -2.3560498e+00
                                            0x3f2cff3b,  //   6.7576951e-01
                                            0xc010d956,  //  -2.2632651e+00
                                            0x401b5680,  //   2.4271545e+00
                                            0x41aa8e55  //   2.1319498e+01
                                            >(x);
            }
        };

        template <class A>
        struct erf_kernel<batch<double, A>>
        {
            using batch_type = batch<double, A>;
            // computes erf(a0)/a0
            // x is sqr(a0) and 0 <= abs(a0) <= 0.65
            static inline batch_type erf1(const batch_type& x)
            {
                return detail::horner<batch_type,
                              0x3ff20dd750429b61ull,  // 1.12837916709551
                              0x3fc16500f106c0a5ull,  // 0.135894887627278
                              0x3fa4a59a4f02579cull,  // 4.03259488531795E-02
                              0x3f53b7664358865aull,  // 1.20339380863079E-03
                              0x3f110512d5b20332ull  // 6.49254556481904E-05
                              >(x) /
                    detail::horner<batch_type,
                           0x3ff0000000000000ull,  // 1
                           0x3fdd0a84eb1ca867ull,  // 0.453767041780003
                           0x3fb64536ca92ea2full,  // 8.69936222615386E-02
                           0x3f8166f75999dbd1ull,  // 8.49717371168693E-03
                           0x3f37ea4332348252ull  // 3.64915280629351E-04
                           >(x);
            }

            // computes erfc(x)*exp(x*x)
            // 0.65 <= abs(x) <= 2.2
            static inline batch_type erfc2(const batch_type& x)
            {
                return detail::horner<batch_type,
                              0x3feffffffbbb552bull,  // 0.999999992049799
                              0x3ff54dfe9b258a60ull,  // 1.33154163936765
                              0x3fec1986509e687bull,  // 0.878115804155882
                              0x3fd53dd7a67c7e9full,  // 0.331899559578213
                              0x3fb2488a6b5cb5e5ull,  // 7.14193832506776E-02
                              0x3f7cf4cfe0aacbb4ull,  // 7.06940843763253E-03
                              0x0ull  // 0
                              >(x) /
                    detail::horner<batch_type,
                           0x3ff0000000000000ull,  // 1
                           0x4003adeae79b9708ull,  // 2.45992070144246
                           0x40053b1052dca8bdull,  // 2.65383972869776
                           0x3ff9e677c2777c3cull,  // 1.61876655543871
                           0x3fe307622fcff772ull,  // 0.594651311286482
                           0x3fc033c113a7deeeull,  // 0.126579413030178
                           0x3f89a996639b0d00ull  // 1.25304936549413E-02
                           >(x);
            }

            // computes erfc(x)*exp(x*x)
            // 2.2 <= abs(x) <= 6
            static inline batch_type erfc3(const batch_type& x)
            {
                return detail::horner<batch_type,
                              0x3fefff5a9e697ae2ull,  //0.99992114009714
                              0x3ff9fa202deb88e5ull,  //1.62356584489367
                              0x3ff44744306832aeull,  //1.26739901455873
                              0x3fe29be1cff90d94ull,  //0.581528574177741
                              0x3fc42210f88b9d43ull,  //0.157289620742839
                              0x3f971d0907ea7a92ull,  //2.25716982919218E-02
                              0x0ll  //0
                              >(x) /
                    detail::horner<batch_type,
                           0x3ff0000000000000ull,  //1
                           0x400602f24bf3fdb6ull,  //2.75143870676376
                           0x400afd487397568full,  //3.37367334657285
                           0x400315ffdfd5ce91ull,  //2.38574194785344
                           0x3ff0cfd4cb6cde9full,  //1.05074004614827
                           0x3fd1d7ab774bb837ull,  //0.278788439273629
                           0x3fa47bd61bbb3843ull  //4.00072964526861E-02
                           >(x);
            }

            // computes erfc(rx)*exp(rx*rx)
            // x >=  6 rx = 1/x
            static inline batch_type erfc4(const batch_type& x)
            {
                return detail::horner<batch_type,
                              0xbc7e4ad1ec7d0000ll,  // -2.627435221016534e-17
                              0x3fe20dd750429a16ll,  // 5.641895835477182e-01
                              0x3db60000e984b501ll,  // 2.000889609806154e-11
                              0xbfd20dd753ae5dfdll,  // -2.820947949598745e-01
                              0x3e907e71e046a820ll,  // 2.457786367990903e-07
                              0x3fdb1494cac06d39ll,  // 4.231311779019112e-01
                              0x3f34a451701654f1ll,  // 3.149699042180451e-04
                              0xbff105e6b8ef1a63ll,  // -1.063940737150596e+00
                              0x3fb505a857e9ccc8ll,  // 8.211757799454056e-02
                              0x40074fbabc514212ll,  // 2.913930388669777e+00
                              0x4015ac7631f7ac4fll,  // 5.418419628850713e+00
                              0xc0457e03041e9d8bll,  // -4.298446704382794e+01
                              0x4055803d26c4ec4fll,  // 8.600373238783617e+01
                              0xc0505fce04ec4ec5ll  // -6.549694941594051e+01
                              >(x);
            }
        };
    }

    template<class A>
    batch<float, A> erf(batch<float, A> const& self, requires<generic>) {
      using batch_type = batch<float, A>;
                batch_type x = abs(self);
                batch_type r1(0.);
                auto test1 = x < batch_type(2.f / 3.f);
                if (any(test1))
                {
                    r1 = self * detail::erf_kernel<batch_type>::erf1(x * x);
                    if (all(test1))
                        return r1;
                }
                batch_type z = x / (batch_type(1.) + x);
                z -= batch_type(0.4f);
                batch_type r2 = batch_type(1.) - exp(-x * x) * detail::erf_kernel<batch_type>::erfc2(z);
                r2 = select(self < batch_type(0.), -r2, r2);
                r1 = select(test1, r1, r2);
#ifndef XSIMD_NO_INFINITIES
                r1 = select(xsimd::isinf(self), sign(self), r1);
#endif
                return r1;
            }
    template<class A> batch<double, A> erf(batch<double, A> const& self, requires<generic>) {
      using batch_type = batch<double, A>;
                batch_type x = abs(self);
                batch_type xx = x * x;
                batch_type lim1 (0.65);
                batch_type lim2 (2.2);
                auto test1 = x < lim1;
                batch_type r1 (0.);
                if (any(test1))
                {
                    r1 = self * detail::erf_kernel<batch_type>::erf1(xx);
                    if (all(test1))
                        return r1;
                }
                auto test2 = x < lim2;
                auto test3 = test2 && !test1;
                batch_type ex = exp(-xx);
                if (any(test3))
                {
                    batch_type z = batch_type(1.) - ex * detail::erf_kernel<batch_type>::erfc2(x);
                    batch_type r2 = select(self < batch_type(0.), -z, z);
                    r1 = select(test1, r1, r2);
                    if (all(test1 || test3))
                        return r1;
                }
                batch_type z = batch_type(1.) - ex * detail::erf_kernel<batch_type>::erfc3(x);
                z = select(self < batch_type(0.), -z, z);
#ifndef XSIMD_NO_INFINITIES
                z = select(xsimd::isinf(self), sign(self), z);
#endif
                return select(test2, r1, z);
    }

    // erfc
    template<class A> batch<float, A> erfc(batch<float, A> const& self, requires<generic>) {
      using batch_type = batch<float, A>;
                batch_type x = abs(self);
                auto test0 = self < batch_type(0.);
                batch_type r1 (0.);
                auto test1 = x < batch_type(2.f / 3.f);
                batch_type z = x / (batch_type(1.) + x);
                if (any(test1))
                {
                    r1 = detail::erf_kernel<batch_type>::erfc3(z);
                    if (all(test1))
                        return select(test0, batch_type(2.) - r1, r1);
                }
                z -= batch_type(0.4f);
                batch_type r2 = exp(-x * x) * detail::erf_kernel<batch_type>::erfc2(z);
                r1 = select(test1, r1, r2);
#ifndef XSIMD_NO_INFINITIES
                r1 = select(x == constants::infinity<batch_type>(), batch_type(0.), r1);
#endif
                return select(test0, batch_type(2.) - r1, r1);
    }
    template<class A> batch<double, A> erfc(batch<double, A> const& self, requires<generic>) {
      using batch_type = batch<double, A>;
                batch_type x = abs(self);
                batch_type xx = x * x;
                batch_type lim1 (0.65);
                batch_type lim2 (2.2);
                auto test0 = self < batch_type(0.);
                auto test1 = x < lim1;
                batch_type r1 (0.);
                if (any(test1))
                {
                    r1 = batch_type(1.) - x * detail::erf_kernel<batch_type>::erf1(xx);
                    if (all(test1))
                        return select(test0, batch_type(2.) - r1, r1);
                }
                auto test2 = x < lim2;
                auto test3 = test2 && !test1;
                batch_type ex = exp(-xx);
                if (any(test3))
                {
                    batch_type z = ex * detail::erf_kernel<batch_type>::erfc2(x);
                    r1 = select(test1, r1, z);
                    if (all(test1 || test3))
                        return select(test0, batch_type(2.) - r1, r1);
                }
                batch_type z = ex * detail::erf_kernel<batch_type>::erfc3(x);
                r1 = select(test2, r1, z);
#ifndef XSIMD_NO_INFINITIES
                r1 = select(x == constants::infinity<batch_type>(), batch_type(0.), r1);
#endif
                return select(test0, batch_type(2.) - r1, r1);
    }

    // estrin
    namespace detail
    {

        template <class B>
        struct estrin
        {
            B x;

            template <typename... Ts>
            inline B operator()(const Ts&... coefs) noexcept
            {
                return eval(coefs...);
            }

          private:
            inline B eval(const B& c0) noexcept
            {
                return c0;
            }

            inline B eval(const B& c0, const B& c1) noexcept
            {
                return fma(x, c1, c0);
            }

            template <size_t... Is, class Tuple>
            inline B eval(::xsimd::detail::index_sequence<Is...>, const Tuple& tuple)
            {
                return estrin{x * x}(std::get<Is>(tuple)...);
            }

            template <class... Args>
            inline B eval(const std::tuple<Args...>& tuple) noexcept
            {
                return eval(::xsimd::detail::make_index_sequence<sizeof...(Args)>(), tuple);
            }

            template <class... Args>
            inline B eval(const std::tuple<Args...>& tuple, const B& c0) noexcept
            {
                return eval(std::tuple_cat(tuple, std::make_tuple(eval(c0))));
            }

            template <class... Args>
            inline B eval(const std::tuple<Args...>& tuple, const B& c0, const B& c1) noexcept
            {
                return eval(std::tuple_cat(tuple, std::make_tuple(eval(c0, c1))));
            }

            template <class... Args, class... Ts>
            inline B eval(const std::tuple<Args...>& tuple, const B& c0, const B& c1, const Ts&... coefs) noexcept
            {
                return eval(std::tuple_cat(tuple, std::make_tuple(eval(c0, c1))), coefs...);
            }

            template <class... Ts>
            inline B eval(const B& c0, const B& c1, const Ts&... coefs) noexcept
            {
                return eval(std::make_tuple(eval(c0, c1)), coefs...);
            }
        };
    }
    template <class T, class A, uint64_t... Coefs>
    batch<T, A> estrin(const batch<T, A>& self) {
      using batch_type = batch<T, A>;
      return detail::estrin<batch_type>{self}(detail::coef<batch_type, Coefs>()...);
    }

    // exp
    namespace detail
    {
        enum exp_reduction_tag { exp_tag, exp2_tag, exp10_tag };

        template <class B, exp_reduction_tag Tag>
        struct exp_reduction_base;

        template <class B>
        struct exp_reduction_base<B, exp_tag>
        {
            static constexpr B maxlog() noexcept
            {
                return constants::maxlog<B>();
            }

            static constexpr B minlog() noexcept
            {
                return constants::minlog<B>();
            }
        };

        template <class B>
        struct exp_reduction_base<B, exp10_tag>
        {
            static constexpr B maxlog() noexcept
            {
                return constants::maxlog10<B>();
            }

            static constexpr B minlog() noexcept
            {
                return constants::minlog10<B>();
            }
        };

        template <class B>
        struct exp_reduction_base<B, exp2_tag>
        {
            static constexpr B maxlog() noexcept
            {
                return constants::maxlog2<B>();
            }

            static constexpr B minlog() noexcept
            {
                return constants::minlog2<B>();
            }
        };

        template <class T, class A, exp_reduction_tag Tag>
        struct exp_reduction;

        template <class A>
        struct exp_reduction<float, A, exp_tag> : exp_reduction_base<batch<float, A>, exp_tag>
        {
          using batch_type = batch<float, A>;
            static inline batch_type approx(const batch_type& x)
            {
                batch_type y = detail::horner<batch_type,
                             0x3f000000,  //  5.0000000e-01
                             0x3e2aa9a5,  //  1.6666277e-01
                             0x3d2aa957,  //  4.1665401e-02
                             0x3c098d8b,  //  8.3955629e-03
                             0x3ab778cf  //  1.3997796e-03
                             >(x);
                return ++fma(y, x * x, x);
            }

            static inline batch_type reduce(const batch_type& a, batch_type& x)
            {
                batch_type k = nearbyint(constants::invlog_2<batch_type>() * a);
                x = fnma(k, constants::log_2hi<batch_type>(), a);
                x = fnma(k, constants::log_2lo<batch_type>(), x);
                return k;
            }
        };

        template <class A>
        struct exp_reduction<float, A, exp10_tag> : exp_reduction_base<batch<float, A>, exp10_tag>
        {
          using batch_type = batch<float, A>;
            static inline batch_type approx(const batch_type& x)
            {
                return ++(detail::horner<batch_type,
                                 0x40135d8e,  //    2.3025851e+00
                                 0x4029a926,  //    2.6509490e+00
                                 0x400237da,  //    2.0346589e+00
                                 0x3f95eb4c,  //    1.1712432e+00
                                 0x3f0aacef,  //    5.4170126e-01
                                 0x3e54dff1  //    2.0788552e-01
                                 >(x) *
                          x);
            }

            static inline batch_type reduce(const batch_type& a, batch_type& x)
            {
                batch_type k = nearbyint(constants::invlog10_2<batch_type>() * a);
                x = fnma(k, constants::log10_2hi<batch_type>(), a);
                x -= k * constants::log10_2lo<batch_type>();
                return k;
            }
        };

        template <class A>
        struct exp_reduction<float, A, exp2_tag> : exp_reduction_base<batch<float, A>, exp2_tag>
        {
            using batch_type = batch<float, A>;
            static inline batch_type approx(const batch_type& x)
            {
                batch_type y = detail::horner<batch_type,
                             0x3e75fdf1,  //    2.4022652e-01
                             0x3d6356eb,  //    5.5502813e-02
                             0x3c1d9422,  //    9.6178371e-03
                             0x3ab01218,  //    1.3433127e-03
                             0x3922c8c4  //    1.5524315e-04
                             >(x);
                return ++fma(y, x * x, x * constants::log_2<batch_type>());
            }

            static inline batch_type reduce(const batch_type& a, batch_type& x)
            {
                batch_type k = nearbyint(a);
                x = (a - k);
                return k;
            }
        };

        template <class A>
        struct exp_reduction<double, A, exp_tag> : exp_reduction_base<batch<double, A>, exp_tag>
        {
          using batch_type = batch<double, A>;
            static inline batch_type approx(const batch_type& x)
            {
                batch_type t = x * x;
                return fnma(t,
                            detail::horner<batch_type,
                                   0x3fc555555555553eull,
                                   0xbf66c16c16bebd93ull,
                                   0x3f11566aaf25de2cull,
                                   0xbebbbd41c5d26bf1ull,
                                   0x3e66376972bea4d0ull>(t),
                            x);
            }

            static inline batch_type reduce(const batch_type& a, batch_type& hi, batch_type& lo, batch_type& x)
            {
                batch_type k = nearbyint(constants::invlog_2<batch_type>() * a);
                hi = fnma(k, constants::log_2hi<batch_type>(), a);
                lo = k * constants::log_2lo<batch_type>();
                x = hi - lo;
                return k;
            }

            static inline batch_type finalize(const batch_type& x, const batch_type& c, const batch_type& hi, const batch_type& lo)
            {
                return batch_type(1.) - (((lo - (x * c) / (batch_type(2.) - c)) - hi));
            }
        };

        template <class A>
        struct exp_reduction<double, A, exp10_tag> : exp_reduction_base<batch<double, A>, exp10_tag>
        {
          using batch_type = batch<double, A>;
            static inline batch_type approx(const batch_type& x)
            {
                batch_type xx = x * x;
                batch_type px = x * detail::horner<batch_type,
                                  0x40a2b4798e134a01ull,
                                  0x40796b7a050349e4ull,
                                  0x40277d9474c55934ull,
                                  0x3fa4fd75f3062dd4ull>(xx);
                batch_type x2 = px / (detail::horner1<batch_type,
                                     0x40a03f37650df6e2ull,
                                     0x4093e05eefd67782ull,
                                     0x405545fdce51ca08ull>(xx) -
                             px);
                return ++(x2 + x2);
            }

            static inline batch_type reduce(const batch_type& a, batch_type&, batch_type&, batch_type& x)
            {
                batch_type k = nearbyint(constants::invlog10_2<batch_type>() * a);
                x = fnma(k, constants::log10_2hi<batch_type>(), a);
                x = fnma(k, constants::log10_2lo<batch_type>(), x);
                return k;
            }

            static inline batch_type finalize(const batch_type&, const batch_type& c, const batch_type&, const batch_type&)
            {
                return c;
            }
        };

        template <class A>
        struct exp_reduction<double, A, exp2_tag> : exp_reduction_base<batch<double, A>, exp2_tag>
        {
          using batch_type = batch<double, A>;
            static inline batch_type approx(const batch_type& x)
            {
                batch_type t = x * x;
                return fnma(t,
                            detail::horner<batch_type,
                                   0x3fc555555555553eull,
                                   0xbf66c16c16bebd93ull,
                                   0x3f11566aaf25de2cull,
                                   0xbebbbd41c5d26bf1ull,
                                   0x3e66376972bea4d0ull>(t),
                            x);
            }

            static inline batch_type reduce(const batch_type& a, batch_type&, batch_type&, batch_type& x)
            {
                batch_type k = nearbyint(a);
                x = (a - k) * constants::log_2<batch_type>();
                return k;
            }

            static inline batch_type finalize(const batch_type& x, const batch_type& c, const batch_type&, const batch_type&)
            {
                return batch_type(1.) + x + x * c / (batch_type(2.) - c);
            }
        };

      template<exp_reduction_tag Tag, class A> batch<float, A> exp(batch<float, A> const& self) {
        using batch_type = batch<float, A>;
        using reducer_t = exp_reduction<float, A, Tag>;
        batch_type x;
        batch_type k = reducer_t::reduce(self, x);
        x = reducer_t::approx(x);
        x = select(self <= reducer_t::minlog(), batch_type(0.), ldexp(x, to_int(k)));
        x = select(self >= reducer_t::maxlog(), constants::infinity<batch_type>(), x);
        return x;
      }
      template<exp_reduction_tag Tag, class A> batch<double, A> exp(batch<double, A> const& self) {
        using batch_type = batch<double, A>;
        using reducer_t = exp_reduction<double, A, Tag>;
        batch_type hi, lo, x;
        batch_type k = reducer_t::reduce(self, hi, lo, x);
        batch_type c = reducer_t::approx(x);
        c = reducer_t::finalize(x, c, hi, lo);
        c = select(self <= reducer_t::minlog(), batch_type(0.), ldexp(c, to_int(k)));
        c = select(self >= reducer_t::maxlog(), constants::infinity<batch_type>(), c);
        return c;
      }
    }

    template<class A, class T> batch<T, A> exp(batch<T, A> const& self, requires<generic>) {
      return detail::exp<detail::exp_tag>(self);
    }

    template<class A, class T> batch<T, A> exp10(batch<T, A> const& self, requires<generic>) {
      return detail::exp<detail::exp10_tag>(self);
    }

    template<class A, class T> batch<T, A> exp2(batch<T, A> const& self, requires<generic>) {
      return detail::exp<detail::exp2_tag>(self);
    }

    namespace detail {
          template<class A>
          static inline batch<float, A> expm1(const batch<float, A>& a)
            {
              using batch_type = batch<float, A>;
                batch_type k = nearbyint(constants::invlog_2<batch_type>() * a);
                batch_type x = fnma(k, constants::log_2hi<batch_type>(), a);
                x = fnma(k, constants::log_2lo<batch_type>(), x);
                batch_type hx = x * batch_type(0.5);
                batch_type hxs = x * hx;
                batch_type r = detail::horner<batch_type,
                             0X3F800000UL,  // 1
                             0XBD08887FUL,  // -3.3333298E-02
                             0X3ACF6DB4UL  // 1.582554
                             >(hxs);
                batch_type t = fnma(r, hx, batch_type(3.));
                batch_type e = hxs * ((r - t) / (batch_type(6.) - x * t));
                e = fms(x, e, hxs);
                using i_type = as_integer_t<batch_type>;
                i_type ik = to_int(k);
                batch_type two2mk = ::xsimd::bitwise_cast<batch_type>((constants::maxexponent<batch_type>() - ik) << constants::nmb<batch_type>());
                batch_type y = batch_type(1.) - two2mk - (e - x);
                return ldexp(y, ik);
            }

            template<class A>
            static inline batch<double, A> expm1(const batch<double, A>& a)
            {
              using batch_type = batch<double, A>;
                batch_type k = nearbyint(constants::invlog_2<batch_type>() * a);
                batch_type hi = fnma(k, constants::log_2hi<batch_type>(), a);
                batch_type lo = k * constants::log_2lo<batch_type>();
                batch_type x = hi - lo;
                batch_type hxs = x * x * batch_type(0.5);
                batch_type r = detail::horner<batch_type,
                             0X3FF0000000000000ULL,
                             0XBFA11111111110F4ULL,
                             0X3F5A01A019FE5585ULL,
                             0XBF14CE199EAADBB7ULL,
                             0X3ED0CFCA86E65239ULL,
                             0XBE8AFDB76E09C32DULL>(hxs);
                batch_type t = batch_type(3.) - r * batch_type(0.5) * x;
                batch_type e = hxs * ((r - t) / (batch_type(6) - x * t));
                batch_type c = (hi - x) - lo;
                e = (x * (e - c) - c) - hxs;
                using i_type = as_integer_t<batch_type>;
                i_type ik = to_int(k);
                batch_type two2mk = ::xsimd::bitwise_cast<batch_type>((constants::maxexponent<batch_type>() - ik) << constants::nmb<batch_type>());
                batch_type ct1 = batch_type(1.) - two2mk - (e - x);
                batch_type ct2 = ++(x - (e + two2mk));
                batch_type y = select(k < batch_type(20.), ct1, ct2);
                return ldexp(y, ik);
            }

    }

    template<class A, class T> batch<T, A> expm1(batch<T, A> const& self, requires<generic>) {
      using batch_type = batch<T, A>;
      return select(self < constants::logeps<batch_type>(),
                    batch_type(-1.),
                    select(self > constants::maxlog<batch_type>(),
                           constants::infinity<batch_type>(),
                           detail::expm1(self)));
    }

    // fdim
    template<class A, class T> batch<T, A> fdim(batch<T, A> const& self, batch<T, A> const& other, requires<generic>) {
      return fmax(batch<T, A>(0), self - other);
    }

    // floor
    template<class A, class T> batch<T, A> floor(batch<T, A> const& self, requires<generic>) {
      batch<T, A> truncated_self = trunc(self);
      return select(truncated_self > self, truncated_self - 1, truncated_self);
    }

    // fma
    template<class A, class T> batch<T, A> fma(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z, requires<generic>) {
      return x * y + z;
    }

    // fmod
    template<class A, class T> batch<T, A> fmod(batch<T, A> const& self, batch<T, A> const& other, requires<generic>) {
      return fnma(trunc(self / other), other, self);
    }

    // fms
    template<class A, class T> batch<T, A> fms(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z, requires<generic>) {
      return x * y - z;
    }

    // fnma
    template<class A, class T> batch<T, A> fnma(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z, requires<generic>) {
      return -x * y + z;
    }

    // fnms
    template<class A, class T> batch<T, A> fnms(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z, requires<generic>) {
      return -x * y - z;
    }

    // frexp
    template <class A, class T>
    batch<T, A> frexp(const batch<T, A>& self, batch<as_integer_t<T>, A>& exp, requires<generic>) {
        using batch_type = batch<T, A>;
        using i_type = batch<as_integer_t<T>, A>;
        i_type m1f = constants::mask1frexp<batch_type>();
        i_type r1 = m1f & ::xsimd::bitwise_cast<i_type>(self);
        batch_type x = self & ::xsimd::bitwise_cast<batch_type>(~m1f);
        exp = (r1 >> constants::nmb<batch_type>()) - constants::maxexponentm1<batch_type>();
        exp = select(bool_cast(self != batch_type(0.)), exp, i_type(typename i_type::value_type(0)));
        return select((self != batch_type(0.)), x | ::xsimd::bitwise_cast<batch_type>(constants::mask2frexp<batch_type>()), batch_type(0.));
    }

    // ge
    template<class A, class T> batch_bool<T, A> ge(batch<T, A> const& self, batch<T, A> const& other, requires<generic>) {
      return other < self;
    }

    // gt
    template<class A, class T> batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other, requires<generic>) {
      return other <= self;
    }

    // horner
    template <class T, class A, uint64_t... Coefs>
    batch<T, A> horner(const batch<T, A>& self) {
      return detail::horner<batch<T, A>, Coefs...>(self);
    }

    // hypot
    template<class A, class T> batch<T, A> hypot(batch<T, A> const& self, batch<T, A> const& other, requires<generic>) {
      return sqrt(fma(self, self, other * other));
    }

    // ipow
    template<class A, class T, class ITy> batch<T, A> ipow(batch<T, A> const& self, ITy other, requires<generic>) {
      static_assert(std::is_integral<ITy>::value, "second argument must be an integer");
      batch<T, A> a = self;
      ITy b = other;
      bool const recip = b < 0;
      batch<T, A> r(static_cast<T>(1));
      while (1)
      {
          if (b & 1)
          {
              r *= a;
          }
          b /= 2;
          if (b == 0)
          {
              break;
          }
          a *= a;
      }
      return recip ? 1 / r : r;
    }

    // is_even
    template<class A, class T> batch_bool<T, A> is_even(batch<T, A> const& self, requires<generic>) {
      return is_flint(self * T(0.5));
    }

    // is_flint
    template<class A, class T> batch_bool<T, A> is_flint(batch<T, A> const& self, requires<generic>) {
      auto frac = select(isnan(self - self), constants::nan<batch<T, A>>(), self - trunc(self));
      return frac == T(0.);
    }

    // is_odd
    template<class A, class T> batch_bool<T, A> is_odd(batch<T, A> const& self, requires<generic>) {
      return is_even(self - T(1.));
    }

    // isinf
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> isinf(batch<T, A> const& , requires<generic>) {
      return batch_bool<T, A>(false);
    }
    template<class A, class T> batch_bool<T, A> isinf(batch<T, A> const& self, requires<generic>) {
      return abs(self) == batch<T, A>(std::numeric_limits<T>::infinity());
    }

    // isfinite
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> isfinite(batch<T, A> const& , requires<generic>) {
      return batch_bool<T, A>(true);
    }
    template<class A, class T> batch_bool<T, A> isfinite(batch<T, A> const& self, requires<generic>) {
      return (self - self) == batch<T, A>(0);
    }

    // isnan
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> isnan(batch<T, A> const& , requires<generic>) {
      return batch_bool<T, A>(false);
    }

    // ldexp
    template <class A, class T>
    batch<T, A> ldexp(const batch<T, A>& self, const batch<as_integer_t<T>, A>& other, requires<generic>) {
        using batch_type = batch<T, A>;
        using itype = as_integer_t<batch_type>;
        itype ik = other + constants::maxexponent<T>();
        ik = ik << constants::nmb<T>();
        return self * ::xsimd::bitwise_cast<batch_type>(ik);
    }

    // le
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> le(batch<T, A> const& self, batch<T, A> const& other, requires<generic>) {
      return (self < other) || (self == other);
    }

    // load_aligned
    namespace detail {
      template<class A, class T_in, class T_out>
      batch<T_out, A> load_aligned(T_in const* mem, convert<T_out>, requires<generic>, with_fast_conversion) {
        using batch_type_in = batch<T_in, A>;
        using batch_type_out = batch<T_out, A>;
        return conversion::fast(batch_type_in::load_aligned(mem), batch_type_out(), A{});
      }
      template<class A, class T_in, class T_out>
      batch<T_out, A> load_aligned(T_in const* mem, convert<T_out>, requires<generic>, with_slow_conversion) {
        static_assert(!std::is_same<T_in, T_out>::value, "there should be a direct load for this type combination");
        using batch_type_out = batch<T_out, A>;
        alignas(A::alignment()) T_out buffer[batch_type_out::size];
        std::copy(mem, mem + batch_type_out::size, std::begin(buffer));
        return batch_type_out::load_aligned(buffer);
      }
    }
    template<class A, class T_in, class T_out>
    batch<T_out, A> load_aligned(T_in const* mem, convert<T_out> cvt, requires<generic>) {
      return detail::load_aligned<A>(mem, cvt, A{}, detail::conversion_type<A, T_in, T_out>{});
    }

    // load_unaligned
    namespace detail {
      template<class A, class T_in, class T_out>
      batch<T_out, A> load_unaligned(T_in const* mem, convert<T_out>, requires<generic>, with_fast_conversion) {
        using batch_type_in = batch<T_in, A>;
        using batch_type_out = batch<T_out, A>;
        return conversion::fast(batch_type_in::load_unaligned(mem), batch_type_out(), A{});
      }

      template<class A, class T_in, class T_out>
      batch<T_out, A> load_unaligned(T_in const* mem, convert<T_out> cvt, requires<generic>, with_slow_conversion) {
        static_assert(!std::is_same<T_in, T_out>::value, "there should be a direct load for this type combination");
        return load_aligned<A>(mem, cvt, generic{}, with_slow_conversion{});
      }
    }
    template<class A, class T_in, class T_out>
    batch<T_out, A> load_unaligned(T_in const* mem, convert<T_out> cvt, requires<generic>) {
      return detail::load_unaligned<A>(mem, cvt, generic{}, detail::conversion_type<A, T_in, T_out>{});
    }

    // log
    template<class A> batch<float, A> log(batch<float, A> const& self, requires<generic>) {
      using batch_type = batch<float, A>;
      using i_type = as_integer_t<batch_type>;
      batch_type x = self;
      i_type k(0);
      auto isnez = (self != batch_type(0.));
#ifndef XSIMD_NO_DENORMALS
      auto test = (self < constants::smallestposval<batch_type>()) && isnez;
      if (any(test))
      {
          k = select(bool_cast(test), k - i_type(23), k);
          x = select(test, x * batch_type(8388608ul), x);
      }
#endif
      i_type ix = ::xsimd::bitwise_cast<i_type>(x);
      ix += 0x3f800000 - 0x3f3504f3;
      k += (ix >> 23) - 0x7f;
      ix = (ix & i_type(0x007fffff)) + 0x3f3504f3;
      x = ::xsimd::bitwise_cast<batch_type>(ix);
      batch_type f = --x;
      batch_type s = f / (batch_type(2.) + f);
      batch_type z = s * s;
      batch_type w = z * z;
      batch_type t1 = w * detail::horner<batch_type, 0x3eccce13, 0x3e789e26>(w);
      batch_type t2 = z * detail::horner<batch_type, 0x3f2aaaaa, 0x3e91e9ee>(w);
      batch_type R = t2 + t1;
      batch_type hfsq = batch_type(0.5) * f * f;
      batch_type dk = to_float(k);
      batch_type r = fma(dk, constants::log_2hi<batch_type>(), fma(s, (hfsq + R), dk * constants::log_2lo<batch_type>()) - hfsq + f);
#ifndef XSIMD_NO_INFINITIES
      batch_type zz = select(isnez, select(self == constants::infinity<batch_type>(), constants::infinity<batch_type>(), r), constants::minusinfinity<batch_type>());
#else
      batch_type zz = select(isnez, r, constants::minusinfinity<batch_type>());
#endif
      return select(!(self >= batch_type(0.)), constants::nan<batch_type>(), zz);
    }

    template<class A> batch<double, A> log(batch<double, A> const& self, requires<generic>) {
               using batch_type = batch<double, A>;
                using i_type = as_integer_t<batch_type>;

                batch_type x = self;
                i_type hx = ::xsimd::bitwise_cast<i_type>(x) >> 32;
                i_type k(0);
                auto isnez = (self != batch_type(0.));
#ifndef XSIMD_NO_DENORMALS
                auto test = (self < constants::smallestposval<batch_type>()) && isnez;
                if (any(test))
                {
                    k = select(bool_cast(test), k - i_type(54), k);
                    x = select(test, x * batch_type(18014398509481984ull), x);
                }
#endif
                hx += 0x3ff00000 - 0x3fe6a09e;
                k += (hx >> 20) - 0x3ff;
                batch_type dk = to_float(k);
                hx = (hx & i_type(0x000fffff)) + 0x3fe6a09e;
                x = ::xsimd::bitwise_cast<batch_type>(hx << 32 | (i_type(0xffffffff) & ::xsimd::bitwise_cast<i_type>(x)));

                batch_type f = --x;
                batch_type hfsq = batch_type(0.5) * f * f;
                batch_type s = f / (batch_type(2.) + f);
                batch_type z = s * s;
                batch_type w = z * z;

                batch_type t1 = w * detail::horner<batch_type,
                                  0x3fd999999997fa04ll,
                                  0x3fcc71c51d8e78afll,
                                  0x3fc39a09d078c69fll>(w);
                batch_type t2 = z * detail::horner<batch_type,
                                  0x3fe5555555555593ll,
                                  0x3fd2492494229359ll,
                                  0x3fc7466496cb03dell,
                                  0x3fc2f112df3e5244ll>(w);
                batch_type R = t2 + t1;
                batch_type r = fma(dk, constants::log_2hi<batch_type>(), fma(s, (hfsq + R), dk * constants::log_2lo<batch_type>()) - hfsq + f);
#ifndef XSIMD_NO_INFINITIES
                batch_type zz = select(isnez, select(self == constants::infinity<batch_type>(), constants::infinity<batch_type>(), r), constants::minusinfinity<batch_type>());
#else
                batch_type zz = select(isnez, r, constants::minusinfinity<batch_type>());
#endif
                return select(!(self >= batch_type(0.)), constants::nan<batch_type>(), zz);
    }

    // log2
    template<class A> batch<float, A> log2(batch<float, A> const& self, requires<generic>) {
      using batch_type = batch<float, A>;
                using i_type = as_integer_t<batch_type>;
                batch_type x = self;
                i_type k(0);
                auto isnez = (self != batch_type(0.));
#ifndef XSIMD_NO_DENORMALS
                auto test = (self < constants::smallestposval<batch_type>()) && isnez;
                if (any(test))
                {
                    k = select(bool_cast(test), k - i_type(25), k);
                    x = select(test, x * batch_type(33554432ul), x);
                }
#endif
                i_type ix = ::xsimd::bitwise_cast<i_type>(x);
                ix += 0x3f800000 - 0x3f3504f3;
                k += (ix >> 23) - 0x7f;
                ix = (ix & i_type(0x007fffff)) + 0x3f3504f3;
                x = ::xsimd::bitwise_cast<batch_type>(ix);
                batch_type f = --x;
                batch_type s = f / (batch_type(2.) + f);
                batch_type z = s * s;
                batch_type w = z * z;
                batch_type t1 = w * detail::horner<batch_type, 0x3eccce13, 0x3e789e26>(w);
                batch_type t2 = z * detail::horner<batch_type, 0x3f2aaaaa, 0x3e91e9ee>(w);
                batch_type R = t1 + t2;
                batch_type hfsq = batch_type(0.5) * f * f;
                batch_type dk = to_float(k);
                batch_type r = fma(fms(s, hfsq + R, hfsq) + f, constants::invlog_2<batch_type>(), dk);
#ifndef XSIMD_NO_INFINITIES
                batch_type zz = select(isnez, select(self == constants::infinity<batch_type>(), constants::infinity<batch_type>(), r), constants::minusinfinity<batch_type>());
#else
                batch_type zz = select(isnez, r, constants::minusinfinity<batch_type>());
#endif
                return select(!(self >= batch_type(0.)), constants::nan<batch_type>(), zz);
    }
    template<class A> batch<double, A> log2(batch<double, A> const& self, requires<generic>) {
      using batch_type = batch<double, A>;
                using i_type = as_integer_t<batch_type>;
                batch_type x = self;
                i_type hx = ::xsimd::bitwise_cast<i_type>(x) >> 32;
                i_type k(0);
                auto isnez = (self != batch_type(0.));
#ifndef XSIMD_NO_DENORMALS
                auto test = (self < constants::smallestposval<batch_type>()) && isnez;
                if (any(test))
                {
                    k = select(bool_cast(test), k - i_type(54), k);
                    x = select(test, x * batch_type(18014398509481984ull), x);
                }
#endif
                hx += 0x3ff00000 - 0x3fe6a09e;
                k += (hx >> 20) - 0x3ff;
                hx = (hx & i_type(0x000fffff)) + 0x3fe6a09e;
                x = ::xsimd::bitwise_cast<batch_type>(hx << 32 | (i_type(0xffffffff) & ::xsimd::bitwise_cast<i_type>(x)));
                batch_type f = --x;
                batch_type s = f / (batch_type(2.) + f);
                batch_type z = s * s;
                batch_type w = z * z;
                batch_type t1 = w * detail::horner<batch_type,
                                  0x3fd999999997fa04ll,
                                  0x3fcc71c51d8e78afll,
                                  0x3fc39a09d078c69fll>(w);
                batch_type t2 = z * detail::horner<batch_type,
                                  0x3fe5555555555593ll,
                                  0x3fd2492494229359ll,
                                  0x3fc7466496cb03dell,
                                  0x3fc2f112df3e5244ll>(w);
                batch_type R = t2 + t1;
                batch_type hfsq = batch_type(0.5) * f * f;
                batch_type hi = f - hfsq;
                hi = hi & ::xsimd::bitwise_cast<batch_type>((constants::allbits<i_type>() << 32));
                batch_type lo = fma(s, hfsq + R, f - hi - hfsq);
                batch_type val_hi = hi * constants::invlog_2hi<batch_type>();
                batch_type val_lo = fma(lo + hi, constants::invlog_2lo<batch_type>(), lo * constants::invlog_2hi<batch_type>());
                batch_type dk = to_float(k);
                batch_type w1 = dk + val_hi;
                val_lo += (dk - w1) + val_hi;
                val_hi = w1;
                batch_type r = val_lo + val_hi;
#ifndef XSIMD_NO_INFINITIES
                batch_type zz = select(isnez, select(self == constants::infinity<batch_type>(), constants::infinity<batch_type>(), r), constants::minusinfinity<batch_type>());
#else
                batch_type zz = select(isnez, r, constants::minusinfinity<batch_type>());
#endif
                return select(!(self >= batch_type(0.)), constants::nan<batch_type>(), zz);
    }

    // log10
    template<class A> batch<float, A> log10(batch<float, A> const& self, requires<generic>) {
      using batch_type = batch<float, A>;
                const batch_type
                    ivln10hi(4.3432617188e-01f),
                    ivln10lo(-3.1689971365e-05f),
                    log10_2hi(3.0102920532e-01f),
                    log10_2lo(7.9034151668e-07f);
                using i_type = as_integer_t<batch_type>;
                batch_type x = self;
                i_type k(0);
                auto isnez = (self != batch_type(0.));
#ifndef XSIMD_NO_DENORMALS
                auto test = (self < constants::smallestposval<batch_type>()) && isnez;
                if (any(test))
                {
                    k = select(bool_cast(test), k - i_type(25), k);
                    x = select(test, x * batch_type(33554432ul), x);
                }
#endif
                i_type ix = ::xsimd::bitwise_cast<i_type>(x);
                ix += 0x3f800000 - 0x3f3504f3;
                k += (ix >> 23) - 0x7f;
                ix = (ix & i_type(0x007fffff)) + 0x3f3504f3;
                x = ::xsimd::bitwise_cast<batch_type>(ix);
                batch_type f = --x;
                batch_type s = f / (batch_type(2.) + f);
                batch_type z = s * s;
                batch_type w = z * z;
                batch_type t1 = w * detail::horner<batch_type, 0x3eccce13, 0x3e789e26>(w);
                batch_type t2 = z * detail::horner<batch_type, 0x3f2aaaaa, 0x3e91e9ee>(w);
                batch_type R = t2 + t1;
                batch_type dk = to_float(k);
                batch_type hfsq = batch_type(0.5) * f * f;
                batch_type hibits = f - hfsq;
                hibits &= ::xsimd::bitwise_cast<batch_type>(i_type(0xfffff000));
                batch_type lobits = fma(s, hfsq + R, f - hibits - hfsq);
                batch_type r = fma(dk, log10_2hi,
                          fma(hibits, ivln10hi,
                              fma(lobits, ivln10hi,
                                  fma(lobits + hibits, ivln10lo, dk * log10_2lo))));
#ifndef XSIMD_NO_INFINITIES
                batch_type zz = select(isnez, select(self == constants::infinity<batch_type>(), constants::infinity<batch_type>(), r), constants::minusinfinity<batch_type>());
#else
                batch_type zz = select(isnez, r, constants::minusinfinity<batch_type>());
#endif
                return select(!(self >= batch_type(0.)), constants::nan<batch_type>(), zz);
    }
    template<class A> batch<double, A> log10(batch<double, A> const& self, requires<generic>) {
      using batch_type = batch<double, A>;
                const batch_type
                    ivln10hi(4.34294481878168880939e-01),
                    ivln10lo(2.50829467116452752298e-11),
                    log10_2hi(3.01029995663611771306e-01),
                    log10_2lo(3.69423907715893078616e-13);
                using i_type = as_integer_t<batch_type>;
                batch_type x = self;
                i_type hx = ::xsimd::bitwise_cast<i_type>(x) >> 32;
                i_type k(0);
                auto isnez = (self != batch_type(0.));
#ifndef XSIMD_NO_DENORMALS
                auto test = (self < constants::smallestposval<batch_type>()) && isnez;
                if (any(test))
                {
                    k = select(bool_cast(test), k - i_type(54), k);
                    x = select(test, x * batch_type(18014398509481984ull), x);
                }
#endif
                hx += 0x3ff00000 - 0x3fe6a09e;
                k += (hx >> 20) - 0x3ff;
                hx = (hx & i_type(0x000fffff)) + 0x3fe6a09e;
                x = ::xsimd::bitwise_cast<batch_type>(hx << 32 | (i_type(0xffffffff) & ::xsimd::bitwise_cast<i_type>(x)));
                batch_type f = --x;
                batch_type dk = to_float(k);
                batch_type s = f / (batch_type(2.) + f);
                batch_type z = s * s;
                batch_type w = z * z;
                batch_type t1 = w * detail::horner<batch_type,
                                  0x3fd999999997fa04ll,
                                  0x3fcc71c51d8e78afll,
                                  0x3fc39a09d078c69fll>(w);
                batch_type t2 = z * detail::horner<batch_type,
                                  0x3fe5555555555593ll,
                                  0x3fd2492494229359ll,
                                  0x3fc7466496cb03dell,
                                  0x3fc2f112df3e5244ll>(w);
                batch_type R = t2 + t1;
                batch_type hfsq = batch_type(0.5) * f * f;
                batch_type hi = f - hfsq;
                hi = hi & ::xsimd::bitwise_cast<batch_type>(constants::allbits<i_type>() << 32);
                batch_type lo = f - hi - hfsq + s * (hfsq + R);
                batch_type val_hi = hi * ivln10hi;
                batch_type y = dk * log10_2hi;
                batch_type val_lo = dk * log10_2lo + (lo + hi) * ivln10lo + lo * ivln10hi;
                batch_type w1 = y + val_hi;
                val_lo += (y - w1) + val_hi;
                val_hi = w1;
                batch_type r = val_lo + val_hi;
#ifndef XSIMD_NO_INFINITIES
                batch_type zz = select(isnez, select(self == constants::infinity<batch_type>(), constants::infinity<batch_type>(), r), constants::minusinfinity<batch_type>());
#else
                batch_type zz = select(isnez, r, constants::minusinfinity<batch_type>());
#endif
                return select(!(self >= batch_type(0.)), constants::nan<batch_type>(), zz);
    }

    // log1p
    template<class A> batch<float, A> log1p(batch<float, A> const& self, requires<generic>) {
      using batch_type = batch<float, A>;
                using i_type = as_integer_t<batch_type>;
                const batch_type uf = self + batch_type(1.);
                auto isnez = (uf != batch_type(0.));
                i_type iu = ::xsimd::bitwise_cast<i_type>(uf);
                iu += 0x3f800000 - 0x3f3504f3;
                i_type k = (iu >> 23) - 0x7f;
                iu = (iu & i_type(0x007fffff)) + 0x3f3504f3;
                batch_type f = --(::xsimd::bitwise_cast<batch_type>(iu));
                batch_type s = f / (batch_type(2.) + f);
                batch_type z = s * s;
                batch_type w = z * z;
                batch_type t1 = w * detail::horner<batch_type, 0x3eccce13, 0x3e789e26>(w);
                batch_type t2 = z * detail::horner<batch_type, 0x3f2aaaaa, 0x3e91e9ee>(w);
                batch_type R = t2 + t1;
                batch_type hfsq = batch_type(0.5) * f * f;
                batch_type dk = to_float(k);
                /* correction term ~ log(1+x)-log(u), avoid underflow in c/u */
                batch_type c = select(bool_cast(k >= i_type(2)), batch_type(1.) - (uf - self), self - (uf - batch_type(1.))) / uf;
                batch_type r = fma(dk, constants::log_2hi<batch_type>(), fma(s, (hfsq + R), dk * constants::log_2lo<batch_type>() + c) - hfsq + f);
#ifndef XSIMD_NO_INFINITIES
                batch_type zz = select(isnez, select(self == constants::infinity<batch_type>(), constants::infinity<batch_type>(), r), constants::minusinfinity<batch_type>());
#else
                batch_type zz = select(isnez, r, constants::minusinfinity<batch_type>());
#endif
                return select(!(uf >= batch_type(0.)), constants::nan<batch_type>(), zz);
    }
    template<class A> batch<double, A> log1p(batch<double, A> const& self, requires<generic>) {
      using batch_type = batch<double, A>;
                using i_type = as_integer_t<batch_type>;
                const batch_type uf = self + batch_type(1.);
                auto isnez = (uf != batch_type(0.));
                i_type hu = ::xsimd::bitwise_cast<i_type>(uf) >> 32;
                hu += 0x3ff00000 - 0x3fe6a09e;
                i_type k = (hu >> 20) - 0x3ff;
                /* correction term ~ log(1+x)-log(u), avoid underflow in c/u */
                batch_type c = select(bool_cast(k >= i_type(2)), batch_type(1.) - (uf - self), self - (uf - batch_type(1.))) / uf;
                hu = (hu & i_type(0x000fffff)) + 0x3fe6a09e;
                batch_type f = ::xsimd::bitwise_cast<batch_type>((hu << 32) | (i_type(0xffffffff) & ::xsimd::bitwise_cast<i_type>(uf)));
                f = --f;
                batch_type hfsq = batch_type(0.5) * f * f;
                batch_type s = f / (batch_type(2.) + f);
                batch_type z = s * s;
                batch_type w = z * z;
                batch_type t1 = w * detail::horner<batch_type,
                                  0x3fd999999997fa04ll,
                                  0x3fcc71c51d8e78afll,
                                  0x3fc39a09d078c69fll>(w);
                batch_type t2 = z * detail::horner<batch_type,
                                  0x3fe5555555555593ll,
                                  0x3fd2492494229359ll,
                                  0x3fc7466496cb03dell,
                                  0x3fc2f112df3e5244ll>(w);
                batch_type R = t2 + t1;
                batch_type dk = to_float(k);
                batch_type r = fma(dk, constants::log_2hi<batch_type>(), fma(s, hfsq + R, dk * constants::log_2lo<batch_type>() + c) - hfsq + f);
#ifndef XSIMD_NO_INFINITIES
                batch_type zz = select(isnez, select(self == constants::infinity<batch_type>(), constants::infinity<batch_type>(), r), constants::minusinfinity<batch_type>());
#else
                batch_type zz = select(isnez, r, constants::minusinfinity<batch_type>());
#endif
                return select(!(uf >= batch_type(0.)), constants::nan<batch_type>(), zz);
    }

    // mul
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> mul(batch<T, A> const& self, batch<T, A> const& other, requires<generic>) {
      constexpr std::size_t size = batch<T, A>::size;
      alignas(A::alignment()) T self_buffer[size];
      alignas(A::alignment()) T other_buffer[size];
      self.store_aligned(&self_buffer[0]);
      other.store_aligned(&other_buffer[0]);
      for(std::size_t i = 0; i < size; ++i)
        self_buffer[i] *= other_buffer[i];
      return batch<T, A>::load_aligned(self_buffer);
    }

    // nearbyint
    template<class A, class T> batch<T, A> nearbyint(batch<T, A> const& self, requires<generic>) {
      using batch_type = batch<T, A>;
      batch_type s = bitofsign(self);
      batch_type v = self ^ s;
      batch_type t2n = constants::twotonmb<batch_type>();
      batch_type d0 = v + t2n;
      return s ^ select(v < t2n, d0 - t2n, v);
    }

    // neq
    template<class A, class T> batch_bool<T, A> neq(batch<T, A> const& self, batch<T, A> const& other, requires<generic>) {
      return !(other == self);
    }

    // nextafter
    namespace detail
    {
        template <class T, class A, bool is_int = std::is_integral<T>::value>
        struct nextafter_kernel
        {
            using batch_type = batch<T, A>;

            static inline batch_type next(batch_type const& b) noexcept
            {
                return b;
            }

            static inline batch_type prev(batch_type const& b) noexcept
            {
                return b;
            }
        };

        template <class T, class A>
        struct bitwise_cast_batch;

        template <class A>
        struct bitwise_cast_batch<float, A>
        {
            using type = batch<int32_t, A>;
        };

        template <class A>
        struct bitwise_cast_batch<double, A>
        {
            using type = batch<int64_t, A>;
        };

        template <class T, class A>
        struct nextafter_kernel<T, A, false>
        {
            using batch_type = batch<T, A>;
            using int_batch = typename bitwise_cast_batch<T, A>::type;
            using int_type = typename int_batch::value_type;

            static inline batch_type next(const batch_type& b) noexcept
            {
                batch_type n = ::xsimd::bitwise_cast<batch_type>(::xsimd::bitwise_cast<int_batch>(b) + int_type(1));
                return select(b == constants::infinity<batch_type>(), b, n);
            }

            static inline batch_type prev(const batch_type& b) noexcept
            {
                batch_type p = ::xsimd::bitwise_cast<batch_type>(::xsimd::bitwise_cast<int_batch>(b) - int_type(1));
                return select(b == constants::minusinfinity<batch_type>(), b, p);
            }
        };
    }
    template<class A, class T> batch<T, A> nextafter(batch<T, A> const& from, batch<T, A> const& to, requires<generic>) {
      using kernel = detail::nextafter_kernel<T, A>;
      return select(from == to, from,
                    select(to > from, kernel::next(from), kernel::prev(from)));
    }

    // pow
    template<class A, class T> batch<T, A> pow(batch<T, A> const& self, batch<T, A> const& other, requires<generic>) {
        using batch_type = batch<T, A>;
        auto negx = self < batch_type(0.);
        batch_type z = exp(other * log(abs(self)));
        z = select(is_odd(other) && negx, -z, z);
        auto invalid = negx && !(is_flint(other) || isinf(other));
        return select(invalid, constants::nan<batch_type>(), z);
    }


    // remainder
    template<class A, class T> batch<T, A> remainder(batch<T, A> const& self, batch<T, A> const& other, requires<generic>) {
      return fnma(nearbyint(self / other), other, self);
    }

    // round
    template<class A, class T> batch<T, A> round(batch<T, A> const& self, requires<generic>) {
      auto v = abs(self);
      auto c = ceil(v);
      auto cp = select(c - 0.5 > v, c - 1, c);
      return select(v > constants::maxflint<batch<T, A>>(), self, copysign(cp, self));
    }

    // sign
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type> batch<T, A> sign(batch<T, A> const& self, requires<generic>) {
      using batch_type = batch<T, A>;
      batch_type res = select(self > batch_type(0), batch_type(1), batch_type(0)) - select(self < batch_type(0), batch_type(1), batch_type(0));
      return res;
    }

    namespace detail {
    template<class T, class A> batch<T, A> signf(batch<T, A> const& self) {
      using batch_type = batch<T, A>;
      batch_type res = select(self > batch_type(0.f), batch_type(1.f), batch_type(0.f)) - select(self < batch_type(0.f), batch_type(1.f), batch_type(0.f));
#ifdef XSIMD_NO_NANS
      return res;
#else
      return select(isnan(self), constants::nan<batch_type>(), res);
#endif
    }
    }

    template<class A> batch<float, A> sign(batch<float, A> const& self, requires<generic>) {
      return detail::signf(self);
    }
    template<class A> batch<double, A> sign(batch<double, A> const& self, requires<generic>) {
      return detail::signf(self);
    }

    // signnz
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type> batch<T, A> signnz(batch<T, A> const& self, requires<generic>) {
      using batch_type = batch<T, A>;
      return (self >> (sizeof(T) * 8 - 1)) | batch_type(1.);
    }

    namespace detail {
    template<class T, class A> batch<T, A> signnzf(batch<T, A> const& self) {
      using batch_type = batch<T, A>;
#ifndef XSIMD_NO_NANS
                return select(isnan(self), constants::nan<batch_type>(), batch_type(1.) | (constants::signmask<batch_type>() & self));
#else
                return batch_type(1.) | (constants::signmask<batch_type>() & self);
#endif
    }
    }

    template<class A> batch<float, A> signnz(batch<float, A> const& self, requires<generic>) {
      return detail::signnzf(self);
    }
    template<class A> batch<double, A> signnz(batch<double, A> const& self, requires<generic>) {
      return detail::signnzf(self);
    }

    // sin
    template<class A, class T> batch<T, A> sin(batch<T, A> const& self, requires<generic>) {
      using batch_type = batch<T, A>;
                const batch_type x = abs(self);
                batch_type xr = constants::nan<batch_type>();
                const batch_type n = detail::trigo_reducer<batch_type, detail::trigo_radian_tag>::reduce(x, xr);
                auto tmp = select(n >= batch_type(2.), batch_type(1.), batch_type(0.));
                auto swap_bit = fma(batch_type(-2.), tmp, n);
                auto sign_bit = bitofsign(self) ^ select(tmp != batch_type(0.), constants::signmask<batch_type>(), batch_type(0.));
                const batch_type z = xr * xr;
                const batch_type se = detail::sin_eval(z, xr);
                const batch_type ce = detail::cos_eval(z);
                const batch_type z1 = select(swap_bit == batch_type(0.), se, ce);
                return z1 ^ sign_bit;
    }

    // sincos
    template<class A, class T> std::pair<batch<T, A>, batch<T, A>> sincos(batch<T, A> const& self, requires<generic>) {
      using batch_type = batch<T, A>;
                const batch_type x = abs(self);
                batch_type xr = constants::nan<batch_type>();
                const batch_type n = detail::trigo_reducer<batch_type>::reduce(x, xr);
                auto tmp = select(n >= batch_type(2.), batch_type(1.), batch_type(0.));
                auto swap_bit = fma(batch_type(-2.), tmp, n);
                const batch_type z = xr * xr;
                const batch_type se = detail::sin_eval(z, xr);
                const batch_type ce = detail::cos_eval(z);
                auto sin_sign_bit = bitofsign(self) ^ select(tmp != batch_type(0.), constants::signmask<batch_type>(), batch_type(0.));
                const batch_type sin_z1 = select(swap_bit == batch_type(0.), se, ce);
                auto cos_sign_bit = select((swap_bit ^ tmp) != batch_type(0.), constants::signmask<batch_type>(), batch_type(0.));
                const batch_type cos_z1 = select(swap_bit != batch_type(0.), se, ce);
                return std::make_pair(sin_z1 ^ sin_sign_bit, cos_z1 ^ cos_sign_bit);
    }

    // sinh
    template<class A> batch<float, A> sinh(batch<float, A> const& self, requires<generic>) {
      using batch_type = batch<float, A>;
                batch_type sqr_self = self * self;
                return detail::horner<batch_type,
                              0x3f800000,  // 1.0f
                              0x3e2aaacc,  // 1.66667160211E-1f
                              0x3c087bbe,  // 8.33028376239E-3f
                              0x39559e2f  // 2.03721912945E-4f
                              >(sqr_self) *
                    self;
    }

    template<class A> batch<double, A> sinh(batch<double, A> const& self, requires<generic>) {
      using batch_type = batch<double, A>;
                batch_type sqrself = self * self;
                return fma(self, (detail::horner<batch_type,
                                      0xc115782bdbf6ab05ull,  //  -3.51754964808151394800E5
                                      0xc0c694b8c71d6182ull,  //  -1.15614435765005216044E4,
                                      0xc064773a398ff4feull,  //  -1.63725857525983828727E2,
                                      0xbfe9435fe8bb3cd6ull  //  -7.89474443963537015605E-1
                                      >(sqrself) /
                               detail::horner1<batch_type,
                                       0xc1401a20e4f90044ull,  //  -2.11052978884890840399E6
                                       0x40e1a7ba7ed72245ull,  //   3.61578279834431989373E4,
                                       0xc0715b6096e96484ull  //  -2.77711081420602794433E2,
                                       >(sqrself)) *
                               sqrself,
                           self);
    }

    // store_aligned
    template<class A, class T_in, class T_out> void store_aligned(T_out *mem, batch<T_in, A> const& self, requires<generic>) {
      static_assert(!std::is_same<T_in, T_out>::value, "there should be a direct store for this type combination");
      alignas(A::alignment()) T_in buffer[batch<T_in, A>::size];
      store_aligned(&buffer[0], self);
      std::copy(std::begin(buffer), std::end(buffer), mem);
    }

    // store_unaligned
    template<class A, class T_in, class T_out> void store_unaligned(T_out *mem, batch<T_in, A> const& self, requires<generic>) {
      static_assert(!std::is_same<T_in, T_out>::value, "there should be a direct store for this type combination");
      return store_aligned<A>(mem, self, generic{});
    }

    // tan
    template<class A, class T> batch<T, A> tan(batch<T, A> const& self, requires<generic>) {
      using batch_type = batch<T, A>;
                const batch_type x = abs(self);
                batch_type xr = constants::nan<batch_type>();
                const batch_type n = detail::trigo_reducer<batch_type>::reduce(x, xr);
                auto tmp = select(n >= batch_type(2.), batch_type(1.), batch_type(0.));
                auto swap_bit = fma(batch_type(-2.), tmp, n);
                auto test = (swap_bit == batch_type(0.));
                const batch_type y = detail::tan_eval(xr, test);
                return y ^ bitofsign(self);
    }

    // tanh
    namespace detail {
        template <class B>
        struct tanh_kernel;

        template <class A>
        struct tanh_kernel<batch<float, A>>
        {
          using batch_type = batch<float, A>;
            static inline batch_type tanh(const batch_type& x)
            {
                batch_type sqrx = x * x;
                return fma(detail::horner<batch_type,
                                  0xbeaaaa99,  //    -3.33332819422E-1F
                                  0x3e088393,  //    +1.33314422036E-1F
                                  0xbd5c1e2d,  //    -5.37397155531E-2F
                                  0x3ca9134e,  //    +2.06390887954E-2F
                                  0xbbbaf0ea  //    -5.70498872745E-3F
                                  >(sqrx) *
                               sqrx,
                           x, x);
            }

            static inline batch_type cotanh(const batch_type& x)
            {
                return batch_type(1.) / tanh(x);
            }
        };

        template <class A>
        struct tanh_kernel<batch<double, A>>
        {
          using batch_type = batch<double, A>;
            static inline batch_type tanh(const batch_type& x)
            {
                batch_type sqrx = x * x;
                return fma(sqrx * p(sqrx) / q(sqrx), x, x);
            }

            static inline batch_type cotanh(const batch_type& x)
            {
                batch_type sqrx = x * x;
                batch_type qval = q(sqrx);
                return qval / (x * fma(p(sqrx), sqrx, qval));
            }

            static inline batch_type p(const batch_type& x)
            {
                return detail::horner<batch_type,
                              0xc0993ac030580563,  // -1.61468768441708447952E3
                              0xc058d26a0e26682d,  // -9.92877231001918586564E1,
                              0xbfeedc5baafd6f4b  // -9.64399179425052238628E-1
                              >(x);
            }

            static inline batch_type q(const batch_type& x)
            {
                return detail::horner1<batch_type,
                               0x40b2ec102442040c,  //  4.84406305325125486048E3
                               0x40a176fa0e5535fa,  //  2.23548839060100448583E3,
                               0x405c33f28a581B86  //  1.12811678491632931402E2,
                               >(x);
            }
        };

    }
    template<class A, class T> batch<T, A> tanh(batch<T, A> const& self, requires<generic>) {
      using batch_type = batch<T, A>;
                batch_type one(1.);
                batch_type x = abs(self);
                auto test = x < (batch_type(5.) / batch_type(8.));
                batch_type bts = bitofsign(self);
                batch_type z = one;
                if (any(test))
                {
                    z = detail::tanh_kernel<batch_type>::tanh(x);
                    if (all(test))
                        return z ^ bts;
                }
                batch_type r = fma(batch_type(-2.), one / (one + exp(x + x)), one);
                return select(test, z, r) ^ bts;
    }

    // trunc
    template<class A, class T> batch<T, A> trunc(batch<T, A> const& self, requires<generic>) {
      return select(abs(self) < constants::maxflint<batch<T, A>>(), to_float(to_int(self)), self);
    }


  }

}

#endif

