#ifndef XSIMD_GENERIC_HPP
#define XSIMD_GENERIC_HPP

#include "../types/xsimd_generic_arch.hpp"
#include "../types/xsimd_utils.hpp"
#include "./xsimd_constants.hpp"

#include <limits>
#include <tuple>


namespace xsimd {
  // Forward declaration. Should we put them in a separate file?
  template<class T, class A>
  batch<T, A> abs(batch<T, A> const& self);
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
        return batch<T, A>((T)0);
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

    // fdim
    template<class A, class T> batch<T, A> fdim(batch<T, A> const& self, batch<T, A> const& other, requires<generic>) {
      return fmax(batch<T, A>((T)0), self - other);
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
        i_type r1 = m1f & bitwise_cast<i_type>(self);
        batch_type x = self & bitwise_cast<batch_type>(~m1f);
        exp = (r1 >> constants::nmb<batch_type>()) - constants::maxexponentm1<batch_type>();
        exp = select(bool_cast(self != batch_type(0.)), exp, i_type(typename i_type::value_type(0)));
        return select((self != batch_type(0.)), x | bitwise_cast<batch_type>(constants::mask2frexp<batch_type>()), batch_type(0.));
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
      return (self - self) == batch<T, A>((T)0);
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
        return self * bitwise_cast<batch_type>(ik);
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
      i_type ix = bitwise_cast<i_type>(x);
      ix += 0x3f800000 - 0x3f3504f3;
      k += (ix >> 23) - 0x7f;
      ix = (ix & i_type(0x007fffff)) + 0x3f3504f3;
      x = bitwise_cast<batch_type>(ix);
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
                i_type hx = bitwise_cast<i_type>(x) >> 32;
                i_type k((typename i_type::value_type)0);
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
                x = bitwise_cast<batch_type>(hx << 32 | (i_type(0xffffffff) & bitwise_cast<i_type>(x)));

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
      batch_type res = select(self > batch_type((T)0), batch_type((T)1), batch_type((T)0)) - select(self < batch_type((T)0), batch_type((T)1), batch_type((T)0));
      return res;
    }

    template<class A> batch<float, A> sign(batch<float, A> const& self, requires<generic>) {
      using batch_type = batch<float, A>;
      batch_type res = select(self > batch_type(0.f), batch_type(1.f), batch_type(0.f)) - select(self < batch_type(0.f), batch_type(1.f), batch_type(0.f));
#ifdef XSIMD_NO_NANS
      return res;
#else
      return select(isnan(self), constants::nan<batch_type>(), res);
#endif
    }
    template<class A> batch<double, A> sign(batch<double, A> const& self, requires<generic>) {
      using batch_type = batch<double, A>;
      batch_type res = select(self > batch_type(0.), batch_type(1.), batch_type(0.)) - select(self < batch_type(0.), batch_type(1.), batch_type(0.));
#ifdef XSIMD_NO_NANS
      return res;
#else
      return select(isnan(self), constants::nan<batch_type>(), res);
#endif
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

    // trunc
    template<class A, class T> batch<T, A> trunc(batch<T, A> const& self, requires<generic>) {
      return select(abs(self) < constants::maxflint<batch<T, A>>(), to_float(to_int(self)), self);
    }


  }

}

#endif

