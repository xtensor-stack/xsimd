#ifndef XSIMD_GENERIC_DETAILS_HPP
#define XSIMD_GENERIC_DETAILS_HPP

#include "../../types/xsimd_generic_arch.hpp"
#include "../../types/xsimd_utils.hpp"
#include "../../math/xsimd_rem_pio2.hpp"
#include "../xsimd_constants.hpp"

#include <limits>
#include <tuple>


namespace xsimd {
  // Forward declaration. Should we put them in a separate file?
  template<class T, class A>
  batch<T, A> abs(batch<T, A> const& self);
  template<class T, class A>
  batch<T, A> abs(batch<std::complex<T>, A> const& self);
  template<class T, class A>
  bool any(batch_bool<T, A> const& self);
  template<class T, class A>
  batch<T, A> atan2(batch<T, A> const& self, batch<T, A> const& other);
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
  batch<T, A> cos(batch<T, A> const& self);
  template<class T, class A>
  batch<T, A> cosh(batch<T, A> const& self);
  template<class T, class A>
  batch<T, A> exp(batch<T, A> const& self);
  template<class T, class A>
  batch<T, A> fma(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z);
  template<class T, class A>
  batch<T, A> fms(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z);
  template<class T, class A>
  batch<T, A> frexp(const batch<T, A>& x, const batch<as_integer_t<T>, A>& e);
  template<class T, class A>
  T hadd(batch<T, A> const&);
  template <class T, class A, uint64_t... Coefs>
  batch<T, A> horner(const batch<T, A>& self);
  template <class T, class A>
  batch<T, A> hypot(const batch<T, A>& self);
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
  batch<std::complex<T>, A> select(batch_bool<T, A> const&, batch<std::complex<T>, A> const& , batch<std::complex<T>, A> const& );
  template<class T, class A>
  batch<T, A> sign(batch<T, A> const& self);
  template<class T, class A>
  batch<T, A> signnz(batch<T, A> const& self);
  template<class T, class A>
  batch<T, A> sin(batch<T, A> const& self);
  template<class T, class A>
  batch<T, A> sinh(batch<T, A> const& self);
  template<class T, class A>
  std::pair<batch<T, A>, batch<T, A>> sincos(batch<T, A> const& self);
  template<class T, class A>
  batch<T, A> sqrt(batch<T, A> const& self);
  template<class T, class A>
  batch<T, A> tan(batch<T, A> const& self);
  template<class T, class A>
  batch<as_float_t<T>, A> to_float(batch<T, A> const& self);
  template<class T, class A>
  batch<as_integer_t<T>, A> to_int(batch<T, A> const& self);
  template<class T, class A>
  batch<T, A> trunc(batch<T, A> const& self);


  namespace kernel {

    namespace detail {
      template<class F, class A, class T, class... Batches>
      batch<T, A> apply(F&& func, batch<T, A> const& self, batch<T, A> const& other) {
        constexpr std::size_t size = batch<T, A>::size;
        alignas(A::alignment()) T self_buffer[size];
        alignas(A::alignment()) T other_buffer[size];
        self.store_aligned(&self_buffer[0]);
        other.store_aligned(&other_buffer[0]);
        for(std::size_t i = 0; i < size; ++i) {
          self_buffer[i] = func(self_buffer[i], other_buffer[i]);
        }
        return batch<T, A>::load_aligned(self_buffer);
      }
    }

    namespace detail {
      // Generic conversion handling machinery. Each architecture must define
      // conversion function when such conversions exits in the form of
      // intrinsic. Then we use that information to automatically decide whether
      // to use scalar or vector conversion when doing load / store / batch_cast
      struct with_fast_conversion{};
      struct with_slow_conversion{};

      template <class A, class From, class To, class = void>
      struct conversion_type_impl
      {
          using type = with_slow_conversion;
      };

      using xsimd::detail::void_t;

      template <class A, class From, class To>
      struct conversion_type_impl<A, From, To,
                void_t<decltype(fast_cast(std::declval<const From&>(), std::declval<const To&>(), std::declval<const A&>()))>>
      {
          using type = with_fast_conversion;
      };

      template <class A, class From, class To>
      using conversion_type = typename conversion_type_impl<A, From, To>::type;
    }

    namespace detail {
    /* origin: boost/simdfunction/horn.hpp*/
    /*
     * ====================================================
     * copyright 2016 NumScale SAS
     *
     * Distributed under the Boost Software License, Version 1.0.
     * (See copy at http://boost.org/LICENSE_1_0.txt)
     * ====================================================
     */
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

    /* origin: boost/simdfunction/horn1.hpp*/
    /*
     * ====================================================
     * copyright 2016 NumScale SAS
     *
     * Distributed under the Boost Software License, Version 1.0.
     * (See copy at http://boost.org/LICENSE_1_0.txt)
     * ====================================================
     */
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



  }

}

#endif

