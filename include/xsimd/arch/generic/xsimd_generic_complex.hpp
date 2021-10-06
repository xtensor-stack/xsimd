#ifndef XSIMD_GENERIC_COMPLEX_HPP
#define XSIMD_GENERIC_COMPLEX_HPP

#include "./xsimd_generic_details.hpp"

namespace xsimd {

  namespace kernel {

    using namespace types;

    namespace detail
    {
      template <class B, bool is_complex>
      struct real_imag_kernel
      {
        using return_type = typename B::real_batch;

        static return_type real(B const& z)
        {
          return z.real();
        }

        static return_type imag(B const& z)
        {
          return z.imag();
        }
      };

      template <class B>
      struct real_imag_kernel<B, false>
      {
        using return_type = B;

        static return_type real(B const& z)
        {
          return z;
        }

        static return_type imag(B const&)
        {
          return B(typename B::value_type(0));
        }
      };
    }

    // real
    template <class A, class T>
    real_batch_type_t<batch<T, A>> real(batch<T, A> const& self, requires_arch<generic>) {
      using batch_type = batch<T, A>;
      constexpr bool is_cplx = xsimd::detail::is_complex<typename batch_type::value_type>::value;
      return detail::real_imag_kernel<batch_type, is_cplx>::real(self);
    }

    // imag
    template <class A, class T>
    real_batch_type_t<batch<T, A>> imag(batch<T, A> const& self, requires_arch<generic>) {
      using batch_type = batch<T, A>;
      constexpr bool is_cplx = xsimd::detail::is_complex<typename batch_type::value_type>::value;
      return detail::real_imag_kernel<batch_type, is_cplx>::imag(self);
    }

    // arg
    template<class A, class T>
    real_batch_type_t<batch<T, A>> arg(batch<T, A> const& self, requires_arch<generic>) {
      return atan2(imag(self), real(self));
    }

    // conj
    template<class A, class T>
    complex_batch_type_t<batch<T, A>> conj(batch<T, A> const& self, requires_arch<generic>) {
      return {real(self), - imag(self)};
    }

    // norm
    template<class A, class T>
    real_batch_type_t<batch<T, A>> norm(batch<T, A> const& self, requires_arch<generic>) {
      return {fma(real(self), real(self), imag(self) * imag(self))};
    }

    // proj
    template<class A, class T>
    complex_batch_type_t<batch<T, A>> proj(batch<T, A> const& self, requires_arch<generic>) {
      using batch_type = complex_batch_type_t<batch<T, A>>;
      using real_batch = typename batch_type::real_batch;
      using real_value_type = typename real_batch::value_type;
      auto cond = xsimd::isinf(real(self)) || xsimd::isinf(imag(self));
      return select(cond,
                    batch_type(constants::infinity<real_batch>(),
                               copysign(real_batch(real_value_type(0)), imag(self))),
                    batch_type(self));
    }
  }
}

#endif

