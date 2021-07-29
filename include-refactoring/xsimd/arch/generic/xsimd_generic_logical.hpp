#ifndef XSIMD_GENERIC_LOGICAL_HPP
#define XSIMD_GENERIC_LOGICAL_HPP

#include "./xsimd_generic_details.hpp"


namespace xsimd {

  namespace kernel {

    using namespace types;

    // ge
    template<class A, class T> batch_bool<T, A> ge(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>) {
      return other <= self;
    }

    // gt
    template<class A, class T> batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>) {
      return other < self;
    }

    // is_even
    template<class A, class T> batch_bool<T, A> is_even(batch<T, A> const& self, requires_arch<generic>) {
      return is_flint(self * T(0.5));
    }

    // is_flint
    template<class A, class T> batch_bool<T, A> is_flint(batch<T, A> const& self, requires_arch<generic>) {
      auto frac = select(isnan(self - self), constants::nan<batch<T, A>>(), self - trunc(self));
      return frac == T(0.);
    }

    // is_odd
    template<class A, class T> batch_bool<T, A> is_odd(batch<T, A> const& self, requires_arch<generic>) {
      return is_even(self - T(1.));
    }

    // isinf
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> isinf(batch<T, A> const& , requires_arch<generic>) {
      return batch_bool<T, A>(false);
    }
    template<class A> batch_bool<float, A> isinf(batch<float, A> const& self, requires_arch<generic>) {
      return abs(self) == std::numeric_limits<float>::infinity();
    }
    template<class A> batch_bool<double, A> isinf(batch<double, A> const& self, requires_arch<generic>) {
      return abs(self) == std::numeric_limits<double>::infinity();
    }

    // isfinite
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> isfinite(batch<T, A> const& , requires_arch<generic>) {
      return batch_bool<T, A>(true);
    }
    template<class A> batch_bool<float, A> isfinite(batch<float, A> const& self, requires_arch<generic>) {
      return (self - self) == 0;
    }
    template<class A> batch_bool<double, A> isfinite(batch<double, A> const& self, requires_arch<generic>) {
      return (self - self) == 0;
    }

    // isnan
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> isnan(batch<T, A> const& , requires_arch<generic>) {
      return batch_bool<T, A>(false);
    }

    // le
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> le(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>) {
      return (self < other) || (self == other);
    }


    // neq
    template<class A, class T> batch_bool<T, A> neq(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>) {
      return !(other == self);
    }
  }
}

#endif

