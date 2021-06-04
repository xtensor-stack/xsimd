#ifndef XSIMD_GENERIC_HPP
#define XSIMD_GENERIC_HPP

#include "../types/xsimd_generic_arch.hpp"
#include "./xsimd_constants.hpp"

#include <limits>


namespace xsimd {
  // Forward declaration. Should we put them in a separate file?
  template<class T, class A>
  batch<T, A> abs(batch<T, A> const& self);
  template<class T, class A>
  batch<T, A> bitofsign(batch<T, A> const& self);
  template<class B, class T, class A>
  B bitwise_cast(batch<T, A> const& self);
  template<class To, class A=default_arch, class From>
  batch<To, A> load_aligned(From const* ptr);
  template<class To, class A=default_arch, class From>
  batch<To, A> load_unaligned(From const* ptr, To*_=(From*)nullptr);
  template<class T, class A>
  batch<T, A> nearbyint(batch<T, A> const& self);
  template<class T, class A>
  batch<T, A> select(batch_bool<T, A> const&, batch<T, A> const& , batch<T, A> const& );
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

    using namespace types;
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
      return ::xsimd::load_aligned<T_out, A>(buffer_out);
    }

    }

    template<class A, class T_out, class T_in>
    batch<T_out, A> batch_cast(batch<T_in, A> const& self, batch<T_out, A> const& out, requires<generic>) {
      return detail::batch_cast(self, out, A{}, detail::conversion_type<A, T_in, T_out>{});
    }

    // bitofsign
    template<class A, class T> batch<T, A> bitofsign(batch<T, A> const& self, requires<generic>) {
      return self & constants::minuszero<batch<T, A>>();
    }

    // clip
    template<class A, class T> batch<T, A> clip(batch<T, A> const& self, batch<T, A> const& lo, batch<T, A> const& hi, requires<generic>) {
      return min(hi, max(self, lo));
    }

    // fdim
    template<class A, class T> batch<T, A> fdim(batch<T, A> const& self, batch<T, A> const& other, requires<generic>) {
      return fmax(batch<T, A>((T)0), self - other);
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

    // isinf
    template<class A, class T> batch_bool<T, A> isinf(batch<T, A> const& self, requires<generic>) {
      return abs(self) == batch<T, A>(std::numeric_limits<T>::infinity());
    }

    // isfinite
    template<class A, class T> batch_bool<T, A> isfinite(batch<T, A> const& self, requires<generic>) {
      return (self - self) == batch<T, A>((T)0);
    }

    // load_aligned
    namespace detail {
      template<class A, class T_in, class T_out>
      batch<T_out, A> load_aligned(T_in const* mem, convert<T_out>, requires<generic>, with_fast_conversion) {
        using batch_type_out = batch<T_out, A>;
        return conversion::fast(::xsimd::load_aligned<T_in, A>(mem), batch_type_out(), A{});
      }
      template<class A, class T_in, class T_out>
      batch<T_out, A> load_aligned(T_in const* mem, convert<T_out>, requires<generic>, with_slow_conversion) {
        static_assert(!std::is_same<T_in, T_out>::value, "there should be a direct load for this type combination");
        using batch_type_out = batch<T_out, A>;
        alignas(A::alignment()) T_out buffer[batch_type_out::size];
        std::copy(mem, mem + batch_type_out::size, std::begin(buffer));
        return ::xsimd::load_aligned<T_out, A>(buffer);
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
        using batch_type_out = batch<T_out, A>;
        return conversion::fast(::xsimd::load_unaligned<T_in, A>(mem), batch_type_out(), A{});
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

    // nearbyint
    template<class A, class T> batch<T, A> nearbyint(batch<T, A> const& self, requires<generic>) {
      using batch_type = batch<T, A>;
      batch_type s = bitofsign(self);
      batch_type v = self ^ s;
      batch_type t2n = constants::twotonmb<batch_type>();
      batch_type d0 = v + t2n;
      return s ^ select(v < t2n, d0 - t2n, v);
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

    // remainder
    template<class A, class T> batch<T, A> remainder(batch<T, A> const& self, batch<T, A> const& other, requires<generic>) {
      return fnma(nearbyint(self / other), other, self);
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

