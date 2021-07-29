#ifndef XSIMD_GENERIC_MEMORY_HPP
#define XSIMD_GENERIC_MEMORY_HPP

#include "./xsimd_generic_details.hpp"


namespace xsimd {

  namespace kernel {

    using namespace types;

    // extract_pair
    template<class A, class T> batch<T, A> extract_pair(batch<T, A> const& self, batch<T, A> const& other, std::size_t i, requires_arch<generic>) {
      constexpr std::size_t size = batch<T, A>::size;
      assert(0<= i && i< size && "index in bounds");

          alignas(A::alignment()) T self_buffer[size];
          self.store_aligned(self_buffer);

          alignas(A::alignment()) T other_buffer[size];
          other.store_aligned(other_buffer);

          alignas(A::alignment()) T concat_buffer[size];

          for (std::size_t j = 0 ; j < (size - i); ++j)
          {
              concat_buffer[j] = self_buffer[i + j];
              if(j < i)
              {
                  concat_buffer[size - 1 - j] = other_buffer[i - 1 - j];
              }
          }
          return batch<T, A>::load_aligned(concat_buffer);
    }

    // load_aligned
    namespace detail {
      template<class A, class T_in, class T_out>
      batch<T_out, A> load_aligned(T_in const* mem, convert<T_out>, requires_arch<generic>, with_fast_conversion) {
        using batch_type_in = batch<T_in, A>;
        using batch_type_out = batch<T_out, A>;
        return fast_cast(batch_type_in::load_aligned(mem), batch_type_out(), A{});
      }
      template<class A, class T_in, class T_out>
      batch<T_out, A> load_aligned(T_in const* mem, convert<T_out>, requires_arch<generic>, with_slow_conversion) {
        static_assert(!std::is_same<T_in, T_out>::value, "there should be a direct load for this type combination");
        using batch_type_out = batch<T_out, A>;
        alignas(A::alignment()) T_out buffer[batch_type_out::size];
        std::copy(mem, mem + batch_type_out::size, std::begin(buffer));
        return batch_type_out::load_aligned(buffer);
      }
    }
    template<class A, class T_in, class T_out>
    batch<T_out, A> load_aligned(T_in const* mem, convert<T_out> cvt, requires_arch<generic>) {
      return detail::load_aligned<A>(mem, cvt, A{}, detail::conversion_type<A, T_in, T_out>{});
    }

    // load_unaligned
    namespace detail {
      template<class A, class T_in, class T_out>
      batch<T_out, A> load_unaligned(T_in const* mem, convert<T_out>, requires_arch<generic>, with_fast_conversion) {
        using batch_type_in = batch<T_in, A>;
        using batch_type_out = batch<T_out, A>;
        return fast_cast(batch_type_in::load_unaligned(mem), batch_type_out(), A{});
      }

      template<class A, class T_in, class T_out>
      batch<T_out, A> load_unaligned(T_in const* mem, convert<T_out> cvt, requires_arch<generic>, with_slow_conversion) {
        static_assert(!std::is_same<T_in, T_out>::value, "there should be a direct load for this type combination");
        return load_aligned<A>(mem, cvt, generic{}, with_slow_conversion{});
      }
    }
    template<class A, class T_in, class T_out>
    batch<T_out, A> load_unaligned(T_in const* mem, convert<T_out> cvt, requires_arch<generic>) {
      return detail::load_unaligned<A>(mem, cvt, generic{}, detail::conversion_type<A, T_in, T_out>{});
    }


    // store
    template<class T, class A>
    void store(batch_bool<T, A> const& self, bool* mem, requires_arch<generic>) {
      using batch_type = batch<T, A>;
      constexpr auto size = batch_bool<T, A>::size;
      alignas(A::alignment()) T buffer[size];
      kernel::store_aligned<A>(&buffer[0], batch_type(self), A{});
      for(std::size_t i = 0; i < size; ++i)
        mem[i] = bool(buffer[i]);
    }


    // store_aligned
    template<class A, class T_in, class T_out> void store_aligned(T_out *mem, batch<T_in, A> const& self, requires_arch<generic>) {
      static_assert(!std::is_same<T_in, T_out>::value, "there should be a direct store for this type combination");
      alignas(A::alignment()) T_in buffer[batch<T_in, A>::size];
      store_aligned(&buffer[0], self);
      std::copy(std::begin(buffer), std::end(buffer), mem);
    }

    // store_unaligned
    template<class A, class T_in, class T_out> void store_unaligned(T_out *mem, batch<T_in, A> const& self, requires_arch<generic>) {
      static_assert(!std::is_same<T_in, T_out>::value, "there should be a direct store for this type combination");
      return store_aligned<A>(mem, self, generic{});
    }

    namespace detail
    {
        template <class A, class T>
        batch<std::complex<T>, A> load_complex(batch<T, A> const& /*hi*/, batch<T, A> const& /*lo*/, requires_arch<generic>)
        {
            throw std::runtime_error("load_complex not implemented");
        }

        template <class A, class T>
        batch<T, A> complex_high(batch<std::complex<T>, A> const& /*src*/, requires_arch<generic>)
        {
            throw std::runtime_error("complex_high not implemented");
        }

        template <class A, class T>
        batch<T, A> complex_low(batch<std::complex<T>, A> const& /*src*/, requires_arch<generic>)
        {
            throw std::runtime_error("complex_low not implemented");
        }
    }

    // load_complex_aligned
    template <class A, class T> batch<std::complex<T>, A> load_complex_aligned(std::complex<T> const* mem, requires_arch<generic>) {
      using real_batch = batch<T, A>;
      T const *buffer = reinterpret_cast<T const *>(mem);
      real_batch hi = real_batch::load_aligned(buffer),
                 lo = real_batch::load_aligned(buffer + real_batch::size);
      return detail::load_complex(hi, lo, A{});
    }

    // load_complex_unaligned
    template <class A, class T> batch<std::complex<T>, A> load_complex_unaligned(std::complex<T> const* mem, requires_arch<generic>) {
      using real_batch = batch<T, A>;
      T const *buffer = reinterpret_cast<T const *>(mem);
      real_batch hi = real_batch::load_unaligned(buffer),
                 lo = real_batch::load_unaligned(buffer + real_batch::size);
      return detail::load_complex(hi, lo, A{});
    }

    // store_complex_aligned
    template <class A, class T> void store_complex_aligned(std::complex<T>* dst, batch<std::complex<T>, A> const& src, requires_arch<generic>) {
        using real_batch = batch<T, A>;
        real_batch hi = detail::complex_high(src, A{});
        real_batch lo = detail::complex_low(src, A{});
        T * buffer = reinterpret_cast<T*>(dst);
        lo.store_aligned(buffer);
        hi.store_aligned(buffer + real_batch::size);
    }

    // store_compelx_unaligned
    template <class A, class T> void store_complex_unaligned(std::complex<T>* dst, batch<std::complex<T>, A> const& src, requires_arch<generic>) {
        using real_batch = batch<T, A>;
        real_batch hi = detail::complex_high(src, A{});
        real_batch lo = detail::complex_low(src, A{});
        T * buffer = reinterpret_cast<T *>(dst);
        lo.store_unaligned(buffer);
        hi.store_unaligned(buffer + real_batch::size);
    }

  }

}

#endif

