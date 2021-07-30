#ifndef XSIMD_GENERIC_FWD_HPP
#define XSIMD_GENERIC_FWD_HPP
namespace xsimd {

  namespace kernel {
    // forward declaration
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> abs(batch<T, A> const& self, requires_arch<generic>);
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_lshift(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>);
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_rshift(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>);
    template<class A, class T> batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>);
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> mul(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>);

  }
}

#endif

