#ifndef XSIMD_API_HPP
#define XSIMD_API_HPP

#include "../types/xsimd_batch.hpp"
#include "../isa/xsimd_isa.hpp"

namespace xsimd {

// high level free functions
//

template<class T, class A>
batch<T, A> abs(batch<T, A> const& self) {
  return kernel::abs<A>(self, A{});
}

template<class T, class Tp>
auto add(T const& self, Tp const& other) -> decltype(self + other){
  return self + other;
}

template<class T, class Tp>
auto bitwise_and(T const& self, Tp const& other) -> decltype(self & other){
  return self & other;
}

// FIXME: I don't understand the use of this
template<class T, class A>
batch_bool<T, A> bitwise_cast(batch_bool<T, A> const& self) {
  return {self};
}

template<class T, class A>
batch<T, A> bitwise_not(batch<T, A> const& self) {
  return kernel::bitwise_not<A>(self, A{});
}

template<class T, class Tp>
auto div(T const& self, Tp const& other) -> decltype(self / other){
  return self / other;
}

template<class T, class A>
batch_bool<T, A> eq(batch<T, A> const& self, batch<T, A> const& other) {
  return self == other;
}

template<class T, class A>
batch<T, A> fabs(batch<T, A> const& self) {
  return kernel::abs<A>(self, A{});
}

template<class T, class A>
batch<T, A> fma(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z) {
  return kernel::fma<A>(x, y, z, A{});
}

template<class T, class A>
batch<T, A> fmax(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::max<A>(self, other, A{});
}

template<class T, class A>
batch<T, A> fmin(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::min<A>(self, other, A{});
}

template<class T, class A>
batch<T, A> fms(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z) {
  return kernel::fms<A>(x, y, z, A{});
}

template<class T, class A>
batch<T, A> fnma(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z) {
  return kernel::fnma<A>(x, y, z, A{});
}

template<class T, class A>
batch<T, A> fnms(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z) {
  return kernel::fnms<A>(x, y, z, A{});
}

template<class T, class A>
batch_bool<T, A> ge(batch<T, A> const& self, batch<T, A> const& other) {
  return self >= other;
}

template<class T, class A>
batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other) {
  return self > other;
}

template<class T, class A>
T hadd(batch<T, A> const& self) {
  return kernel::hadd<A>(self, A{});
}

template<class T, class A>
batch<T, A> haddp(batch<T, A> const* self) {
  return kernel::haddp<A>(self, A{});
}

template<class T, class A>
batch_bool<T, A> le(batch<T, A> const& self, batch<T, A> const& other) {
  return self <= other;
}

template<class To, class A=default_arch, class From>
batch<To, A> load(From* ptr) {
  return kernel::load_aligned<A>(ptr, kernel::convert<To>{}, A{});
}

template<class To, class A=default_arch, class From>
batch<To, A> load_aligned(From* ptr) {
  return kernel::load_aligned<A>(ptr, kernel::convert<To>{}, A{});
}

template<class To, class A=default_arch, class From>
batch<To, A> load_unaligned(From* ptr) {
  return kernel::load_unaligned<A>(ptr, kernel::convert<To>{}, A{});
}

template<class T, class A>
batch_bool<T, A> lt(batch<T, A> const& self, batch<T, A> const& other) {
  return self < other;
}

template<class T, class A>
batch<T, A> max(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::max<A>(self, other, A{});
}

template<class T, class A>
batch<T, A> min(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::min<A>(self, other, A{});
}

template<class T, class Tp>
auto mul(T const& self, Tp const& other) -> decltype(self * other){
  return self * other;
}

template<class T, class A>
batch_bool<T, A> neq(batch<T, A> const& self, batch<T, A> const& other) {
  return self != other;
}

template<class T, class A>
batch<T, A> neg(batch<T, A> const& self) {
  return -self;
}

template<class T, class A>
batch<T, A> pos(batch<T, A> const& self) {
  return +self;
}

template<class T, class A>
batch<T, A> sqrt(batch<T, A> const& self) {
  return kernel::sqrt<A>(self, A{});
}

template<class T, class A>
void store(T* mem, batch<T, A> const& val) {
  return kernel::store_aligned<A>(mem, val, A{});
}

template<class T, class A>
void store_aligned(T* mem, batch<T, A> const& val) {
  return kernel::store_aligned<A>(mem, val, A{});
}

template<class T, class A>
void store_unaligned(T* mem, batch<T, A> const& val) {
  return kernel::store_unaligned<A>(mem, val, A{});
}

template<class T, class Tp>
auto sadd(T const& self, Tp const& other) -> decltype(self + other) {
  using A = typename decltype(self + other)::arch_type;
  return kernel::sadd<A>(self, other, A{});
}

template<class T, class Tp>
auto ssub(T const& self, Tp const& other) -> decltype(self - other) {
  using A = typename decltype(self - other)::arch_type;
  return kernel::ssub<A>(self, other, A{});
}

template<class T, class Tp>
auto sub(T const& self, Tp const& other) -> decltype(self - other){
  return self - other;
}

}

#endif

