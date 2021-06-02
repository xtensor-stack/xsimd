#ifndef XSIMD_API_HPP
#define XSIMD_API_HPP

#include "../types/xsimd_batch.hpp"
#include "../arch/xsimd_isa.hpp"

#include <limits>

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

template<class T, class A>
bool all(batch<T, A> const& self) {
  return kernel::all<A>(self, A{});
}

template<class T, class A>
bool any(batch<T, A> const& self) {
  return kernel::any<A>(self, A{});
}

template<class T_out, class T_in, class A>
batch<T_out, A> batch_cast(batch<T_in, A> const & self) {
  return kernel::batch_cast<A>(self, batch<T_out, A>{}, A{});
}

template<class T, class A>
batch<T, A> bitofsign(batch<T, A> const& self) {
  return kernel::bitofsign<A>(self, A{});
}

template<class T, class Tp>
auto bitwise_and(T const& self, Tp const& other) -> decltype(self & other){
  return self & other;
}

template<class B, class T, class A>
B bitwise_cast(batch<T, A> const& self) {
  return kernel::bitwise_cast<A>(self, B{}, A{});
}

template<class T, class A>
batch<T, A> bitwise_not(batch<T, A> const& self) {
  return kernel::bitwise_not<A>(self, A{});
}

template<class T, class Tp>
auto bitwise_or(T const& self, Tp const& other) -> decltype(self | other){
  return self | other;
}

template<class T, class Tp>
auto bitwise_xor(T const& self, Tp const& other) -> decltype(self ^ other){
  return self ^ other;
}

template<class A, class T>
batch<T, A> clip(batch<T, A> const& self, batch<T, A> const& lo, batch<T, A> const& hi) {
  return kernel::clip(self, lo, hi, A{});
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
batch<T, A> fdim(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::fdim<A>(self, other, A{});
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
batch<T, A> fmod(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::fmod<A>(self, other, A{});
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

template<class B>
B infinity() {
  using T = typename B::value_type;
  return B(std::numeric_limits<T>::infinity());
}

template<class T, class A>
batch_bool<T, A> isinf(batch<T, A> const& self) {
  return kernel::isinf<A>(self, A{});
}

template<class T, class A>
batch_bool<T, A> isfinite(batch<T, A> const& self) {
  return kernel::isfinite<A>(self, A{});
}

template<class T, class A>
batch_bool<T, A> le(batch<T, A> const& self, batch<T, A> const& other) {
  return self <= other;
}

template<class To, class A=default_arch, class From>
batch<To, A> load(From* ptr) {
  return kernel::load_aligned<A>(ptr, kernel::convert<To>{}, A{});
}

template<class To, class A, class From>
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

template<class B>
B minusinfinity() {
  using T = typename B::value_type;
  return B(-std::numeric_limits<T>::infinity());
}

template<class T, class Tp>
auto mul(T const& self, Tp const& other) -> decltype(self * other){
  return self * other;
}

template<class T, class A>
batch<T, A> nearbyint(batch<T, A> const& self) {
  return kernel::nearbyint<A>(self, A{});
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
batch<T, A> nextafter(batch<T, A> const& from, batch<T, A> const& to) {
  return kernel::nextafter<A>(from, to, A{});
}

template<class T, class A>
batch<T, A> pos(batch<T, A> const& self) {
  return +self;
}

template<class T, class A>
batch<T, A> remainder(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::remainder<A>(self, other, A{});
}

template<class T, class A>
batch<T, A> select(batch_bool<T, A> const& cond, batch<T, A> const& true_br, batch<T, A> const& false_br) {
  return kernel::select<A>(cond, true_br, false_br, A{});
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

template<class T, class A>
batch<as_float_t<T>, A> to_float(batch<T, A> const& self) {
  return kernel::to_float<A>(self, A{});
}

template<class T, class A>
batch<as_integer_t<T>, A> to_int(batch<T, A> const& self) {
  return kernel::to_int<A>(self, A{});
}

template<class T, class A>
batch<T, A> trunc(batch<T, A> const& self) {
  return kernel::trunc<A>(self, A{});
}

// high level functions - batch_bool
//
template<class T, class A>
batch<T, A> bitwise_cast(batch_bool<T, A> const& self) {
  return {self.data};
}

template<class T, class A>
bool all(batch_bool<T, A> const& self) {
  return kernel::all<A>(bitwise_cast(self), A{});
}

template<class T, class A>
bool any(batch_bool<T, A> const& self) {
  return kernel::any<A>(bitwise_cast(self), A{});
}


}

#endif

