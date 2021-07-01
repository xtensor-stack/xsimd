#ifndef XSIMD_BATCH_HPP
#define XSIMD_BATCH_HPP

#include "../config/xsimd_arch.hpp"
#include "../memory/xsimd_alignment.hpp"
#include "./xsimd_utils.hpp"

#include <cassert>

namespace xsimd {

template<class T, class A=default_arch>
struct batch : types::simd_register<T, A> {

  static constexpr std::size_t size = sizeof(types::simd_register<T, A>) / sizeof(T);

  using value_type = T;
  using arch_type = A;
  using register_type = typename types::simd_register<T, A>::register_type;
  using batch_bool_type = batch_bool<T, A>;

  // constructors
  batch() : types::simd_register<T, A>{} {}
  batch(T val);
  batch(std::initializer_list<T> data) : batch(data.begin(), detail::make_index_sequence<size>()) {}
  explicit batch(batch_bool_type b);
  batch(register_type reg) : types::simd_register<T, A>({reg}) {}

  template<class U>
  static batch broadcast(U val) { return batch(static_cast<T>(val)); }

  // memory operators
  template<class U>
  void store_aligned(U * mem) const;
  template<class U>
  void store_unaligned(U * mem) const;
  template<class U>
  void store(U * mem, aligned_mode) const { return store_aligned(mem); }
  template<class U>
  void store(U * mem, unaligned_mode) const { return store_unaligned(mem); }

  template<class U>
  static batch load_aligned(U const* mem) ;
  template<class U>
  static batch load_unaligned(U const* mem);
  template<class U>
  static batch load(U const* mem, aligned_mode) { return load_aligned(mem); }
  template<class U>
  static batch load(U const* mem, unaligned_mode) { return load_unaligned(mem); }

  T get(std::size_t i) const {
    alignas(A::alignment()) T buffer[size];
    store_aligned(&buffer[0]);
    return buffer[i];
  }

  // unary operators
  batch_bool<T, A> operator!() const;
  batch<T, A> operator~() const;
  batch operator-() const;
  batch operator+() const { return *this; }

  // comparison operators
  batch_bool<T, A> operator==(batch const& other) const;
  batch_bool<T, A> operator!=(batch const& other) const;
  batch_bool<T, A> operator>=(batch const& other) const;
  batch_bool<T, A> operator<=(batch const& other) const;
  batch_bool<T, A> operator>(batch const& other) const;
  batch_bool<T, A> operator<(batch const& other) const;

  // arithmetic operators. They are defined as friend to enable automatic
  // conversion of parameters from scalar to batch
  friend batch<T, A> operator+(batch<T, A> const& self, batch<T, A> const& other) {
    return batch<T, A>(self) += other;
  }
  friend batch<T, A> operator-(batch<T, A> const& self, batch<T, A> const& other) {
    return batch<T, A>(self) -= other;
  }
  friend batch<T, A> operator*(batch<T, A> const& self, batch<T, A> const& other) {
    return batch<T, A>(self) *= other;
  }
  friend batch<T, A> operator/(batch<T, A> const& self, batch<T, A> const& other) {
    return batch<T, A>(self) /= other;
  }
  friend batch<T, A> operator%(batch<T, A> const& self, batch<T, A> const& other) {
    return batch<T, A>(self) %= other;
  }

  friend batch<T, A> operator&(batch<T, A> const& self, batch<T, A> const& other) {
    return batch<T, A>(self) &= other;
  }

  friend batch<T, A> operator|(batch<T, A> const& self, batch<T, A> const& other) {
    return batch<T, A>(self) |= other;
  }

  friend batch<T, A> operator^(batch<T, A> const& self, batch<T, A> const& other) {
    return batch<T, A>(self) ^= other;
  }

  friend batch<T, A> operator>>(batch<T, A> const& self, batch<T, A> const& other) {
    return batch<T, A>(self) >>= other;
  }

  friend batch<T, A> operator<<(batch<T, A> const& self, batch<T, A> const& other) {
    return batch<T, A>(self) <<= other;
  }

  friend batch<T, A> operator>>(batch<T, A> const& self, int32_t other) {
    return batch<T, A>(self) >>= other;
  }

  friend batch<T, A> operator<<(batch<T, A> const& self, int32_t other) {
    return batch<T, A>(self) <<= other;
  }

  // Update operators
  batch<T, A>& operator+=(batch const& other);
  batch<T, A>& operator-=(batch const& other);
  batch<T, A>& operator*=(batch const& other);
  batch<T, A>& operator/=(batch const& other);
  batch<T, A>& operator%=(batch const& other);
  batch<T, A>& operator&=(batch const& other);
  batch<T, A>& operator|=(batch const& other);
  batch<T, A>& operator^=(batch const& other);
  batch<T, A>& operator>>=(int32_t other);
  batch<T, A>& operator>>=(batch const& other);
  batch<T, A>& operator<<=(int32_t other);
  batch<T, A>& operator<<=(batch const& other);

  // incr/decr
  batch<T, A>& operator++() { return operator+=(1);}
  batch<T, A>& operator--() { return operator-=(1);}
  batch<T, A> operator++(int) { batch copy(*this); operator+=(1); return copy;}
  batch<T, A> operator--(int) { batch copy(*this); operator-=(1); return copy;}

  private:
  template<size_t... Is>
  batch(T const* data, detail::index_sequence<Is...>);
};

template<class T, class A=default_arch>
struct batch_bool : types::simd_register<T, A> {
  static constexpr std::size_t size = sizeof(types::simd_register<T, A>) / sizeof(T);

  using value_type = bool;
  using register_type = typename types::simd_register<T, A>::register_type;
  using batch_type = batch<T, A>;

  batch_bool() : types::simd_register<T, A>{} {}
  batch_bool(bool val);
  batch_bool(register_type reg) : types::simd_register<T, A>({reg}) {}
  batch_bool(std::initializer_list<bool> data) : batch_bool(data.begin(), detail::make_index_sequence<size>()) {}

  void store_aligned(bool * mem) const;
  void store_unaligned(bool * mem) const;
  static batch_bool load_aligned(bool const * mem);
  static batch_bool load_unaligned(bool const * mem);

  batch_bool operator~() const;
  batch_bool operator!() const { return operator==(batch_bool(false)); }
  batch_bool operator==(batch_bool const& other) const;
  batch_bool operator!=(batch_bool const& other) const;
  batch_bool operator&(batch_bool const& other) const;
  batch_bool operator|(batch_bool const& other) const;
  batch_bool operator&&(batch_bool const& other) const {
    return operator&(other);
  }
  batch_bool operator||(batch_bool const& other) const {
    return operator|(other);
  }

  bool get(std::size_t i) const {
    alignas(A::alignment()) bool buffer[size];
    store_aligned(&buffer[0]);
    return buffer[i];
  }

  private:
  template<size_t... Is>
  batch_bool(bool const* data, detail::index_sequence<Is...>);


};

}

#include "../types/xsimd_batch_constant.hpp"
#include "../arch/xsimd_isa.hpp"

namespace xsimd {

// batch implementation

template<class T, class A>
batch<T, A>::batch(T val) : types::simd_register<T, A>(kernel::broadcast<A>(val, A{})) {}

template<class T, class A>
batch<T, A>::batch(batch_bool<T, A> b) : batch(batch(b.data) & batch(1)) {}

template<class T, class A>
template<size_t... Is>
batch<T, A>::batch(T const*data, detail::index_sequence<Is...>) : batch(kernel::set<A>(batch{}, A{}, data[Is]...)) {}

template<class T, class A>
template<class U>
void batch<T, A>::store_aligned(U* mem) const {
  kernel::store_aligned<A>(mem, *this, A{});
}

template<class T, class A>
template<class U>
void batch<T, A>::store_unaligned(U* mem) const {
  kernel::store_unaligned<A>(mem, *this, A{});
}

template<class T, class A>
template<class U>
batch<T, A> batch<T, A>::load_aligned(U const* mem) {
  return kernel::load_aligned<A>(mem, kernel::convert<T>{}, A{});
}

template<class T, class A>
template<class U>
batch<T, A> batch<T, A>::load_unaligned(U const* mem) {
  return kernel::load_unaligned<A>(mem, kernel::convert<T>{}, A{});
}

template<class T, class A>
batch_bool<T, A> batch<T, A>::operator!() const { return kernel::eq<A>(*this, batch(0), A{}); }

template<class T, class A>
batch<T, A> batch<T, A>::operator~() const { return kernel::bitwise_not<A>(*this, A{}); }

template<class T, class A>
batch<T, A> batch<T, A>::operator-() const { return kernel::neg<A>(*this, A{}); }

template<class T, class A>
batch_bool<T, A> batch<T, A>::operator==(batch<T, A> const& other) const { return kernel::eq<A>(*this, other, A{}); }

template<class T, class A>
batch_bool<T, A> batch<T, A>::operator!=(batch<T, A> const& other) const { return kernel::neq<A>(*this, other, A{}); }

template<class T, class A>
batch_bool<T, A> batch<T, A>::operator>=(batch<T, A> const& other) const { return kernel::ge<A>(*this, other, A{}); }

template<class T, class A>
batch_bool<T, A> batch<T, A>::operator<=(batch<T, A> const& other) const { return kernel::le<A>(*this, other, A{}); }

template<class T, class A>
batch_bool<T, A> batch<T, A>::operator>(batch<T, A> const& other) const { return kernel::gt<A>(*this, other, A{}); }

template<class T, class A>
batch_bool<T, A> batch<T, A>::operator<(batch<T, A> const& other) const { return kernel::lt<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator+=(batch<T, A> const& other) { return *this = kernel::add<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator-=(batch<T, A> const& other) { return *this = kernel::sub<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator*=(batch<T, A> const& other) { return *this = kernel::mul<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator/=(batch<T, A> const& other) { return *this = kernel::div<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator%=(batch<T, A> const& other) { return *this = kernel::mod<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator&=(batch<T, A> const& other) { return *this = kernel::bitwise_and<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator|=(batch<T, A> const& other) { return *this = kernel::bitwise_or<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator^=(batch<T, A> const& other) { return *this = kernel::bitwise_xor<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator>>=(batch<T, A> const& other) { return *this = kernel::bitwise_rshift<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator<<=(batch<T, A> const& other) { return *this = kernel::bitwise_lshift<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator>>=(int32_t other) { return *this = kernel::bitwise_rshift<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator<<=(int32_t other) { return *this = kernel::bitwise_lshift<A>(*this, other, A{}); }

// batch_bool implementation
template<class T, class A>
template<size_t... Is>
batch_bool<T, A>::batch_bool(bool const*data, detail::index_sequence<Is...>) : batch_bool(kernel::set<A>(batch_bool{}, A{}, data[Is]...)) {}

template<class T, class A>
batch_bool<T, A> batch_bool<T, A>::operator~() const {
  return kernel::bitwise_not<A>(*this, A{}).data;
}

template<class T, class A>
batch_bool<T, A> batch_bool<T, A>::operator==(batch_bool<T, A> const& other) const {
  return kernel::eq<A>(*this, other, A{}).data;
}

template<class T, class A>
batch_bool<T, A> batch_bool<T, A>::operator!=(batch_bool<T, A> const& other) const {
  return kernel::neq<A>(*this, other, A{}).data;
}

template<class T, class A>
batch_bool<T, A> batch_bool<T, A>::operator&(batch_bool<T, A> const& other) const {
  return kernel::bitwise_and<A>(*this, other, A{}).data;
}

template<class T, class A>
batch_bool<T, A> batch_bool<T, A>::operator|(batch_bool<T, A> const& other) const {
  return kernel::bitwise_or<A>(*this, other, A{}).data;
}

template<class T, class A>
batch_bool<T, A>::batch_bool(bool val) : types::simd_register<T, A>(val?
    (batch_type(0) == batch_type(0)):
    (batch_type(0) != batch_type(0))) {
}


template<class T, class A>
void batch_bool<T, A>::store_aligned(bool* mem) const {
  alignas(A::alignment()) T buffer[size];
  kernel::store_aligned<A>(&buffer[0], batch_type(*this), A{});
  for(std::size_t i = 0; i < size; ++i)
    mem[i] = bool(buffer[i]);
}

template<class T, class A>
void batch_bool<T, A>::store_unaligned(bool* mem) const {
  store_aligned(mem);
}

template<class T, class A>
batch_bool<T, A> batch_bool<T, A>::load_aligned(bool const* mem) {
  batch_type ref(0);
  alignas(A::alignment()) T buffer[size];
  for(std::size_t i = 0; i < size; ++i)
    buffer[i] = mem[i] ? 1 : 0;
  return ref != batch_type::load_aligned(&buffer[0]);
}

template<class T, class A>
batch_bool<T, A> batch_bool<T, A>::load_unaligned(bool const* mem) {
  return load_aligned(mem);
}

}

#endif
