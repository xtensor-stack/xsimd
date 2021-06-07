#ifndef XSIMD_BATCH_HPP
#define XSIMD_BATCH_HPP

#include "../config/xsimd_arch.hpp"
#include "../memory/xsimd_alignment.hpp"

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
  batch(T const* mem);
  batch(T const (&arr)[size]);
  batch(std::initializer_list<T> data);
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

  friend batch<T, A> operator&(batch<T, A> const& self, batch<T, A> const& other) {
    return batch<T, A>(self) &= other;
  }

  friend batch<T, A> operator|(batch<T, A> const& self, batch<T, A> const& other) {
    return batch<T, A>(self) |= other;
  }

  friend batch<T, A> operator^(batch<T, A> const& self, batch<T, A> const& other) {
    return batch<T, A>(self) ^= other;
  }

  // Update operators
  batch<T, A>& operator+=(batch const& other);
  batch<T, A>& operator-=(batch const& other);
  batch<T, A>& operator*=(batch const& other);
  batch<T, A>& operator/=(batch const& other);
  batch<T, A>& operator&=(batch const& other);
  batch<T, A>& operator|=(batch const& other);
  batch<T, A>& operator^=(batch const& other);

};

template<class T, class A=default_arch>
struct batch_bool : types::simd_register<T, A> {
  static constexpr std::size_t size = sizeof(types::simd_register<T, A>) / sizeof(T);

  using value_type = bool;
  using register_type = typename types::simd_register<T, A>::register_type;

  batch_bool() : types::simd_register<T, A>{} {}
  batch_bool(bool val);
  batch_bool(bool const* val);
  batch_bool(register_type reg) : types::simd_register<T, A>({reg}) {}
  batch_bool(std::initializer_list<bool> data);
  operator batch<T, A>() const;

  void store_aligned(bool * mem) const;
  void store_unaligned(bool * mem) const;
  static batch_bool load_aligned(bool const * mem);
  static batch_bool load_unaligned(bool const * mem);

  batch_bool operator~() const;
  batch_bool operator!() const { return operator~(); }
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


};

}

#include "../types/xsimd_batch_constant.hpp"
#include "../arch/xsimd_isa.hpp"

namespace xsimd {

// batch implementation

template<class T, class A>
batch<T, A>::batch(T val) : types::simd_register<T, A>(kernel::broadcast<A>(val, A{})) {}

template<class T, class A>
batch<T, A>::batch(T const* mem) : types::simd_register<T, A>(kernel::load_unaligned<A>(mem, kernel::convert<T>{}, A{})) {}

template<class T, class A>
batch<T, A>::batch(T const (&arr)[size]) : types::simd_register<T, A>(kernel::load_unaligned<A>(&arr[0], A{})) {}

template<class T, class A>
batch<T, A>::batch(std::initializer_list<T> data) : batch(data.begin()) { assert(data.size() == size); }

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
batch_bool<T, A> batch<T, A>::operator!() const { return kernel::eq<A>(*this, batch((T)0), A{}); }

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
batch<T, A>& batch<T, A>::operator&=(batch<T, A> const& other) { return *this = kernel::bitwise_and<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator|=(batch<T, A> const& other) { return *this = kernel::bitwise_or<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator^=(batch<T, A> const& other) { return *this = kernel::bitwise_xor<A>(*this, other, A{}); }


// batch_bool implementation
template<class T, class A>
batch_bool<T, A>::batch_bool(bool const* mem) : types::simd_register<T, A>(batch_bool::load_unaligned(mem).data) {}
template<class T, class A>
batch_bool<T, A>::batch_bool(std::initializer_list<bool> data) : batch_bool(data.begin()) { assert(data.size() == size); }

template<class T, class A>
batch_bool<T, A> batch_bool<T, A>::operator~() const { return kernel::bitwise_not<A>(*this, A{}).data; }

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
    (batch<T, A>(T(0)) == batch<T, A>(T(0))):
    (batch<T, A>(T(0)) != batch<T, A>(T(0)))) {
}


template<class T, class A>
void batch_bool<T, A>::store_aligned(bool* mem) const {
  alignas(A::alignment()) T buffer[size];
  kernel::store_aligned<A>(&buffer[0], *this, A{});
  for(std::size_t i = 0; i < size; ++i)
    mem[i] = bool(buffer[i]);
}

template<class T, class A>
void batch_bool<T, A>::store_unaligned(bool* mem) const {
  store_aligned(mem);
}

template<class T, class A>
batch_bool<T, A> batch_bool<T, A>::load_aligned(bool const* mem) {
  batch<T, A> ref((T)0);
  alignas(A::alignment()) T buffer[size];
  for(std::size_t i = 0; i < size; ++i)
    buffer[i] = mem[i] ? (T)1 : (T)0;
  return ref != batch<T, A>::load_aligned(&buffer[0]);
}

template<class T, class A>
batch_bool<T, A> batch_bool<T, A>::load_unaligned(bool const* mem) {
  return load_aligned(mem);
}

template<class T, class A>
batch_bool<T, A>::operator batch<T, A>() const {
  return batch<T, A>(this->data) & batch<T, A>((T)1);
}

}

#endif
