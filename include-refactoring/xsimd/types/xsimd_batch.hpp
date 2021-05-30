#ifndef XSIMD_BATCH_HPP
#define XSIMD_BATCH_HPP

#include "../config/xsimd_arch.hpp"

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

  static batch broadcast(T val) XSIMD_DEPRECATED("use xsimd::batch(val) instead") { return batch(val); }

  // memory operators
  void store_aligned(T * mem) const;
  void store_unaligned(T * mem) const;
  void load_aligned(T const* mem) { *this = from_aligned(mem); }
  void load_unaligned(T const* mem) { *this = from_unaligned(mem); }
  static batch from_aligned(T const* mem) XSIMD_DEPRECATED("use xsimd::load_aligned(mem) instead") ;
  static batch from_unaligned(T const* mem) XSIMD_DEPRECATED ("use xsimd::load_unaligned(mem) instead") { return {mem}; }

  T operator[](std::size_t i) const {
    alignas(A::alignment) T buffer[size];
    store_unaligned(&buffer[0]);
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

  // Update operators
  batch<T, A>& operator+=(batch const& other);
  batch<T, A>& operator-=(batch const& other);
  batch<T, A>& operator*=(batch const& other);
  batch<T, A>& operator/=(batch const& other);
  batch<T, A>& operator&=(batch const& other);

};

template<class T, class A=default_arch>
struct batch_bool : types::simd_register<T, A> {
  static constexpr std::size_t size = sizeof(types::simd_register<T, A>) / sizeof(T);

  using value_type = bool;
  using register_type = typename types::simd_register<T, A>::register_type;

  batch_bool() : types::simd_register<T, A>{} {}
  batch_bool(bool val);
  batch_bool(register_type reg) : types::simd_register<T, A>({reg}) {}
  operator batch<T, A>() const;

  void store_aligned(bool * mem) const;
  void store_unaligned(bool * mem) const;

  batch_bool operator~() const;

};

}

#include "../isa/xsimd_isa.hpp"

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
void batch<T, A>::store_aligned(T* mem) const {
  kernel::store_aligned<A>(mem, *this, A{});
}

template<class T, class A>
void batch<T, A>::store_unaligned(T* mem) const {
  kernel::store_unaligned<A>(mem, *this, A{});
}

template<class T, class A>
batch<T, A> batch<T, A>::from_aligned(T const* mem) {
  return kernel::load_aligned<A>(mem, kernel::convert<T>{}, A{});
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


// batch_bool implementation

template<class T, class A>
batch_bool<T, A> batch_bool<T, A>::operator~() const { return kernel::bitwise_not<A>(*this, A{}).data; }

template<class T, class A>
batch_bool<T, A>::batch_bool(bool val) : types::simd_register<T, A>(val?
    (batch<T, A>(this->data) == batch<T, A>(this->data)):
    (batch<T, A>(this->data) != batch<T, A>(this->data))) {
}


template<class T, class A>
void batch_bool<T, A>::store_aligned(bool* mem) const {
  alignas(A::alignment) T buffer[size];
  kernel::store_aligned<A>(&buffer[0], *this, A{});
  for(std::size_t i = 0; i < size; ++i)
    mem[i] = bool(buffer[i]);
}

template<class T, class A>
void batch_bool<T, A>::store_unaligned(bool* mem) const {
  store_aligned(mem);
}

template<class T, class A>
batch_bool<T, A>::operator batch<T, A>() const {
  return batch<T, A>(this->data) & batch<T, A>((T)1);
}

}

#endif
