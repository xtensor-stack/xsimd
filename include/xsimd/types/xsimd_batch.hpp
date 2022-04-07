/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_BATCH_HPP
#define XSIMD_BATCH_HPP

#include <cassert>
#include <complex>

#include "../config/xsimd_arch.hpp"
#include "../memory/xsimd_alignment.hpp"
#include "./xsimd_utils.hpp"

namespace xsimd
{

    /**
     * @brief batch of integer or floating point values.
     *
     * Abstract representation of an SIMD register for floating point or integral
     * value.
     *
     * @tparam T the type of the underlying values.
     * @tparam A the architecture this batch is tied too.
     **/
    template <class T, class A = default_arch>
    class batch : public types::simd_register<T, A>
    {
    public:
        static constexpr std::size_t size = sizeof(types::simd_register<T, A>) / sizeof(T);

        using value_type = T;
        using arch_type = A;
        using register_type = typename types::simd_register<T, A>::register_type;
        using batch_bool_type = batch_bool<T, A>;

        // constructors
        batch() = default;
        batch(T val) noexcept;
        batch(std::initializer_list<T> data) noexcept;
        explicit batch(batch_bool_type const& b) noexcept;
        batch(register_type reg) noexcept;

        template <class U>
        static XSIMD_NO_DISCARD batch broadcast(U val) noexcept;

        // memory operators
        template <class U>
        void store_aligned(U* mem) const noexcept;
        template <class U>
        void store_unaligned(U* mem) const noexcept;
        template <class U>
        void store(U* mem, aligned_mode) const noexcept;
        template <class U>
        void store(U* mem, unaligned_mode) const noexcept;

        template <class U>
        static XSIMD_NO_DISCARD batch load_aligned(U const* mem) noexcept;
        template <class U>
        static XSIMD_NO_DISCARD batch load_unaligned(U const* mem) noexcept;
        template <class U>
        static XSIMD_NO_DISCARD batch load(U const* mem, aligned_mode) noexcept;
        template <class U>
        static XSIMD_NO_DISCARD batch load(U const* mem, unaligned_mode) noexcept;

        T get(std::size_t i) const noexcept;

        // comparison operators
        inline batch_bool_type operator==(batch const& other) const noexcept;
        inline batch_bool_type operator!=(batch const& other) const noexcept;
        inline batch_bool_type operator>=(batch const& other) const noexcept;
        inline batch_bool_type operator<=(batch const& other) const noexcept;
        inline batch_bool_type operator>(batch const& other) const noexcept;
        inline batch_bool_type operator<(batch const& other) const noexcept;

        // Update operators
        inline batch& operator+=(batch const& other) noexcept;
        inline batch& operator-=(batch const& other) noexcept;
        inline batch& operator*=(batch const& other) noexcept;
        inline batch& operator/=(batch const& other) noexcept;
        inline batch& operator%=(batch const& other) noexcept;
        inline batch& operator&=(batch const& other) noexcept;
        inline batch& operator|=(batch const& other) noexcept;
        inline batch& operator^=(batch const& other) noexcept;
        inline batch& operator>>=(int32_t other) noexcept;
        inline batch& operator>>=(batch const& other) noexcept;
        inline batch& operator<<=(int32_t other) noexcept;
        inline batch& operator<<=(batch const& other) noexcept;

        // incr/decr operators
        inline batch& operator++() noexcept;
        inline batch& operator--() noexcept;
        inline batch operator++(int) noexcept;
        inline batch operator--(int) noexcept;

        // unary operators
        inline batch_bool_type operator!() const noexcept;
        inline batch operator~() const noexcept;
        inline batch operator-() const noexcept;
        inline batch operator+() const noexcept;

        // arithmetic operators. They are defined as friend to enable automatic
        // conversion of parameters from scalar to batch. Inline implementation
        // is required to avoid warnings.
        friend batch operator+(batch const& self, batch const& other) noexcept
        {
            return batch(self) += other;
        }

        friend batch operator-(batch const& self, batch const& other) noexcept
        {
            return batch(self) -= other;
        }

        friend batch operator*(batch const& self, batch const& other) noexcept
        {
            return batch(self) *= other;
        }

        friend batch operator/(batch const& self, batch const& other) noexcept
        {
            return batch(self) /= other;
        }

        friend batch operator%(batch const& self, batch const& other) noexcept
        {
            return batch(self) %= other;
        }

        friend batch operator&(batch const& self, batch const& other) noexcept
        {
            return batch(self) &= other;
        }

        friend batch operator|(batch const& self, batch const& other) noexcept
        {
            return batch(self) |= other;
        }

        friend batch operator^(batch const& self, batch const& other) noexcept
        {
            return batch(self) ^= other;
        }

        friend batch operator>>(batch const& self, batch const& other) noexcept
        {
            return batch(self) >>= other;
        }

        friend batch operator<<(batch const& self, batch const& other) noexcept
        {
            return batch(self) <<= other;
        }

        friend batch operator>>(batch const& self, int32_t other) noexcept
        {
            return batch(self) >>= other;
        }

        friend batch operator<<(batch const& self, int32_t other) noexcept
        {
            return batch(self) <<= other;
        }

        friend batch operator&&(batch const& self, batch const& other) noexcept
        {
            return batch(self).logical_and(other);
        }

        friend batch operator||(batch const& self, batch const& other) noexcept
        {
            return batch(self).logical_or(other);
        }

    private:
        template <size_t... Is>
        batch(T const* data, detail::index_sequence<Is...>) noexcept;

        batch logical_and(batch const& other) const noexcept;
        batch logical_or(batch const& other) const noexcept;
    };

    template <class T, class A>
    constexpr std::size_t batch<T, A>::size;

    /**
     * @brief batch of predicate over scalar or complex values.
     *
     * Abstract representation of a predicate over SIMD register for scalar or
     * complex values.
     *
     * @tparam T the type of the predicated values.
     * @tparam A the architecture this batch is tied too.
     **/
    template <class T, class A = default_arch>
    class batch_bool : public types::get_bool_simd_register_t<T, A>
    {
    public:
        static constexpr std::size_t size = sizeof(types::simd_register<T, A>) / sizeof(T);

        using value_type = bool;
        using base_type = types::get_bool_simd_register_t<T, A>;
        using register_type = typename base_type::register_type;
        using batch_type = batch<T, A>;

        // constructors
        batch_bool() = default;
        batch_bool(bool val) noexcept;
        batch_bool(register_type reg) noexcept;
        batch_bool(std::initializer_list<bool> data) noexcept;

        template <class Tp>
        batch_bool(Tp const*) = delete;

        // memory operators
        void store_aligned(bool* mem) const noexcept;
        void store_unaligned(bool* mem) const noexcept;
        static XSIMD_NO_DISCARD batch_bool load_aligned(bool const* mem) noexcept;
        static XSIMD_NO_DISCARD batch_bool load_unaligned(bool const* mem) noexcept;

        bool get(std::size_t i) const noexcept;

        // comparison operators
        batch_bool operator==(batch_bool const& other) const noexcept;
        batch_bool operator!=(batch_bool const& other) const noexcept;

        // logical operators
        batch_bool operator~() const noexcept;
        batch_bool operator!() const noexcept;
        batch_bool operator&(batch_bool const& other) const noexcept;
        batch_bool operator|(batch_bool const& other) const noexcept;
        batch_bool operator^(batch_bool const& other) const noexcept;
        batch_bool operator&&(batch_bool const& other) const noexcept;
        batch_bool operator||(batch_bool const& other) const noexcept;

        // update operators
        batch_bool& operator&=(batch_bool const& other) const noexcept { return (*this) = (*this) & other; }
        batch_bool& operator|=(batch_bool const& other) const noexcept { return (*this) = (*this) | other; }
        batch_bool& operator^=(batch_bool const& other) const noexcept { return (*this) = (*this) ^ other; }

    private:
        template <size_t... Is>
        batch_bool(bool const* data, detail::index_sequence<Is...>) noexcept;

        template <class U, class... V, size_t I, size_t... Is>
        static register_type make_register(detail::index_sequence<I, Is...>, U u, V... v) noexcept;

        template <class... V>
        static register_type make_register(detail::index_sequence<>, V... v) noexcept;
    };

    template <class T, class A>
    constexpr std::size_t batch_bool<T, A>::size;

    /**
     * @brief batch of complex values.
     *
     * Abstract representation of an SIMD register for complex values.
     *
     * @tparam T the type of the underlying values.
     * @tparam A the architecture this batch is tied too.
     **/
    template <class T, class A>
    class batch<std::complex<T>, A>
    {
    public:
        using value_type = std::complex<T>;
        using real_batch = batch<T, A>;
        using arch_type = A;
        static constexpr std::size_t size = real_batch::size;
        using batch_bool_type = batch_bool<T, A>;

        // constructors
        batch() = default;
        batch(value_type const& val) noexcept;
        batch(real_batch const& real, real_batch const& imag) noexcept;

        batch(real_batch const& real) noexcept;
        batch(T val) noexcept;
        batch(std::initializer_list<value_type> data) noexcept;
        explicit batch(batch_bool_type const& b) noexcept;

        // memory operators
        static XSIMD_NO_DISCARD batch load_aligned(const T* real_src, const T* imag_src = nullptr) noexcept;
        static XSIMD_NO_DISCARD batch load_unaligned(const T* real_src, const T* imag_src = nullptr) noexcept;
        void store_aligned(T* real_dst, T* imag_dst) const noexcept;
        void store_unaligned(T* real_dst, T* imag_dst) const noexcept;

        static XSIMD_NO_DISCARD batch load_aligned(const value_type* src) noexcept;
        static XSIMD_NO_DISCARD batch load_unaligned(const value_type* src) noexcept;
        void store_aligned(value_type* dst) const noexcept;
        void store_unaligned(value_type* dst) const noexcept;

        template <class U>
        static XSIMD_NO_DISCARD batch load(U const* mem, aligned_mode) noexcept;
        template <class U>
        static XSIMD_NO_DISCARD batch load(U const* mem, unaligned_mode) noexcept;
        template <class U>
        void store(U* mem, aligned_mode) const noexcept;
        template <class U>
        void store(U* mem, unaligned_mode) const noexcept;

        real_batch real() const noexcept;
        real_batch imag() const noexcept;

        value_type get(std::size_t i) const noexcept;

#ifdef XSIMD_ENABLE_XTL_COMPLEX
        // xtl-related methods
        template <bool i3ec>
        batch(xtl::xcomplex<T, T, i3ec> const& val) noexcept;
        template <bool i3ec>
        batch(std::initializer_list<xtl::xcomplex<T, T, i3ec>> data) noexcept;

        template <bool i3ec>
        static XSIMD_NO_DISCARD batch load_aligned(const xtl::xcomplex<T, T, i3ec>* src) noexcept;
        template <bool i3ec>
        static XSIMD_NO_DISCARD batch load_unaligned(const xtl::xcomplex<T, T, i3ec>* src) noexcept;
        template <bool i3ec>
        void store_aligned(xtl::xcomplex<T, T, i3ec>* dst) const noexcept;
        template <bool i3ec>
        void store_unaligned(xtl::xcomplex<T, T, i3ec>* dst) const noexcept;
#endif

        // comparison operators
        batch_bool<T, A> operator==(batch const& other) const noexcept;
        batch_bool<T, A> operator!=(batch const& other) const noexcept;

        // Update operators
        batch& operator+=(batch const& other) noexcept;
        batch& operator-=(batch const& other) noexcept;
        batch& operator*=(batch const& other) noexcept;
        batch& operator/=(batch const& other) noexcept;

        // incr/decr operators
        batch& operator++() noexcept;
        batch& operator--() noexcept;
        batch operator++(int) noexcept;
        batch operator--(int) noexcept;

        // unary operators
        batch_bool_type operator!() const noexcept;
        batch operator~() const noexcept;
        batch operator-() const noexcept;
        batch operator+() const noexcept;

        // arithmetic operators. They are defined as friend to enable automatic
        // conversion of parameters from scalar to batch
        friend batch operator+(batch const& self, batch const& other) noexcept
        {
            return batch(self) += other;
        }

        friend batch operator-(batch const& self, batch const& other) noexcept
        {
            return batch(self) -= other;
        }

        friend batch operator*(batch const& self, batch const& other) noexcept
        {
            return batch(self) *= other;
        }

        friend batch operator/(batch const& self, batch const& other) noexcept
        {
            return batch(self) /= other;
        }

    private:
        real_batch m_real;
        real_batch m_imag;
    };

    template <class T, class A>
    constexpr std::size_t batch<std::complex<T>, A>::size;

}

#include "../arch/xsimd_isa.hpp"
#include "../types/xsimd_batch_constant.hpp"

namespace xsimd
{

    /**
     * Create a batch with all element initialized to \c val.
     *
     * @param val broadcast value
     */
    template <class T, class A>
    inline batch<T, A>::batch(T val) noexcept
        : types::simd_register<T, A>(kernel::broadcast<A>(val, A {}))
    {
    }

    /**
     * Create a batch with elements initialized from \c data.
     * It is an error to have `data.size() != size.
     *
     * @param data sequence of elements
     */
    template <class T, class A>
    inline batch<T, A>::batch(std::initializer_list<T> data) noexcept
        : batch(data.begin(), detail::make_index_sequence<size>())
    {
        assert(data.size() == size && "consistent initialization");
    }

    /**
     * Converts a \c bool_batch to a \c batch where each element is
     * set to 0xFF..FF (resp. 0x00..00) if the corresponding element is `true`
     * (resp. `false`).
     *
     * @param b batch of bool
     */
    template <class T, class A>
    inline batch<T, A>::batch(batch_bool<T, A> const& b) noexcept
        : batch(kernel::from_bool(b, A {}))
    {
    }

    /**
     * Wraps a compatible native simd register as a \c batch. This is generally not needed but
     * becomes handy when doing architecture-specific operations.
     *
     * @param reg native simd register to wrap
     */
    template <class T, class A>
    inline batch<T, A>::batch(register_type reg) noexcept
        : types::simd_register<T, A>({ reg })
    {
    }

    template <class T, class A>
    template <size_t... Is>
    inline batch<T, A>::batch(T const* data, detail::index_sequence<Is...>) noexcept
        : batch(kernel::set<A>(batch {}, A {}, data[Is]...))
    {
    }

    template <class T, class A>
    template <class U>
    inline XSIMD_NO_DISCARD batch<T, A> batch<T, A>::broadcast(U val) noexcept
    {
        return batch(static_cast<T>(val));
    }

    /**************************
     * batch memory operators *
     **************************/

    /**
     * Copy content of this batch to the buffer \c mem. The
     * memory needs to be aligned.
     * @param mem the memory buffer to read
     */
    template <class T, class A>
    template <class U>
    inline void batch<T, A>::store_aligned(U* mem) const noexcept
    {
        kernel::store_aligned<A>(mem, *this, A {});
    }

    /**
     * Copy content of this batch to the buffer \c mem. The
     * memory does not need to be aligned.
     * @param mem the memory buffer to write to
     */
    template <class T, class A>
    template <class U>
    inline void batch<T, A>::store_unaligned(U* mem) const noexcept
    {
        kernel::store_unaligned<A>(mem, *this, A {});
    }

    template <class T, class A>
    template <class U>
    inline void batch<T, A>::store(U* mem, aligned_mode) const noexcept
    {
        return store_aligned(mem);
    }

    template <class T, class A>
    template <class U>
    inline void batch<T, A>::store(U* mem, unaligned_mode) const noexcept
    {
        return store_unaligned(mem);
    }

    /**
     * Loading from aligned memory. May involve a conversion if \c U is different
     * from \c T.
     *
     * @param mem the memory buffer to read from.
     * @return a new batch instance.
     */
    template <class T, class A>
    template <class U>
    inline batch<T, A> batch<T, A>::load_aligned(U const* mem) noexcept
    {
        return kernel::load_aligned<A>(mem, kernel::convert<T> {}, A {});
    }

    /**
     * Loading from unaligned memory. May involve a conversion if \c U is different
     * from \c T.
     *
     * @param mem the memory buffer to read from.
     * @return a new batch instance.
     */
    template <class T, class A>
    template <class U>
    inline batch<T, A> batch<T, A>::load_unaligned(U const* mem) noexcept
    {
        return kernel::load_unaligned<A>(mem, kernel::convert<T> {}, A {});
    }

    template <class T, class A>
    template <class U>
    inline batch<T, A> batch<T, A>::load(U const* mem, aligned_mode) noexcept
    {
        return load_aligned(mem);
    }

    template <class T, class A>
    template <class U>
    inline batch<T, A> batch<T, A>::load(U const* mem, unaligned_mode) noexcept
    {
        return load_unaligned(mem);
    }

    template <class T, class A>
    inline T batch<T, A>::get(std::size_t i) const noexcept
    {
        alignas(A::alignment()) T buffer[size];
        store_aligned(&buffer[0]);
        return buffer[i];
    }

    /******************************
     * batch comparison operators *
     ******************************/

    template <class T, class A>
    inline batch_bool<T, A> batch<T, A>::operator==(batch<T, A> const& other) const noexcept
    {
        return kernel::eq<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch_bool<T, A> batch<T, A>::operator!=(batch<T, A> const& other) const noexcept
    {
        return kernel::neq<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch_bool<T, A> batch<T, A>::operator>=(batch<T, A> const& other) const noexcept
    {
        return kernel::ge<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch_bool<T, A> batch<T, A>::operator<=(batch<T, A> const& other) const noexcept
    {
        return kernel::le<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch_bool<T, A> batch<T, A>::operator>(batch<T, A> const& other) const noexcept
    {
        return kernel::gt<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch_bool<T, A> batch<T, A>::operator<(batch<T, A> const& other) const noexcept
    {
        return kernel::lt<A>(*this, other, A {});
    }

    /**************************
     * batch update operators *
     **************************/

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator+=(batch<T, A> const& other) noexcept
    {
        return *this = kernel::add<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator-=(batch<T, A> const& other) noexcept
    {
        return *this = kernel::sub<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator*=(batch<T, A> const& other) noexcept
    {
        return *this = kernel::mul<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator/=(batch<T, A> const& other) noexcept
    {
        return *this = kernel::div<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator%=(batch<T, A> const& other) noexcept
    {
        return *this = kernel::mod<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator&=(batch<T, A> const& other) noexcept
    {
        return *this = kernel::bitwise_and<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator|=(batch<T, A> const& other) noexcept
    {
        return *this = kernel::bitwise_or<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator^=(batch<T, A> const& other) noexcept
    {
        return *this = kernel::bitwise_xor<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator>>=(batch<T, A> const& other) noexcept
    {
        return *this = kernel::bitwise_rshift<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator<<=(batch<T, A> const& other) noexcept
    {
        return *this = kernel::bitwise_lshift<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator>>=(int32_t other) noexcept
    {
        return *this = kernel::bitwise_rshift<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator<<=(int32_t other) noexcept
    {
        return *this = kernel::bitwise_lshift<A>(*this, other, A {});
    }

    /*****************************
     * batch incr/decr operators *
     *****************************/

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator++() noexcept
    {
        return operator+=(1);
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator--() noexcept
    {
        return operator-=(1);
    }

    template <class T, class A>
    inline batch<T, A> batch<T, A>::operator++(int) noexcept
    {
        batch<T, A> copy(*this);
        operator+=(1);
        return copy;
    }

    template <class T, class A>
    inline batch<T, A> batch<T, A>::operator--(int) noexcept
    {
        batch copy(*this);
        operator-=(1);
        return copy;
    }

    /*************************
     * batch unary operators *
     *************************/

    template <class T, class A>
    inline batch_bool<T, A> batch<T, A>::operator!() const noexcept
    {
        return kernel::eq<A>(*this, batch(0), A {});
    }

    template <class T, class A>
    inline batch<T, A> batch<T, A>::operator~() const noexcept
    {
        return kernel::bitwise_not<A>(*this, A {});
    }

    template <class T, class A>
    inline batch<T, A> batch<T, A>::operator-() const noexcept
    {
        return kernel::neg<A>(*this, A {});
    }

    template <class T, class A>
    inline batch<T, A> batch<T, A>::operator+() const noexcept
    {
        return *this;
    }

    /************************
     * batch private method *
     ************************/

    template <class T, class A>
    inline batch<T, A> batch<T, A>::logical_and(batch<T, A> const& other) const noexcept
    {
        return kernel::logical_and<A>(*this, other, A());
    }

    template <class T, class A>
    inline batch<T, A> batch<T, A>::logical_or(batch<T, A> const& other) const noexcept
    {
        return kernel::logical_or<A>(*this, other, A());
    }

    /***************************
     * batch_bool constructors *
     ***************************/

    template <class T, class A>
    template <size_t... Is>
    inline batch_bool<T, A>::batch_bool(bool const* data, detail::index_sequence<Is...>) noexcept
        : batch_bool(kernel::set<A>(batch_bool {}, A {}, data[Is]...))
    {
    }

    template <class T, class A>
    inline batch_bool<T, A>::batch_bool(register_type reg) noexcept
        : types::get_bool_simd_register_t<T, A>({ reg })
    {
    }

    template <class T, class A>
    inline batch_bool<T, A>::batch_bool(std::initializer_list<bool> data) noexcept
        : batch_bool(data.begin(), detail::make_index_sequence<size>())
    {
    }

    /*******************************
     * batch_bool memory operators *
     *******************************/

    template <class T, class A>
    inline void batch_bool<T, A>::store_aligned(bool* mem) const noexcept
    {
        kernel::store(*this, mem, A {});
    }

    template <class T, class A>
    inline void batch_bool<T, A>::store_unaligned(bool* mem) const noexcept
    {
        store_aligned(mem);
    }

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::load_aligned(bool const* mem) noexcept
    {
        batch_type ref(0);
        alignas(A::alignment()) T buffer[size];
        for (std::size_t i = 0; i < size; ++i)
            buffer[i] = mem[i] ? 1 : 0;
        return ref != batch_type::load_aligned(&buffer[0]);
    }

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::load_unaligned(bool const* mem) noexcept
    {
        return load_aligned(mem);
    }

    template <class T, class A>
    inline bool batch_bool<T, A>::get(std::size_t i) const noexcept
    {
        alignas(A::alignment()) bool buffer[size];
        store_aligned(&buffer[0]);
        return buffer[i];
    }

    /***********************************
     * batch_bool comparison operators *
     ***********************************/

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::operator==(batch_bool<T, A> const& other) const noexcept
    {
        return kernel::eq<A>(*this, other, A {}).data;
    }

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::operator!=(batch_bool<T, A> const& other) const noexcept
    {
        return kernel::neq<A>(*this, other, A {}).data;
    }

    /********************************
     * batch_bool logical operators *
     ********************************/

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::operator~() const noexcept
    {
        return kernel::bitwise_not<A>(*this, A {}).data;
    }

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::operator!() const noexcept
    {
        return operator==(batch_bool(false));
    }

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::operator&(batch_bool<T, A> const& other) const noexcept
    {
        return kernel::bitwise_and<A>(*this, other, A {}).data;
    }

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::operator|(batch_bool<T, A> const& other) const noexcept
    {
        return kernel::bitwise_or<A>(*this, other, A {}).data;
    }

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::operator^(batch_bool<T, A> const& other) const noexcept
    {
        return kernel::bitwise_xor<A>(*this, other, A {}).data;
    }

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::operator&&(batch_bool const& other) const noexcept
    {
        return operator&(other);
    }

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::operator||(batch_bool const& other) const noexcept
    {
        return operator|(other);
    }

    /******************************
     * batch_bool private methods *
     ******************************/

    template <class T, class A>
    inline batch_bool<T, A>::batch_bool(bool val) noexcept
        : base_type { make_register(detail::make_index_sequence<size - 1>(), val) }
    {
    }

    template <class T, class A>
    template <class U, class... V, size_t I, size_t... Is>
    inline auto batch_bool<T, A>::make_register(detail::index_sequence<I, Is...>, U u, V... v) noexcept -> register_type
    {
        return make_register(detail::index_sequence<Is...>(), u, u, v...);
    }

    template <class T, class A>
    template <class... V>
    inline auto batch_bool<T, A>::make_register(detail::index_sequence<>, V... v) noexcept -> register_type
    {
        return kernel::set<A>(batch_bool<T, A>(), A {}, v...).data;
    }

    /*******************************
     * batch<complex> constructors *
     *******************************/

    template <class T, class A>
    inline batch<std::complex<T>, A>::batch(value_type const& val) noexcept
        : m_real(val.real())
        , m_imag(val.imag())
    {
    }

    template <class T, class A>
    inline batch<std::complex<T>, A>::batch(real_batch const& real, real_batch const& imag) noexcept
        : m_real(real)
        , m_imag(imag)
    {
    }

    template <class T, class A>
    inline batch<std::complex<T>, A>::batch(real_batch const& real) noexcept
        : m_real(real)
        , m_imag(0)
    {
    }

    template <class T, class A>
    inline batch<std::complex<T>, A>::batch(T val) noexcept
        : m_real(val)
        , m_imag(0)
    {
    }

    template <class T, class A>
    inline batch<std::complex<T>, A>::batch(std::initializer_list<value_type> data) noexcept
    {
        *this = load_unaligned(data.begin());
    }

    template <class T, class A>
    inline batch<std::complex<T>, A>::batch(batch_bool_type const& b) noexcept
        : m_real(b)
        , m_imag(0)
    {
    }

    /***********************************
     * batch<complex> memory operators *
     ***********************************/

    template <class T, class A>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::load_aligned(const T* real_src, const T* imag_src) noexcept
    {
        return { batch<T, A>::load_aligned(real_src), imag_src ? batch<T, A>::load_aligned(imag_src) : batch<T, A>(0) };
    }
    template <class T, class A>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::load_unaligned(const T* real_src, const T* imag_src) noexcept
    {
        return { batch<T, A>::load_unaligned(real_src), imag_src ? batch<T, A>::load_unaligned(imag_src) : batch<T, A>(0) };
    }

    template <class T, class A>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::load_aligned(const value_type* src) noexcept
    {
        return kernel::load_complex_aligned<A>(src, kernel::convert<value_type> {}, A {});
    }

    template <class T, class A>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::load_unaligned(const value_type* src) noexcept
    {
        return kernel::load_complex_unaligned<A>(src, kernel::convert<value_type> {}, A {});
    }

    template <class T, class A>
    inline void batch<std::complex<T>, A>::store_aligned(value_type* dst) const noexcept
    {
        return kernel::store_complex_aligned(dst, *this, A {});
    }

    template <class T, class A>
    inline void batch<std::complex<T>, A>::store_unaligned(value_type* dst) const noexcept
    {
        return kernel::store_complex_unaligned(dst, *this, A {});
    }

    template <class T, class A>
    inline void batch<std::complex<T>, A>::store_aligned(T* real_dst, T* imag_dst) const noexcept
    {
        m_real.store_aligned(real_dst);
        m_imag.store_aligned(imag_dst);
    }

    template <class T, class A>
    inline void batch<std::complex<T>, A>::store_unaligned(T* real_dst, T* imag_dst) const noexcept
    {
        m_real.store_unaligned(real_dst);
        m_imag.store_unaligned(imag_dst);
    }

    template <class T, class A>
    template <class U>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::load(U const* mem, aligned_mode) noexcept
    {
        return load_aligned(mem);
    }

    template <class T, class A>
    template <class U>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::load(U const* mem, unaligned_mode) noexcept
    {
        return load_unaligned(mem);
    }

    template <class T, class A>
    template <class U>
    inline void batch<std::complex<T>, A>::store(U* mem, aligned_mode) const noexcept
    {
        return store_aligned(mem);
    }

    template <class T, class A>
    template <class U>
    inline void batch<std::complex<T>, A>::store(U* mem, unaligned_mode) const noexcept
    {
        return store_unaligned(mem);
    }

    template <class T, class A>
    inline auto batch<std::complex<T>, A>::real() const noexcept -> real_batch
    {
        return m_real;
    }

    template <class T, class A>
    inline auto batch<std::complex<T>, A>::imag() const noexcept -> real_batch
    {
        return m_imag;
    }

    template <class T, class A>
    inline auto batch<std::complex<T>, A>::get(std::size_t i) const noexcept -> value_type
    {
        alignas(A::alignment()) value_type buffer[size];
        store_aligned(&buffer[0]);
        return buffer[i];
    }

    /**************************************
     * batch<complex> xtl-related methods *
     **************************************/

#ifdef XSIMD_ENABLE_XTL_COMPLEX

    template <class T, class A>
    template <bool i3ec>
    inline batch<std::complex<T>, A>::batch(xtl::xcomplex<T, T, i3ec> const& val) noexcept
        : m_real(val.real())
        , m_imag(val.imag())
    {
    }

    template <class T, class A>
    template <bool i3ec>
    inline batch<std::complex<T>, A>::batch(std::initializer_list<xtl::xcomplex<T, T, i3ec>> data) noexcept
    {
        *this = load_unaligned(data.begin());
    }

    // Memory layout of an xcomplex and std::complex are the same when xcomplex
    // stores values and not reference. Unfortunately, this breaks strict
    // aliasing...

    template <class T, class A>
    template <bool i3ec>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::load_aligned(const xtl::xcomplex<T, T, i3ec>* src) noexcept
    {
        return load_aligned(reinterpret_cast<std::complex<T> const*>(src));
    }

    template <class T, class A>
    template <bool i3ec>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::load_unaligned(const xtl::xcomplex<T, T, i3ec>* src) noexcept
    {
        return load_unaligned(reinterpret_cast<std::complex<T> const*>(src));
    }

    template <class T, class A>
    template <bool i3ec>
    inline void batch<std::complex<T>, A>::store_aligned(xtl::xcomplex<T, T, i3ec>* dst) const noexcept
    {
        store_aligned(reinterpret_cast<std::complex<T>*>(dst));
    }

    template <class T, class A>
    template <bool i3ec>
    inline void batch<std::complex<T>, A>::store_unaligned(xtl::xcomplex<T, T, i3ec>* dst) const noexcept
    {
        store_unaligned(reinterpret_cast<std::complex<T>*>(dst));
    }

#endif

    /***************************************
     * batch<complex> comparison operators *
     ***************************************/

    template <class T, class A>
    inline batch_bool<T, A> batch<std::complex<T>, A>::operator==(batch const& other) const noexcept
    {
        return m_real == other.m_real && m_imag == other.m_imag;
    }

    template <class T, class A>
    inline batch_bool<T, A> batch<std::complex<T>, A>::operator!=(batch const& other) const noexcept
    {
        return m_real != other.m_real || m_imag != other.m_imag;
    }

    /***********************************
     * batch<complex> update operators *
     ***********************************/

    template <class T, class A>
    inline batch<std::complex<T>, A>& batch<std::complex<T>, A>::operator+=(batch const& other) noexcept
    {
        m_real += other.m_real;
        m_imag += other.m_imag;
        return *this;
    }

    template <class T, class A>
    inline batch<std::complex<T>, A>& batch<std::complex<T>, A>::operator-=(batch const& other) noexcept
    {
        m_real -= other.m_real;
        m_imag -= other.m_imag;
        return *this;
    }

    template <class T, class A>
    inline batch<std::complex<T>, A>& batch<std::complex<T>, A>::operator*=(batch const& other) noexcept
    {
        real_batch new_real = real() * other.real() - imag() * other.imag();
        real_batch new_imag = real() * other.imag() + imag() * other.real();
        m_real = new_real;
        m_imag = new_imag;
        return *this;
    }

    template <class T, class A>
    inline batch<std::complex<T>, A>& batch<std::complex<T>, A>::operator/=(batch const& other) noexcept
    {
        real_batch a = real();
        real_batch b = imag();
        real_batch c = other.real();
        real_batch d = other.imag();
        real_batch e = c * c + d * d;
        m_real = (c * a + d * b) / e;
        m_imag = (c * b - d * a) / e;
        return *this;
    }

    /**************************************
     * batch<complex> incr/decr operators *
     **************************************/

    template <class T, class A>
    inline batch<std::complex<T>, A>& batch<std::complex<T>, A>::operator++() noexcept
    {
        return operator+=(1);
    }

    template <class T, class A>
    inline batch<std::complex<T>, A>& batch<std::complex<T>, A>::operator--() noexcept
    {
        return operator-=(1);
    }

    template <class T, class A>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::operator++(int) noexcept
    {
        batch copy(*this);
        operator+=(1);
        return copy;
    }

    template <class T, class A>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::operator--(int) noexcept
    {
        batch copy(*this);
        operator-=(1);
        return copy;
    }

    /**********************************
     * batch<complex> unary operators *
     **********************************/

    template <class T, class A>
    inline batch_bool<T, A> batch<std::complex<T>, A>::operator!() const noexcept
    {
        return operator==(batch(0));
    }

    template <class T, class A>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::operator~() const noexcept
    {
        return { ~m_real, ~m_imag };
    }

    template <class T, class A>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::operator-() const noexcept
    {
        return { -m_real, -m_imag };
    }

    template <class T, class A>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::operator+() const noexcept
    {
        return { +m_real, +m_imag };
    }

    /**********************************
     * size type aliases
     **********************************/

    namespace details
    {
        template <typename T, std::size_t N, class ArchList>
        struct sized_batch;

        template <typename T, std::size_t N>
        struct sized_batch<T, N, xsimd::arch_list<>>
        {
            using type = void;
        };

        template <typename T, std::size_t N, class Arch, class... Archs>
        struct sized_batch<T, N, xsimd::arch_list<Arch, Archs...>>
        {
            using type = typename std::conditional<xsimd::batch<T, Arch>::size == N, xsimd::batch<T, Arch>,
                                                   typename sized_batch<T, N, xsimd::arch_list<Archs...>>::type>::type;
        };
    }

    /**
     * @brief type utility to select a batch of given type and size
     *
     * If one of the available architectures has a native vector type of the
     * given type and size, sets the @p type member to the appropriate batch
     * type. Otherwise set its to @p void.
     *
     * @tparam T the type of the underlying values.
     * @tparam N the number of elements of that type in the batch.
     **/
    template <typename T, std::size_t N>
    struct make_sized_batch
    {
        using type = typename details::sized_batch<T, N, supported_architectures>::type;
    };

    template <typename T, std::size_t N>
    using make_sized_batch_t = typename make_sized_batch<T, N>::type;
}

#endif
