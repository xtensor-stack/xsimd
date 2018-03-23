/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_SHIM_HPP
#define XSIMD_SHIM_HPP

#include <cmath>
#include <iostream>

#include "xsimd_base.hpp"

namespace xsimd
{

    namespace detail
    {
        template <class T>
        struct get_int_type
        {
            using signed_type = typename std::make_signed<T>::type;
            using unsigned_type = typename std::make_unsigned<T>::type;
        };

        template <>
        struct get_int_type<float>
        {
            using signed_type = int32_t;
            using unsigned_type = uint32_t;
        };

        template <>
        struct get_int_type<double>
        {
            using signed_type = int64_t;
            using unsigned_type = uint64_t;
        };

        template <class T>
        struct get_float_type;

        template <>
        struct get_float_type<int32_t>
        {
            using type = float;
        };

        template <>
        struct get_float_type<int64_t>
        {
            using type = double;
        };
    }

    /*************************
     * batch_bool<T, N> *
     *************************/

    template <class T, std::size_t N>
    struct simd_batch_traits<batch_bool<T, N>>
    {
        using value_type = typename detail::get_int_type<T>::unsigned_type;
        static constexpr std::size_t size = N;
        using batch_bool_type = batch_bool<T, N>;
    };

    template <class T, std::size_t N>
    class batch_bool : public simd_batch_bool<batch_bool<T, N>>
    {

    public:
        using b_type = typename detail::get_int_type<T>::unsigned_type;
        using type = std::array<b_type, N>;

        batch_bool();
        template <class... Args>
        explicit batch_bool(Args... b);
        batch_bool(const batch<T, N>& rhs);
        batch_bool& operator=(const batch<T, N>& rhs);

        b_type& operator[](std::size_t index);
        const b_type& operator[](std::size_t index) const;

        operator type() const;

    private:

        type m_storage;
    };

    template <class T, std::size_t N>
    batch_bool<T, N> operator&(const batch_bool<T, N>& lhs, const batch_bool<T, N>& rhs);
    template <class T, std::size_t N>
    batch_bool<T, N> operator|(const batch_bool<T, N>& lhs, const batch_bool<T, N>& rhs);
    template <class T, std::size_t N>
    batch_bool<T, N> operator^(const batch_bool<T, N>& lhs, const batch_bool<T, N>& rhs);
    template <class T, std::size_t N>
    batch_bool<T, N> operator~(const batch_bool<T, N>& rhs);
    template <class T, std::size_t N>
    batch_bool<T, N> bitwise_andnot(const batch_bool<T, N>& lhs, const batch_bool<T, N>& rhs);

    template <class T, std::size_t N>
    batch_bool<T, N> operator==(const batch_bool<T, N>& lhs, const batch_bool<T, N>& rhs);
    template <class T, std::size_t N>
    batch_bool<T, N> operator!=(const batch_bool<T, N>& lhs, const batch_bool<T, N>& rhs);

    template <class T, std::size_t N>
    bool all(const batch_bool<T, N>& rhs);
    template <class T, std::size_t N>
    bool any(const batch_bool<T, N>& rhs);

    /********************
     * batch<T, N> impl *
     ********************/

    template <class T, std::size_t N>
    struct simd_batch_traits<batch<T, N>>
    {
        using value_type = T;
        static constexpr std::size_t size = N;
        using batch_bool_type = batch_bool<T, N>;
    };

    template <class T, std::size_t N>
    class batch : public simd_batch<batch<T, N>>
    {

    public:
        using self_type = batch<T, N>;
        using storage_type = std::array<T, N>;
        using value_type = T;

        batch();
        batch(T src);
        batch(const batch<T, N>& src);
        batch(const storage_type& src);

        explicit batch(T* src);
        batch(T* src, aligned_mode);
        batch(T* src, unaligned_mode);
        batch& operator=(const storage_type& rhs);
        batch& operator=(const T& rhs);

        operator storage_type() const;

        batch& load_aligned(T* src);
        batch& load_unaligned(T* src);

        void store_aligned(T* dst) const;
        void store_unaligned(T* dst) const;

        T& operator[](std::size_t index);
        const T& operator[](std::size_t index) const;

    private:

        alignas(32) storage_type m_storage;
        value_type* m_value = (value_type*) __builtin_assume_aligned(&m_storage[0], 32);
    };

    template <class T, std::size_t N>
    batch<T, N> operator-(const batch<T, N>& rhs);
    template <class T, std::size_t N>
    batch<T, N> operator+(const batch<T, N>& lhs, const batch<T, N>& rhs);
    template <class T, std::size_t N>
    batch<T, N> operator-(const batch<T, N>& lhs, const batch<T, N>& rhs);
    template <class T, std::size_t N>
    batch<T, N> operator*(const batch<T, N>& lhs, const batch<T, N>& rhs);
    template <class T, std::size_t N>
    batch<T, N> operator/(const batch<T, N>& lhs, const batch<T, N>& rhs);

    template <class T, std::size_t N>
    batch_bool<T, N> operator==(const batch<T, N>& lhs, const batch<T, N>& rhs);
    template <class T, std::size_t N>
    batch_bool<T, N> operator!=(const batch<T, N>& lhs, const batch<T, N>& rhs);
    template <class T, std::size_t N>
    batch_bool<T, N> operator<(const batch<T, N>& lhs, const batch<T, N>& rhs);
    template <class T, std::size_t N>
    batch_bool<T, N> operator<=(const batch<T, N>& lhs, const batch<T, N>& rhs);

    template <class T, std::size_t N>
    batch<T, N> operator&(const batch<T, N>& lhs, const batch<T, N>& rhs);
    template <class T, std::size_t N>
    batch<T, N> operator|(const batch<T, N>& lhs, const batch<T, N>& rhs);
    template <class T, std::size_t N>
    batch<T, N> operator^(const batch<T, N>& lhs, const batch<T, N>& rhs);
    template <class T, std::size_t N>
    batch<T, N> operator~(const batch<T, N>& rhs);
    template <class T, std::size_t N>
    batch<T, N> bitwise_andnot(const batch<T, N>& lhs, const batch<T, N>& rhs);
    template <class T, std::size_t N>
    batch<T, N> operator<<(const batch<T, N>& rhs, const int n);
    template <class T, std::size_t N>
    batch<T, N> operator>>(const batch<T, N>& rhs, const int n);

    template <class T, std::size_t N>
    batch<T, N> min(const batch<T, N>& lhs, const batch<T, N>& rhs);
    template <class T, std::size_t N>
    batch<T, N> max(const batch<T, N>& lhs, const batch<T, N>& rhs);
    template <class T, std::size_t N>
    batch<T, N> fmin(const batch<T, N>& lhs, const batch<T, N>& rhs);
    template <class T, std::size_t N>
    batch<T, N> fmax(const batch<T, N>& lhs, const batch<T, N>& rhs);

    template <class T, std::size_t N>
    batch<T, N> abs(const batch<T, N>& rhs);
    template <class T, std::size_t N>
    batch<T, N> sqrt(const batch<T, N>& rhs);

    template <class T, std::size_t N>
    batch<T, N> fma(const batch<T, N>& x, const batch<T, N>& y, const batch<T, N>& z);
    template <class T, std::size_t N>
    batch<T, N> fms(const batch<T, N>& x, const batch<T, N>& y, const batch<T, N>& z);
    template <class T, std::size_t N>
    batch<T, N> fnma(const batch<T, N>& x, const batch<T, N>& y, const batch<T, N>& z);
    template <class T, std::size_t N>
    batch<T, N> fnms(const batch<T, N>& x, const batch<T, N>& y, const batch<T, N>& z);

    template <class T, std::size_t N>
    double hadd(const batch<T, N>& rhs);
    template <class T, std::size_t N>
    batch<T, N> haddp(const batch<T, N>* row);

    template <class T, std::size_t N>
    batch<T, N> select(const batch_bool<T, N>& cond, const batch<T, N>& a, const batch<T, N>& b);

    template <class T, std::size_t N>
    batch_bool<T, N> is_nan(const batch<T, N>& x);

    /****************************************
     * batch_bool<T, N> implementation *
     ****************************************/

    template <class T, std::size_t N>
    inline batch_bool<T, N>::batch_bool()
    {
    }

    template <class T, std::size_t N>
    template <class... Args>
    inline batch_bool<T, N>::batch_bool(Args... b)
        : m_storage{b...}
    {
    }

    // template <class T, std::size_t N>
    // inline batch_bool<T, N>::batch_bool(const T& rhs)
    //     : m_storage(rhs)
    // {
    // }

    // template <class T, std::size_t N>
    // inline batch_bool<T, N>& batch_bool<T, N>::operator=(const T& rhs)
    // {
    //     m_storage = rhs;
    //     return *this;
    // }

#define OP_MACRO(op)                                                                     \
    batch<T, N> tmp;                                                                     \
    for (int i = 0; i < N; ++i)                                                  \
    {                                                                                    \
        tmp[i] = lhs[i] op rhs[i];                                                       \
    }                                                                                    \
    return tmp;                                                                          \

#define UNARY_OP_MACRO(op)                                                               \
    batch<T, N> tmp;                                                                     \
    for (int i = 0; i < N; ++i)                                                  \
    {                                                                                    \
        tmp[i] = op rhs[i];                                                              \
    }                                                                                    \
    return tmp;                                                                          \

#define BOOL_OP_MACRO(op)                                                                \
    using int_t = typename detail::get_int_type<T>::unsigned_type;                       \
    batch_bool<T, N> tmp;                                                                \
    for (std::size_t i = 0; i < N; ++i)                                                  \
    {                                                                                    \
        tmp[i] = lhs[i] op rhs[i];                                                       \
    }                                                                                    \
    return tmp;                                                                          \

#define BITWISE_OP_MACRO(op)                                                             \
    using int_t = typename detail::get_int_type<T>::unsigned_type;                       \
    batch<T, N> tmp;                                                                     \
    for (std::size_t i = 0; i < N; ++i)                                                  \
    {                                                                                    \
        int_t it = (*reinterpret_cast<const int_t*>(&lhs[i])) op (*reinterpret_cast<const int_t*>(&rhs[i])); \
        tmp[i] = *reinterpret_cast<T*>(&it);                                             \
    }                                                                                    \
    return tmp;                                                                          \

#define BOOL_MACRO(op)                                                                   \
    using int_t = typename detail::get_int_type<T>::unsigned_type;                       \
    batch_bool<T, N> tmp;                                                                \
    for (std::size_t i = 0; i < N; ++i)                                                  \
    {                                                                                    \
        tmp[i] = static_cast<T>(static_cast<int_t>(lhs) op static_cast<int_t>(rhs));     \
    }                                                                                    \
    return tmp;                                                                          \

#define BOOL_MACRO_UNARY(op)                                                             \
    using int_t = typename detail::get_int_type<T>::unsigned_type;                       \
    batch_bool<T, N> tmp;                                                                \
    for (std::size_t i = 0; i < N; ++i)                                                  \
    {                                                                                    \
        tmp[i] = static_cast<T>(op static_cast<int_t>(lhs));                             \
    }                                                                                    \
    return tmp;                                                                          \

    template <class T, std::size_t N>
    inline batch_bool<T, N>::operator type() const
    {
        return m_storage;
    }

    template <class T, std::size_t N>
    inline typename batch_bool<T, N>::b_type& batch_bool<T, N>::operator[](std::size_t index)
    {
        return m_storage[index];
    }

    template <class T, std::size_t N>
    inline const typename batch_bool<T, N>::b_type& batch_bool<T, N>::operator[](std::size_t index) const
    {
        return m_storage[index];
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> operator&(const batch_bool<T, N>& lhs, const batch_bool<T, N>& rhs)
    {
        BOOL_MACRO(&);
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> operator|(const batch_bool<T, N>& lhs, const batch_bool<T, N>& rhs)
    {
        BOOL_MACRO(|);
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> operator^(const batch_bool<T, N>& lhs, const batch_bool<T, N>& rhs)
    {
        BOOL_MACRO(^);
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> operator~(const batch_bool<T, N>& rhs)
    {
        using int_t = typename detail::get_int_type<T>::unsigned_type;
        batch_bool<T, N> tmp;
        for (std::size_t i = 0; i < N; ++i)
        {
            int_t it = ~(*reinterpret_cast<const int_t*>(&rhs[i]));
            tmp[i] = *reinterpret_cast<T*>(&it);
        }
        return tmp;
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> bitwise_andnot(const batch_bool<T, N>& lhs, const batch_bool<T, N>& rhs)
    {
        lhs & (~rhs);
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> operator==(const batch_bool<T, N>& lhs, const batch_bool<T, N>& rhs)
    {
        BOOL_MACRO(==);
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> operator!=(const batch_bool<T, N>& lhs, const batch_bool<T, N>& rhs)
    {
        BOOL_MACRO(!=)
    }

    template <class T, std::size_t N>
    inline bool all(const batch_bool<T, N>& rhs)
    {
        bool result = true;
        for (std::size_t i = 0; i < N; ++i)
        {
            result = result && (bool(rhs[i]));
        }
        return result;
    }

    template <class T, std::size_t N>
    inline bool any(const batch_bool<T, N>& rhs)
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            if (rhs[i])
            {
                return true;
            }
        }
        return false;
    }

    /*************************************
     * template <class T, std::size_t N> *
     * batch<T, N> implementation        *
     *************************************/

    template <class T, std::size_t N>
    inline batch<T, N>::batch()
    {
    }

    template <class T, std::size_t N>
    inline batch<T, N>::batch(T rhs)
    {
        std::fill(m_storage.begin(), m_storage.end(), rhs);
        m_value = &m_storage[0];
    }

    template <class T, std::size_t N>
    inline batch<T, N>::batch(T* src)
        : m_value(src)
    {
        // std::copy(src, src + N, m_storage.begin());
    }

    template <class T, std::size_t N>
    inline batch<T, N>::batch(T* src, aligned_mode)
        : m_value((T*) __builtin_assume_aligned(src, 32))
    {
    }

    template <class T, std::size_t N>
    inline batch<T, N>::batch(T* src, unaligned_mode)
        : batch(src)
    {
    }

    template <class T, std::size_t N>
    inline batch<T, N>::batch(const storage_type& d)
        : m_storage({d}), m_value(&m_storage[0])
    {
    }

    template <class T, std::size_t N>
    inline batch<T, N>::batch(const batch<T, N>& lhs)
        : batch(lhs.m_storage)
    {
    }

    template <class T, std::size_t N>
    inline batch<T, N>& batch<T, N>::operator=(const T& rhs)
    {
        std::fill(m_storage.begin(), m_storage.end(), rhs);
        m_value = &m_storage[0];
        return *this;
    }

    template <class T, std::size_t N>
    inline batch<T, N>::operator storage_type() const
    {
        if (&m_storage[0] != m_value)
        {
            std::copy(m_value, m_value + N, m_storage);
        }
        return m_storage;
    }

    template <class T, std::size_t N>
    __attribute__((always_inline)) inline batch<T, N>& batch<T, N>::load_aligned(T* src)
    {

        m_value = (T*) __builtin_assume_aligned(src, 32);;
        // std::copy(src, src + N, m_storage.begin());
        return *this;
    }

    template <class T, std::size_t N>
    inline batch<T, N>& batch<T, N>::load_unaligned(T* src)
    {
        m_value = src;
        return *this;
    }

    template <class T, std::size_t N>
    __attribute__((always_inline)) inline void store_aligned_impl(T* __restrict__ dst, const T* __restrict__ src)
    {
        T* dst_a = (T*) __builtin_assume_aligned(dst, 32);
        const T* src_a = (const T*) __builtin_assume_aligned(src, 32);
        for (std::size_t i = 0; i < N; ++i)
        {
            dst_a[i] = src_a[i];
        }
    }

    template <class T, std::size_t N>
    __attribute__((always_inline)) inline void batch<T, N>::store_aligned(T* dst) const
    {
        store_aligned_impl<T, N>(dst, m_value);
        // auto dst_a = (T*) __builtin_assume_aligned(dst, 32);
        // for (std::size_t i = 0; i < N; ++i)
        // {
        //     dst[i] = m_value[i];
        // }
        // std::copy(m_value, m_value + N, dst);
    }

    template <class T, std::size_t N>
    inline void batch<T, N>::store_unaligned(T* dst) const
    {
        store_aligned(dst);
    }

    template <class T, std::size_t N>
    inline T& batch<T, N>::operator[](std::size_t index)
    {
        return m_value[index];
    }

    template <class T, std::size_t N>
    inline const T& batch<T, N>::operator[](std::size_t index) const
    {
        return m_value[index];
    }

    template <class T, std::size_t N>
    inline batch<T, N> operator-(const batch<T, N>& rhs)
    {
        UNARY_OP_MACRO(-);
    }

    template <class T, std::size_t N>
    __attribute__((always_inline)) batch<T, N> operator+(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        batch<T, N> tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = lhs[i] + rhs[i];
        }
        return tmp;
        // OP_MACRO(+);
    }

    template <class T, std::size_t N>
    inline batch<T, N> operator-(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        OP_MACRO(-);
    }

    template <class T, std::size_t N>
    inline batch<T, N> operator*(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        OP_MACRO(*);
    }

    template <class T, std::size_t N>
    inline batch<T, N> operator/(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        OP_MACRO(/);
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> operator==(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        BOOL_OP_MACRO(==);
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> operator!=(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        BOOL_OP_MACRO(!=);
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> operator<(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        BOOL_OP_MACRO(<);
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> operator<=(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        BOOL_OP_MACRO(<=);
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> operator>(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        BOOL_OP_MACRO(>);
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> operator>=(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        BOOL_OP_MACRO(>=);
    }

    template <class T, std::size_t N>
    inline batch<T, N> operator&(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        BITWISE_OP_MACRO(&);
    }

    template <class T, std::size_t N>
    inline batch<T, N> operator|(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        BITWISE_OP_MACRO(|);
    }

    template <class T, std::size_t N>
    inline batch<T, N> operator^(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        BITWISE_OP_MACRO(^);
    }

#define EXPR_MACRO(expr)                                                                 \
    batch<T, N> tmp;                                                                     \
    for (std::size_t i = 0; i < N; ++i)                                                  \
    {                                                                                    \
        tmp[i] = expr;                                                                   \
    }                                                                                    \
    return tmp;                                                                          \

    template <class T, std::size_t N>
    inline batch<T, N> operator~(const batch<T, N>& rhs)
    {
        using int_t = typename detail::get_int_type<T>::unsigned_type;
        EXPR_MACRO(*reinterpret_cast<const T*>(~(*reinterpret_cast<const int_t*>(&rhs[i]))))
    }

    template <class T, std::size_t N>
    inline batch<T, N> bitwise_andnot(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        return lhs & (~rhs);
    }

    template <class T, std::size_t N>
    inline batch<T, N> operator<<(const batch<T, N>& lhs, const int n)
    {
        EXPR_MACRO(lhs[i] << n);
    }

    template <class T, std::size_t N>
    inline batch<T, N> operator>>(const batch<T, N>& lhs, const int n)
    {
        EXPR_MACRO(lhs[i] >> n);
    }

    template <class T, std::size_t N>
    inline batch<T, N> min(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        EXPR_MACRO(lhs[i] < rhs[i] ? lhs[i] : rhs[i]);
    }

    template <class T, std::size_t N>
    inline batch<T, N> max(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        EXPR_MACRO(lhs[i] > rhs[i] ? lhs[i] : rhs[i]);
    }

    template <class T, std::size_t N>
    inline batch<T, N> fmin(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        return min(lhs, rhs);
    }

    template <class T, std::size_t N>
    inline batch<T, N> fmax(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        return max(lhs, rhs);
    }

    template <class T, std::size_t N>
    inline batch<T, N> abs(const batch<T, N>& rhs)
    {
        EXPR_MACRO(std::abs(rhs[i]));
    }

    template <class T, std::size_t N>
    inline batch<T, N> sqrt(const batch<T, N>& rhs)
    {
        EXPR_MACRO(std::sqrt(rhs[i]));
    }

    template <class T, std::size_t N>
    inline batch<T, N> fma(const batch<T, N>& x, const batch<T, N>& y, const batch<T, N>& z)
    {
        EXPR_MACRO(x[i] * y[i] + z[i]);
    }

    template <class T, std::size_t N>
    inline batch<T, N> fms(const batch<T, N>& x, const batch<T, N>& y, const batch<T, N>& z)
    {
        EXPR_MACRO(x[i] * y[i] - z[i]);
    }

    template <class T, std::size_t N>
    inline batch<T, N> fnma(const batch<T, N>& x, const batch<T, N>& y, const batch<T, N>& z)
    {
        EXPR_MACRO(-x[i] * y[i] + z[i]);
    }

    template <class T, std::size_t N>
    inline batch<T, N> fnms(const batch<T, N>& x, const batch<T, N>& y, const batch<T, N>& z)
    {
        EXPR_MACRO(-x[i] * y[i] - z[i]);
    }

    template <class T, std::size_t N>
    inline T hadd(const batch<T, N>& rhs)
    {
        T result;
        for (std::size_t i = 0; i < N; ++i)
        {
            result += rhs[i];
        }
        return result;
    }
    // TODO add tests for these functions
    template <class T, std::size_t N>
    inline batch<T, N> haddp(const batch<T, N>* row)
    {
        batch<T, N> result;
        for (std::size_t i = 0; i < N; ++i)
        {
            for (std::size_t j = 0; j < N; ++j)
            {
                result[i] += row[i][j];
            }
        }
        return result;
    }

    template <class T, std::size_t N>
    inline batch<T, N> select(const batch_bool<T, N>& cond, const batch<T, N>& a, const batch<T, N>& b)
    {
        EXPR_MACRO(cond[i] ? a[i] : b[i]);
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> is_nan(const batch<T, N>& x)
    {
        batch_bool<T, N> result;
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = std::isnan(x[i]);
        }
        return result;
    }

    // rounding

    template <class T, std::size_t N>
    batch<typename detail::get_int_type<T>::signed_type, N> to_int(const batch<T, N>& x)
    {
        using int_type = typename detail::get_int_type<T>::signed_type;
        using result_type = batch<int_type, N>;
        result_type result;
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = static_cast<int_type>(x[i]);
        }
        return result;
    }

    template <class T, std::size_t N>
    batch<typename detail::get_float_type<T>::type, N> to_float(const batch<T, N>& x)
    {
        using float_type = typename detail::get_float_type<T>::type;
        using result_type = batch<float_type, N>;
        result_type result;
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = static_cast<float_type>(x[i]);
        }
        return result;
    }

    namespace detail
    {
        template <class T>
        struct get_opposite
        {
            using type = typename get_float_type<T>::type;
        };

        template <>
        struct get_opposite<double>
        {
            using type = int64_t;
        };

        template <>
        struct get_opposite<float>
        {
            using type = int32_t;
        };

        template <class T>
        using get_opposite_t = typename get_opposite<T>::type;

        template <class F, class T, std::size_t N>
        struct batch_caster
        {
            batch<T, N> bitwise(const batch<F, N>& x)
            {
                return batch<T, N>(const_cast<T*>(reinterpret_cast<const T*>(&x[0])));
            }
        };
    }

    template <class T, class F, std::size_t N>
    T bitwise_cast(const batch<F, N>& x)
    {
        return detail::batch_caster<F, typename simd_batch_traits<T>::value_type, N>().bitwise(x);
    }

    // template <class T, std::size_t N>
    // batch<T, N> trunc(const batch<T, N>& x)
    // {
    //     EXPR_MACRO(std::trunc(x[i]));
    // }

    template <class T, std::size_t N>
    batch<T, N> nearbyint(const batch<T, N>& x)
    {
        EXPR_MACRO(std::nearbyint(x[i]));
    }

    template <class T, std::size_t N>
    batch<T, N> floor(const batch<T, N>& x)
    {
        EXPR_MACRO(std::floor(x[i]));
    }
}

#endif