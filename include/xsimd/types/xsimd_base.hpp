/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_BASE_HPP
#define XSIMD_BASE_HPP

#include <cstddef>

namespace xsimd
{
    template <class T, std::size_t N>
    class batch_bool;

    template <class T, std::size_t N>
    class batch;

    /*******************
     * simd_batch_bool *
     *******************/

    template <class X>
    class simd_batch_bool
    {

    public:

        X& operator&=(const X& rhs);
        X& operator|=(const X& rhs);
        X& operator^=(const X& rhs);

        X& operator()();
        const X& operator()() const;

    protected:

        simd_batch_bool() = default;
        ~simd_batch_bool() = default;

        simd_batch_bool(const simd_batch_bool&) = default;
        simd_batch_bool& operator=(const simd_batch_bool&) = default;

        simd_batch_bool(simd_batch_bool&&) = default;
        simd_batch_bool& operator=(simd_batch_bool&&) = default;
    };

    template <class X>
    X operator&&(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>& rhs);

    template <class X>
    X operator&&(const simd_batch_bool<X>& lhs, bool rhs);

    template <class X>
    X operator&&(bool lhs, const simd_batch_bool<X>& rhs);

    template <class X>
    X operator||(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>& rhs);

    template <class X>
    X operator||(const simd_batch_bool<X>& lhs, bool rhs);

    template <class X>
    X operator||(bool lhs, const simd_batch_bool<X>& rhs);

    template <class X>
    X operator!(const simd_batch_bool<X>& rhs);

    /**************
     * simd_batch *
     **************/

    template <class X>
    struct simd_batch_traits;

    template <class X>
    class simd_batch
    {

    public:

        using value_type = typename simd_batch_traits<X>::value_type;
        static std::size_t constexpr size = simd_batch_traits<X>::size;

        X& operator+=(const X& rhs);
        X& operator+=(const value_type& rhs);

        X& operator-=(const X& rhs);
        X& operator-=(const value_type& rhs);

        X& operator*=(const X& rhs);
        X& operator*=(const value_type& rhs);

        X& operator/=(const X& rhs);
        X& operator/=(const value_type& rhs);

        X& operator&=(const X& rhs);
        X& operator|=(const X& rhs);
        X& operator^=(const X& rhs);

        X& operator++();
        X& operator++(int);

        X& operator--();
        X& operator--(int);

        X& operator()();
        const X& operator()() const;

    protected:

        simd_batch() = default;
        ~simd_batch() = default;

        simd_batch(const simd_batch&) = default;
        simd_batch& operator=(const simd_batch&) = default;

        simd_batch(simd_batch&&) = default;
        simd_batch& operator=(simd_batch&&) = default;
    };

    template <class X>
    X operator+(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs);

    template <class X>
    X operator+(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs);

    template <class X>
    X operator-(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs);

    template <class X>
    X operator-(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs);

    template <class X>
    X operator*(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs);

    template <class X>
    X operator*(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs);

    template <class X>
    X operator/(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs);

    template <class X>
    X operator/(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs);

    template <class X>
    typename simd_batch_traits<X>::batch_bool_type 
    operator>(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    typename simd_batch_traits<X>::batch_bool_type
    operator>=(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    typename simd_batch_traits<X>::batch_bool_type
    operator!(const simd_batch<X>& rhs);


    /**********************************
     * simd_batch_bool implementation *
     **********************************/

    template <class X>
    inline X& simd_batch_bool<X>::operator&=(const X& rhs)
    {
        (*this)() = (*this)() & rhs;
        return (*this)();
    }

    template <class X>
    inline X& simd_batch_bool<X>::operator|=(const X& rhs)
    {
        (*this)() = (*this)() | rhs;
        return (*this)();
    }

    template <class X>
    inline X& simd_batch_bool<X>::operator^=(const X& rhs)
    {
        (*this)() = (*this)() ^ rhs;
        return (*this)();
    }

    template <class X>
    inline X& simd_batch_bool<X>::operator()()
    {
        return *static_cast<X*>(this);
    }

    template <class X>
    const X& simd_batch_bool<X>::operator()() const
    {
        return *static_cast<const X*>(this);
    }

    template <class X>
    inline X operator&&(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>& rhs)
    {
        return lhs() & rhs();
    }

    template <class X>
    inline X operator&&(const simd_batch_bool<X>& lhs, bool rhs)
    {
        return lhs() & X(rhs);
    }

    template <class X>
    inline X operator&&(bool lhs, const simd_batch_bool<X>& rhs)
    {
        return X(lhs) & rhs();
    }

    template <class X>
    inline X operator||(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>& rhs)
    {
        return lhs() | rhs();
    }

    template <class X>
    inline X operator||(const simd_batch_bool<X>& lhs, bool rhs)
    {
        return lhs() | X(rhs);
    }

    template <class X>
    inline X operator||(bool lhs, const simd_batch_bool<X>& rhs)
    {
        return X(lhs) | rhs();
    }

    template <class X>
    inline X operator!(const simd_batch_bool<X>& rhs)
    {
        return rhs() == 0;
    }


    /*****************************
     * simd_batch implementation *
     *****************************/
 
    template <class X>
    inline X& simd_batch<X>::operator+=(const X& rhs)
    {
        (*this)() = (*this)() + rhs;
        return (*this)();
    }

    template <class X>
    inline X& simd_batch<X>::operator+=(const value_type& rhs)
    {
        (*this)() = (*this)() + X(rhs);
        return (*this)();
    }

    template <class X>
    inline X& simd_batch<X>::operator-=(const X& rhs)
    {
        (*this)() = (*this)() - rhs;
        return (*this)();
    }

    template <class X>
    inline X& simd_batch<X>::operator-=(const value_type& rhs)
    {
        (*this)() = (*this)() - X(rhs);
        return (*this)();
    }

    template <class X>
    inline X& simd_batch<X>::operator*=(const X& rhs)
    {
        (*this)() = (*this)() * rhs;
        return (*this)();
    }

    template <class X>
    inline X& simd_batch<X>::operator*=(const value_type& rhs)
    {
        (*this)() = (*this)() * X(rhs);
        return (*this)();
    }

    template <class X>
    inline X& simd_batch<X>::operator/=(const X& rhs)
    {
        (*this)() = (*this)() / rhs;
        return (*this)();
    }

    template <class X>
    inline X& simd_batch<X>::operator/=(const value_type& rhs)
    {
        (*this)() = (*this)() / X(rhs);
        return (*this)();
    }

    template <class X>
    inline X& simd_batch<X>::operator&=(const X& rhs)
    {
        (*this)() = (*this)() & rhs;
        return (*this)();
    }

    template <class X>
    inline X& simd_batch<X>::operator|=(const X& rhs)
    {
        (*this)() = (*this)() | rhs;
        return (*this)();
    }

    template <class X>
    inline X& simd_batch<X>::operator^=(const X& rhs)
    {
        (*this)() = (*this)() ^ rhs;
        return (*this)();
    }

    template <class X>
    inline X& simd_batch<X>::operator++()
    {
        (*this)() += value_type(1);
        return (*this)();
    }

    template <class X>
    inline X& simd_batch<X>::operator++(int)
    {
        X tmp = (*this)();
        (*this)() += value_type(1);
        return tmp;
    }

    template <class X>
    inline X& simd_batch<X>::operator--()
    {
        (*this)() -= value_type(1);
        return (*this)();
    }

    template <class X>
    inline X& simd_batch<X>::operator--(int)
    {
        X tmp = (*this)();
        (*this)() -= value_type(1);
        return tmp;
    }

    template <class X>
    inline X& simd_batch<X>::operator()()
    {
        return *static_cast<X*>(this);
    }

    template <class X>
    inline const X& simd_batch<X>::operator()() const
    {
        return *static_cast<const X*>(this);
    }

    template <class X>
    inline X operator+(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs)
    {
        return lhs() + X(rhs);
    }

    template <class X>
    inline X operator+(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs)
    {
        return X(lhs) + rhs();
    }

    template <class X>
    inline X operator-(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs)
    {
        return lhs() - X(rhs);
    }

    template <class X>
    inline X operator-(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs)
    {
        return X(lhs) - rhs();
    }

    template <class X>
    inline X operator*(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs)
    {
        return lhs() * X(rhs);
    }

    template <class X>
    inline X operator*(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs)
    {
        return X(lhs) * rhs();
    }

    template <class X>
    inline X operator/(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs)
    {
        return lhs() / X(rhs);
    }

    template <class X>
    inline X operator/(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs)
    {
        return X(lhs) / rhs();
    }

    template <class X>
    inline typename simd_batch_traits<X>::batch_bool_type
    operator>(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        return rhs() < lhs();
    }

    template <class X>
    inline typename simd_batch_traits<X>::batch_bool_type
    operator>=(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        return rhs() <= lhs();
    }

    template <class X>
    inline typename simd_batch_traits<X>::batch_bool_type
    operator!(const simd_batch<X>& rhs)
    {
        return rhs() == X(0);
    }
}

#endif

