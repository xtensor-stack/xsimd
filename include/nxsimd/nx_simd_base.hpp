//
// Copyright (c) 2016 Johan Mabille
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
//

#ifndef NX_SIMD_BASE_HPP
#define NX_SIMD_BASE_HPP

namespace nxsimd
{

    //Boolean simd CRTP base class
    template <class X>
    class simd_vector_bool
    {

    public:

        X& operator&=(const X& rhs);
        X& operator|=(const X& rhs);
        X& operator^=(const X& rhs);

        X& operator()();
        const X& operator()() const;

    protected:

        simd_vector_bool() = default;
        ~simd_vector_bool() = default;

        simd_vector_bool(const simd_vector_bool&) = default;
        simd_vector_bool& operator=(const simd_vector_bool) = default;

        simd_vector_bool(simd_vector_bool&&) = default;
        simd_vector_bool operator=(simd_vector_bool&&) = default;
    };

    template <class X>
    X operator&&(const simd_vector_bool<X>& lhs, const simd_vector_bool<X>& rhs);

    template <class X>
    X operator&&(const simd_vector_bool<X>& lhs, bool rhs);

    template <class X>
    X operator&&(bool lhs, const simd_vector_bool<X>& rhs);

    template <class X>
    X operator||(const simd_vector_bool<X>& lhs, const simd_vector_bool<X>& rhs);

    template <class X>
    X operator||(const simd_vector_bool<X>& lhs, bool rhs);

    template <class X>
    X operator||(bool lhs, const simd_vector_bool<X>& rhs);

    template <class X>
    X operator!(const simd_vector_bool<X>& rhs);


    template <class X>
    struct simd_vector_traits;


    // Numeric simd CRTP class
    template <class X>
    class simd_vector
    {

    public:

        using value_type = typename simd_vector_traits<X>::value_type;

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

        simd_vector() = default;
        ~simd_vector() = default;

        simd_vector(const simd_vector&) = default;
        simd_vector& operator=(const simd_vector&) = default;

        simd_vector(simd_vector&&) = default;
        simd_vector& operator=(simd_vector&&) = default;
    };

    template <class X>
    X operator+(const simd_vector<X>& lhs, const typename simd_vector_traits<X>::value_type& rhs);

    template <class X>
    X operator+(const typename simd_vector<X>::value_type& lhs, const simd_vector<X>& rhs);

    template <class X>
    X operator-(const simd_vector<X>& lhs, const typename simd_vector_traits<X>::value_type& rhs);

    template <class X>
    X operator-(const typename simd_vector<X>::value_type& lhs, const simd_vector<X>& rhs);

    template <class X>
    X operator*(const simd_vector<X>& lhs, const typename simd_vector_traits<X>::value_type& rhs);

    template <class X>
    X operator*(const typename simd_vector<X>::value_type& lhs, const simd_vector<X>& rhs);

    template <class X>
    X operator/(const simd_vector<X>& lhs, const typename simd_vector_traits<X>::value_type& rhs);

    template <class X>
    X operator/(const typename simd_vector<X>::value_type& lhs, const simd_vector<X>& rhs);

    template <class X>
    typename simd_vector_traits<X>::vector_bool 
    operator>(const simd_vector<X>& lhs, const simd_vector<X>& rhs);

    template <class X>
    typename simd_vector_traits<X>::vector_bool
    operator>=(const simd_vector<X>& lhs, const simd_vector<X>& rhs);

    template <class X>
    typename simd_vector_traits<X>::vector_bool
    operator!(const simd_vector<X>& rhs);


    /*************************************
     * simd_vector_bool implementation
     *************************************/

    template <class X>
    inline X& simd_vector_bool<X>::operator&=(const X& rhs)
    {
        (*this)() = (*this)() & rhs;
        return (*this)();
    }

    template <class X>
    inline X& simd_vector_bool<X>::operator|=(const X& rhs)
    {
        (*this)() = (*this)() | rhs;
        return (*this)();
    }

    template <class X>
    inline X& simd_vector_bool<X>::operator^=(const X& rhs)
    {
        (*this)() = (*this)() ^ rhs;
        return (*this)();
    }

    template <class X>
    inline X& simd_vector_bool<X>::operator()()
    {
        return *static_cast<X*>(this);
    }

    template <class X>
    const X& simd_vector_bool<X>::operator()() const
    {
        return *static_cast<const X*>(this);
    }

    template <class X>
    X operator&&(const simd_vector_bool<X>& lhs, const simd_vector_bool<X>& rhs)
    {
        return lhs() & rhs();
    }

    template <class X>
    X operator&&(const simd_vector_bool<X>& lhs, bool rhs)
    {
        return lhs() & X(rhs);
    }

    template <class X>
    X operator&&(bool lhs, const simd_vector_bool<X>& rhs)
    {
        return X(lhs) & rhs();

    template <class X>
    X operator||(const simd_vector_bool<X>& lhs, const simd_vector_bool<X>& rhs)
    {
        return lhs() | rhs();
    }

    template <class X>
    X operator||(const simd_vector_bool<X>& lhs, bool rhs)
    {
        return lhs() | X(rhs);
    }

    template <class X>
    X operator||(bool lhs, const simd_vector_bool<X>& rhs)
    {
        return X(lhs) | rhs();
    }

    template <class X>
    X operator!(const simd_vector_bool<X>& rhs)
    {
        return rhs() == 0;
    }


    /*************************************
     * simd_vector_base implementation
     *************************************/
 
    template <class X>
    inline X& simd_vector<X>::operator+=(const X& rhs)
    {
        (*this)() = (*this)() + rhs;
        return *this;
    }

    template <class X>
    inline X& simd_vector<X>::operator+=(const value_type& rhs)
    {
        (*this)() = (*this)() + X(rhs);
        return *this;
    }

    template <class X>
    inline X& simd_vector<X>::operator-=(const X& rhs)
    {
        (*this)() = (*this)() - rhs;
        return *this;
    }

    template <class X>
    inline X& simd_vector<X>::operator-=(const value_type& rhs)
    {
        (*this)() = (*this)() - X(rhs);
        return *this;
    }

    template <class X>
    inline X& simd_vector<X>::operator*=(const X& rhs)
    {
        (*this)() = (*this)() * rhs;
        return *this;
    }

    template <class X>
    inline X& simd_vector<X>::operator*=(const value_type& rhs)
    {
        (*this)() = (*this)() * X(rhs);
        return *this;
    }

    template <class X>
    inline X& simd_vector<X>::operator/=(const X& rhs)
    {
        (*this)() = (*this)() / rhs;
        return *this;
    }

    template <class X>
    inline X& simd_vector<X>::operator/=(const value_type& rhs)
    {
        (*this)() = (*this)() / X(rhs);
        return *this;
    }

    template <class X>
    inline X& simd_vector<X>::operator&=(const X& rhs)
    {
        (*this)() = (*this)() & rhs;
        return (*this);
    }

    template <class X>
    inline X& simd_vector<X>::operator|=(const X& rhs)
    {
        (*this)() = (*this)() | rhs;
        return (*this);
    }

    template <class X>
    inline X& simd_vector<X>::operator^=(const X& rhs)
    {
        (*this)() = (*this)() ^ rhs;
        return *this;
    }

    template <class X>
    inline X& simd_vector<X>::operator++()
    {
        (*this)() += value_type(1);
        return (*this);
    }

    template <class X>
    inline X& simd_vector<X>::operator++(int)
    {
        X tmp = (*this)();
        (*this)() += value_type(1);
        return tmp;
    }

    template <class X>
    inline X& simd_vector<X>::operator--()
    {
        (*this)() -= value_type(1);
        return (*this);
    }

    template <class X>
    inline X& simd_vector<X>::operator--(int)
    {
        X tmp = (*this)();
        (*this)() -= value_type(1);
        return tmp;
    }

    template <class X>
    inline X& simd_vector<X>::operator()()
    {
        return *static_cast<X*>(this);
    }

    template <class X>
    inline const X& simd_vector<X>::operator()() const
    {
        return *static_cast<const X*>(this);
    }

}

#endif

