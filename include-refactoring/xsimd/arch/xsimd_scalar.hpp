#ifndef XSIMD_SCALAR_HPP
#define XSIMD_SCALAR_HPP

#include <cmath>
#include <limits>

namespace xsimd
{
    namespace detail
    {
        template <class C>
        inline C expm1_complex_scalar_impl(const C& val)
        {
            using T = typename C::value_type;
            T isin = std::sin(val.imag());
            T rem1 = std::expm1(val.real());
            T re = rem1 + T(1.);
            T si = std::sin(val.imag() * T(0.5));
            return std::complex<T>(rem1 - T(2.) * re *si * si, re * isin);
        }
    }

    template <class T>
    inline std::complex<T> expm1(const std::complex<T>& val)
    {
        return detail::expm1_complex_scalar_impl(val);
    }

    namespace detail
    {
        template <class C>
        inline C log1p_complex_scalar_impl(const C& val)
        {
            using T = typename C::value_type;
            C u = C(1.) + val;
            return u == C(1.) ? val : (u.real() <= T(0.) ? log(u) : log(u) * val / (u - C(1.)));
        }
    }

    template <class T>
    inline std::complex<T> log1p(const std::complex<T>& val)
    {
        return detail::log1p_complex_scalar_impl(val);
    }

    template <class T>
    std::complex<T> log2(const std::complex<T>& val)
    {
        return log(val) / std::log(T(2));
    }

    template<typename T, class = typename std::enable_if<std::is_scalar<T>::value>::type>
    T sadd(const T& lhs, const T& rhs)
    {
        if (std::numeric_limits<T>::is_signed)
        {
            if ((lhs > 0) && (rhs > std::numeric_limits<T>::max() - lhs))
            {
                return std::numeric_limits<T>::max();
            }
            else if ((lhs < 0) && (rhs < std::numeric_limits<T>::lowest() - lhs))
            {
                return std::numeric_limits<T>::lowest();
            }
            else {
                return lhs + rhs;
            }
        }
        else
        {
            if (rhs > std::numeric_limits<T>::max() - lhs)
            {
                return std::numeric_limits<T>::max();
            }
            else
            {
                return lhs + rhs;
            }

        }
    }

    template<typename T, class = typename std::enable_if<std::is_scalar<T>::value>::type>
    T ssub(const T& lhs, const T& rhs)
    {
        if (std::numeric_limits<T>::is_signed)
        {
            return sadd(lhs, (T)-rhs);
        }
        else
        {
            if (lhs < rhs)
            {
                return std::numeric_limits<T>::lowest();
            }
            else
            {
                return lhs - rhs;
            }

        }
    }
    template <class T, class = typename std::enable_if<std::is_scalar<T>::value>::type>
    inline T sign(const T& v)
    {
        return v < T(0) ? T(-1.) : v == T(0) ? T(0.) : T(1.);
    }
    namespace detail
    {
        template <class C>
        inline C sign_complex_scalar_impl(const C& v)
        {
            using value_type = typename C::value_type;
            if (v.real())
            {
                return C(sign(v.real()), value_type(0));
            }
            else
            {
                return C(sign(v.imag()), value_type(0));
            }
        }
    }

    template <class T>
    inline std::complex<T> sign(const std::complex<T>& v)
    {
        return detail::sign_complex_scalar_impl(v);
    }

}

#endif
