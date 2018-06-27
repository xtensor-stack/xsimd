/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_COMPLEX_BASE_HPP
#define XSIMD_COMPLEX_BASE_HPP

#include <complex>
#include <cstddef>
#include <ostream>

#ifdef XSIMD_ENABLE_XTL_COMPLEX
#include "xtl/xcomplex.hpp"
#endif

#include "xsimd_base.hpp"
#include "xsimd_utils.hpp"

namespace xsimd
{
    /*****************************
     * complex_batch_bool_traits *
     *****************************/

    template <class C, class R, std::size_t N, std::size_t Align>
    struct complex_batch_bool_traits
    {
        using value_type = C;
        static constexpr std::size_t size = N;
        using batch_type = batch<C, N>;
        static constexpr std::size_t align = Align;
        using real_batch = batch_bool<R, N>;
    };

    /***************************
     * simd_complex_batch_bool *
     ***************************/

    template <class X>
    class simd_complex_batch_bool : public simd_batch_bool<X>
    {
    public:

        using value_type = typename simd_batch_traits<X>::value_type;
        static constexpr std::size_t size = simd_batch_traits<X>::size;
        using real_batch = typename simd_batch_traits<X>::real_batch;

        simd_complex_batch_bool() = default;
        simd_complex_batch_bool(bool b);
        simd_complex_batch_bool(const real_batch& b);

        const real_batch& value() const;

        bool operator[](std::size_t index) const;

    private:

        real_batch m_value;
    };

    /************************
     * complex_batch_traits *
     ************************/

    template <class C, class R, std::size_t N, std::size_t Align>
    struct complex_batch_traits
    {
        using value_type = C;
        static constexpr std::size_t size = N;
        using batch_bool_type = batch_bool<C, N>;
        static constexpr std::size_t align = Align;
        using real_batch = batch<R, N>;
    };

    /**********************
     * simd_complex_batch *
     **********************/

    template <class T>
    struct is_ieee_compliant;
    
    template <class T>
    struct is_ieee_compliant<std::complex<T>>
        : std::integral_constant<bool, std::numeric_limits<std::complex<T>>::is_iec559>
    {
    };

#ifdef XSIMD_ENABLE_XTL_COMPLEX
    template <class T>
    struct is_ieee_compliant<xtl::xcomplex<T, T, false>> : std::false_type
    {
    };
#endif

    template <class X>
    class simd_complex_batch
    {
    public:

        using value_type = typename simd_batch_traits<X>::value_type;
        static constexpr std::size_t size = simd_batch_traits<X>::size;
        using real_batch = typename simd_batch_traits<X>::real_batch;
        using real_value_type = typename value_type::value_type;

        simd_complex_batch() = default;
        explicit simd_complex_batch(const value_type& v);
        simd_complex_batch(const real_batch& re, const real_batch& im);

        real_batch& real();
        real_batch& imag();
        
        const real_batch& real() const;
        const real_batch& imag() const;

        X& operator+=(const X& rhs);
        X& operator+=(const value_type& rhs);

        X& operator-=(const X& rhs);
        X& operator-=(const value_type& rhs);

        X& operator*=(const X& rhs);
        X& operator*=(const value_type& rhs);

        X& operator/=(const X& rhs);
        X& operator/=(const value_type& rhs);

        template <class T>
        X& load_aligned(const T* real_src, const T* imag_src);
        template <class T>
        X& load_unaligned(const T* real_src, const T* imag_src);

        template <class T>
        void store_aligned(T* real_dst, T* imag_dst) const;
        template <class T>
        void store_unaligned(T* real_dst, T* imag_dst) const;

        template <class T>
        X& load_aligned(const T* src);
        template <class T>
        X& load_unaligned(const T* src);

        template <class T>
        void store_aligned(T* dst) const;
        template <class T>
        void store_unaligned(T* dst) const;

        value_type operator[](std::size_t index) const;

        X& operator()();
        const X& operator()() const;

    protected:

        real_batch m_real;
        real_batch m_imag;
    };

    template <class X>
    X operator-(const simd_complex_batch<X>& rhs);

    template <class X>
    X operator+(const simd_complex_batch<X>& lhs, const simd_complex_batch<X>& rhs);
    template <class X>
    X operator+(const simd_complex_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs);
    template <class X>
    X operator+(const typename simd_batch_traits<X>::value_type& lhs, const simd_complex_batch<X>& rhs);

    template <class X>
    X operator-(const simd_complex_batch<X>& lhs, const simd_complex_batch<X>& rhs);
    template <class X>
    X operator-(const simd_complex_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs);
    template <class X>
    X operator-(const typename simd_batch_traits<X>::value_type& lhs, const simd_complex_batch<X>& rhs);

    template <class X>
    X operator*(const simd_complex_batch<X>& lhs, const simd_complex_batch<X>& rhs);
    template <class X>
    X operator*(const simd_complex_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs);
    template <class X>
    X operator*(const typename simd_batch_traits<X>::value_type& lhs, const simd_complex_batch<X>& rhs);

    template <class X>
    X operator/(const simd_complex_batch<X>& lhs, const simd_complex_batch<X>& rhs);
    template <class X>
    X operator/(const simd_complex_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs);
    template <class X>
    X operator/(const typename simd_batch_traits<X>::value_type& lhs, const simd_complex_batch<X>& rhs);

    template <class X>
    typename simd_batch_traits<X>::value_type
    hadd(const simd_complex_batch<X>& rhs);

    template <class X>
    X select(const typename simd_batch_traits<X>::batch_bool_type& cond,
             const simd_complex_batch<X>& a,
             const simd_complex_batch<X>& b);

    template <class X>
    typename simd_batch_traits<X>::batch_bool_type
    operator==(const simd_complex_batch<X>& lhs, const simd_complex_batch<X>& rhs);

    template <class X>
    typename simd_batch_traits<X>::batch_bool_type
    operator!=(const simd_complex_batch<X>& lhs, const simd_complex_batch<X>& rhs);

    template <class X>
    std::ostream& operator<<(std::ostream& out, const simd_complex_batch<X>& rhs);

    /*******************************************
     * xsimd_complex_batch_bool implementation *
     *******************************************/

    template <class X>
    inline simd_complex_batch_bool<X>::simd_complex_batch_bool(bool b)
        : m_value(b)
    {
    }

    template <class X>
    inline simd_complex_batch_bool<X>::simd_complex_batch_bool(const real_batch& b)
        : m_value(b)
    {
    }

    template <class X>
    inline auto simd_complex_batch_bool<X>::value() const -> const real_batch&
    {
        return m_value;
    }

    template <class X>
    inline bool simd_complex_batch_bool<X>::operator[](std::size_t index) const
    {
        return m_value[index];
    }

    namespace detail
    {
        template <class T, std::size_t N>
        struct batch_bool_complex_kernel
        {
            using batch_type = batch_bool<T, N>;

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
                return lhs.value() & rhs.value();
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
                return lhs.value() | rhs.value();
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
                return lhs.value() ^ rhs.value();
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
                return ~(rhs.value());
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
                return xsimd::bitwise_andnot(lhs.value(), rhs.value());
            }

            static batch_type equal(const batch_type& lhs, const batch_type& rhs)
            {
                return lhs.value() == rhs.value();
            }

            static batch_type not_equal(const batch_type& lhs, const batch_type& rhs)
            {
                return lhs.value() != rhs.value();
            }

            static bool all(const batch_type& rhs)
            {
                return xsimd::all(rhs.value());
            }

            static bool any(const batch_type& rhs)
            {
                return xsimd::any(rhs.value());
            }
        };

        template <class T, std::size_t N>
        struct batch_bool_kernel<std::complex<T>, N>
            : batch_bool_complex_kernel<std::complex<T>, N>
        {
        };

#ifdef XSIMD_ENABLE_XTL_COMPLEX
        template <class T, std::size_t N, bool ieee_compliant>
        struct batch_bool_kernel<xtl::xcomplex<T, T, ieee_compliant>, N>
            : batch_bool_complex_kernel<xtl::xcomplex<T, T, ieee_compliant>, N>
        {
        };
#endif
    }

    /**************************************
     * xsimd_complex_batch implementation *
     **************************************/

    template <class X>
    inline simd_complex_batch<X>::simd_complex_batch(const value_type& v)
        : m_real(v.real()), m_imag(v.imag())
    {
    }

    template <class X>
    inline simd_complex_batch<X>::simd_complex_batch(const real_batch& re, const real_batch& im)
        : m_real(re), m_imag(im)
    {
    }

    template <class X>
    inline auto simd_complex_batch<X>::real() -> real_batch&
    {
        return m_real;
    }

    template <class X>
    inline auto simd_complex_batch<X>::imag() -> real_batch&
    {
        return m_imag;
    }

    template <class X>
    inline auto simd_complex_batch<X>::real() const -> const real_batch&
    {
        return m_real;
    }

    template <class X>
    inline auto simd_complex_batch<X>::imag() const -> const real_batch&
    {
        return m_imag;
    }

    template <class X>
    inline X& simd_complex_batch<X>::operator+=(const X& rhs)
    {
        m_real += rhs.real();
        m_imag += rhs.imag();
        return (*this)();
    }

    template <class X>
    inline X& simd_complex_batch<X>::operator+=(const value_type& rhs)
    {
        return (*this)() += X(rhs);
    }

    template <class X>
    inline X& simd_complex_batch<X>::operator-=(const X& rhs)
    {
        m_real -= rhs.real();
        m_imag -= rhs.imag();
        return (*this)();
    }

    template <class X>
    inline X& simd_complex_batch<X>::operator-=(const value_type& rhs)
    {
        return (*this)() -= X(rhs);
    }

    namespace detail
    {
        template <class X, bool ieee_compliant>
        struct complex_batch_multiplier
        {
            using real_batch = typename simd_batch_traits<X>::real_batch;

            inline static X mul(const X& lhs, const X& rhs)
            {
                real_batch a = lhs.real();
                real_batch b = lhs.imag();
                real_batch c = rhs.real();
                real_batch d = rhs.imag();
                return X(a*c - b*d, a*d + b*c);
            }

            inline static X div(const X& lhs, const X& rhs)
            {
                real_batch a = lhs.real();
                real_batch b = lhs.imag();
                real_batch c = rhs.real();
                real_batch d = rhs.imag();
                real_batch e = c*c + d*d;
                return X((c*a + d*b) / e, (c*b - d*a) / e);
            }
        };
    }

    template <class X>
    inline X& simd_complex_batch<X>::operator*=(const X& rhs)
    {
        using kernel = detail::complex_batch_multiplier<X, is_ieee_compliant<value_type>::value>;
        (*this)() = kernel::mul((*this)(), rhs);
        return (*this)();
    }

    template <class X>
    inline X& simd_complex_batch<X>::operator*=(const value_type& rhs)
    {
        return (*this)() *= X(rhs);
    }

    template <class X>
    inline X& simd_complex_batch<X>::operator/=(const X& rhs)
    {
        using kernel = detail::complex_batch_multiplier<X, is_ieee_compliant<value_type>::value>;
        (*this)() = kernel::div((*this)(), rhs);
        return (*this)();
    }

    template <class X>
    inline X& simd_complex_batch<X>::operator/=(const value_type& rhs)
    {
        return (*this)() /= X(rhs);
    }

    template <class X>
    template <class T>
    inline X& simd_complex_batch<X>::load_aligned(const T* real_src, const T* imag_src)
    {
        m_real.load_aligned(real_src);
        m_imag.load_aligned(imag_src);
        return (*this)();
    }

    template <class X>
    template <class T>
    inline X& simd_complex_batch<X>::load_unaligned(const T* real_src, const T* imag_src)
    {
        m_real.load_unaligned(real_src);
        m_imag.load_unaligned(imag_src);
        return (*this)();
    }
    
    template <class X>
    template <class T>
    inline void simd_complex_batch<X>::store_aligned(T* real_dst, T* imag_dst) const
    {
        m_real.store_aligned(real_dst);
        m_imag.store_aligned(imag_dst);
    }

    template <class X>
    template <class T>
    inline void simd_complex_batch<X>::store_unaligned(T* real_dst, T* imag_dst) const
    {
        m_real.store_unaligned(real_dst);
        m_imag.store_unaligned(imag_dst);
    }

    template <class X>
    template <class T>
    inline X& simd_complex_batch<X>::load_aligned(const T* src)
    {
        using tmp_value_type = typename T::value_type;
        const tmp_value_type* rbuf = reinterpret_cast<const tmp_value_type*>(src);
        real_batch hi, lo;
        hi.load_aligned(rbuf);
        lo.load_aligned(rbuf + size);
        return (*this)().load_complex(hi, lo);
    }

    template <class X>
    template <class T>
    inline X& simd_complex_batch<X>::load_unaligned(const T* src)
    {
        using tmp_value_type = typename T::value_type;
        const tmp_value_type* rbuf = reinterpret_cast<const tmp_value_type*>(src);
        real_batch hi, lo;
        hi.load_unaligned(rbuf);
        lo.load_unaligned(rbuf + size);
        return (*this)().load_complex(hi, lo);
    }

    template <class X>
    template <class T>
    void simd_complex_batch<X>::store_aligned(T* dst) const
    {
        real_batch hi = (*this)().get_complex_high();
        real_batch lo = (*this)().get_complex_low();
        using tmp_value_type = typename T::value_type;
        tmp_value_type* rbuf = reinterpret_cast<tmp_value_type*>(dst);
        hi.store_aligned(rbuf);
        lo.store_aligned(rbuf + size);
    }

    template <class X>
    template <class T>
    void simd_complex_batch<X>::store_unaligned(T* dst) const
    {
        real_batch hi = (*this)().get_complex_high();
        real_batch lo = (*this)().get_complex_low();
        using tmp_value_type = typename T::value_type;
        tmp_value_type* rbuf = reinterpret_cast<tmp_value_type*>(dst);
        hi.store_unaligned(rbuf);
        lo.store_unaligned(rbuf + size);
    }

    template <class X>
    inline auto simd_complex_batch<X>::operator[](std::size_t index) const -> value_type
    {
        return value_type(m_real[index], m_imag[index]);
    }

    template <class X>
    inline X& simd_complex_batch<X>::operator()()
    {
        return *static_cast<X*>(this);
    }

    template <class X>
    inline const X& simd_complex_batch<X>::operator()() const
    {
        return *static_cast<const X*>(this);
    }

    /********************************
     * simd_complex_batch operators *
     ********************************/

    template <class X>
    inline X operator-(const simd_complex_batch<X>& rhs)
    {
        return X(-rhs().real(), -rhs().imag());
    }

    template <class X>
    inline X operator+(const simd_complex_batch<X>& lhs, const simd_complex_batch<X>& rhs)
    {
        X tmp(lhs());
        return tmp += rhs();
    }

    template <class X>
    inline X operator+(const simd_complex_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs)
    {
        X tmp(lhs());
        return tmp += X(rhs);
    }

    template <class X>
    inline X operator+(const typename simd_batch_traits<X>::value_type& lhs, const simd_complex_batch<X>& rhs)
    {
        X tmp(lhs);
        return tmp += rhs();
    }

    template <class X>
    inline X operator-(const simd_complex_batch<X>& lhs, const simd_complex_batch<X>& rhs)
    {
        X tmp(lhs());
        return tmp -= rhs();
    }

    template <class X>
    inline X operator-(const simd_complex_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs)
    {
        X tmp(lhs());
        return tmp -= X(rhs);
    }

    template <class X>
    inline X operator-(const typename simd_batch_traits<X>::value_type& lhs, const simd_complex_batch<X>& rhs)
    {
        X tmp(lhs);
        return tmp -= rhs();
    }

    template <class X>
    inline X operator*(const simd_complex_batch<X>& lhs, const simd_complex_batch<X>& rhs)
    {
        X tmp(lhs());
        return tmp *= rhs();
    }

    template <class X>
    inline X operator*(const simd_complex_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs)
    {
        X tmp(lhs());
        return tmp *= X(rhs);
    }

    template <class X>
    inline X operator*(const typename simd_batch_traits<X>::value_type& lhs, const simd_complex_batch<X>& rhs)
    {
        X tmp(lhs);
        return tmp *= rhs();
    }

    template <class X>
    inline X operator/(const simd_complex_batch<X>& lhs, const simd_complex_batch<X>& rhs)
    {
        X tmp(lhs());
        return tmp /= rhs();
    }

    template <class X>
    inline X operator/(const simd_complex_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs)
    {
        X tmp(lhs());
        return tmp /= X(rhs);
    }
    template <class X>
    inline X operator/(const typename simd_batch_traits<X>::value_type& lhs, const simd_complex_batch<X>& rhs)
    {
        X tmp(lhs);
        return tmp /= rhs();
    }

    template <class X>
    inline typename simd_batch_traits<X>::value_type
    hadd(const simd_complex_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        return value_type(hadd(rhs.real()), hadd(rhs.imag()));
    }

    template <class X>
    inline X select(const typename simd_batch_traits<X>::batch_bool_type& cond,
                    const simd_complex_batch<X>& a,
                    const simd_complex_batch<X>& b)
    {
        return X(select(cond, a.real(), b.real()), select(cond, a.imag(), b.imag()));
    }

    template <class X>
    inline typename simd_batch_traits<X>::batch_bool_type
    operator==(const simd_complex_batch<X>& lhs, const simd_complex_batch<X>& rhs)
    {
        return (lhs.real() == rhs.real()) && (lhs.imag() == rhs.imag());
    }

    template <class X>
    inline typename simd_batch_traits<X>::batch_bool_type
    operator!=(const simd_complex_batch<X>& lhs, const simd_complex_batch<X>& rhs)
    {
        return !(lhs == rhs);
    }

    template <class X>
    inline std::ostream& operator<<(std::ostream& out, const simd_complex_batch<X>& rhs)
    {
        out << '(';
        std::size_t s = simd_complex_batch<X>::size;
        for (std::size_t i = 0; i < s - 1; ++i)
        {
            out << "(" << rhs()[i].real() << "," << rhs()[i].imag() << "), ";
        }
        out << "(" << rhs()[s - 1].real() << "," << rhs()[s - 1].imag() << "))";
        return out;
    }
}

#endif

