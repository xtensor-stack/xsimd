/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_NEON_BOOL_HPP
#define XSIMD_NEON_BOOL_HPP

#include "xsimd_base.hpp"

#define EXPAND(...) __VA_ARGS__

#define CASE(i, op)              \
    case i: return op(lhs, i);   \

#define REPEAT_32(op, addx)      \
    CASE( 1 + addx, EXPAND(op)); \
    CASE( 2 + addx, EXPAND(op)); \
    CASE( 3 + addx, EXPAND(op)); \
    CASE( 4 + addx, EXPAND(op)); \
    CASE( 5 + addx, EXPAND(op)); \
    CASE( 6 + addx, EXPAND(op)); \
    CASE( 7 + addx, EXPAND(op)); \
    CASE( 8 + addx, EXPAND(op)); \
    CASE( 9 + addx, EXPAND(op)); \
    CASE(10 + addx, EXPAND(op)); \
    CASE(11 + addx, EXPAND(op)); \
    CASE(12 + addx, EXPAND(op)); \
    CASE(13 + addx, EXPAND(op)); \
    CASE(14 + addx, EXPAND(op)); \
    CASE(15 + addx, EXPAND(op)); \
    CASE(16 + addx, EXPAND(op)); \
    CASE(17 + addx, EXPAND(op)); \
    CASE(18 + addx, EXPAND(op)); \
    CASE(19 + addx, EXPAND(op)); \
    CASE(20 + addx, EXPAND(op)); \
    CASE(21 + addx, EXPAND(op)); \
    CASE(22 + addx, EXPAND(op)); \
    CASE(23 + addx, EXPAND(op)); \
    CASE(24 + addx, EXPAND(op)); \
    CASE(25 + addx, EXPAND(op)); \
    CASE(26 + addx, EXPAND(op)); \
    CASE(27 + addx, EXPAND(op)); \
    CASE(28 + addx, EXPAND(op)); \
    CASE(29 + addx, EXPAND(op)); \
    CASE(30 + addx, EXPAND(op)); \
    CASE(31 + addx, EXPAND(op)); \

namespace xsimd
{
    /********************
     * batch_bool<T, 4> *
     ********************/
    
    template <class T>
    struct simd_batch_traits<batch_bool<T, 4>>
    {
        using value_type = T;
        static constexpr std::size_t size = 4;
        using batch_type = batch<T, 4>;
        static constexpr std::size_t align = XSIMD_DEFAULT_ALIGNMENT;
    };

    template <class T>
    class batch_bool<T, 4> : public simd_batch_bool<batch_bool<T, 4>>
    {
    public:

        using simd_type = uint32x4_t;

        batch_bool();
        explicit batch_bool(bool b);
        batch_bool(bool b0, bool b1, bool b2, bool b3);
        batch_bool(const simd_type& rhs);
        template <class V>
        batch_bool(const batch_bool<V, 4>& rhs);

        batch_bool& operator=(const simd_type& rhs);

        operator simd_type() const;

        bool operator[](std::size_t index) const;

    private:

        simd_type m_value;
    };

    /********************
     * batch_bool<T, 2> *
     ********************/

    template <class T>
    struct simd_batch_traits<batch_bool<T, 2>>
    {
        using value_type = T;
        static constexpr std::size_t size = 2;
        using batch_type = batch<T, 2>;
        static constexpr std::size_t align = XSIMD_DEFAULT_ALIGNMENT;
    };

    template <class T>
    class batch_bool<T, 2> : public simd_batch_bool<batch_bool<T, 2>>
    {
    public:

        using simd_type = uint64x2_t;

        batch_bool();
        explicit batch_bool(bool b);
        batch_bool(bool b0, bool b1);
        batch_bool(const simd_type& rhs);
        template <class V>
        batch_bool(const batch_bool<V, 2>& rhs);

        batch_bool& operator=(const simd_type& rhs);

        operator simd_type() const;

        bool operator[](std::size_t index) const;

    private:

        simd_type m_value;
    };

    /***********************************
     * batch_bool<T, 4> implementation *
     ***********************************/

    template <class T>
    inline batch_bool<T, 4>::batch_bool()
    {
    }

    template <class T>
    inline batch_bool<T, 4>::batch_bool(bool b)
        : m_value(vdupq_n_u32(
            static_cast<uint32_t>(-(int)b))
        )
    {
    }

    template <class T>
    inline batch_bool<T, 4>::batch_bool(bool b1, bool b2, bool b3, bool b4)
        : m_value{
            static_cast<uint32_t>(-int(b1)),
            static_cast<uint32_t>(-int(b2)),
            static_cast<uint32_t>(-int(b3)),
            static_cast<uint32_t>(-int(b4))}
    {
    }

    template <class T>
    inline batch_bool<T, 4>::batch_bool(const simd_type& rhs)
        : m_value(rhs)
    {
    }

    template <class T>
    template <class V>
    inline batch_bool<T, 4>::batch_bool(const batch_bool<V, 4>& rhs)
        : m_value(static_cast<simd_type>(rhs))
    {
    }

    template <class T>
    inline batch_bool<T, 4>& batch_bool<T, 4>::operator=(const simd_type& rhs)
    {
        m_value = rhs;
        return *this;
    }

    template <class T>
    inline batch_bool<T, 4>::operator uint32x4_t() const
    {
        return m_value;
    }

    template <class T>
    inline bool batch_bool<T, 4>::operator[](std::size_t index) const
    {
        return static_cast<bool>(m_value[index]);
    }

    namespace detail
    {
        template <class T>
        struct batch_bool_kernel<T, 4>
        {
            using batch_type = batch_bool<T, 4>;

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
                return vandq_u32(lhs, rhs);
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
                return vorrq_u32(lhs, rhs);
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
                return veorq_u32(lhs, rhs);
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
                return vmvnq_u32(rhs);
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
                return vbicq_u32(lhs, rhs);
            }

            static batch_type equal(const batch_type& lhs, const batch_type& rhs)
            {
                return vceqq_u32(lhs, rhs);
            }

            static batch_type not_equal(const batch_type& lhs, const batch_type& rhs)
            {
                return veorq_u32(lhs, rhs);
            }

            static bool all(const batch_type& rhs)
            {
                uint32x2_t tmp = vand_u32(vget_low_u32(rhs), vget_high_u32(rhs));
                return vget_lane_u32(vpmin_u32(tmp, tmp), 0) != 0;
            }

            static bool any(const batch_type& rhs)
            {
                uint32x2_t tmp = vorr_u32(vget_low_u32(rhs), vget_high_u32(rhs));
                return vget_lane_u32(vpmax_u32(tmp, tmp), 0);
            }
        };
    }

    /***********************************
     * batch_bool<T, 2> implementation *
     ***********************************/

    template <class T>
    inline batch_bool<T, 2>::batch_bool()
    {
    }

    template <class T>
    inline batch_bool<T, 2>::batch_bool(bool b)
        : m_value(vdupq_n_u64(static_cast<uint64_t>(-(int)b)))
    {
    }

    template <class T>
    inline batch_bool<T, 2>::batch_bool(bool b1, bool b2)
        : m_value{
            static_cast<uint64_t>(-int(b1)),
            static_cast<uint64_t>(-int(b2))}
    {
    }

    template <class T>
    inline batch_bool<T, 2>::batch_bool(const simd_type& rhs)
        : m_value(rhs)
    {
    }

    template <class T>
    template <class V>
    inline batch_bool<T, 2>::batch_bool(const batch_bool<V, 2>& rhs)
        : m_value(static_cast<simd_type>(rhs))
    {
    }

    template <class T>
    inline batch_bool<T, 2>& batch_bool<T, 2>::operator=(const simd_type& rhs)
    {
        m_value = rhs;
        return *this;
    }

    template <class T>
    inline batch_bool<T, 2>::operator uint64x2_t() const
    {
        return m_value;
    }

    template <class T>
    inline bool batch_bool<T, 2>::operator[](std::size_t index) const
    {
        return static_cast<bool>(m_value[index]);
    }

    namespace detail
    {
        template <class T>
        struct batch_bool_kernel<T, 2>
        {
            using batch_type = batch_bool<T, 2>;

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
                return vandq_u64(lhs, rhs);
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
                return vorrq_u64(lhs, rhs);
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
                return veorq_u64(lhs, rhs);
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
                return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(rhs)));
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
                // According to Eigen
                return vbicq_u64(lhs, rhs);
            }

            static batch_type equal(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
                return vceqq_u64(lhs, rhs);
#else
                return vreinterpretq_u64_u32(vceqq_u32(vreinterpretq_u32_u64(lhs), vreinterpretq_u32_u64(rhs)));
#endif
            }

            static batch_type not_equal(const batch_type& lhs, const batch_type& rhs)
            {
                return veorq_u64(lhs, rhs);
            }

            static bool all(const batch_type& rhs)
            {
                uint64x1_t tmp = vand_u64(vget_low_u64(rhs), vget_high_u64(rhs));
                return vget_lane_u64(tmp, 0) != 0;
            }

            static bool any(const batch_type& rhs)
            {
                uint64x1_t tmp = vorr_u64(vget_low_u64(rhs), vget_high_u64(rhs));
                return bool(vget_lane_u64(tmp, 0));
            }
        };
    }
}
#endif
